import hashlib
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import supervision as sv
from fastapi import BackgroundTasks
from pydantic import BaseModel, ConfigDict, Field, field_validator

from inference.core.cache.base import BaseCache
from inference.core.env import DEVICE_ID
from inference.core.managers.metrics import get_system_info
from inference.core.roboflow_api import (
    get_roboflow_workspace,
    send_inference_results_to_model_monitoring,
)
from inference.core.version import __version__
from inference.core.workflows.execution_engine.constants import (
    CLASS_NAME_KEY,
    INFERENCE_ID_KEY,
    PREDICTION_TYPE_KEY,
)
from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    CLASSIFICATION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    ROBOFLOW_MODEL_ID_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

SHORT_DESCRIPTION = "Periodically report an aggregated sample of inference results to Roboflow Model Monitoring."

LONG_DESCRIPTION = """
Periodically aggregate and report a curated sample of inference predictions to Roboflow Model Monitoring by collecting predictions in memory, grouping by class, selecting the most confident prediction per class, and sending aggregated results at configurable intervals to enable efficient video processing monitoring, production analytics, and model performance tracking workflows with minimal performance overhead.

## How This Block Works

This block aggregates predictions over time and sends representative samples to Roboflow Model Monitoring at regular intervals, reducing API calls and maintaining video processing performance. The block:

1. Receives predictions and configuration:
   - Takes predictions from any supported model type (object detection, instance segmentation, keypoint detection, or classification)
   - Receives model ID for identification in Model Monitoring
   - Accepts frequency parameter specifying reporting interval in seconds
   - Receives execution mode flag (fire-and-forget)
2. Validates Roboflow API key:
   - Checks that a valid Roboflow API key is available (required for API access)
   - Raises an error if API key is missing with instructions on how to retrieve one
3. Collects predictions in memory:
   - Stores predictions in an in-memory aggregator organized by model ID
   - Accumulates predictions between reporting intervals
   - Maintains state for the duration of the workflow execution session
4. Checks reporting interval:
   - Uses cache to track last report time based on unique aggregator key
   - Calculates time elapsed since last report
   - Compares elapsed time to configured frequency threshold
   - Skips reporting if interval has not been reached (returns status message)
5. Consolidates predictions when reporting:
   - Formats all collected predictions for Model Monitoring
   - Groups predictions by class name across all collected data
   - For each class, sorts predictions by confidence (highest first)
   - Selects the most confident prediction per class as representative sample
   - Creates a curated set of predictions (one per class with highest confidence)
6. Retrieves workspace information:
   - Gets workspace ID from Roboflow API using the provided API key
   - Uses caching (15-minute expiration) to avoid repeated API calls
   - Caches workspace name using MD5 hash of API key as cache key
7. Sends aggregated data to Model Monitoring:
   - Constructs inference data payload with timestamp, source info, device ID, and server version
   - Includes system information (if available) for monitoring context
   - Sends aggregated predictions (one per class) to Roboflow Model Monitoring API
   - Flushes in-memory aggregator after sending (starts fresh collection)
   - Updates last report time in cache
8. Executes synchronously or asynchronously:
   - **Asynchronous mode (fire_and_forget=True)**: Submits task to background thread pool or FastAPI background tasks, allowing workflow to continue without waiting for API call to complete
   - **Synchronous mode (fire_and_forget=False)**: Waits for API call to complete and returns immediate status, useful for debugging and error handling
9. Returns status information:
   - Outputs error_status indicating success (False) or failure (True)
   - Outputs message with reporting status or error details
   - Provides feedback on whether aggregation was sent or skipped

The block is optimized for video processing workflows where sending every prediction would create excessive API calls and impact performance. By aggregating predictions and selecting representative samples (most confident per class), the block provides meaningful monitoring data while minimizing overhead. The interval-based reporting ensures regular updates to Model Monitoring without constant API calls.

## Common Use Cases

#### 🔍 Why Use This Block?

This block is a game-changer for projects relying on video processing in Workflows. 
With its aggregation process, it identifies the most confident predictions across classes and sends 
them at regular intervals in small messages to Roboflow backend - ensuring that video processing 
performance is impacted to the least extent.

Perfect for:

* Monitoring production line performance in real-time 🏭.

* Debugging and validating your model’s performance over time ⏱️.

* Providing actionable insights from inference workflows with minimal overhead 🔧.

#### 🚨 Limitations

* The block is should not be relied on when running Workflow in `inference` server or via HTTP request to Roboflow 
hosted platform, as the internal state is not persisted in a memory that would be accessible for all requests to
the server, causing aggregation to **only have a scope of single request**. We will solve that problem in future 
releases if proven to be serious limitation for clients.

## Connecting to Other Blocks

This block receives predictions and outputs status information:

- **After model blocks** (Object Detection Model, Instance Segmentation Model, Classification Model, Keypoint Detection Model) to aggregate and report predictions to Model Monitoring (e.g., aggregate detection results, report classification outputs, monitor model predictions), enabling model-to-monitoring workflows
- **After filtering or analytics blocks** (DetectionsFilter, ContinueIf, OverlapFilter) to aggregate filtered or analyzed results for monitoring (e.g., aggregate filtered detections, report analytics results, monitor processed predictions), enabling analysis-to-monitoring workflows
- **In video processing workflows** to efficiently monitor video analysis with minimal performance impact (e.g., aggregate video frame detections, report video processing results, monitor video analysis performance), enabling video monitoring workflows
- **After preprocessing or transformation blocks** to monitor transformed predictions (e.g., aggregate transformed detections, report processed results, monitor transformation outputs), enabling transformation-to-monitoring workflows
- **In production deployment workflows** to track model performance in production environments (e.g., monitor production inference, track deployment performance, report production metrics), enabling production monitoring workflows
- **As a sink block** to send aggregated monitoring data without blocking workflow execution (e.g., background monitoring reporting, non-blocking analytics, efficient data collection), enabling sink-to-monitoring workflows

## Requirements

This block requires a valid Roboflow API key configured in the environment or workflow configuration. The API key is required to authenticate with Roboflow API and access Model Monitoring features. Visit https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key to learn how to retrieve an API key. The block maintains in-memory state for aggregation, which means it works best for long-running workflows (like video processing with InferencePipeline). The block should not be relied upon when running workflows in inference server or via HTTP requests to Roboflow hosted platform, as the internal state is only accessible for single requests and aggregation scope is limited to single request execution. The block aggregates data for all video feeds connected to a single InferencePipeline process (cannot separate aggregations per video feed). The frequency parameter must be at least 1 second. For more information on Model Monitoring at Roboflow, see https://docs.roboflow.com/deploy/model-monitoring.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Model Monitoring Inference Aggregator",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "sink",
            "ui_manifest": {
                "section": "advanced",
                "icon": "fal fa-chart-line",
                "blockPriority": 8.5,
                "requires_rf_key": True,
            },
        }
    )
    type: Literal["roboflow_core/model_monitoring_inference_aggregator@v1"]
    predictions: Selector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
            KEYPOINT_DETECTION_PREDICTION_KIND,
            CLASSIFICATION_PREDICTION_KIND,
        ]
    ) = Field(
        description="Model predictions (object detection, instance segmentation, keypoint detection, or classification) to aggregate and report to Roboflow Model Monitoring. Predictions are collected in memory, grouped by class name, and the most confident prediction per class is selected as a representative sample. Predictions accumulate between reporting intervals based on the frequency setting. Supported prediction types: supervision Detections objects or classification prediction dictionaries.",
        examples=["$steps.object_detection.predictions", "$steps.classification.predictions", "$steps.instance_segmentation.predictions"],
    )
    model_id: Selector(kind=[ROBOFLOW_MODEL_ID_KIND]) = Field(
        description="Roboflow model ID (format: 'project/version') to associate with the predictions in Model Monitoring. This identifies which model generated the predictions being reported. The model ID is included in the monitoring data sent to Roboflow, allowing you to track performance per model in the Model Monitoring dashboard.",
        examples=["my_project/3", "production_model/1", "detection_model/5"],
    )
    frequency: Union[
        int,
        Selector(kind=[STRING_KIND]),
    ] = Field(
        default=5,
        description="Reporting frequency in seconds. Specifies how often aggregated predictions are sent to Roboflow Model Monitoring. For example, if set to 5, the block collects predictions for 5 seconds, then sends the aggregated sample (one most confident prediction per class) to Model Monitoring. Must be at least 1 second. Lower values provide more frequent updates but increase API calls. Higher values reduce API calls but provide less frequent updates. Default: 5 seconds. Works well for video processing where you want regular but not excessive reporting.",
        examples=[3, 5, 10, 30, 60],
    )
    unique_aggregator_key: str = Field(
        description="Unique key used internally to track the aggregation session and cache last report time. This key must be unique for each instance of this block in your workflow. The key is used to create cache entries that track when the last report was sent, enabling interval-based reporting. This field is automatically generated and hidden in the UI.",
        examples=["session-1v73kdhfse"],
        json_schema_extra={"hidden": True},
    )
    fire_and_forget: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=True,
        description="Execution mode flag. When True (default), the block runs asynchronously in the background, allowing the workflow to continue processing without waiting for the API call to complete. This provides faster workflow execution but errors are not immediately available. When False, the block runs synchronously and waits for the API call to complete, returning immediate status and error information. Use False for debugging and error handling, True for production workflows where performance is prioritized.",
        examples=[True, False],
    )

    @field_validator("frequency")
    @classmethod
    def ensure_frequency_is_correct(cls, value: Any) -> Any:
        if isinstance(value, int) and value < 1:
            raise ValueError("`frequency` cannot be lower than 1.")
        return value

    @field_validator("model_id")
    @classmethod
    def ensure_model_id_is_correct(cls, value: Any) -> Any:
        if isinstance(value, str) and value == "":
            raise ValueError("`model_id` cannot be empty.")
        return value

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="error_status", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="message", kind=[STRING_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class ParsedPrediction(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=(),
    )

    model_id: str
    class_name: str
    confidence: float
    inference_id: str
    model_type: str


class PredictionsAggregator(object):

    def __init__(self):
        self._raw_predictions: dict[str, List[Union[sv.Detections, dict]]] = {}

    def collect(self, value: Union[sv.Detections, dict], model_id: str) -> None:
        # TODO: push into global state, otherwise for HTTP server use,
        #   state would at most have 1 prediction!!!
        if model_id not in self._raw_predictions:
            self._raw_predictions[model_id] = []
        self._raw_predictions[model_id].append(value)

    def get_and_flush(self) -> List[ParsedPrediction]:
        predictions = self._consolidate()
        self._raw_predictions = {}
        return predictions

    def _consolidate(self) -> List[ParsedPrediction]:
        formatted_predictions = []
        for model_id, predictions in self._raw_predictions.items():
            formatted_predictions.extend(
                format_predictions_for_model_monitoring(predictions, model_id)
            )
        class_groups: Dict[str, List[ParsedPrediction]] = defaultdict(list)
        for prediction in formatted_predictions:
            class_name = prediction.class_name
            class_groups[class_name].append(prediction)
        representative_predictions = []
        for class_name, predictions in class_groups.items():
            predictions.sort(key=lambda x: x.confidence, reverse=True)
            representative_predictions.append(predictions[0])
        return representative_predictions


class ModelMonitoringInferenceAggregatorBlockV1(WorkflowBlock):

    def __init__(
        self,
        cache: BaseCache,
        api_key: Optional[str],
        background_tasks: Optional[BackgroundTasks],
        thread_pool_executor: Optional[ThreadPoolExecutor],
    ):
        if api_key is None:
            raise ValueError(
                "ModelMonitoringInferenceAggregator block cannot run without Roboflow API key. "
                "If you do not know how to get API key - visit "
                "https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key to learn how to "
                "retrieve one."
            )
        self._api_key = api_key
        self._cache = cache
        self._background_tasks = background_tasks
        self._thread_pool_executor = thread_pool_executor
        self._predictions_aggregator = PredictionsAggregator()

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["api_key", "cache", "background_tasks", "thread_pool_executor"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        fire_and_forget: bool,
        predictions: Union[sv.Detections, dict],
        frequency: int,
        unique_aggregator_key: str,
        model_id: str,
    ) -> BlockResult:
        self._last_report_time_cache_key = f"workflows:steps_cache:roboflow_core/model_monitoring_inference_aggregator@v1:{unique_aggregator_key}:last_report_time"
        if predictions:
            self._predictions_aggregator.collect(predictions, model_id)
        if not self._is_in_reporting_range(frequency):
            return {
                "error_status": False,
                "message": "Not in reporting range, skipping report. (Ok)",
            }
        preds = self._predictions_aggregator.get_and_flush()
        registration_task = partial(
            send_to_model_monitoring_request,
            cache=self._cache,
            last_report_time_cache_key=self._last_report_time_cache_key,
            api_key=self._api_key,
            predictions=preds,
        )
        error_status = False
        message = "Reporting happens in the background task"
        if fire_and_forget and self._background_tasks:
            self._background_tasks.add_task(registration_task)
        elif fire_and_forget and self._thread_pool_executor:
            self._thread_pool_executor.submit(registration_task)
        else:
            error_status, message = registration_task()
        self._cache.set(self._last_report_time_cache_key, datetime.now().isoformat())
        return {
            "error_status": error_status,
            "message": message,
        }

    def _is_in_reporting_range(self, frequency: int) -> bool:
        now = datetime.now()
        last_report_time_str = self._cache.get(self._last_report_time_cache_key)
        if last_report_time_str is None:
            self._cache.set(self._last_report_time_cache_key, now.isoformat())
            v = self._cache.get(self._last_report_time_cache_key)
            last_report_time = now
        else:
            last_report_time = datetime.fromisoformat(last_report_time_str)
        time_elapsed = int((now - last_report_time).total_seconds())
        return time_elapsed >= int(frequency)


# TODO: maybe make this a helper or decorator, it's used in multiple places
def get_workspace_name(
    api_key: str,
    cache: BaseCache,
) -> str:
    api_key_hash = hashlib.md5(api_key.encode("utf-8")).hexdigest()
    cache_key = f"workflows:api_key_to_workspace:{api_key_hash}"
    cached_workspace_name = cache.get(cache_key)
    if cached_workspace_name:
        return cached_workspace_name
    workspace_name_from_api = get_roboflow_workspace(api_key=api_key)
    cache.set(key=cache_key, value=workspace_name_from_api, expire=900)
    return workspace_name_from_api


def send_to_model_monitoring_request(
    cache: BaseCache,
    last_report_time_cache_key: str,
    api_key: str,
    predictions: List[ParsedPrediction],
) -> Tuple[bool, str]:
    workspace_id = get_workspace_name(api_key=api_key, cache=cache)
    try:
        inference_data = {
            "timestamp": datetime.now().isoformat(),
            "source": "workflow",
            "source_info": "ModelMonitoringInferenceAggregatorBlockV1",
            "inference_results": [],
            "device_id": DEVICE_ID,
            "inference_server_version": __version__,
        }
        system_info = get_system_info()
        if system_info:
            for key, value in system_info.items():
                inference_data[key] = value
        inference_data["inference_results"] = [p.model_dump() for p in predictions]
        send_inference_results_to_model_monitoring(
            api_key, workspace_id, inference_data
        )
        cache.set(last_report_time_cache_key, datetime.now().isoformat())
        return (
            False,
            "Data sent successfully",
        )
    except Exception as error:
        logging.warning(f"Could not upload inference data. Reason: {error}")
        return (
            True,
            f"Error while uploading inference data. Error type: {type(error)}. Details: {error}",
        )


def format_predictions_for_model_monitoring(
    predictions_list: List[Union[sv.Detections, dict]],
    model_id: str,
) -> List[ParsedPrediction]:
    results = []
    for predictions in predictions_list:
        if isinstance(predictions, sv.Detections):
            for detection in predictions:
                _, _, confidence, _, _, data = detection
                prediction = ParsedPrediction(
                    class_name=data.get("class_name", ""),
                    confidence=(confidence if confidence is not None else 0.0),
                    inference_id=data.get(INFERENCE_ID_KEY, ""),
                    model_type=data.get(PREDICTION_TYPE_KEY, ""),
                    model_id=model_id,
                )
                results.append(prediction)
        elif isinstance(predictions, dict):
            detections = predictions.get("predictions", [])
            prediction_type = predictions.get(PREDICTION_TYPE_KEY, "")
            inference_id = predictions.get(INFERENCE_ID_KEY, "")
            if isinstance(detections, list):
                for d in detections:
                    pred_instance = ParsedPrediction(
                        class_name=d.get(CLASS_NAME_KEY, ""),
                        confidence=d.get("confidence", 0.0),
                        inference_id=inference_id,
                        model_type=prediction_type,
                        model_id=model_id,
                    )
                    results.append(pred_instance)
            elif isinstance(detections, dict):
                for class_name, details in detections.items():
                    pred_instance = ParsedPrediction(
                        class_name=class_name,
                        confidence=details.get("confidence", 0.0),
                        inference_id=inference_id,
                        model_type=prediction_type,
                        model_id=model_id,
                    )
                    results.append(pred_instance)
    return results
