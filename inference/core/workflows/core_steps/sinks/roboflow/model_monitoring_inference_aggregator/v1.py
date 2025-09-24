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
This block 📊 **transforms inference data reporting** to a whole new level by 
periodically aggregating and sending a curated sample of predictions to 
**[Roboflow Model Monitoring](https://docs.roboflow.com/deploy/model-monitoring)**.

#### ✨ Key Features
* **Effortless Aggregation:** Collects and organizes predictions in-memory, ensuring only the most relevant 
and confident predictions are reported.

* **Customizable Reporting Intervals:** Choose how frequently (in seconds) data should be sent—ensuring 
optimal balance between granularity and resource efficiency.

* **Debug-Friendly Mode:** Fine-tune operations by enabling or disabling asynchronous background execution.

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

* This block do not have ability to separate aggregations for multiple videos processed by `InferencePipeline` - 
effectively aggregating data for **all video feeds connected to single process running `InferencePipeline`**. 
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
        description="Model predictions to report to Roboflow Model Monitoring.",
        examples=["$steps.my_step.predictions"],
    )
    model_id: Selector(kind=[ROBOFLOW_MODEL_ID_KIND]) = Field(
        description="Model ID to report to Roboflow Model Monitoring.",
        examples=["my_project/3"],
    )
    frequency: Union[
        int,
        Selector(kind=[STRING_KIND]),
    ] = Field(
        default=5,
        description="Frequency of reporting (in seconds). For example, if 5 is provided, the "
        "block will report an aggregated sample of predictions every 5 seconds.",
        examples=["3", "5"],
    )
    unique_aggregator_key: str = Field(
        description="Unique key used internally to track the session of inference results reporting. "
        "Must be unique for each step in your Workflow.",
        examples=["session-1v73kdhfse"],
        json_schema_extra={"hidden": True},
    )
    fire_and_forget: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=True,
        description="Boolean flag to run the block asynchronously (True) for faster workflows or  "
        "synchronously (False) for debugging and error handling.",
        examples=[True],
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
