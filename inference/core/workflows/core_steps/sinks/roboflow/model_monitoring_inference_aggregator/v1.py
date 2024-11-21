import hashlib
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from typing import Any, List, Literal, Optional, Tuple, Type, Union

import supervision as sv
from fastapi import BackgroundTasks
from pydantic import ConfigDict, Field, field_validator

from inference.core.cache.base import BaseCache
from inference.core.env import DEVICE_ID
from inference.core.managers.metrics import get_system_info
from inference.core.roboflow_api import (
    get_roboflow_workspace,
    send_inference_results_to_model_monitoring,
)
from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    CLASSIFICATION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

SHORT_DESCRIPTION = "Periodically report an aggregated sample of inference results to Roboflow Model Monitoring"

LONG_DESCRIPTION = """
This block periodically reports an aggregated sample of inference results to Roboflow Model Monitoring.

It aggregates predictions in memory between reports and then sends a representative sample of predictions at a regular interval specified by the `frequency` parameter.

This is particularly useful when using InferencePipeline, which doesn't automatically report results to Model Monitoring.

For more details on Model Monitoring at Roboflow, visit: https://docs.roboflow.com/deploy/model-monitoring.
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
        }
    )
    type: Literal["roboflow_core/model_monitoring_inference_aggregator@v1",]
    predictions: Selector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
            KEYPOINT_DETECTION_PREDICTION_KIND,
            CLASSIFICATION_PREDICTION_KIND,
        ]
    ) = Field(
        description="Reference data to extract property from",
        examples=["$steps.my_step.predictions"],
    )
    frequency: Union[
        str,
        Selector(kind=[STRING_KIND]),
    ] = Field(
        default=5,
        description="Frequency of reporting (in seconds). For example, if 5 is provided, the block will report an aggregated sample of predictions every 5 seconds.",
        examples=["3", "5"],
    )
    fire_and_forget: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=True,
        description="Boolean flag dictating if sink is supposed to be executed in the background, "
        "not waiting on status of registration before end of workflow run. Use `True` if best-effort "
        "registration is needed, use `False` while debugging and if error handling is needed",
        examples=[True],
    )

    @field_validator("frequency")
    @classmethod
    def ensure_frequency_is_correct(cls, value: Any) -> Any:
        if isinstance(value, int) and value < 1:
            raise ValueError("`frequency` cannot be lower than 1.")
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


@dataclass
class Prediction(object):
    class_name: str
    confidence: float
    inference_id: str
    model_type: str


class PredictionsAggregator(object):
    _raw_predictions: List[Union[sv.Detections, dict]]

    def __init__(self):
        self._raw_predictions = []

    def collect(self, value: Union[sv.Detections, dict]) -> None:
        self._raw_predictions.append(value)

    def _consolidate(self) -> List[Prediction]:
        formatted_predictions = []
        for detections in self._raw_predictions:
            f = format_sv_detections_for_model_monitoring(detections)
            formatted_predictions.extend(f)

        class_groups = defaultdict(list)
        for prediction in formatted_predictions:
            class_name = prediction["class_name"]
            class_groups[class_name].append(prediction)

        representative_predictions = []
        for class_name, predictions in class_groups.items():
            predictions.sort(key=lambda x: x["confidence"], reverse=True)
            representative_predictions.append(predictions[0])

        return representative_predictions

    def get_and_flush(self) -> List[Prediction]:
        predictions = self._consolidate()
        self._raw_predictions = []
        return predictions


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
        self._last_report_time_cache_key = "roboflow_model_monitoring_last_report_time"
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
    ) -> BlockResult:
        if predictions:
            self._predictions_aggregator.collect(predictions)
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
    predictions: List[Prediction],
) -> Tuple[bool, str]:
    workspace_id = get_workspace_name(api_key=api_key, cache=cache)
    try:
        inference_data = {
            "timestamp": datetime.now().isoformat(),
            "source": "workflow",
            "source_info": "ModelMonitoringInferenceAggregatorBlockV1",
            "inference_results": [],
        }
        if DEVICE_ID:
            inference_data["device_id"] = DEVICE_ID
        system_info = get_system_info()
        if system_info:
            for key, value in system_info.items():
                inference_data[key] = value
        inference_data["inference_results"] = predictions
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


def format_sv_detections_for_model_monitoring(
    detections: Union[sv.Detections, dict],
) -> List[Prediction]:
    results = []
    if isinstance(detections, sv.Detections):
        num_detections = len(detections.data.get("detection_id", []))
        for i in range(num_detections):
            prediction = Prediction(
                class_name=detections.data.get("class_name", [""])[i],
                confidence=(
                    detections.confidence[i]
                    if detections.confidence is not None
                    else 0.0
                ),
                inference_id=detections.data.get("inference_id", [""])[i],
                model_type=detections.data.get("prediction_type", [""])[i],
            )
            results.append(prediction.__dict__)
    elif isinstance(detections, dict):
        predictions = detections.get("predictions", [])
        if isinstance(predictions, list):
            for prediction in predictions:
                pred_instance = Prediction(
                    class_name=prediction.get("class", ""),
                    confidence=prediction.get("confidence", 0.0),
                    inference_id=detections.get("inference_id", ""),
                    model_type=detections.get("prediction_type", ""),
                )
                results.append(pred_instance.__dict__)
        elif isinstance(predictions, dict):
            for class_name, details in predictions.items():
                pred_instance = Prediction(
                    class_name=class_name,
                    confidence=details.get("confidence", 0.0),
                    inference_id=detections.get("inference_id", ""),
                    model_type=detections.get("prediction_type", ""),
                )
                results.append(pred_instance.__dict__)
    return results
