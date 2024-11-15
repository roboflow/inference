import hashlib
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import List, Literal, Optional, Tuple, Type, Union
from inference.core.managers.metrics import get_system_info
import supervision as sv
from fastapi import BackgroundTasks
from pydantic import ConfigDict, Field

from inference.core.cache.base import BaseCache
from inference.core.roboflow_api import (
    export_inference_to_model_monitoring,
    get_roboflow_workspace,
)
from inference.core.env import DEVICE_ID
from inference.core.workflows.execution_engine.constants import INFERENCE_ID_KEY
from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    CLASSIFICATION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    STRING_KIND,
    StepOutputSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)


LAST_REPORT_TIME_CACHE_KEY = "roboflow_model_monitoring_last_report_time"
WORKSPACE_NAME_CACHE_EXPIRE = 900  # 15 min

SHORT_DESCRIPTION = "Periodically export inference results to Roboflow Model Monitoring"

LONG_DESCRIPTION = """
Block allows users to report inference results to Roboflow Model Monitoring.

This is useful for reporting results to Model Monitoring when using InferencePipeline, which
does not automatically report results to Model Monitoring. This block only reports back a sample
of the results, which are specified by the `frequency` parameter.

For more information on Model Monitoring at Roboflow, see https://docs.roboflow.com/deploy/model-monitoring.
"""

BLOCK_NAME = "Roboflow Model Monitoring Exporter"


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": BLOCK_NAME,
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "sink",
        }
    )
    type: Literal[
        "roboflow_core/roboflow_model_monitoring_exporter@v1",
        BLOCK_NAME,
    ]
    predictions: StepOutputSelector(
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
        WorkflowParameterSelector(kind=[STRING_KIND]),
        StepOutputSelector(kind=[STRING_KIND]),
    ] = Field(
        description="Frequency of reporting (in seconds). For example, if 5 is provided, the block will report an inference result to Model Monitoring every 5 seconds.",
        examples=["3", "5"],
    )
    fire_and_forget: Union[bool, WorkflowParameterSelector(kind=[BOOLEAN_KIND])] = (
        Field(
            default=True,
            description="Boolean flag dictating if sink is supposed to be executed in the background, "
            "not waiting on status of registration before end of workflow run. Use `True` if best-effort "
            "registration is needed, use `False` while debugging and if error handling is needed",
            examples=[True],
        )
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="error_status", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="message", kind=[STRING_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class RoboflowModelMonitoringExporterBlockV1(WorkflowBlock):

    def __init__(
        self,
        cache: BaseCache,
        api_key: Optional[str],
        background_tasks: Optional[BackgroundTasks],
        thread_pool_executor: Optional[ThreadPoolExecutor],
        inference_results_batch_size: int = 10,
    ):
        self._api_key = api_key
        self._cache = cache
        self._background_tasks = background_tasks
        self._thread_pool_executor = thread_pool_executor
        self.inference_results_batch_size = inference_results_batch_size

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["api_key", "cache", "background_tasks", "thread_pool_executor"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def is_in_reporting_range(self, frequency: int) -> bool:
        now = datetime.now()
        last_report_time_str = self._cache.get(LAST_REPORT_TIME_CACHE_KEY)
        print(
            f"debug info last_report_time_str: {last_report_time_str}, now: {now}, frequency: {frequency}"
        )
        if last_report_time_str is None:
            self._cache.set(LAST_REPORT_TIME_CACHE_KEY, now.isoformat())
            last_report_time = now
        else:
            last_report_time = datetime.fromisoformat(last_report_time_str)
        print(
            f"now: {now}, last_report_time: {last_report_time}, diff: {now - last_report_time}, seconds: {int((now - last_report_time).total_seconds())}"
        )
        time_elapsed = int((now - last_report_time).total_seconds())
        print(
            f"Is in reporting range: last_time: {last_report_time}, current_time: {now}, difference: {time_elapsed} >= {frequency}"
        )
        return time_elapsed >= int(frequency)

    def run(
        self,
        fire_and_forget: bool,
        predictions: Union[sv.Detections, dict],
        frequency: int = 3,
    ) -> BlockResult:
        if not self.is_in_reporting_range(frequency):
            return {
                "error_status": False,
                "message": "Not in reporting range, skipping report. (Ok)",
            }
        if self._api_key is None:
            raise ValueError(
                "RoboflowModelMonitoringExporter block cannot run without Roboflow API key. "
                "If you do not know how to get API key - visit "
                "https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key to learn how to "
                "retrieve one."
            )
        registration_task = partial(
            export_to_model_monitoring_request,
            cache=self._cache,
            api_key=self._api_key,
            predictions=predictions,
        )
        error_status = False
        message = "Reporting happens in the background task"
        if fire_and_forget and self._background_tasks:
            self._background_tasks.add_task(registration_task)
        elif fire_and_forget and self._thread_pool_executor:
            self._thread_pool_executor.submit(registration_task)
        else:
            error_status, message = registration_task()
        self._cache.set(LAST_REPORT_TIME_CACHE_KEY, datetime.now().isoformat())
        return {
            "error_status": error_status,
            "message": message,
        }


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
    cache.set(
        key=cache_key, value=workspace_name_from_api, expire=WORKSPACE_NAME_CACHE_EXPIRE
    )
    return workspace_name_from_api


def export_to_model_monitoring_request(
    cache: BaseCache,
    api_key: str,
    predictions: Union[sv.Detections, dict],
) -> Tuple[bool, str]:
    workspace_id = get_workspace_name(api_key=api_key, cache=cache)
    try:
        inference_data = {
            "timestamp": datetime.now().isoformat(),
            "source": "workflow",
            "source_info": BLOCK_NAME,
            "inference_results": [],
        }
        if DEVICE_ID:
            inference_data["device_id"] = DEVICE_ID
        system_info = get_system_info()
        if system_info:
            for key, value in system_info.items():
                inference_data[key] = value
        if isinstance(predictions, sv.Detections):
            inference_data["inference_results"] = (
                format_sv_detections_for_model_monitoring(predictions)
            )
        elif isinstance(predictions, dict):
            pass
        export_inference_to_model_monitoring(api_key, workspace_id, inference_data)
        cache.set(LAST_REPORT_TIME_CACHE_KEY, datetime.now().isoformat())
        return (
            False,
            "Data exported successfully",
        )
    except Exception as error:
        logging.warning(f"Could not upload inference data. Reason: {error}")
        return (
            True,
            f"Error while uploading inference data. Error type: {type(error)}. Details: {error}",
        )


def format_sv_detections_for_model_monitoring(
    detections: Union[sv.Detections, dict],
) -> List[dict]:
    results = []
    if isinstance(detections, sv.Detections):
        num_detections = len(detections.data.get("detection_id", []))
        for i in range(num_detections):
            formatted_det = {
                "class": detections.data.get("class_name", [""])[i],
                "confidence": (
                    detections.confidence[i]
                    if detections.confidence is not None
                    else ""
                ),
                "inference_id": detections.data.get("inference_id", [""])[i],
                "model_type": detections.data.get("prediction_type", [""])[i],
            }
            results.append(formatted_det)
    elif isinstance(detections, dict):
        # TODO: when are detections of type dict?
        logging.warning(f"Detections are not sv.Detections. Detections: {detections}")
    return results
