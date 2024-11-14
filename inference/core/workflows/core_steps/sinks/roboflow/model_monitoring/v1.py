import hashlib
import json
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import List, Literal, Optional, Tuple, Type, Union

import numpy as np
import supervision as sv
from fastapi import BackgroundTasks
from pydantic import ConfigDict, Field

from inference.core.cache.base import BaseCache
from inference.core.roboflow_api import report_inference_to_model_monitoring
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

LAST_INFERENCES_CACHE_KEY = (
    "roboflow_model_monitoring_reporting_recent_inference_results"
)
LAST_REPORT_TIME_CACHE_KEY = "roboflow_model_monitoring_last_report_time"

SHORT_DESCRIPTION = "Add custom metadata to Roboflow Model Monitoring dashboard"

LONG_DESCRIPTION = """
Block allows users to add custom metadata to each inference result in Roboflow Model Monitoring dashboard.

This is useful for adding information specific to your use case. For example, if you want to be able to
filter inferences by a specific label such as location, you can attach those labels to each inference with this block.

For more information on Model Monitoring at Roboflow, see https://docs.roboflow.com/deploy/model-monitoring.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Model Monitoring Reporting",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "sink",
        }
    )
    type: Literal[
        "roboflow_core/roboflow_model_monitoring_reporting@v1",
        "RoboflowModelMonitoringReporting",
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
    # TODO: maybe add the ability to specify only reporting if a certain detection is present
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


class RoboflowModelMonitoringReportingBlockV1(WorkflowBlock):

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
        self._latest_inference_results = []

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["api_key", "cache", "background_tasks", "thread_pool_executor"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def _reset_buffer(self):
        self._latest_inference_results = []

    def _append_to_buffer(self, inference_data: dict):
        self._latest_inference_results.append(inference_data)
        if len(self._latest_inference_results) > self.inference_results_batch_size:
            self._latest_inference_results = self._latest_inference_results[
                -self.inference_results_batch_size :
            ]
        print(f"Latest inference results: {len(self._latest_inference_results)}")

    def is_in_reporting_range(self, frequency: int) -> bool:
        now = datetime.now()
        last_report_time_str = self._cache.get(LAST_REPORT_TIME_CACHE_KEY)
        if last_report_time_str is None:
            self._cache.set(LAST_REPORT_TIME_CACHE_KEY, now.isoformat())
            last_report_time = now
        else:
            print("Getting last report time")
            last_report_time = datetime.fromisoformat(last_report_time_str)
        difference = int((now - last_report_time).total_seconds())
        return difference < int(frequency)

    def run(
        self,
        fire_and_forget: bool,
        predictions: sv.Detections,
        frequency: int = 3,
    ) -> BlockResult:
        if predictions is not None and len(predictions) > 0:
            self._append_to_buffer(predictions)
        if not self.is_in_reporting_range(frequency):
            return {
                "error_status": False,
                "message": "Not in reporting range, skipping report. (Ok)",
            }
        if self._api_key is None:
            raise ValueError(
                "RoboflowModelMonitoringReporting block cannot run without Roboflow API key. "
                "If you do not know how to get API key - visit "
                "https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key to learn how to "
                "retrieve one."
            )
        inference_ids: List[str] = predictions.data.get(INFERENCE_ID_KEY, [])
        if len(inference_ids) == 0:
            return {
                "error_status": True,
                "message": "RoboflowModelMonitoringReporting block cannot run without inference_ids. "
                "This is known bug (https://github.com/roboflow/inference/issues/567). "
                "Please provide a report for the problem under mentioned issue.",
            }
        inference_ids: List[str] = list(set(inference_ids))
        # registration_task = partial(
        #     report_to_model_monitoring_request,
        #     cache=self._cache,
        #     api_key=self._api_key,
        #     inference_data=self._latest_inference_results,
        # )
        error_status = False
        message = "Reporting happens in the background task"
        print(f"Reporting task")
        error_status, message = report_to_model_monitoring_request(
            cache=self._cache,
            api_key=self._api_key,
            inference_data=self._latest_inference_results,
        )
        self._reset_buffer()
        # self._last_report_time = now
        # self._cache.set(LAST_REPORT_TIME_CACHE_KEY, now.isoformat())
        return {
            "error_status": error_status,
            "message": message,
        }


def report_to_model_monitoring_request(
    cache: BaseCache,
    api_key: str,
    inference_data: dict,
) -> Tuple[bool, str]:
    try:
        # report_inference_to_model_monitoring(
        #     api_key=api_key,
        #     inference_data=inference_data,
        # )
        print(f"Inference data: {inference_data}")
        cache.set(LAST_REPORT_TIME_CACHE_KEY, datetime.now().isoformat())
        return (
            False,
            "Inference data upload was successful",
        )
    except Exception as error:
        logging.warning(f"Could not upload inference data. Reason: {error}")
        return (
            True,
            f"Error while uploading inference data. Error type: {type(error)}. Details: {error}",
        )
