from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from typing import Literal, Optional, Type, Union

import supervision as sv
from fastapi import BackgroundTasks

from inference.core.cache.base import BaseCache
from inference.core.workflows.core_steps.sinks.noop import (
    DisableSink,
    disabled_sink_response,
    versioned_sink_manifest_config,
)
from inference.core.workflows.core_steps.sinks.roboflow.model_monitoring_inference_aggregator.v1 import (
    BlockManifest as BlockManifestV1,
)
from inference.core.workflows.core_steps.sinks.roboflow.model_monitoring_inference_aggregator.v1 import (
    ModelMonitoringInferenceAggregatorBlockV1,
    PredictionsAggregator,
    send_to_model_monitoring_request,
)
from inference.core.workflows.prototypes.block import BlockResult, WorkflowBlockManifest


class BlockManifest(BlockManifestV1):
    model_config = versioned_sink_manifest_config(BlockManifestV1, version="v2")
    type: Literal["roboflow_core/model_monitoring_inference_aggregator@v2"]
    disable_sink: DisableSink = False


class ModelMonitoringInferenceAggregatorBlockV2(
    ModelMonitoringInferenceAggregatorBlockV1
):
    def __init__(
        self,
        cache: BaseCache,
        api_key: Optional[str],
        background_tasks: Optional[BackgroundTasks],
        thread_pool_executor: Optional[ThreadPoolExecutor],
    ):
        self._api_key = api_key
        self._cache = cache
        self._background_tasks = background_tasks
        self._thread_pool_executor = thread_pool_executor
        self._predictions_aggregator = PredictionsAggregator()

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
        disable_sink: bool = False,
    ) -> BlockResult:
        if disable_sink:
            return disabled_sink_response()
        if self._api_key is None:
            raise ValueError(
                "ModelMonitoringInferenceAggregator block cannot run without Roboflow API key. "
                "If you do not know how to get API key - visit "
                "https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key to learn how to "
                "retrieve one."
            )
        self._last_report_time_cache_key = (
            "workflows:steps_cache:"
            "roboflow_core/model_monitoring_inference_aggregator@v2:"
            f"{unique_aggregator_key}:last_report_time"
        )
        if predictions:
            self._predictions_aggregator.collect(predictions, model_id)
        if not self._is_in_reporting_range(frequency):
            return {
                "error_status": False,
                "message": "Not in reporting range, skipping report. (Ok)",
            }
        predictions_to_report = self._predictions_aggregator.get_and_flush()
        registration_task = partial(
            send_to_model_monitoring_request,
            cache=self._cache,
            last_report_time_cache_key=self._last_report_time_cache_key,
            api_key=self._api_key,
            predictions=predictions_to_report,
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
