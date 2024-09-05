import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import List, Literal, Optional, Tuple, Type, Union

import numpy as np
import supervision as sv
from fastapi import BackgroundTasks
from pydantic import ConfigDict, Field

from inference.core.cache.base import BaseCache
from inference.core.roboflow_api import add_custom_metadata, get_roboflow_workspace
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

SHORT_DESCRIPTION = "Add custom metadata to Roboflow Model Monitoring dashboard"

LONG_DESCRIPTION = """
Block allows users to add custom metadata to each inference result in Roboflow Model Monitoring dashboard.

This is useful for adding information specific to your use case. For example, if you want to be able to
filter inferences by a specific label such as location, you can attach those labels to each inference with this block.

For more information on Model Monitoring at Roboflow, see https://docs.roboflow.com/deploy/model-monitoring.
"""

WORKSPACE_NAME_CACHE_EXPIRE = 900  # 15 min


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Roboflow Custom Metadata",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "sink",
        }
    )
    type: Literal["roboflow_core/roboflow_custom_metadata@v1", "RoboflowCustomMetadata"]
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
    field_value: Union[
        str,
        WorkflowParameterSelector(kind=[STRING_KIND]),
        StepOutputSelector(kind=[STRING_KIND]),
    ] = Field(
        description="This is the name of the metadata field you are creating",
        examples=["toronto", "pass", "fail"],
    )
    field_name: str = Field(
        description="Name of the field to be updated in Roboflow Customer Metadata",
        examples=["The name of the value of the field"],
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


class RoboflowCustomMetadataBlockV1(WorkflowBlock):

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

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["api_key", "cache", "background_tasks", "thread_pool_executor"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        fire_and_forget: bool,
        field_name: str,
        field_value: str,
        predictions: sv.Detections,
    ) -> BlockResult:
        if self._api_key is None:
            raise ValueError(
                "RoboflowCustomMetadata block cannot run without Roboflow API key. "
                "If you do not know how to get API key - visit "
                "https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key to learn how to "
                "retrieve one."
            )
        inference_ids: List[str] = predictions.data.get(INFERENCE_ID_KEY, [])
        if len(inference_ids) == 0:
            return {
                "error_status": True,
                "message": "Custom metadata upload failed because no inference_ids were received. "
                "This is known bug (https://github.com/roboflow/inference/issues/567). "
                "Please provide a report for the problem under mentioned issue.",
            }
        inference_ids: List[str] = list(set(inference_ids))
        registration_task = partial(
            add_custom_metadata_request,
            cache=self._cache,
            api_key=self._api_key,
            inference_ids=inference_ids,
            field_name=field_name,
            field_value=field_value,
        )
        error_status = False
        message = "Registration happens in the background task"
        if fire_and_forget and self._background_tasks:
            self._background_tasks.add_task(registration_task)
        elif fire_and_forget and self._thread_pool_executor:
            self._thread_pool_executor.submit(registration_task)
        else:
            error_status, message = registration_task()
        return {
            "error_status": error_status,
            "message": message,
        }


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


def add_custom_metadata_request(
    cache: BaseCache,
    api_key: str,
    inference_ids: List[str],
    field_name: str,
    field_value: str,
) -> Tuple[bool, str]:
    workspace_id = get_workspace_name(api_key=api_key, cache=cache)
    try:
        add_custom_metadata(
            api_key=api_key,
            workspace_id=workspace_id,
            inference_ids=inference_ids,
            field_name=field_name,
            field_value=field_value,
        )
        return (
            False,
            "Custom metadata upload was successful",
        )
    except Exception as error:
        logging.warning(
            f"Could not add custom metadata for inference IDs: {inference_ids}. Reason: {error}"
        )
        return (
            True,
            f"Error while custom metadata registration. Error type: {type(error)}. Details: {error}",
        )
