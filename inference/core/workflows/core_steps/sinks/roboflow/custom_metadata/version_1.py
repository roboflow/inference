import hashlib
from functools import partial
from typing import List, Literal, Optional, Type, Union

import numpy as np
import supervision as sv
from fastapi import BackgroundTasks
from pydantic import ConfigDict, Field

from inference.core.cache.base import BaseCache
from inference.core.roboflow_api import add_custom_metadata, get_roboflow_workspace
from inference.core.workflows.execution_engine.constants import INFERENCE_ID_KEY
from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    BATCH_OF_CLASSIFICATION_PREDICTION_KIND,
    BATCH_OF_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    BATCH_OF_KEYPOINT_DETECTION_PREDICTION_KIND,
    BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND,
    BATCH_OF_STRING_KIND,
    BOOLEAN_KIND,
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
            BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND,
            BATCH_OF_INSTANCE_SEGMENTATION_PREDICTION_KIND,
            BATCH_OF_KEYPOINT_DETECTION_PREDICTION_KIND,
            BATCH_OF_CLASSIFICATION_PREDICTION_KIND,
        ]
    ) = Field(
        description="Reference data to extract property from",
        examples=["$steps.my_step.predictions"],
    )
    field_value: Union[
        str,
        WorkflowParameterSelector(kind=[STRING_KIND]),
        StepOutputSelector(kind=[BATCH_OF_STRING_KIND]),
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
        )
    )

    @classmethod
    def accepts_batch_input(cls) -> bool:
        return True

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="error_status", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="message", kind=[STRING_KIND]),
            OutputDefinition(
                name="predictions", kind=[BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND]
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return "~=1.0.0"


class RoboflowCustomMetadataBlockV1(WorkflowBlock):

    def __init__(
        self,
        cache: BaseCache,
        api_key: Optional[str],
        background_tasks: Optional[BackgroundTasks],
    ):
        self._api_key = api_key
        self._cache = cache
        self._background_tasks = background_tasks

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["api_key", "cache", "background_tasks"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    async def run(
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
        inference_ids: List[np.ndarray] = [p[INFERENCE_ID_KEY] for p in predictions]
        if len(inference_ids) == 0:
            return [
                {
                    "error_status": True,
                    "predictions": predictions,
                    "message": "Custom metadata upload failed because no inference_ids were received",
                }
            ]
        inference_ids: List[str] = list(set(np.concatenate(inference_ids).tolist()))
        if field_name is None:
            return [
                {
                    "error_status": True,
                    "predictions": predictions,
                    "message": "Custom metadata upload failed because no field_name was inputted",
                }
            ]
        if field_value is None or len(field_value) == 0:
            return [
                {
                    "error_status": True,
                    "predictions": predictions,
                    "message": "Custom metadata upload failed because no field_value was received",
                }
            ]
        registration_task = partial(
            add_custom_metadata_request,
            cache=self._cache,
            api_key=self._api_key,
            inference_ids=inference_ids,
            field_name=field_name,
            field_value=field_value[0],
        )
        if fire_and_forget and self._background_tasks:
            self._background_tasks.add_task(registration_task)
        else:
            registration_task()
        return [
            {
                "error_status": False,
                "predictions": predictions,
                "message": "Custom metadata upload was successful",
            }
        ]


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
):
    workspace_id = get_workspace_name(api_key=api_key, cache=cache)
    was_added = False
    try:
        was_added = add_custom_metadata(
            api_key=api_key,
            workspace_id=workspace_id,
            inference_ids=inference_ids,
            field_name=field_name,
            field_value=field_value,
        )
    except Exception as e:
        pass
    return was_added
