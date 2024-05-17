from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

from fastapi import BackgroundTasks
from pydantic import AliasChoices, ConfigDict, Field

from inference.core.utils.image_utils import load_image
from inference.core.workflows.core_steps.sinks.active_learning.entities import (
    DisabledActiveLearningConfiguration,
    EnabledActiveLearningConfiguration,
)
from inference.core.workflows.core_steps.sinks.active_learning.middleware import (
    WorkflowsActiveLearningMiddleware,
)
from inference.core.workflows.entities.base import OutputDefinition
from inference.core.workflows.entities.types import (
    BATCH_OF_BOOLEAN_KIND,
    BATCH_OF_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    BATCH_OF_KEYPOINT_DETECTION_PREDICTION_KIND,
    BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND,
    BATCH_OF_STRING_KIND,
    BOOLEAN_KIND,
    ROBOFLOW_PROJECT_KIND,
    FlowControl,
    StepOutputImageSelector,
    StepOutputSelector,
    WorkflowImageSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import (
    WorkflowBlock,
    WorkflowBlockManifest,
)

SHORT_DESCRIPTION = "TODO"

LONG_DESCRIPTION = """
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "sink",
        }
    )
    type: Literal["RoboflowDataCollector"]
    target_project: Union[
        WorkflowParameterSelector(kind=[ROBOFLOW_PROJECT_KIND]), str
    ] = Field(
        description="name of Roboflow dataset / project to be used as target for collected data",
        examples=["my_dataset", "$inputs.target_al_dataset"],
    )
    images: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        description="Reference at image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("images", "image"),
    )
    disable_sink: Union[bool, WorkflowParameterSelector(kind=[BOOLEAN_KIND])] = Field(
        default=False,
        description="boolean flag that can be also reference to input - to arbitrarily disable "
        "data collection for specific request",
        examples=[True, "$inputs.disable_active_learning"],
    )
    fire_and_forget: Union[bool, WorkflowParameterSelector(kind=[BOOLEAN_KIND])] = (
        Field(
            default=True,
            description="Boolean flag dictating if sink is supposed to be executed in the background, "
            "not waiting on status of registration before end of workflow run. Use `True` if best-effort "
            "registration is needed, use `False` while debugging and if error handling is needed",
        )
    )
    predictions: Optional[
        StepOutputSelector(
            kind=[
                BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND,
                BATCH_OF_INSTANCE_SEGMENTATION_PREDICTION_KIND,
                BATCH_OF_KEYPOINT_DETECTION_PREDICTION_KIND,
            ]
        )
    ] = Field(
        default=None,
        description="Reference to detection-like predictions",
        examples=["$steps.object_detection_model.predictions"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="error_status", kind=[BATCH_OF_BOOLEAN_KIND]),
            OutputDefinition(name="error_message", kind=[BATCH_OF_STRING_KIND]),
        ]


class RoboflowDataCollectorBlock(WorkflowBlock):

    def __init__(
        self,
        background_tasks: Optional[BackgroundTasks],
        api_key: Optional[str],
    ):
        self._background_tasks = background_tasks
        self._api_key = api_key

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["background_tasks", "api_key"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    async def run_locally(
        self,
        images: List[dict],
        predictions: List[List[dict]],
        prediction_type: List[str],
        target_dataset: str,
        target_dataset_api_key: Optional[str],
        disable_active_learning: bool,
        active_learning_configuration: Optional[
            Union[
                EnabledActiveLearningConfiguration, DisabledActiveLearningConfiguration
            ]
        ],
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FlowControl]]:
        prediction_type = set(prediction_type)
        if len(prediction_type) > 1:
            raise ValueError(
                f"Active Learning data collection step requires only single prediction "
                f"type to be part of ingest. Detected: {prediction_type}."
            )
        prediction_type = next(iter(prediction_type))
        predictions_output_name = (
            "predictions" if "classification" not in prediction_type else "top"
        )
        decoded_images = [load_image(e)[0] for e in images]
        images_meta = [
            {"width": i.shape[1], "height": i.shape[0]} for i in decoded_images
        ]
        active_learning_compatible_predictions = [
            {"image": image_meta, predictions_output_name: prediction}
            for image_meta, prediction in zip(images_meta, predictions)
        ]
        self._active_learning_middleware.register(
            # this should actually be asyncio, but that requires a lot of backend components redesign
            dataset_name=target_dataset,
            images=images,
            predictions=active_learning_compatible_predictions,
            api_key=target_dataset_api_key or self._api_key,
            active_learning_disabled_for_request=disable_active_learning,
            prediction_type=prediction_type,
            background_tasks=self._background_tasks,
            active_learning_configuration=active_learning_configuration,
        )
        return []
