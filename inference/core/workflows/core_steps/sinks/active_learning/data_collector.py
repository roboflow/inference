from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

from fastapi import BackgroundTasks
from pydantic import AliasChoices, ConfigDict, Field
from typing_extensions import Annotated

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
    BATCH_OF_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    BATCH_OF_KEYPOINT_DETECTION_PREDICTION_KIND,
    BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND,
    BATCH_OF_PREDICTION_TYPE_KIND,
    BATCH_OF_TOP_CLASS_KIND,
    BOOLEAN_KIND,
    ROBOFLOW_PROJECT_KIND,
    STRING_KIND,
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

SHORT_DESCRIPTION = (
    "Collect data and predictions that flow through workflows for use "
    "in active learning."
)

LONG_DESCRIPTION = """
Sample images and model predictions from a workflow and upload them back to a Roboflow 
project.

This block is useful for:

1. Gathering data for use in training a new model, from scratch, or;
2. Gathering data to improve an existing model.

This block uses an Active Learning Configuration to determine how to configure active 
learning. The Configuration specification allows you to determine a sampling strategy, 
such s random sampling or threshold sampling.

To learn more about active learning configurations, refer to the Inference Active 
Learning Configuration documentation.
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
    type: Literal["ActiveLearningDataCollector"]
    images: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        description="Reference an image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("images", "image"),
    )
    predictions: StepOutputSelector(
        kind=[
            BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND,
            BATCH_OF_INSTANCE_SEGMENTATION_PREDICTION_KIND,
            BATCH_OF_KEYPOINT_DETECTION_PREDICTION_KIND,
            BATCH_OF_TOP_CLASS_KIND,
        ]
    ) = Field(
        description="Reference to detection-like predictions",
        examples=["$steps.object_detection_model.predictions"],
    )
    prediction_type: Annotated[
        StepOutputSelector(kind=[BATCH_OF_PREDICTION_TYPE_KIND]),
        Field(
            description="Type of `predictions`. Must be output from the step referred in `predictions` field",
            examples=["$steps.detection.prediction_type"],
        ),
    ]
    target_dataset: Union[
        WorkflowParameterSelector(kind=[ROBOFLOW_PROJECT_KIND]), str
    ] = Field(
        description="name of Roboflow dataset / project to be used as target for collected data",
        examples=["my_dataset", "$inputs.target_al_dataset"],
    )
    target_dataset_api_key: Union[
        WorkflowParameterSelector(kind=[STRING_KIND]), Optional[str]
    ] = Field(
        default=None,
        description="API key to be used for data registration. This may help in a scenario when data applicable for Universe models predictions to be saved in private workspaces and for models that were trained in the same workspace (not necessarily within the same project))",
    )
    disable_active_learning: Union[
        bool, WorkflowParameterSelector(kind=[BOOLEAN_KIND])
    ] = Field(
        default=False,
        description="boolean flag that can be also reference to input - to arbitrarily disable data collection for specific request - overrides all AL config",
        examples=[True, "$inputs.disable_active_learning"],
    )
    active_learning_configuration: Optional[
        Union[EnabledActiveLearningConfiguration, DisabledActiveLearningConfiguration]
    ] = Field(
        default=None,
        description="Optional configuration of Active Learning data sampling in the exact format explained in Active Learning docs.",
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return []


class ActiveLearningDataCollectorBlock(WorkflowBlock):

    def __init__(
        self,
        active_learning_middleware: WorkflowsActiveLearningMiddleware,
        background_tasks: Optional[BackgroundTasks],
        api_key: Optional[str],
    ):
        self._active_learning_middleware = active_learning_middleware
        self._background_tasks = background_tasks
        self._api_key = api_key

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["active_learning_middleware", "background_tasks", "api_key"]

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
