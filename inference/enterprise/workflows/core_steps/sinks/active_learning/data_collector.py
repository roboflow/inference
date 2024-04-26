from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

from fastapi import BackgroundTasks
from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt, confloat
from typing_extensions import Annotated

from inference.core.utils.image_utils import load_image
from inference.enterprise.workflows.entities.base import OutputDefinition
from inference.enterprise.workflows.entities.types import (
    BOOLEAN_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    PREDICTION_TYPE_KIND,
    ROBOFLOW_PROJECT_KIND,
    STRING_KIND,
    TOP_CLASS_KIND,
    FlowControl,
    InferenceImageSelector,
    InferenceParameterSelector,
    OutputStepImageSelector,
    StepOutputSelector,
)
from inference.enterprise.workflows.errors import ExecutionGraphError
from inference.enterprise.workflows.prototypes.block import (
    WorkflowBlock,
    WorkflowBlockManifest,
)


class DisabledActiveLearningConfiguration(BaseModel):
    enabled: Literal[False]


class LimitDefinition(BaseModel):
    type: Literal["minutely", "hourly", "daily"]
    value: PositiveInt


class RandomSamplingConfig(BaseModel):
    type: Literal["random"]
    name: str
    traffic_percentage: confloat(ge=0.0, le=1.0)
    tags: List[str] = Field(default_factory=lambda: [])
    limits: List[LimitDefinition] = Field(default_factory=lambda: [])


class CloseToThresholdSampling(BaseModel):
    type: Literal["close_to_threshold"]
    name: str
    probability: confloat(ge=0.0, le=1.0)
    threshold: confloat(ge=0.0, le=1.0)
    epsilon: confloat(ge=0.0, le=1.0)
    max_batch_images: Optional[int] = Field(default=None)
    only_top_classes: bool = Field(default=True)
    minimum_objects_close_to_threshold: int = Field(default=1)
    selected_class_names: Optional[List[str]] = Field(default=None)
    tags: List[str] = Field(default_factory=lambda: [])
    limits: List[LimitDefinition] = Field(default_factory=lambda: [])


class ClassesBasedSampling(BaseModel):
    type: Literal["classes_based"]
    name: str
    probability: confloat(ge=0.0, le=1.0)
    selected_class_names: List[str]
    tags: List[str] = Field(default_factory=lambda: [])
    limits: List[LimitDefinition] = Field(default_factory=lambda: [])


class DetectionsBasedSampling(BaseModel):
    type: Literal["detections_number_based"]
    name: str
    probability: confloat(ge=0.0, le=1.0)
    more_than: Optional[NonNegativeInt]
    less_than: Optional[NonNegativeInt]
    selected_class_names: Optional[List[str]] = Field(default=None)
    tags: List[str] = Field(default_factory=lambda: [])
    limits: List[LimitDefinition] = Field(default_factory=lambda: [])


class ActiveLearningBatchingStrategy(BaseModel):
    batches_name_prefix: str
    recreation_interval: Literal["never", "daily", "weekly", "monthly"]
    max_batch_images: Optional[int] = Field(default=None)


ActiveLearningStrategyType = Annotated[
    Union[
        RandomSamplingConfig,
        CloseToThresholdSampling,
        ClassesBasedSampling,
        DetectionsBasedSampling,
    ],
    Field(discriminator="type"),
]


class EnabledActiveLearningConfiguration(BaseModel):
    enabled: bool
    persist_predictions: bool
    sampling_strategies: List[ActiveLearningStrategyType]
    batching_strategy: ActiveLearningBatchingStrategy
    tags: List[str] = Field(default_factory=lambda: [])
    max_image_size: Optional[Tuple[PositiveInt, PositiveInt]] = Field(default=None)
    jpeg_compression_level: int = Field(default=95, gt=0, le=100)


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
    image: Union[InferenceImageSelector, OutputStepImageSelector] = Field(
        description="Reference at image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )
    predictions: StepOutputSelector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
            KEYPOINT_DETECTION_PREDICTION_KIND,
            TOP_CLASS_KIND,
        ]
    ) = Field(
        description="Reference to detection-like predictions",
        examples=["$steps.object_detection_model.predictions"],
    )
    prediction_type: Annotated[
        StepOutputSelector(kind=[PREDICTION_TYPE_KIND]),
        Field(
            description="Type of `predictions`. Must be output from the step referred in `predictions` field",
            examples=["$steps.detection.prediction_type"],
        ),
    ]
    target_dataset: Union[
        InferenceParameterSelector(kind=[ROBOFLOW_PROJECT_KIND]), str
    ] = Field(
        description="name of Roboflow dataset / project to be used as target for collected data",
        examples=["my_dataset", "$inputs.target_al_dataset"],
    )
    target_dataset_api_key: Union[
        InferenceParameterSelector(kind=[STRING_KIND]), Optional[str]
    ] = Field(
        default=None,
        description="API key to be used for data registration. This may help in a scenario when data applicable for Universe models predictions to be saved in private workspaces and for models that were trained in the same workspace (not necessarily within the same project))",
    )
    disable_active_learning: Union[
        bool, InferenceParameterSelector(kind=[BOOLEAN_KIND])
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
    def get_input_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return []

    async def run_locally(
        self,
        image: List[dict],
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
            raise ExecutionGraphError(
                f"Active Learning data collection step requires only single prediction "
                f"type to be part of ingest. Detected: {prediction_type}."
            )
        prediction_type = next(iter(prediction_type))
        predictions_output_name = (
            "predictions" if "classification" not in prediction_type else "top"
        )
        decoded_images = [load_image(e)[0] for e in image]
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
            images=image,
            predictions=active_learning_compatible_predictions,
            api_key=target_dataset_api_key or self._api_key,
            active_learning_disabled_for_request=disable_active_learning,
            prediction_type=prediction_type,
            background_tasks=self._background_tasks,
            active_learning_configuration=active_learning_configuration,
        )
        return []
