from abc import ABC, abstractmethod
from typing import List, Type, Union

import supervision as sv
from pydantic import AliasChoices, ConfigDict, Field

from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    BATCH_OF_IMAGES_KIND,
    BATCH_OF_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    BATCH_OF_KEYPOINT_DETECTION_PREDICTION_KIND,
    BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND,
    BOOLEAN_KIND,
    StepOutputImageSelector,
    StepOutputSelector,
    WorkflowImageSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

OUTPUT_IMAGE_KEY: str = "image"


class VisualizationManifest(WorkflowBlockManifest, ABC):
    model_config = ConfigDict(
        json_schema_extra={
            "license": "Apache-2.0",
            "block_type": "visualization",
        }
    )
    predictions: StepOutputSelector(
        kind=[
            BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND,
            BATCH_OF_INSTANCE_SEGMENTATION_PREDICTION_KIND,
            BATCH_OF_KEYPOINT_DETECTION_PREDICTION_KIND,
        ]
    ) = Field(  # type: ignore
        description="Predictions",
        examples=["$steps.object_detection_model.predictions"],
    )
    image: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        title="Input Image",
        description="The input image for this step.",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("image", "images"),
    )

    copy_image: Union[bool, WorkflowParameterSelector(kind=[BOOLEAN_KIND])] = Field(  # type: ignore
        description="Duplicate the image contents (vs overwriting the image in place). Deselect for chained visualizations that should stack on previous ones where the intermediate state is not needed.",
        default=True,
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name=OUTPUT_IMAGE_KEY,
                kind=[
                    BATCH_OF_IMAGES_KIND,
                ],
            ),
        ]


class VisualizationBlock(WorkflowBlock, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    @abstractmethod
    def get_manifest(cls) -> Type[VisualizationManifest]:
        pass

    @abstractmethod
    def getAnnotator(self, *args, **kwargs) -> sv.annotators.base.BaseAnnotator:
        pass

    @abstractmethod
    def run(
        self,
        image: WorkflowImageData,
        predictions: sv.Detections,
        copy_image: bool,
        *args,
        **kwargs
    ) -> BlockResult:
        pass
