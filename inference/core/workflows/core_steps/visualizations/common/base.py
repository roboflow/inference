from abc import ABC, abstractmethod
from typing import List, Type, Union

import supervision as sv
from pydantic import AliasChoices, ConfigDict, Field

from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    IMAGE_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
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
    image: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        title="Input Image",
        description="The input image for this step.",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("image", "images"),
    )
    copy_image: Union[bool, WorkflowParameterSelector(kind=[BOOLEAN_KIND])] = Field(  # type: ignore
        description="Duplicate the image contents (vs overwriting the image in place). Deselect for chained visualizations that should stack on previous ones where the intermediate state is not needed.",
        default=True,
        examples=[True, False],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name=OUTPUT_IMAGE_KEY,
                kind=[
                    IMAGE_KIND,
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
        self, image: WorkflowImageData, copy_image: bool, *args, **kwargs
    ) -> BlockResult:
        pass


class PredictionsVisualizationManifest(VisualizationManifest, ABC):
    predictions: StepOutputSelector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
            KEYPOINT_DETECTION_PREDICTION_KIND,
        ]
    ) = Field(  # type: ignore
        description="Predictions",
        examples=["$steps.object_detection_model.predictions"],
    )


class PredictionsVisualizationBlock(VisualizationBlock, ABC):
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
