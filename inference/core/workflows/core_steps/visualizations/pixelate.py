from typing import List, Literal, Optional, Type, Union

import supervision as sv
from pydantic import AliasChoices, ConfigDict, Field

from inference.core.workflows.entities.base import OutputDefinition, WorkflowImageData
from inference.core.workflows.entities.types import (
    BATCH_OF_IMAGES_KIND,
    BATCH_OF_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    BATCH_OF_KEYPOINT_DETECTION_PREDICTION_KIND,
    BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND,
    BOOLEAN_KIND,
    INTEGER_KIND,
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

TYPE: str = "PixelateVisualization"
SHORT_DESCRIPTION = "Pixelates detected objects in an image."
LONG_DESCRIPTION = """
The `PixelateVisualization` block pixelates detected
objects in an image using Supervision's `sv.PixelateAnnotator`.
"""


class PixelateManifest(WorkflowBlockManifest):
    type: Literal[f"{TYPE}"]
    model_config = ConfigDict(
        json_schema_extra={
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
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

    pixel_size: Union[int, WorkflowParameterSelector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Size of the pixelation.",
        default=20,
        examples=[20, "$inputs.pixel_size"],
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


class PixelateVisualizationBlock(WorkflowBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.annotatorCache = {}

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return PixelateManifest

    def getAnnotator(
        self,
        pixel_size: int,
    ) -> sv.annotators.base.BaseAnnotator:
        key = "_".join(map(str, [pixel_size]))

        if key not in self.annotatorCache:
            self.annotatorCache[key] = sv.PixelateAnnotator(pixel_size=pixel_size)
        return self.annotatorCache[key]

    async def run(
        self,
        image: WorkflowImageData,
        predictions: sv.Detections,
        copy_image: bool,
        pixel_size: Optional[int],
    ) -> BlockResult:
        annotator = self.getAnnotator(
            pixel_size,
        )

        annotated_image = annotator.annotate(
            scene=image.numpy_image.copy() if copy_image else image.numpy_image,
            detections=predictions,
        )

        output = WorkflowImageData(
            parent_metadata=image.parent_metadata,
            workflow_root_ancestor_metadata=image.workflow_root_ancestor_metadata,
            numpy_image=annotated_image,
        )

        return {OUTPUT_IMAGE_KEY: output}
