from dataclasses import replace
from typing import List, Literal, Optional, Type, Union

import supervision as sv
from pydantic import AliasChoices, ConfigDict, Field

from inference.core.workflows.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.entities.types import (
    # IMAGE_KIND,
    # OBJECT_DETECTION_PREDICTION_KIND,
    # INSTANCE_SEGMENTATION_PREDICTION_KIND,
    # KEYPOINT_DETECTION_PREDICTION_KIND,
    BATCH_OF_IMAGES_KIND,
    BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND,
    BATCH_OF_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    BATCH_OF_KEYPOINT_DETECTION_PREDICTION_KIND,
    INTEGER_KIND,
    FloatZeroToOne,
    FLOAT_ZERO_TO_ONE_KIND,
    BOOLEAN_KIND,
    StepOutputImageSelector,
    StepOutputSelector,
    WorkflowImageSelector,
    WorkflowParameterSelector
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

OUTPUT_IMAGE_KEY: str = "image"
TYPE: str = "BoundingBoxVisualization"
SHORT_DESCRIPTION = (
    "Draws a box around detected objects in an image."
)
LONG_DESCRIPTION = """
The `BoundingBoxVisualization` block draws a box around detected
objects in an image using Supervision's `sv.RoundBoxAnnotator`.
"""


class BoundingBoxManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "visualization",
        }
    )
    type: Literal[f"{TYPE}"]
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

    copy_image: Union[bool, WorkflowParameterSelector(kind=[BOOLEAN_KIND])] = Field(
        description="Duplicate the image contents (vs overwriting the image in place). Deselect for chained visualizations that should stack on previous ones where the intermediate state is not needed.",
        default=True
    )

    thickness: Union[int, WorkflowParameterSelector(kind=[INTEGER_KIND])] = Field(
        description="Thickness of the bounding box in pixels.",
        default=1,
    )

    roundness: Union[FloatZeroToOne, WorkflowParameterSelector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = Field(
        description="Roundness of the corners of the bounding box.",
        default=0.0,
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

annotatorCache = {}

def getAnnotator(
    thickness:int,
    roundness: float
):
    key = f"{thickness}_{roundness}"
    if key not in annotatorCache:
        if roundness == 0:
            annotatorCache[key] = sv.BoxAnnotator(
                thickness=thickness
            )
        else:
            annotatorCache[key] = sv.RoundBoxAnnotator(
                thickness=thickness,
                roundness=roundness
            )
    return annotatorCache[key]
class BoundingBoxVisualizationBlock(WorkflowBlock):
    def __init__(self):
        pass

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BoundingBoxManifest

    async def run(
        self,
        image: WorkflowImageData,
        predictions: sv.Detections,
        copy_image: bool,
        thickness: Optional[int],
        roundness: Optional[float]
    ) -> BlockResult:
        annotator = getAnnotator(thickness, roundness)

        annotated_image = annotator.annotate(
            scene=image.numpy_image.copy() if copy_image else image.numpy_image,
            detections=predictions
        )

        output = WorkflowImageData(
            parent_metadata=image.parent_metadata,
            workflow_root_ancestor_metadata=image.workflow_root_ancestor_metadata,
            numpy_image=annotated_image,
        )

        return {
            OUTPUT_IMAGE_KEY: output
        }
