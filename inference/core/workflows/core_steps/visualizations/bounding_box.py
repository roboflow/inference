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
    StepOutputImageSelector,
    StepOutputSelector,
    WorkflowImageSelector
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

class BoundingBoxVisualizationBlock(WorkflowBlock):
    def __init__(self):
        self.annotator = None

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BoundingBoxManifest

    async def run(
        self,
        image: WorkflowImageData,
        predictions: sv.Detections
    ) -> BlockResult:
        if self.annotator is None:
            self.annotator = sv.RoundBoxAnnotator(
                thickness=3
            )

        annotated_image = self.annotator.annotate(
            scene=image.numpy_image,
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
