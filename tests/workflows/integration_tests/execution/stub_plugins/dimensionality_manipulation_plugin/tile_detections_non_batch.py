"""
This is just example, test implementation, please do not assume it being fully functional.
"""

from typing import List, Literal, Type, Union

import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.utils.drawing import create_tiles
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    IMAGE_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    StepOutputImageSelector,
    StepOutputSelector,
    WorkflowImageSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "short_description": "",
            "long_description": "",
            "license": "Apache-2.0",
            "block_type": "transformation",
        }
    )
    type: Literal["TileDetectionsNonBatch"]
    crops: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        description="Reference an image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )
    crops_predictions: StepOutputSelector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
            KEYPOINT_DETECTION_PREDICTION_KIND,
        ]
    ) = Field(
        description="Reference to predictions of detection-like model, that can be based of cropping "
        "(detection must define RoI - eg: bounding box)",
        examples=["$steps.my_object_detection_model.predictions"],
    )

    @classmethod
    def get_output_dimensionality_offset(
        cls,
    ) -> int:
        return -1

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="visualisations", kind=[IMAGE_KIND]),
        ]


class TileDetectionsNonBatchBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        crops: Batch[WorkflowImageData],
        crops_predictions: Batch[sv.Detections],
    ) -> BlockResult:
        annotator = sv.BoxAnnotator()
        visualisations = []
        for image, prediction in zip(crops, crops_predictions):
            annotated_image = annotator.annotate(
                image.numpy_image.copy(),
                prediction,
            )
            visualisations.append(annotated_image)
        tile = create_tiles(visualisations)
        return {"visualisations": tile}
