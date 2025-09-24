"""
This is just example, test implementation, please do not assume it being fully functional.
"""

from copy import deepcopy
from typing import Dict, List, Literal, Optional, Type, Union

import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
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
    type: Literal["StitchDetectionsBatch"]
    images: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        description="Reference an image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )
    images_predictions: StepOutputSelector(
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
    def accepts_batch_input(cls) -> bool:
        return True

    @classmethod
    def get_input_dimensionality_offsets(
        cls,
    ) -> Dict[str, int]:
        return {
            "images": 0,
            "images_predictions": 1,
        }

    @classmethod
    def get_dimensionality_reference_property(cls) -> Optional[str]:
        return "images"

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="predictions",
                kind=[
                    OBJECT_DETECTION_PREDICTION_KIND,
                    INSTANCE_SEGMENTATION_PREDICTION_KIND,
                    KEYPOINT_DETECTION_PREDICTION_KIND,
                ],
            ),
        ]


class StitchDetectionsBatchBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        images: Batch[WorkflowImageData],
        images_predictions: Batch[Batch[sv.Detections]],
    ) -> BlockResult:
        result = []
        for image, image_predictions in zip(images, images_predictions):
            image_predictions = [deepcopy(p) for p in image_predictions if len(p)]
            for p in image_predictions:
                coords = p["parent_coordinates"][0]
                p.xyxy += np.concatenate((coords, coords))
            merged_prediction = sv.Detections.merge(image_predictions)
            result.append({"predictions": merged_prediction})
        return result
