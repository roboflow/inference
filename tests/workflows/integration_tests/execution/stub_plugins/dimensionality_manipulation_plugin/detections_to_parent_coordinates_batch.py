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
    type: Literal["DetectionsToParentsCoordinatesBatch"]
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
        return "images_predictions"

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


class DetectionsToParentCoordinatesBatchBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    @classmethod
    def accepts_batch_input(cls) -> bool:
        return True

    def run(
        self,
        images: Batch[WorkflowImageData],
        images_predictions: Batch[Batch[sv.Detections]],
    ) -> BlockResult:
        result = []
        for i, (image, image_predictions) in enumerate(zip(images, images_predictions)):
            print("Processing image", i)
            parent_id = image.parent_metadata.parent_id
            parent_coordinates = image.parent_metadata.origin_coordinates
            transformed_predictions = []
            for j, prediction in enumerate(image_predictions):
                print(
                    f"Processing prediction {j} - start {len(prediction)} - {prediction['parent_id']}"
                )
                prediction_copy = deepcopy(prediction)
                prediction_copy["parent_id"] = np.array([parent_id] * len(prediction))
                if parent_coordinates:
                    offset = [0, 0]
                    dimensions = [
                        parent_coordinates.origin_height,
                        parent_coordinates.origin_width,
                    ]
                    prediction_copy["parent_coordinates"] = np.array(
                        [offset] * len(prediction)
                    )
                    prediction_copy["parent_dimensions"] = np.array(
                        [dimensions] * len(prediction)
                    )
                print(
                    f"Processing prediction {j} - end {len(prediction_copy)} - {prediction_copy['parent_id']}"
                )
                transformed_predictions.append({"predictions": prediction_copy})
            result.append(transformed_predictions)
        return result
