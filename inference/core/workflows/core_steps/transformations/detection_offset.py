import uuid
from copy import deepcopy
from typing import Any, Dict, List, Literal, Type, Union

import numpy as np
import supervision as sv
from pydantic import AliasChoices, ConfigDict, Field, PositiveInt
from typing_extensions import Annotated

from inference.core.workflows.constants import DETECTION_ID_KEY, PARENT_ID_KEY
from inference.core.workflows.entities.base import OutputDefinition
from inference.core.workflows.entities.types import (
    BATCH_OF_IMAGE_METADATA_KIND,
    BATCH_OF_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    BATCH_OF_KEYPOINT_DETECTION_PREDICTION_KIND,
    BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND,
    BATCH_OF_PARENT_ID_KIND,
    BATCH_OF_PREDICTION_TYPE_KIND,
    INTEGER_KIND,
    StepOutputSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import (
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
Apply a fixed offset to the width and height of a detection.

You can use this block to add padding around the result of a detection. This is useful 
to ensure that you can analyze bounding boxes that may be within the region of an 
object instead of being around an object.
"""

SHORT_DESCRIPTION = "Apply a fixed offset on the width and height of detections."


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
        }
    )
    type: Literal["DetectionOffset"]
    name: str = Field(description="Unique name of step in workflows")
    predictions: StepOutputSelector(
        kind=[
            BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND,
            BATCH_OF_INSTANCE_SEGMENTATION_PREDICTION_KIND,
            BATCH_OF_KEYPOINT_DETECTION_PREDICTION_KIND,
        ]
    ) = Field(
        description="Reference to detection-like predictions",
        examples=["$steps.object_detection_model.predictions"],
    )
    offset_width: Union[PositiveInt, WorkflowParameterSelector(kind=[INTEGER_KIND])] = (
        Field(
            description="Offset for boxes width",
            examples=[10, "$inputs.offset_x"],
            validation_alias=AliasChoices("offset_width", "offset_x"),
        )
    )
    offset_height: Union[
        PositiveInt, WorkflowParameterSelector(kind=[INTEGER_KIND])
    ] = Field(
        description="Offset for boxes height",
        examples=[10, "$inputs.offset_y"],
        validation_alias=AliasChoices("offset_height", "offset_y"),
    )
    image_metadata: Annotated[
        StepOutputSelector(kind=[BATCH_OF_IMAGE_METADATA_KIND]),
        Field(
            description="Metadata of image used to create `predictions`. Must be output from the step referred in `predictions` field",
            examples=["$steps.detection.image"],
        ),
    ]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="predictions",
                kind=[
                    BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND,
                    BATCH_OF_INSTANCE_SEGMENTATION_PREDICTION_KIND,
                    BATCH_OF_KEYPOINT_DETECTION_PREDICTION_KIND,
                ],
            ),
            OutputDefinition(name="image", kind=[BATCH_OF_IMAGE_METADATA_KIND]),
            OutputDefinition(name="parent_id", kind=[BATCH_OF_PARENT_ID_KIND]),
        ]


class DetectionOffsetBlock(WorkflowBlock):
    # TODO: This block breaks parent coordinates.
    # Issue report: https://github.com/roboflow/inference/issues/380

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    async def run_locally(
        self,
        predictions: List[sv.Detections],
        offset_width: int,
        offset_height: int,
        image_metadata: List[dict],
    ) -> List[Dict[str, Union[sv.Detections, Any]]]:
        offset_predictions = []
        for detections in predictions:
            offset_detections = deepcopy(detections)
            offset_detections.xyxy = np.array(
                [
                    (
                        x1 - offset_width // 2,
                        y1 - offset_height // 2,
                        x2 + offset_width // 2,
                        y2 + offset_height // 2,
                    )
                    for (x1, y1, x2, y2) in offset_detections.xyxy
                ]
            )
            # parent ID remains unchanged
            offset_predictions.append(offset_detections)
        return [
            {
                "predictions": offset_prediction,
                PARENT_ID_KEY: prediction[PARENT_ID_KEY],
                "image": image,
            }
            for offset_prediction, image, prediction in zip(
                offset_predictions, image_metadata, predictions
            )
        ]


def offset_detections(
    detections: sv.Detections,
    offset_width: int,
    offset_height: int,
    parent_id_key: str = PARENT_ID_KEY,
    detection_id_key: str = DETECTION_ID_KEY,
) -> sv.Detections:
    _detections = deepcopy(detections)
    _detections.xyxy = [
        (
            x1 - offset_width // 2,
            y1 - offset_height // 2,
            x2 + offset_width // 2,
            y2 + offset_height // 2,
        )
        for (x1, y1, x2, y2) in _detections.xyxy
    ]
    _detections[parent_id_key] = detections[detection_id_key].copy()
    _detections[detection_id_key] = [str(uuid.uuid4()) for _ in detections]
    return _detections
