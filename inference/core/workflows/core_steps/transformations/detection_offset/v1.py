import uuid
from copy import deepcopy
from typing import List, Literal, Optional, Type, Union

import numpy as np
import supervision as sv
from pydantic import AliasChoices, ConfigDict, Field, PositiveInt

from inference.core.workflows.execution_engine.constants import (
    DETECTION_ID_KEY,
    PARENT_ID_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
)
from inference.core.workflows.execution_engine.entities.types import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INTEGER_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    StepOutputSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
Apply a fixed offset to the width and height of a detection.

You can use this block to add padding around the result of a detection. This is useful 
to ensure that you can analyze bounding boxes that may be within the region of an 
object instead of being around an object.
"""

SHORT_DESCRIPTION = "Apply a padding around the width and height of detections."


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Detection Offset",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
        }
    )
    type: Literal["roboflow_core/detection_offset@v1", "DetectionOffset"]
    predictions: StepOutputSelector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
            KEYPOINT_DETECTION_PREDICTION_KIND,
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

    @classmethod
    def accepts_batch_input(cls) -> bool:
        return True

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

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class DetectionOffsetBlockV1(WorkflowBlock):
    # TODO: This block breaks parent coordinates.
    # Issue report: https://github.com/roboflow/inference/issues/380

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        predictions: Batch[sv.Detections],
        offset_width: int,
        offset_height: int,
    ) -> BlockResult:
        return [
            {
                "predictions": offset_detections(
                    detections=detections,
                    offset_width=offset_width,
                    offset_height=offset_height,
                )
            }
            for detections in predictions
        ]


def offset_detections(
    detections: sv.Detections,
    offset_width: int,
    offset_height: int,
    parent_id_key: str = PARENT_ID_KEY,
    detection_id_key: str = DETECTION_ID_KEY,
) -> sv.Detections:
    if len(detections) == 0:
        return detections
    _detections = deepcopy(detections)
    image_dimensions = detections.data["image_dimensions"]
    _detections.xyxy = np.array(
        [
            (
                max(0, x1 - offset_width // 2),
                max(0, y1 - offset_height // 2),
                min(image_dimensions[i][1], x2 + offset_width // 2),
                min(image_dimensions[i][0], y2 + offset_height // 2),
            )
            for i, (x1, y1, x2, y2) in enumerate(_detections.xyxy)
        ]
    )
    _detections[parent_id_key] = detections[detection_id_key].copy()
    _detections[detection_id_key] = [str(uuid.uuid4()) for _ in detections]
    return _detections
