from typing import Any, Dict, List, Literal, Optional, Type
from uuid import uuid4

import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

OUTPUT_KEY: str = "predictions"

SHORT_DESCRIPTION = "Merge multiple detections into a single bounding box."
LONG_DESCRIPTION = """
The `DetectionsMerge` block combines multiple detections into a single bounding box that encompasses all input detections.
This is useful when you want to:
- Merge overlapping or nearby detections of the same object
- Create a single region that contains multiple detected objects
- Simplify multiple detections into one larger detection

The resulting detection will have:
- A bounding box that contains all input detections
- The classname of the merged detection is set to "merged_detection" by default, but can be customized via the `class_name` parameter
- The confidence is set to the lowest confidence among all detections
"""


class DetectionsMergeManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Detections Merge",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
            "ui_manifest": {
                "section": "transformation",
                "icon": "fal fa-object-union",
                "blockPriority": 5,
            },
        }
    )
    type: Literal["roboflow_core/detections_merge@v1"]
    predictions: Selector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
            KEYPOINT_DETECTION_PREDICTION_KIND,
        ]
    ) = Field(
        description="Object detection predictions to merge into a single bounding box.",
        examples=["$steps.object_detection_model.predictions"],
    )
    class_name: str = Field(
        default="merged_detection",
        description="The class name to assign to the merged detection.",
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name=OUTPUT_KEY, kind=[OBJECT_DETECTION_PREDICTION_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


def calculate_union_bbox(detections: sv.Detections) -> np.ndarray:
    """Calculate a single bounding box that contains all input detections."""
    if len(detections) == 0:
        return np.array([], dtype=np.float32).reshape(0, 4)

    # Get all bounding boxes
    xyxy = detections.xyxy

    # Calculate the union by taking min/max coordinates
    x1 = np.min(xyxy[:, 0])
    y1 = np.min(xyxy[:, 1])
    x2 = np.max(xyxy[:, 2])
    y2 = np.max(xyxy[:, 3])

    return np.array([[x1, y1, x2, y2]])


def get_lowest_confidence_index(detections: sv.Detections) -> int:
    """Get the index of the detection with the lowest confidence."""
    if detections.confidence is None:
        return 0
    return int(np.argmin(detections.confidence))


class DetectionsMergeBlockV1(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return DetectionsMergeManifest

    def run(
        self,
        predictions: sv.Detections,
        class_name: str = "merged_detection",
    ) -> BlockResult:
        if predictions is None or len(predictions) == 0:
            return {
                OUTPUT_KEY: sv.Detections(
                    xyxy=np.array([], dtype=np.float32).reshape(0, 4)
                )
            }

        # Calculate the union bounding box
        union_bbox = calculate_union_bbox(predictions)

        # Get the index of the detection with lowest confidence
        lowest_conf_idx = get_lowest_confidence_index(predictions)

        # Create a new detection with the union bbox and ensure numpy arrays for all fields
        merged_detection = sv.Detections(
            xyxy=union_bbox,
            confidence=(
                np.array([predictions.confidence[lowest_conf_idx]], dtype=np.float32)
                if predictions.confidence is not None
                else None
            ),
            class_id=np.array(
                [0], dtype=np.int32
            ),  # Fixed class_id of 0 for merged detection
            data={
                "class_name": np.array([class_name]),
                "detection_id": np.array([str(uuid4())]),
            },
        )

        return {OUTPUT_KEY: merged_detection}
