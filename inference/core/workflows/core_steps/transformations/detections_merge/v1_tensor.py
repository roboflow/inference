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
Combine multiple detection predictions into a single merged detection with a union bounding box that encompasses all input detections, simplifying multiple detections into one larger detection region for overlapping object consolidation, region creation from multiple objects, and detection simplification workflows.

## How This Block Works

This block merges multiple detections into a single detection by calculating a union bounding box that contains all input detections. The block:

1. Receives detection predictions (object detection, instance segmentation, or keypoint detection) containing multiple detections
2. Validates input (handles empty detections by returning an empty detection result)
3. Calculates the union bounding box from all input detections:
   - Extracts all bounding box coordinates (xyxy format) from input detections
   - Finds the minimum x and y coordinates (leftmost and topmost points) across all boxes
   - Finds the maximum x and y coordinates (rightmost and bottommost points) across all boxes
   - Creates a single bounding box that completely encompasses all input detections
4. Determines the merged detection's confidence:
   - Finds the detection with the lowest confidence score among all input detections
   - Uses this lowest confidence as the merged detection's confidence (conservative approach)
   - Handles cases where confidence scores may not be present
5. Creates a new merged detection with:
   - The calculated union bounding box (encompasses all input detections)
   - A customizable class name (default: "merged_detection", configurable via class_name parameter)
   - The lowest confidence from input detections (conservative confidence assignment)
   - A fixed class_id of 0 for the merged detection
   - A newly generated detection ID (unique identifier for the merged detection)
6. Returns the single merged detection containing all input detections within its bounding box

The block creates a unified bounding box representation of multiple detections, useful for consolidating overlapping or nearby detections into a single region. The union bounding box approach ensures all original detections are completely contained within the merged detection. By using the lowest confidence, the block adopts a conservative approach, ensuring the merged detection's confidence reflects the least certain input detection. The merged detection can be customized with a class name to indicate its merged nature or to represent a specific category.

## Common Use Cases

- **Overlapping Detection Consolidation**: Merge multiple overlapping detections of the same or related objects into a single unified detection (e.g., merge overlapping detections of the same person from multiple frames, consolidate duplicate detections from different models, combine overlapping object parts into one detection), enabling overlapping detection simplification
- **Multi-Object Region Creation**: Create a single bounding box region that encompasses multiple detected objects for area-based analysis (e.g., create a region containing multiple people for crowd analysis, merge detections of objects in a scene into one region, combine multiple detections into a single monitoring zone), enabling multi-object region workflows
- **Nearby Detection Grouping**: Group nearby detections together into a single merged detection (e.g., merge detections of objects close to each other, group nearby detections into clusters, combine adjacent detections for simplified processing), enabling spatial grouping workflows
- **Detection Simplification**: Simplify multiple detections into one larger detection for downstream processing (e.g., reduce multiple detections to one for simpler analysis, consolidate detections for easier visualization, merge detections for streamlined workflows), enabling detection simplification workflows
- **Zone Definition from Detections**: Create zone boundaries from multiple detection locations (e.g., define zones based on detection locations, create regions from detected object positions, establish boundaries from detection clusters), enabling zone creation from detections
- **Redundant Detection Removal**: Merge redundant or duplicate detections into a single representation (e.g., combine duplicate detections from different stages, merge redundant object detections, consolidate repeated detections), enabling redundant detection consolidation workflows

## Connecting to Other Blocks

This block receives multiple detection predictions and produces a single merged detection:

- **After detection blocks** (e.g., Object Detection, Instance Segmentation, Keypoint Detection) to merge multiple detections into one unified detection for simplified processing, enabling detection consolidation workflows
- **After filtering blocks** (e.g., Detections Filter) to merge filtered detections that meet specific criteria into a single detection (e.g., merge filtered detections by class, combine detections after filtering, consolidate filtered results), enabling filtered detection consolidation
- **Before crop blocks** to create a single crop region from multiple detections (e.g., crop a region containing multiple objects, extract area encompassing multiple detections, create unified crop region), enabling multi-detection region extraction
- **Before zone-based blocks** (e.g., Polygon Zone, Dynamic Zone) to define zones based on merged detection regions (e.g., create zones from merged detection areas, establish monitoring zones from merged detections, define regions from consolidated detections), enabling zone creation from merged detections
- **Before visualization blocks** to display simplified merged detections instead of multiple individual detections (e.g., visualize consolidated detection regions, display merged bounding boxes, show simplified detection representation), enabling simplified visualization outputs
- **Before analysis blocks** that benefit from simplified detection representation (e.g., analyze merged detection regions, process consolidated detections, work with simplified detection data), enabling simplified detection analysis workflows
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
        description="Detection predictions containing multiple detections to merge into a single detection. Supports object detection, instance segmentation, or keypoint detection predictions. All input detections will be combined into one merged detection with a union bounding box that encompasses all input detections. If empty detections are provided, the block returns an empty detection result. The merged detection will contain all input detections within its bounding box boundaries.",
        examples=["$steps.object_detection_model.predictions"],
    )
    class_name: str = Field(
        default="merged_detection",
        description="Class name to assign to the merged detection. The merged detection will use this class name in its data. Default is 'merged_detection' to indicate that this is a merged detection. You can customize this to represent a specific category or to indicate the purpose of the merged detection (e.g., 'crowd', 'group', 'region'). This class name will be stored in the detection's data dictionary.",
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
