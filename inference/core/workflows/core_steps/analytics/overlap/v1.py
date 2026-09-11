from functools import lru_cache
from typing import List, Optional, Union

import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field
from supervision.detection.utils.internal import get_data_item
from typing_extensions import Literal, Type

from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

OUTPUT_KEY: str = "overlaps"
SHORT_DESCRIPTION = "Filter objects overlapping some other class"
LONG_DESCRIPTION = """
Filter detection predictions to keep only objects that overlap with instances of a specified class, enabling spatial relationship filtering to identify objects that are positioned relative to other objects (e.g., people on bicycles, items on pallets, objects in containers).

## How This Block Works

This block filters detections based on spatial overlap relationships with a specified overlap class. The block:

1. Takes detection predictions (object detection or instance segmentation) and an overlap class name as input
2. Separates detections into two groups:
   - **Overlap class detections**: Objects matching the specified `overlap_class_name` (e.g., "bicycle", "pallet", "car")
   - **Other detections**: All remaining objects that may overlap with the overlap class
3. For each overlap class detection, identifies other detections that spatially overlap with it using one of two overlap modes:
   - **Center Overlap**: Checks if the center point of other detections falls within the overlap class bounding box (more precise, requires the center to be inside)
   - **Any Overlap**: Checks if there's any spatial intersection between bounding boxes (more lenient, any overlap counts)
4. Collects all detections that overlap with any overlap class instance
5. Filters out the overlap class detections themselves from the output
6. Returns only the overlapping detections (objects that are positioned relative to the overlap class)

The block effectively answers: "Which objects are overlapping with instances of class X?" For example, if you specify "bicycle" as the overlap class, the block finds people or other objects that overlap with bicycles, but removes the bicycles themselves from the output. This enables workflows to identify objects that have spatial relationships with specific reference classes, such as identifying items on surfaces, objects in containers, or people on vehicles.

## Common Use Cases

- **Person-on-Vehicle Detection**: Identify people on bicycles, motorcycles, or other vehicles by using the vehicle class as the overlap class (e.g., filter for people overlapping with "bicycle" detections), enabling detection of riders, passengers, or people using vehicles
- **Items on Surfaces**: Find objects positioned on pallets, tables, or shelves by using the surface class as the overlap class (e.g., filter for items overlapping with "pallet" detections), enabling inventory tracking, object counting on surfaces, or surface occupancy analysis
- **Objects in Containers**: Identify items inside containers, boxes, or vehicles by using the container class as the overlap class (e.g., filter for objects overlapping with "container" detections), enabling content detection, loading verification, or container monitoring
- **Spatial Relationship Filtering**: Filter detections based on proximity or containment relationships (e.g., find all objects that are inside or overlapping with a specific class), enabling conditional processing based on spatial arrangements
- **Nested Object Detection**: Identify objects that are part of or attached to other objects (e.g., find equipment attached to vehicles, accessories on people), enabling detection of composite objects or object relationships
- **Zone-Based Filtering**: Use overlap class as a reference zone to find objects that intersect with specific regions (e.g., filter objects overlapping with "parking_space" class), enabling zone-based analysis and conditional detection filtering

## Connecting to Other Blocks

The filtered overlapping detections from this block can be connected to:

- **Detection model blocks** (e.g., Object Detection Model, Instance Segmentation Model) to receive predictions that are filtered to show only objects overlapping with a specified reference class, enabling spatial relationship analysis
- **Visualization blocks** (e.g., Bounding Box Visualization, Polygon Visualization, Label Visualization) to display only the overlapping objects, highlighting objects that have spatial relationships with the reference class
- **Counting and analytics blocks** (e.g., Line Counter, Time in Zone, Velocity) to count or analyze only overlapping objects (e.g., count people on bicycles, track items on pallets), providing metrics for spatially related objects
- **Data storage blocks** (e.g., Local File Sink, CSV Formatter, Roboflow Dataset Upload, Webhook Sink) to save or transmit filtered overlapping detection results, storing data about objects with specific spatial relationships
- **Filtering blocks** (e.g., Detections Filter) to apply additional filtering criteria to the overlapping detections, enabling multi-stage filtering workflows
- **Flow control blocks** (e.g., Continue If) to conditionally trigger downstream processing based on whether overlapping objects are detected, enabling workflows that respond to spatial relationships
"""


class OverlapManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Overlap Filter",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "analytics",
            "ui_manifest": {
                "section": "flow_control",
                "icon": "far fa-square-o",
                "blockPriority": 1.5,
            },
        }
    )
    type: Literal["roboflow_core/overlap@v1"]
    predictions: Selector(
        kind=[OBJECT_DETECTION_PREDICTION_KIND, INSTANCE_SEGMENTATION_PREDICTION_KIND]
    ) = Field(  # type: ignore
        description="Detection predictions (object detection or instance segmentation) containing objects that may overlap with the specified overlap class. The block identifies detections matching the overlap_class_name and finds other detections that spatially overlap with them. Only the overlapping detections (not the overlap class itself) are returned in the output.",
        examples=["$steps.object_detection_model.predictions"],
    )
    overlap_type: Literal["Center Overlap", "Any Overlap"] = Field(
        default="Center Overlap",
        description="Method for determining spatial overlap between detections. 'Center Overlap' checks if the center point of other detections falls within the overlap class bounding box (more precise, requires center to be inside). 'Any Overlap' checks if there's any spatial intersection between bounding boxes (more lenient, any overlap counts). Center Overlap is stricter and better for containment relationships, while Any Overlap is more inclusive and better for detecting any proximity or partial overlap.",
        examples=["Center Overlap", "Any Overlap"],
    )
    overlap_class_name: Union[str] = Field(
        description="Class name of the reference objects used for overlap detection. Detections matching this class name are used as reference points, and other detections that overlap with these reference objects are kept in the output. The overlap class detections themselves are removed from the results. Example: Use 'bicycle' to find people or objects overlapping with bicycles; use 'pallet' to find items on pallets.",
        json_schema_extra={
            "hide_description": True,
        },
    )

    @classmethod
    @lru_cache(maxsize=None)
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name=OUTPUT_KEY,
                kind=[
                    OBJECT_DETECTION_PREDICTION_KIND,
                ],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class OverlapBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return OverlapManifest

    @classmethod
    def coords_overlap(
        cls,
        overlap: list[int],
        other: list[int],
        overlap_type: Literal["Center Overlap", "Any Overlap"],
    ):
        """Check bbox-to-bbox overlap (fallback when masks unavailable)."""
        # coords are [x1, y1, x2, y2]
        if overlap_type == "Center Overlap":
            size = [other[2] - other[0], other[3] - other[1]]
            x, y = [other[0] + size[0] / 2, other[1] + size[1] / 2]
            return (
                x > overlap[0] and x < overlap[2] and y > overlap[1] and y < overlap[3]
            )
        else:
            return not (
                other[2] < overlap[0]
                or other[0] > overlap[2]
                or other[3] < overlap[1]
                or other[1] > overlap[3]
            )

    @classmethod
    def mask_overlap(
        cls,
        overlap_mask: np.ndarray,
        other_mask: np.ndarray,
        overlap_type: Literal["Center Overlap", "Any Overlap"],
    ) -> bool:
        """Check mask-to-mask overlap for instance segmentation.

        Args:
            overlap_mask: Binary mask for overlap class detection (H, W)
            other_mask: Binary mask for other detection (H, W)
            overlap_type: Type of overlap to check

        Returns:
            True if masks overlap according to overlap_type

        Note:
            For "Center Overlap": Checks if the center of other's bounding box
            falls within the overlap mask (mask-aware center check).
            For "Any Overlap": Checks if masks have any pixel intersection.
        """
        if overlap_type == "Center Overlap":
            # Find center of other_mask's bounding box
            other_coords = np.argwhere(other_mask > 0)
            if len(other_coords) == 0:
                return False  # Empty mask

            # Center of bounding box in (row, col) format
            min_row, min_col = other_coords.min(axis=0)
            max_row, max_col = other_coords.max(axis=0)
            center_row = (min_row + max_row) // 2
            center_col = (min_col + max_col) // 2

            # Check if center point is inside overlap mask
            if (
                0 <= center_row < overlap_mask.shape[0]
                and 0 <= center_col < overlap_mask.shape[1]
            ):
                return overlap_mask[center_row, center_col] > 0
            return False

        else:  # "Any Overlap"
            # Check if there's any pixel intersection
            intersection = np.logical_and(overlap_mask > 0, other_mask > 0)
            return np.any(intersection)

    def run(
        self,
        predictions: sv.Detections,
        overlap_type: Literal["Center Overlap", "Any Overlap"],
        overlap_class_name: str,
    ) -> BlockResult:
        """Filter predictions to keep only objects overlapping with overlap_class_name.

        Uses mask-aware overlap detection when masks are available (instance segmentation),
        falls back to bounding box overlap for object detection.
        """
        # Check if masks are available
        has_masks = predictions.mask is not None and len(predictions.mask) > 0

        # Separate overlap class detections from others
        overlaps = []
        overlap_indices = []
        others = {}
        for i in range(len(predictions.xyxy)):
            data = get_data_item(predictions.data, i)
            if data["class_name"] == overlap_class_name:
                if has_masks:
                    overlaps.append((i, predictions.mask[i]))
                else:
                    overlaps.append((i, predictions.xyxy[i]))
                overlap_indices.append(i)
            else:
                if has_masks:
                    others[i] = predictions.mask[i]
                else:
                    others[i] = predictions.xyxy[i]

        # Find overlapping objects
        idx = set()
        for overlap_idx, overlap_data in overlaps:
            if not others:
                break

            if has_masks:
                # Use mask-aware overlap detection
                overlapped = {
                    k
                    for k in others
                    if OverlapBlockV1.mask_overlap(overlap_data, others[k], overlap_type)
                }
            else:
                # Fall back to bbox overlap
                overlapped = {
                    k
                    for k in others
                    if OverlapBlockV1.coords_overlap(
                        predictions.xyxy[overlap_idx], others[k], overlap_type
                    )
                }

            # Once overlapped, don't need to check again
            for k in overlapped:
                del others[k]

            idx = idx.union(overlapped)

        return {OUTPUT_KEY: predictions[list(idx)]}
