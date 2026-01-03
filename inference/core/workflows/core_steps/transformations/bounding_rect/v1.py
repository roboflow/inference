from typing import List, Literal, Optional, Tuple, Type

import cv2 as cv
import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.workflows.execution_engine.constants import (
    BOUNDING_RECT_ANGLE_KEY_IN_SV_DETECTIONS,
    BOUNDING_RECT_HEIGHT_KEY_IN_SV_DETECTIONS,
    BOUNDING_RECT_RECT_KEY_IN_SV_DETECTIONS,
    BOUNDING_RECT_WIDTH_KEY_IN_SV_DETECTIONS,
)
from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

OUTPUT_KEY: str = "detections_with_rect"

SHORT_DESCRIPTION = "Find the minimal bounding box surrounding the detected polygon."
LONG_DESCRIPTION = """
Calculate the minimum rotated bounding rectangle around polygon segmentation masks, converting complex polygon shapes into simplified rectangular bounding boxes with orientation information for zone creation, region simplification, and rectangular area approximation based on detected object shapes.

## How This Block Works

This block processes instance segmentation predictions with polygon masks and calculates the minimum rotated bounding rectangle (smallest rectangle that can enclose the polygon) for each detection. The block:

1. Receives instance segmentation predictions containing polygon masks (validates that masks are present)
2. Processes each detection's polygon mask individually
3. Extracts the largest contour from the polygon mask (handles multi-contour masks by selecting the largest)
4. Calculates the minimum rotated bounding rectangle using OpenCV's minAreaRect:
   - Finds the smallest rotated rectangle that can completely enclose the polygon
   - Determines the rectangle's center point, dimensions (width, height), and rotation angle
   - Computes the four corner points of the rotated rectangle
5. Updates the detection's mask to be a rectangular mask matching the calculated bounding rectangle (converts the rotated rectangle polygon back to a mask format)
6. Updates the detection's axis-aligned bounding box (xyxy) to the bounding box of the rotated rectangle (fits the rotated rectangle into an axis-aligned box)
7. Stores additional rectangle metadata in the detection data:
   - Rectangle corner coordinates (rotated rectangle points)
   - Rectangle width and height (dimensions of the rotated rectangle)
   - Rectangle angle (rotation angle in degrees)
8. Merges all processed detections and returns them with updated masks, bounding boxes, and rectangle metadata

The block transforms complex polygon shapes into simplified rectangular representations, preserving orientation information through the rotation angle. This is particularly useful when you need to create zones or regions based on detected object shapes (e.g., sports fields, road segments, marked areas) and want to simplify them to rectangular approximations. The minimum rotated rectangle provides the most compact rectangular representation of the polygon, potentially at an angle to minimize area.

## Common Use Cases

- **Zone Creation from Object Shapes**: Convert detected polygon shapes into rectangular zones for area monitoring or analysis (e.g., create zones from basketball court detections, generate road segment zones from road markings, create rectangular regions from zebra crossing detections), enabling zone-based workflows from complex shapes
- **Region Simplification**: Simplify complex polygon shapes to rectangular approximations for easier processing (e.g., simplify irregular segmentation masks to rectangles, convert complex shapes to rectangular regions, approximate polygon areas with rectangles), enabling simplified region processing
- **Rotated Region Detection**: Detect and extract rotated rectangular regions from polygon detections (e.g., find rotated parking spaces from segmentation, detect angled road markings as rectangles, extract rotated objects as rectangular zones), enabling rotation-aware region extraction
- **Area Approximation**: Approximate polygon areas with compact rectangular bounding boxes (e.g., approximate sports field areas with minimal rectangles, estimate region sizes using rotated bounding boxes, calculate compact rectangular areas from complex shapes), enabling area estimation with rectangular approximations
- **Shape Normalization**: Normalize polygon shapes to rectangular representations for standardized processing (e.g., normalize detected shapes to rectangles for consistent analysis, standardize polygon regions to rectangular format, convert variable shapes to uniform rectangular regions), enabling shape normalization workflows
- **Multi-Object Zone Extraction**: Extract rectangular zones from multiple detected polygon objects (e.g., create zones from multiple road segment detections, generate rectangular regions from multiple field detections, extract zones from various marked area detections), enabling multi-object zone creation workflows

## Connecting to Other Blocks

This block receives instance segmentation predictions with polygon masks and produces detections with rectangular masks and bounding boxes:

- **After instance segmentation blocks** to convert polygon masks to rectangular bounding boxes for zone creation or simplified processing, enabling rectangular zone generation from complex shapes
- **Before zone-based blocks** (e.g., Polygon Zone, Dynamic Zone) to prepare rectangular regions for zone-based workflows (e.g., create zones from simplified rectangles, use rectangular approximations for zone monitoring, enable zone workflows with rectangular regions), enabling zone-based workflows with simplified shapes
- **After filtering blocks** (e.g., Detections Filter) to process only specific polygon detections before converting to rectangles (e.g., filter detections by class before rectangular conversion, select specific polygon types for rectangle extraction, prepare filtered detections for zone creation), enabling selective rectangle extraction
- **Before crop blocks** to extract rectangular regions from polygon detections (e.g., crop rotated rectangular regions from polygon shapes, extract rectangular areas from complex detections, prepare rectangular crop regions from polygons), enabling rectangular region extraction
- **Before visualization blocks** to display simplified rectangular representations of complex polygons (e.g., visualize rectangular approximations of polygons, display rotated bounding rectangles, show simplified rectangular zones), enabling rectangular visualization outputs
- **Before analysis blocks** that work better with rectangular regions than complex polygons (e.g., analyze rectangular zones instead of polygons, process simplified rectangular regions, work with normalized rectangular shapes), enabling simplified region analysis workflows
"""


class BoundingRectManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Bounding Rectangle",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
            "ui_manifest": {
                "section": "transformation",
                "icon": "fal fa-rectangles-mixed",
                "blockPriority": 5,
            },
        }
    )
    type: Literal["roboflow_core/bounding_rect@v1"]
    predictions: Selector(
        kind=[
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
        ]
    ) = Field(  # type: ignore
        description="Instance segmentation predictions containing polygon masks. The block requires masks to be present - it will raise an error if masks are missing. Each detection's polygon mask will be processed to calculate the minimum rotated bounding rectangle. The mask should contain polygon shapes that you want to convert to rectangular bounding boxes. Detections are processed individually, with the largest contour extracted from each mask. The block outputs detections with updated masks (rectangular), updated bounding boxes (axis-aligned boxes of the rotated rectangles), and additional rectangle metadata stored in detection.data (rectangle coordinates, width, height, angle).",
        examples=["$segmentation.predictions"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name=OUTPUT_KEY, kind=[INSTANCE_SEGMENTATION_PREDICTION_KIND]
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


def calculate_minimum_bounding_rectangle(
    mask: np.ndarray,
) -> Tuple[np.array, float, float, float]:
    contours = sv.mask_to_polygons(mask)
    largest_contour = max(contours, key=len)

    rect = cv.minAreaRect(largest_contour)
    box = cv.boxPoints(rect)
    box = np.array(box, dtype=int)

    width, height = rect[1]
    angle = rect[2]
    return box, width, height, angle


class BoundingRectBlockV1(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BoundingRectManifest

    def run(
        self,
        predictions: sv.Detections,
    ) -> BlockResult:
        if predictions.mask is None:
            raise ValueError(
                "Mask missing. This block operates on output from segmentation model."
            )
        to_be_merged = []
        for i in range(len(predictions)):
            # copy
            det = predictions[i]

            rect, width, height, angle = calculate_minimum_bounding_rectangle(
                det.mask[0]
            )

            det.mask = np.array(
                [
                    sv.polygon_to_mask(
                        polygon=np.around(rect).astype(np.int32),
                        resolution_wh=(det.mask[0].shape[1], det.mask[0].shape[0]),
                    ).astype(bool)
                ]
            )
            det.xyxy = np.array(
                [sv.polygon_to_xyxy(polygon=np.around(rect).astype(np.int32))]
            )

            det[BOUNDING_RECT_RECT_KEY_IN_SV_DETECTIONS] = np.array(
                [rect], dtype=np.float16
            )
            det[BOUNDING_RECT_WIDTH_KEY_IN_SV_DETECTIONS] = np.array(
                [width], dtype=np.float16
            )
            det[BOUNDING_RECT_HEIGHT_KEY_IN_SV_DETECTIONS] = np.array(
                [height], dtype=np.float16
            )
            det[BOUNDING_RECT_ANGLE_KEY_IN_SV_DETECTIONS] = np.array(
                [angle], dtype=np.float16
            )

            to_be_merged.append(det)

        return {OUTPUT_KEY: sv.Detections.merge(to_be_merged)}
