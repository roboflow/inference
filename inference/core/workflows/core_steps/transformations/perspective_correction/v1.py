import math
from typing import List, Optional, Tuple, Union

import cv2 as cv
import numpy as np
import supervision as sv
from pydantic import AliasChoices, ConfigDict, Field
from typing_extensions import Literal, Type

from inference.core.workflows.execution_engine.constants import (
    KEYPOINTS_XY_KEY_IN_SV_DETECTIONS,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    IMAGE_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

OUTPUT_DETECTIONS_KEY: str = "corrected_coordinates"
OUTPUT_IMAGE_KEY: str = "warped_image"
OUTPUT_EXTENDED_TRANSFORMED_RECT_WIDTH_KEY: str = "extended_transformed_rect_width"
OUTPUT_EXTENDED_TRANSFORMED_RECT_HEIGHT_KEY: str = "extended_transformed_rect_height"
TYPE: str = "PerspectiveCorrection"
SHORT_DESCRIPTION = (
    "Adjust detection coordinates from a polygon-defined plane "
    "to a straight rectangular plane with specified width and height."
)
LONG_DESCRIPTION = """
Transform detection coordinates and optionally images from a perspective view to a top-down orthographic view using perspective transformation, correcting camera angle distortions to enable accurate measurements, top-down analysis, and coordinate normalization for scenarios where objects are viewed at an angle (e.g., surveillance cameras, aerial imagery, or tilted camera setups).

## How This Block Works

This block corrects perspective distortion by transforming coordinates from a perspective view (where objects appear smaller when further away and angles are distorted) to a top-down orthographic view (as if the camera were directly above the scene). The block:

1. Receives input data: images and/or detections (object detection or instance segmentation predictions), along with perspective polygons defining regions to transform
2. Processes perspective polygons:
   - Selects the largest polygon from provided polygons (if multiple are provided)
   - Sorts polygon vertices in clockwise order and orients them starting from the leftmost bottom vertex
   - Ensures proper polygon ordering for transformation matrix calculation
3. Optionally extends perspective polygons to contain all detections:
   - If `extend_perspective_polygon_by_detections_anchor` is set, extends the polygon to ensure all detection anchor points (or entire bounding boxes if "ALL" is specified) are contained within the polygon
   - Calculates extension amounts needed to contain detections outside the original polygon
   - Adjusts polygon vertices to create a larger region that encompasses all detections
4. Generates perspective transformation matrix:
   - Maps the source polygon (4 vertices in the perspective view) to a destination rectangle (top-down view) with specified width and height
   - Uses OpenCV's `getPerspectiveTransform` to compute the 3x3 transformation matrix
   - Handles extended dimensions when polygon extension is enabled
5. Applies perspective transformation to detections (if provided):
   - Transforms bounding box coordinates from perspective view to top-down coordinates
   - Transforms instance segmentation masks by converting masks to polygons, transforming polygon vertices, and converting back to masks in the new coordinate space
   - Transforms keypoint coordinates for keypoint detection predictions
   - Updates all coordinate data to reflect the corrected perspective
6. Optionally warps images (if `warp_image` is True):
   - Applies the perspective transformation to the entire image using OpenCV's `warpPerspective`
   - Produces a top-down view of the image with corrected perspective
   - Outputs the warped image at the specified transformed rectangle dimensions (plus any extensions)
7. Returns corrected outputs:
   - `corrected_coordinates`: Detections with transformed coordinates in the top-down coordinate space
   - `warped_image`: Perspective-corrected image (if image warping is enabled)
   - `extended_transformed_rect_width` and `extended_transformed_rect_height`: Final dimensions including any polygon extensions

The transformation effectively "unwarps" the perspective distortion, making coordinates and images appear as if viewed from directly above. This is useful for accurate measurements, area calculations, distance measurements, and spatial analysis where perspective distortion would otherwise introduce errors.

## Common Use Cases

- **Top-Down Analysis**: Correct perspective distortion for top-down analysis and measurement (e.g., surveillance camera analysis, overhead view generation, top-down coordinate normalization), enabling top-down analysis workflows
- **Accurate Measurements**: Enable accurate distance, area, and size measurements by removing perspective distortion (e.g., measure object sizes in real-world units, calculate areas accurately, measure distances without distortion), enabling measurement workflows
- **Spatial Analysis**: Perform spatial analysis and coordinate-based operations on corrected coordinates (e.g., zone-based analysis, spatial tracking, coordinate-based filtering), enabling spatial analysis workflows
- **Aerial and Overhead Imagery**: Process aerial imagery or overhead camera feeds with perspective correction (e.g., drone imagery analysis, overhead camera processing, satellite image analysis), enabling aerial analysis workflows
- **Quality Control and Inspection**: Correct perspective for quality control and inspection workflows (e.g., manufacturing inspection, product quality checks, defect detection with accurate measurements), enabling quality control workflows
- **Indoor Navigation and Mapping**: Correct perspective for indoor navigation and mapping applications (e.g., floor plan generation, indoor mapping, navigation systems), enabling mapping workflows

## Connecting to Other Blocks

This block receives images and/or detections and produces perspective-corrected outputs:

- **After detection models** to correct coordinates for accurate analysis (e.g., object detection with perspective correction, instance segmentation with corrected coordinates), enabling detection-to-correction workflows
- **After zone or polygon definition blocks** to use defined regions as perspective polygons (e.g., use polygon zones as perspective regions, apply correction to specific regions), enabling zone-to-correction workflows
- **Before measurement blocks** to enable accurate measurements on corrected coordinates (e.g., distance measurement with corrected coordinates, size measurement on top-down view, area calculation on corrected coordinates), enabling correction-to-measurement workflows
- **Before analytics blocks** to perform analytics on corrected coordinates (e.g., zone analytics with corrected coordinates, tracking with top-down view, path analysis with corrected paths), enabling correction-to-analytics workflows
- **Before visualization blocks** to visualize corrected coordinates and warped images (e.g., display top-down view, visualize corrected detections, show perspective-corrected results), enabling correction-to-visualization workflows
- **In workflow outputs** to provide perspective-corrected final outputs (e.g., top-down coordinate outputs, corrected detection outputs, warped image outputs), enabling correction-to-output workflows

## Requirements

This block requires either images or predictions (detections) as input. The `perspective_polygons` parameter must contain at least one polygon with exactly 4 vertices defining the region to transform. Polygons can be provided as a list of 4 coordinate pairs `[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]` or as NumPy arrays. If multiple polygons are provided, the largest polygon (by area) is selected for each batch element. The `transformed_rect_width` and `transformed_rect_height` parameters define the dimensions of the output top-down rectangle. The block uses OpenCV's perspective transformation functions, which require proper polygon ordering and valid coordinate data. If polygon extension is enabled, the output dimensions are automatically adjusted to include the extended regions.
"""
ALL_POSITIONS = "ALL"


class PerspectiveCorrectionManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Perspective Correction",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
            "ui_manifest": {
                "section": "advanced",
                "icon": "fal fa-toolbox",
                "blockPriority": 2,
                "opencv": True,
            },
        }
    )
    type: Literal["roboflow_core/perspective_correction@v1", "PerspectiveCorrection"]
    predictions: Optional[
        Selector(
            kind=[
                OBJECT_DETECTION_PREDICTION_KIND,
                INSTANCE_SEGMENTATION_PREDICTION_KIND,
            ]
        )
    ] = Field(  # type: ignore
        description="Optional object detection or instance segmentation predictions to transform. If provided, bounding boxes, masks, and keypoints are transformed to the top-down coordinate space. If not provided, only image warping is performed (if enabled). Either predictions or images must be provided.",
        default=None,
        examples=[
            "$steps.object_detection_model.predictions",
            "$steps.instance_segmentation_model.predictions",
        ],
    )
    images: Selector(kind=[IMAGE_KIND]) = Field(
        title="Image to Crop",
        description="Input images to optionally warp to top-down view. Required if warp_image is True. Images are transformed using the perspective transformation matrix to produce top-down views. If only images are provided (no predictions), only image warping is performed.",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("images", "image"),
    )
    perspective_polygons: Union[list, Selector(kind=[LIST_OF_VALUES_KIND])] = Field(  # type: ignore
        description="Perspective polygons defining regions to transform from perspective view to top-down view. Each polygon must consist of exactly 4 vertices (coordinates). Format: list of 4 coordinate pairs [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] or NumPy arrays. If multiple polygons are provided for a batch element, the largest polygon (by area) is selected. The polygon defines the source region in the perspective view that will be mapped to the destination rectangle.",
        examples=[
            "$steps.perspective_wrap.zones",
            [[100, 100], [500, 100], [500, 400], [100, 400]],
        ],
    )
    transformed_rect_width: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Width of the destination rectangle in the top-down view (in pixels). The perspective polygon is transformed to fit this width. Coordinates are scaled to match this dimension. If polygon extension is enabled, the actual output width may be larger to accommodate extended regions.",
        default=1000,
        examples=[1000, 1920],
    )
    transformed_rect_height: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Height of the destination rectangle in the top-down view (in pixels). The perspective polygon is transformed to fit this height. Coordinates are scaled to match this dimension. If polygon extension is enabled, the actual output height may be larger to accommodate extended regions.",
        default=1000,
        examples=[1000, 1080],
    )
    extend_perspective_polygon_by_detections_anchor: Union[str, Selector(kind=[STRING_KIND])] = Field(  # type: ignore
        description=f"Optional setting to extend the perspective polygon to contain all detection anchor points. If set to a Position value ({', '.join(sv.Position.list())}), extends the polygon to contain that anchor point from all detections. If set to '{ALL_POSITIONS}', extends to contain entire bounding boxes (all corners). Empty string (default) disables extension. Extension ensures all detections are within the transformed region, automatically adjusting polygon boundaries and output dimensions.",
        default="",
        examples=["CENTER", "BOTTOM_CENTER", "ALL"],
    )
    warp_image: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(  # type: ignore
        description="If True, applies perspective transformation to the input image, producing a warped image in the top-down view. The warped image shows the perspective-corrected view at the specified transformed rectangle dimensions (plus any extensions). If False (default), only detection coordinates are transformed, and the original image is returned unchanged. Images must be provided if this is True.",
        default=False,
        examples=[False, True],
    )

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images", "predictions"]

    @classmethod
    def get_parameters_accepting_batches_and_scalars(cls) -> List[str]:
        return [
            "perspective_polygons",
            "transformed_rect_width",
            "transformed_rect_height",
        ]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name=OUTPUT_DETECTIONS_KEY,
                kind=[
                    OBJECT_DETECTION_PREDICTION_KIND,
                    INSTANCE_SEGMENTATION_PREDICTION_KIND,
                ],
            ),
            OutputDefinition(
                name=OUTPUT_IMAGE_KEY,
                kind=[
                    IMAGE_KIND,
                ],
            ),
            OutputDefinition(
                name=OUTPUT_EXTENDED_TRANSFORMED_RECT_WIDTH_KEY,
                kind=[
                    INTEGER_KIND,
                ],
            ),
            OutputDefinition(
                name=OUTPUT_EXTENDED_TRANSFORMED_RECT_HEIGHT_KEY,
                kind=[
                    INTEGER_KIND,
                ],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


def pick_largest_perspective_polygons(
    perspective_polygons_batch: Union[
        List[np.ndarray],
        List[List[np.ndarray]],
        List[List[List[int]]],
        List[List[List[List[int]]]],
    ],
) -> List[np.ndarray]:
    if not isinstance(perspective_polygons_batch, (list, Batch)):
        raise ValueError("Unexpected type of input")
    if not perspective_polygons_batch:
        raise ValueError("Unexpected empty batch")
    if len(perspective_polygons_batch) == 4 and all(
        isinstance(p, list) and len(p) == 2 for p in perspective_polygons_batch
    ):
        perspective_polygons_batch = [perspective_polygons_batch]

    largest_perspective_polygons: List[np.ndarray] = []
    for polygons in perspective_polygons_batch:
        if polygons is None:
            continue
        if not isinstance(polygons, list) and not isinstance(polygons, np.ndarray):
            raise ValueError("Unexpected type of batch element")
        if len(polygons) == 0:
            raise ValueError("Unexpected empty batch element")
        if isinstance(polygons, np.ndarray):
            if polygons.shape != (4, 2):
                raise ValueError("Unexpected shape of batch element")
            largest_perspective_polygons.append(polygons)
            continue
        if len(polygons) == 4 and all(
            isinstance(p, list) and len(p) == 2 for p in polygons
        ):
            largest_perspective_polygons.append(np.array(polygons))
            continue
        polygons = [p if isinstance(p, np.ndarray) else np.array(p) for p in polygons]
        polygons = [p for p in polygons if p.shape == (4, 2)]
        if not polygons:
            raise ValueError("No batch element consists of 4 vertices")
        polygons = [np.around(p).astype(np.int32) for p in polygons]
        largest_polygon = max(polygons, key=lambda p: cv.contourArea(p))
        largest_perspective_polygons.append(largest_polygon)
    return largest_perspective_polygons


def sort_polygon_vertices_clockwise(polygon: np.ndarray) -> np.ndarray:
    x_center = min(polygon[:, 0]) / 2 + max(polygon[:, 0]) / 2
    y_center = min(polygon[:, 1]) / 2 + max(polygon[:, 1]) / 2
    angle = lambda p: math.atan2(x_center - p[0], y_center - p[1])
    return np.array(sorted(polygon.tolist(), key=angle, reverse=True))


def roll_polygon_vertices_to_start_from_leftmost_bottom(
    polygon: np.ndarray,
) -> np.ndarray:
    x_min = min(polygon[:, 0])
    x_max = max(polygon[:, 0])
    y_min = min(polygon[:, 1])
    y_max = max(polygon[:, 1])
    leftmost_bottom_rect = [
        [x_min, y_max],
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
    ]
    min_dist = sum(
        ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        for (x1, y1), (x2, y2) in zip(leftmost_bottom_rect, polygon)
    )
    closest = polygon
    for shift in range(4):
        rolled = np.roll(polygon, shift=(0, shift), axis=(0, 0))
        dist = sum(
            ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            for (x1, y1), (x2, y2) in zip(leftmost_bottom_rect, rolled)
        )
        if dist < min_dist or (dist == min_dist and rolled[0][0] < closest[0][0]):
            min_dist = dist
            closest = rolled
    return closest


def ccw(x1, y1, x2, y2, x3, y3):
    return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) > 0


def calculate_line_coeffs(
    x1: int, y1: int, x2: int, y2: int
) -> Tuple[Optional[float], float]:
    if x1 == x2:
        return None, x1
    # Solved a and b for ax + b = y
    return (y2 - y1) / (x2 - x1), (y1 * x2 - y2 * x1) / (x2 - x1)


def calculate_line_intercept_to_contain_point(
    a: Optional[float],
    x: int,
    y: int,
) -> float:
    if a is None:
        return x
    return y - a * x


def solve_line_intersection(
    a1: Optional[float],
    b1: float,
    a2: Optional[float],
    b2: float,
) -> Tuple[float, float]:
    if a1 is None and a2 is None:
        raise ValueError("Both lines are vertical")
    if a1 is None:
        x = b1
        y = a2 * x + b2
    elif a2 is None:
        x = b2
        y = a1 * x + b1
    else:
        x = (b2 - b1) / (a1 - a2)
        y = a1 * x + b1
    return x, y


def calculate_vertices_to_contain_point(
    vertex_1: np.ndarray,
    vertex_2: np.ndarray,
    vertex_3_from_1: np.ndarray,
    vertex_4_from_2: np.ndarray,
    x: int,
    y: int,
) -> Tuple[np.ndarray, np.ndarray]:
    a, _ = calculate_line_coeffs(
        x1=vertex_1[0],
        y1=vertex_1[1],
        x2=vertex_2[0],
        y2=vertex_2[1],
    )
    b = calculate_line_intercept_to_contain_point(
        a=a,
        x=x,
        y=y,
    )
    a_3_1, b_3_1 = calculate_line_coeffs(
        x1=vertex_1[0],
        y1=vertex_1[1],
        x2=vertex_3_from_1[0],
        y2=vertex_3_from_1[1],
    )
    vertex_1 = (
        np.array(
            solve_line_intersection(
                a1=a_3_1,
                b1=b_3_1,
                a2=a,
                b2=b,
            )
        )
        .round()
        .astype(int)
    )
    a_4_2, b_4_2 = calculate_line_coeffs(
        x1=vertex_2[0],
        y1=vertex_2[1],
        x2=vertex_4_from_2[0],
        y2=vertex_4_from_2[1],
    )
    vertex_2 = (
        np.array(
            solve_line_intersection(
                a1=a_4_2,
                b1=b_4_2,
                a2=a,
                b2=b,
            )
        )
        .round()
        .astype(int)
    )
    return vertex_1, vertex_2


def extend_perspective_polygon(
    polygon: List[np.ndarray],
    detections: sv.Detections,
    bbox_position: Union[sv.Position, Literal[ALL_POSITIONS]],
) -> Tuple[np.ndarray, float, float, float, float]:
    if not bbox_position:
        return polygon
    bottom_left, top_left, top_right, bottom_right = polygon
    extended_width = 0
    extended_height = 0
    original_width = max(
        (
            (bottom_left[0] - bottom_right[0]) ** 2
            + (bottom_left[1] - bottom_right[1]) ** 2
        )
        ** 0.5,
        ((top_left[0] - top_right[0]) ** 2 + (top_left[1] - top_right[1]) ** 2) ** 0.5,
    )
    original_height = max(
        ((bottom_left[0] - top_left[0]) ** 2 + (bottom_left[1] - top_left[1]) ** 2)
        ** 0.5,
        ((bottom_right[0] - top_right[0]) ** 2 + (bottom_right[1] - top_right[1]) ** 2)
        ** 0.5,
    )
    for i in range(len(detections)):
        det = detections[i]
        # extend to the left
        points = []
        if bbox_position == ALL_POSITIONS:
            points.append(
                det.get_anchors_coordinates(anchor=sv.Position.BOTTOM_LEFT)[0]
            )
            points.append(det.get_anchors_coordinates(anchor=sv.Position.TOP_LEFT)[0])
        else:
            points.append(det.get_anchors_coordinates(anchor=bbox_position)[0])
        for x, y in points:
            if (
                cv.pointPolygonTest(
                    np.array([bottom_left, top_left, top_right, bottom_right]),
                    (x, y),
                    False,
                )
                >= 0
            ):
                continue
            if not ccw(
                x1=bottom_left[0],
                y1=bottom_left[1],
                x2=top_left[0],
                y2=top_left[1],
                x3=x,
                y3=y,
            ):
                original_bottom_left = bottom_left
                original_top_left = top_left
                bottom_left, top_left = calculate_vertices_to_contain_point(
                    vertex_1=original_bottom_left,
                    vertex_2=original_top_left,
                    vertex_3_from_1=bottom_right,
                    vertex_4_from_2=top_right,
                    x=x,
                    y=y,
                )
                extended_width += (
                    (bottom_left[0] - original_bottom_left[0]) ** 2
                    + (bottom_left[1] - original_bottom_left[1]) ** 2
                ) ** 0.5
        # extend to the right
        points = []
        if bbox_position == ALL_POSITIONS:
            points.append(
                det.get_anchors_coordinates(anchor=sv.Position.BOTTOM_RIGHT)[0]
            )
            points.append(det.get_anchors_coordinates(anchor=sv.Position.TOP_RIGHT)[0])
        else:
            points.append(det.get_anchors_coordinates(anchor=bbox_position)[0])
        for x, y in points:
            if (
                cv.pointPolygonTest(
                    np.array([bottom_left, top_left, top_right, bottom_right]),
                    (x, y),
                    False,
                )
                >= 0
            ):
                continue
            if not ccw(
                x1=top_right[0],
                y1=top_right[1],
                x2=bottom_right[0],
                y2=bottom_right[1],
                x3=x,
                y3=y,
            ):
                original_bottom_right = bottom_right
                original_top_right = top_right
                top_right, bottom_right = calculate_vertices_to_contain_point(
                    vertex_1=original_top_right,
                    vertex_2=original_bottom_right,
                    vertex_3_from_1=top_left,
                    vertex_4_from_2=bottom_left,
                    x=x,
                    y=y,
                )
                extended_width += (
                    (bottom_right[0] - original_bottom_right[0]) ** 2
                    + (bottom_right[1] - original_bottom_right[1]) ** 2
                ) ** 0.5
        # extend to the bottom
        points = []
        if bbox_position == ALL_POSITIONS:
            points.append(
                det.get_anchors_coordinates(anchor=sv.Position.BOTTOM_RIGHT)[0]
            )
            points.append(
                det.get_anchors_coordinates(anchor=sv.Position.BOTTOM_LEFT)[0]
            )
        else:
            points.append(det.get_anchors_coordinates(anchor=bbox_position)[0])
        for x, y in points:
            if (
                cv.pointPolygonTest(
                    np.array([bottom_left, top_left, top_right, bottom_right]),
                    (x, y),
                    False,
                )
                >= 0
            ):
                continue
            if not ccw(
                x1=bottom_right[0],
                y1=bottom_right[1],
                x2=bottom_left[0],
                y2=bottom_left[1],
                x3=x,
                y3=y,
            ):
                original_bottom_right = bottom_right
                original_bottom_left = bottom_left
                bottom_right, bottom_left = calculate_vertices_to_contain_point(
                    vertex_1=original_bottom_right,
                    vertex_2=original_bottom_left,
                    vertex_3_from_1=top_right,
                    vertex_4_from_2=top_left,
                    x=x,
                    y=y,
                )
                extended_height += (
                    (bottom_left[0] - original_bottom_left[0]) ** 2
                    + (bottom_left[1] - original_bottom_left[1]) ** 2
                ) ** 0.5
        # extend to the top
        points = []
        if bbox_position == ALL_POSITIONS:
            points.append(det.get_anchors_coordinates(anchor=sv.Position.TOP_RIGHT)[0])
            points.append(det.get_anchors_coordinates(anchor=sv.Position.TOP_LEFT)[0])
        else:
            points.append(det.get_anchors_coordinates(anchor=bbox_position)[0])
        for x, y in points:
            if (
                cv.pointPolygonTest(
                    np.array([bottom_left, top_left, top_right, bottom_right]),
                    (x, y),
                    False,
                )
                >= 0
            ):
                continue
            if not ccw(
                x1=top_left[0],
                y1=top_left[1],
                x2=top_right[0],
                y2=top_right[1],
                x3=x,
                y3=y,
            ):
                original_top_left = top_left
                original_top_right = top_right
                top_left, top_right = calculate_vertices_to_contain_point(
                    vertex_1=original_top_left,
                    vertex_2=original_top_right,
                    vertex_3_from_1=bottom_left,
                    vertex_4_from_2=bottom_right,
                    x=x,
                    y=y,
                )
                extended_height += (
                    (top_left[0] - original_top_left[0]) ** 2
                    + (top_left[1] - original_top_left[1]) ** 2
                ) ** 0.5
    return (
        np.array(
            [
                bottom_left,
                top_left,
                top_right,
                bottom_right,
            ]
        ),
        original_width,
        original_height,
        extended_width,
        extended_height,
    )


def generate_transformation_matrix(
    src_polygon: np.ndarray,
    transformed_rect_width: int,
    transformed_rect_height: int,
    detections: Optional[sv.Detections] = None,
    detections_anchor: Optional[Union[sv.Position, Literal[ALL_POSITIONS]]] = None,
) -> Tuple[np.ndarray, float, float]:
    polygon_with_vertices_clockwise = sort_polygon_vertices_clockwise(
        polygon=src_polygon
    )
    src_polygon = roll_polygon_vertices_to_start_from_leftmost_bottom(
        polygon=polygon_with_vertices_clockwise
    )
    original_width = transformed_rect_width
    original_height = transformed_rect_height
    extended_width = 0
    extended_height = 0
    if detections and detections_anchor:
        (
            src_polygon,
            original_width,
            original_height,
            extended_width,
            extended_height,
        ) = extend_perspective_polygon(
            polygon=src_polygon,
            detections=detections,
            bbox_position=(
                sv.Position(detections_anchor)
                if detections_anchor != ALL_POSITIONS
                else detections_anchor
            ),
        )
    extended_width = extended_width * transformed_rect_width / max(original_width, 1)
    extended_height = (
        extended_height * transformed_rect_height / max(original_height, 1)
    )
    src_polygon = src_polygon.astype(np.float32)
    dst_polygon = np.array(
        [
            [0, transformed_rect_height + int(round(extended_height)) - 1],
            [0, 0],
            [transformed_rect_width + int(round(extended_width)) - 1, 0],
            [
                transformed_rect_width + int(round(extended_width)) - 1,
                transformed_rect_height + int(round(extended_height)) - 1,
            ],
        ]
    ).astype(dtype=np.float32)
    # https://docs.opencv.org/4.9.0/da/d54/group__imgproc__transform.html#ga20f62aa3235d869c9956436c870893ae
    return (
        cv.getPerspectiveTransform(
            src=src_polygon,
            dst=dst_polygon,
        ),
        extended_width,
        extended_height,
    )


def correct_detections(
    detections: sv.Detections,
    perspective_transformer: np.array,
    transformed_rect_width: float,
    transformed_rect_height: float,
) -> sv.Detections:
    corrected_detections: List[sv.Detections] = []
    for i in range(len(detections)):
        # copy
        detection = detections[i]
        mask = np.array(detection.mask)
        if (
            not np.array_equal(mask, np.array(None))
            and len(mask) > 0
            and isinstance(mask[0], np.ndarray)
        ):
            polygon = np.array(sv.mask_to_polygons(mask[0]), dtype=np.float32)
            # https://docs.opencv.org/4.9.0/d2/de8/group__core__array.html#gad327659ac03e5fd6894b90025e6900a7
            corrected_polygon: np.ndarray = cv.perspectiveTransform(
                src=polygon, m=perspective_transformer
            ).reshape(-1, 2)
            detection.mask = np.array(
                [
                    sv.polygon_to_mask(
                        polygon=np.around(corrected_polygon).astype(np.int32),
                        resolution_wh=(
                            int(round(transformed_rect_width)),
                            int(round(transformed_rect_height)),
                        ),
                    ).astype(bool)
                ]
            )
            detection.xyxy = np.array(
                [
                    np.around(sv.polygon_to_xyxy(polygon=corrected_polygon)).astype(
                        np.int32
                    )
                ]
            )
        else:
            xmin, ymin, xmax, ymax = np.around(detection.xyxy[0]).tolist()
            polygon = np.array(
                [[[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]],
                dtype=np.float32,
            )
            # https://docs.opencv.org/4.9.0/d2/de8/group__core__array.html#gad327659ac03e5fd6894b90025e6900a7
            corrected_polygon: np.ndarray = cv.perspectiveTransform(
                src=polygon, m=perspective_transformer
            ).reshape(-1, 2)
            detection.xyxy = np.array(
                [
                    np.around(sv.polygon_to_xyxy(polygon=corrected_polygon)).astype(
                        np.int32
                    )
                ]
            )
        if KEYPOINTS_XY_KEY_IN_SV_DETECTIONS in detection.data:
            corrected_key_points = cv.perspectiveTransform(
                src=np.array(
                    [detection.data[KEYPOINTS_XY_KEY_IN_SV_DETECTIONS][0]],
                    dtype=np.float32,
                ),
                m=perspective_transformer,
            ).reshape(-1, 2)
            detection[KEYPOINTS_XY_KEY_IN_SV_DETECTIONS] = np.array(
                [np.around(corrected_key_points).astype(np.int32)], dtype="object"
            )
        corrected_detections.append(detection)
    return sv.Detections.merge(corrected_detections)


class PerspectiveCorrectionBlockV1(WorkflowBlock):
    def __init__(self):
        self.perspective_transformers: List[Tuple[np.ndarray, float, float]] = []

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return PerspectiveCorrectionManifest

    def run(
        self,
        images: Batch[WorkflowImageData],
        predictions: Optional[Batch[sv.Detections]],
        perspective_polygons: Union[
            List[np.ndarray],
            List[List[np.ndarray]],
            List[List[List[int]]],
            List[List[List[List[int]]]],
        ],
        transformed_rect_width: Union[int, List[int], np.ndarray],
        transformed_rect_height: Union[int, List[int], np.ndarray],
        extend_perspective_polygon_by_detections_anchor: Union[
            sv.Position, Literal[ALL_POSITIONS]
        ],
        warp_image: Optional[bool],
    ) -> BlockResult:
        if not predictions and not images:
            raise ValueError(
                "Either predictions or images are required to apply perspective correction."
            )
        if warp_image and not images:
            raise ValueError(
                "images are required to warp image into requested perspective."
            )
        if not predictions:
            predictions = [None] * len(images)
        batch_size = len(predictions) if predictions else len(images)
        if isinstance(transformed_rect_height, int):
            transformed_rect_height = [transformed_rect_height] * batch_size
        if isinstance(transformed_rect_width, int):
            transformed_rect_width = [transformed_rect_width] * batch_size

        if (
            not self.perspective_transformers
            or extend_perspective_polygon_by_detections_anchor
        ):
            self.perspective_transformers = []
            largest_perspective_polygons = pick_largest_perspective_polygons(
                perspective_polygons
            )

            if len(largest_perspective_polygons) == 1 and batch_size > 1:
                largest_perspective_polygons = largest_perspective_polygons * batch_size

            if len(largest_perspective_polygons) != batch_size:
                raise ValueError(
                    f"Predictions batch size ({batch_size}) does not match number of perspective polygons ({largest_perspective_polygons})"
                )
            for polygon, detections, width, height in zip(
                largest_perspective_polygons,
                predictions,
                list(transformed_rect_width),
                list(transformed_rect_height),
            ):
                if polygon is None:
                    self.perspective_transformers.append(None)
                    continue
                self.perspective_transformers.append(
                    generate_transformation_matrix(
                        src_polygon=polygon,
                        detections=detections,
                        transformed_rect_width=width,
                        transformed_rect_height=height,
                        detections_anchor=extend_perspective_polygon_by_detections_anchor,
                    )
                )

        result = []
        for detections, perspective_transformer_w_h, image, width, height in zip(
            predictions,
            self.perspective_transformers,
            images,
            transformed_rect_width,
            transformed_rect_height,
        ):
            perspective_transformer, extended_width, extended_height = (
                perspective_transformer_w_h
            )
            result_image = image
            if warp_image:
                # https://docs.opencv.org/4.9.0/da/d54/group__imgproc__transform.html#gaf73673a7e8e18ec6963e3774e6a94b87
                warped_image = cv.warpPerspective(
                    src=image.numpy_image,
                    M=perspective_transformer,
                    dsize=(
                        int(round(width)) + int(round(extended_width)),
                        int(round(height)) + int(round(extended_height)),
                    ),
                )
                result_image = WorkflowImageData.copy_and_replace(
                    origin_image_data=image,
                    numpy_image=warped_image,
                )

            if detections is None:
                result.append(
                    {
                        OUTPUT_DETECTIONS_KEY: None,
                        OUTPUT_IMAGE_KEY: result_image,
                        OUTPUT_EXTENDED_TRANSFORMED_RECT_WIDTH_KEY: width
                        + int(round(extended_width)),
                        OUTPUT_EXTENDED_TRANSFORMED_RECT_HEIGHT_KEY: height
                        + int(round(extended_height)),
                    }
                )
                continue

            corrected_detections = correct_detections(
                detections=detections,
                perspective_transformer=perspective_transformer,
                transformed_rect_width=width + int(round(extended_width)),
                transformed_rect_height=height + int(round(extended_height)),
            )

            result.append(
                {
                    OUTPUT_DETECTIONS_KEY: corrected_detections,
                    OUTPUT_IMAGE_KEY: result_image,
                    OUTPUT_EXTENDED_TRANSFORMED_RECT_WIDTH_KEY: width
                    + int(round(extended_width)),
                    OUTPUT_EXTENDED_TRANSFORMED_RECT_HEIGHT_KEY: height
                    + int(round(extended_height)),
                }
            )
        return result
