from typing import List, Literal, Optional, Tuple, Type, Union

import cv2 as cv
import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.workflows.execution_engine.constants import (
    POLYGON_KEY_IN_SV_DETECTIONS,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    FLOAT_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

OUTPUT_KEY: str = "zones"
OUTPUT_KEY_DETECTIONS: str = "predictions"
OUTPUT_KEY_SIMPLIFICATION_CONVERGED: str = "simplification_converged"
TYPE: str = "roboflow_core/dynamic_zone@v1"
SHORT_DESCRIPTION = (
    "Simplify polygons so they are geometrically convex "
    "and contain only the requested amount of vertices."
)
LONG_DESCRIPTION = """
The `DynamicZoneBlock` is a transformer block designed to simplify polygon
so it's geometrically convex and then reduce number of vertices to requested amount.
This block is best suited when Zone needs to be created based on shape of detected object
(i.e. basketball field, road segment, zebra crossing etc.)
Input detections should be filtered and contain only desired classes of interest.
"""


class DynamicZonesManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Dynamic Zone",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
            "ui_manifest": {
                "section": "advanced",
                "icon": "fal fa-square-dashed",
                "blockPriority": 3,
                "opencv": True,
            },
        }
    )
    type: Literal[f"{TYPE}", "DynamicZone"]
    predictions: Selector(
        kind=[
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
        ]
    ) = Field(  # type: ignore
        description="",
        examples=["$segmentation.predictions"],
    )
    required_number_of_vertices: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Keep simplifying polygon until number of vertices matches this number",
        examples=[4, "$inputs.vertices"],
    )
    scale_ratio: Union[float, Selector(kind=[FLOAT_KIND])] = Field(  # type: ignore
        default=1,
        description="Expand resulting polygon along imaginary line from centroid to edge by this ratio",
        examples=[1.05, "$inputs.scale_ratio"],
    )
    apply_least_squares: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(  # type: ignore
        default=False,
        description="Apply least squares algorithm to fit resulting polygon edges to base contour",
        examples=[True, "$inputs.apply_least_squares"],
    )
    midpoint_fraction: Union[float, Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = Field(  # type: ignore
        default=1,
        description="Fraction of vertices to keep in the middle of each edge before fitting least squares line. "
        "This parameter is useful when vertices of convex polygon are not aligned with edge that would be otherwise fitted to points closer to the center of each edge.",
        examples=[0.9, "$inputs.midpoint_fraction"],
    )

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["predictions"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name=OUTPUT_KEY, kind=[LIST_OF_VALUES_KIND]),
            OutputDefinition(
                name=OUTPUT_KEY_DETECTIONS, kind=[INSTANCE_SEGMENTATION_PREDICTION_KIND]
            ),
            OutputDefinition(
                name=OUTPUT_KEY_SIMPLIFICATION_CONVERGED, kind=[BOOLEAN_KIND]
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


def calculate_simplified_polygon(
    contours: List[np.ndarray], required_number_of_vertices: int, max_steps: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    largest_contour = max(contours, key=len)

    # https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga014b28e56cb8854c0de4a211cb2be656
    convex_contour = cv.convexHull(
        points=largest_contour,
        returnPoints=True,
        clockwise=True,
    )
    # https://docs.opencv.org/4.9.0/d3/dc0/group__imgproc__shape.html#ga8d26483c636be6b35c3ec6335798a47c
    perimeter = cv.arcLength(curve=convex_contour, closed=True)
    upper_epsilon = perimeter
    lower_epsilon = 0.0000001
    epsilon = lower_epsilon + upper_epsilon / 2
    # https://docs.opencv.org/4.9.0/d3/dc0/group__imgproc__shape.html#ga0012a5fdaea70b8a9970165d98722b4c
    simplified_polygon = cv.approxPolyDP(
        curve=convex_contour, epsilon=epsilon, closed=True
    )
    for _ in range(max_steps):
        if len(simplified_polygon) == required_number_of_vertices:
            break
        if len(simplified_polygon) > required_number_of_vertices:
            lower_epsilon = epsilon
        else:
            upper_epsilon = epsilon
        epsilon = lower_epsilon + (upper_epsilon - lower_epsilon) / 2
        simplified_polygon = cv.approxPolyDP(
            curve=convex_contour, epsilon=epsilon, closed=True
        )
    while len(simplified_polygon.shape) > 2:
        simplified_polygon = np.concatenate(simplified_polygon)
    return simplified_polygon, largest_contour


def calculate_least_squares_polygon(
    contour: np.ndarray, polygon: np.ndarray, midpoint_fraction: float = 1
) -> np.ndarray:
    def find_closest_index(point: np.ndarray, contour: np.ndarray) -> int:
        dists = np.linalg.norm(contour - point, axis=1)
        return np.argmin(dists)

    def pick_contour_points_between_vertices(
        point_1: np.ndarray, point_2: np.ndarray, contour: np.ndarray
    ) -> np.ndarray:
        i1 = find_closest_index(point_1, contour)
        i2 = find_closest_index(point_2, contour)

        if i1 <= i2:
            return contour[i1 : i2 + 1]
        else:
            return np.concatenate((contour[i1:], contour[: i2 + 1]), axis=0)

    def least_squares_line(points: np.ndarray) -> Optional[Tuple[float, float]]:
        if len(points) < 2:
            return None
        x = points[:, 0]
        y = points[:, 1]
        A = np.vstack([x, np.ones_like(x)]).T
        a, b = np.linalg.lstsq(A, y, rcond=None)[0]
        return (a, b)

    def intersect_lines(
        line_1: Optional[Tuple[float, float]], line_2: Optional[Tuple[float, float]]
    ) -> Optional[np.ndarray]:
        if line_1 is None or line_2 is None:
            return None
        a_1, b_1 = line_1
        a_2, b_2 = line_2
        if np.isclose(a_1, a_2):
            return None
        x = (b_2 - b_1) / (a_1 - a_2)
        y = a_1 * x + b_1
        return np.array([x, y])

    pairs = [[polygon[-1], polygon[0]]] + list(zip(polygon[:-1], polygon[1:]))

    lines = []
    for point_1, point_2 in pairs:
        segment_points = pick_contour_points_between_vertices(point_1, point_2, contour)
        if midpoint_fraction < 1:
            number_of_points = int(round(len(segment_points) * midpoint_fraction))
            if number_of_points > 2:
                number_of_points_to_discard = (
                    len(segment_points) - number_of_points
                ) // 2
                segment_points = segment_points[
                    number_of_points_to_discard : len(segment_points)
                    - number_of_points_to_discard
                ]
        line_params = least_squares_line(segment_points)
        lines.append(line_params)

    intersections = []
    for i in range(len(lines)):
        line_1 = lines[i]
        line_2 = lines[(i + 1) % len(lines)]
        pt = intersect_lines(line_1, line_2)
        intersections.append(pt)

    return np.array(intersections, dtype=float).round().astype(int)


def scale_polygon(polygon: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1:
        return polygon

    M = cv.moments(polygon)

    if M["m00"] == 0:
        return polygon

    centroid_x = M["m10"] / M["m00"]
    centroid_y = M["m01"] / M["m00"]

    shifted = polygon - [centroid_x, centroid_y]
    scaled = shifted * scale
    result = scaled + [centroid_x, centroid_y]

    return result.round().astype(np.int32)


class DynamicZonesBlockV1(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return DynamicZonesManifest

    def run(
        self,
        predictions: Batch[sv.Detections],
        required_number_of_vertices: int,
        scale_ratio: float = 1,
        apply_least_squares: bool = False,
        midpoint_fraction: float = 1,
    ) -> BlockResult:
        result = []
        for detections in predictions:
            if detections is None:
                result.append({OUTPUT_KEY: None})
                continue
            simplified_polygons = []
            updated_detections = []
            if detections.mask is None:
                result.append({OUTPUT_KEY: []})
                continue
            all_converged = True
            for i, mask in enumerate(detections.mask):
                # copy
                updated_detection = detections[i]

                contours = sv.mask_to_polygons(mask)
                simplified_polygon, largest_contour = calculate_simplified_polygon(
                    contours=contours,
                    required_number_of_vertices=required_number_of_vertices,
                )
                if apply_least_squares:
                    simplified_polygon = calculate_least_squares_polygon(
                        contour=largest_contour,
                        polygon=simplified_polygon,
                        midpoint_fraction=midpoint_fraction,
                    )
                vertices_count, _ = simplified_polygon.shape
                if vertices_count < required_number_of_vertices:
                    all_converged = False
                    for _ in range(required_number_of_vertices - vertices_count):
                        simplified_polygon = np.append(
                            simplified_polygon,
                            [simplified_polygon[-1]],
                            axis=0,
                        )
                elif vertices_count > required_number_of_vertices:
                    all_converged = False
                    simplified_polygon = simplified_polygon[
                        :required_number_of_vertices
                    ]
                updated_detection[POLYGON_KEY_IN_SV_DETECTIONS] = np.array(
                    [simplified_polygon]
                )
                simplified_polygon = scale_polygon(
                    polygon=simplified_polygon,
                    scale=scale_ratio,
                )
                simplified_polygons.append(simplified_polygon)
                updated_detection.mask = np.array(
                    [
                        sv.polygon_to_mask(
                            polygon=simplified_polygon,
                            resolution_wh=mask.shape[::-1],
                        )
                    ]
                )
                updated_detections.append(updated_detection)
            result.append(
                {
                    OUTPUT_KEY: simplified_polygons,
                    OUTPUT_KEY_DETECTIONS: sv.Detections.merge(updated_detections),
                    OUTPUT_KEY_SIMPLIFICATION_CONVERGED: all_converged,
                }
            )
        if not result:
            result.append(
                {
                    OUTPUT_KEY: [],
                    OUTPUT_KEY_DETECTIONS: None,
                    OUTPUT_KEY_SIMPLIFICATION_CONVERGED: False,
                }
            )
        return result
