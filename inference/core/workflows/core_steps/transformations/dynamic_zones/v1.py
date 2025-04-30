from typing import List, Literal, Optional, Type, Union

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
    FLOAT_KIND,
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
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


def calculate_simplified_polygon(
    mask: np.ndarray, required_number_of_vertices: int, max_steps: int = 1000
) -> np.ndarray:
    # Extract polygons from mask and select the largest
    contours = sv.mask_to_polygons(mask)
    largest_contour = max(contours, key=len)

    # Compute convex hull of the largest polygon
    convex_contour = cv.convexHull(
        points=largest_contour,
        returnPoints=True,
        clockwise=True,
    )
    # Calculate the perimeter for epsilons
    perimeter = cv.arcLength(curve=convex_contour, closed=True)
    lower_epsilon = 1e-7  # avoid degenerate case
    upper_epsilon = perimeter

    # Store best found solution closest to required_number_of_vertices
    best_polygon = None
    best_diff = float("inf")

    for _ in range(max_steps):
        epsilon = (lower_epsilon + upper_epsilon) / 2
        # Try to simplify the convex contour
        simplified_polygon = cv.approxPolyDP(
            curve=convex_contour, epsilon=epsilon, closed=True
        )
        num_vertices = len(simplified_polygon)

        # Track best approximation
        diff = abs(num_vertices - required_number_of_vertices)
        if diff < best_diff:
            best_polygon = simplified_polygon
            best_diff = diff
            if diff == 0:
                break  # Exact match

        # Binary search adjustment
        if num_vertices > required_number_of_vertices:
            lower_epsilon = epsilon
        else:
            upper_epsilon = epsilon

        # If further search won't yield a different vertex count, break early
        if upper_epsilon - lower_epsilon < 1e-7:
            break

    # Flatten the result to 2D array (N,2) for output
    polygon_2d = best_polygon.reshape(-1, 2)
    return polygon_2d


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
        scale_ratio: float,
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
            for i, mask in enumerate(detections.mask):
                # copy
                updated_detection = detections[i]

                simplified_polygon = calculate_simplified_polygon(
                    mask=mask,
                    required_number_of_vertices=required_number_of_vertices,
                )
                vertices_count, _ = simplified_polygon.shape
                if vertices_count < required_number_of_vertices:
                    for _ in range(required_number_of_vertices - vertices_count):
                        simplified_polygon = np.append(
                            simplified_polygon,
                            [simplified_polygon[-1]],
                            axis=0,
                        )
                updated_detection[POLYGON_KEY_IN_SV_DETECTIONS] = np.array(
                    [simplified_polygon]
                )
                if len(simplified_polygon) == required_number_of_vertices:
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
                }
            )
        return result
