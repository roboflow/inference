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
    contours = sv.mask_to_polygons(mask)
    # Skip sorting, just use argmax for direct access to the largest
    largest_contour = contours[np.argmax([c.shape[0] for c in contours])]

    # https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga014b28e56cb8854c0de4a211cb2be656
    convex_contour = cv.convexHull(
        largest_contour,
        returnPoints=True,
        clockwise=True,
    )
    # https://docs.opencv.org/4.9.0/d3/dc0/group__imgproc__shape.html#ga8d26483c636be6b35c3ec6335798a47c
    perimeter = cv.arcLength(convex_contour, closed=True)
    upper_epsilon = perimeter
    lower_epsilon = 1e-7
    epsilon = lower_epsilon + upper_epsilon / 2

    simplified_polygon = cv.approxPolyDP(convex_contour, epsilon=epsilon, closed=True)

    for _ in range(max_steps):
        n = len(simplified_polygon)
        if n == required_number_of_vertices:
            break
        if n > required_number_of_vertices:
            lower_epsilon = epsilon
        else:
            upper_epsilon = epsilon
        epsilon = lower_epsilon + (upper_epsilon - lower_epsilon) * 0.5
        simplified_polygon = cv.approxPolyDP(
            convex_contour, epsilon=epsilon, closed=True
        )

    # Remove extra nesting (e.g. shape Nx1x2 -> Nx2)
    if len(simplified_polygon.shape) == 3 and simplified_polygon.shape[1] == 1:
        simplified_polygon = simplified_polygon[:, 0, :]

    return simplified_polygon


def scale_polygon(polygon: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1:
        return polygon

    M = cv.moments(polygon)
    if M["m00"] == 0:
        return polygon

    centroid_x = M["m10"] / M["m00"]
    centroid_y = M["m01"] / M["m00"]

    shifted = polygon - np.array([centroid_x, centroid_y])
    scaled = shifted * scale
    result = scaled + np.array([centroid_x, centroid_y])

    return np.round(result).astype(np.int32)


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
            if detections.mask is None:
                result.append({OUTPUT_KEY: []})
                continue

            simplified_polygons = []
            updated_detections = []

            masks = detections.mask
            n_masks = len(masks)

            # Pre-fetch updated detections as a list, avoid repeated attribute access
            updated_detection_list = list(detections)
            for i in range(n_masks):
                mask = masks[i]
                updated_detection = updated_detection_list[i]

                simplified_polygon = calculate_simplified_polygon(
                    mask=mask,
                    required_number_of_vertices=required_number_of_vertices,
                )

                # Pad polygon if too small
                vertices_count = simplified_polygon.shape[0]
                if vertices_count < required_number_of_vertices:
                    last_point = simplified_polygon[-1][np.newaxis, :]
                    repeat_times = required_number_of_vertices - vertices_count
                    simplified_polygon = np.vstack(
                        [
                            simplified_polygon,
                            np.repeat(last_point, repeat_times, axis=0),
                        ]
                    )

                # Assign polygon
                updated_detection[POLYGON_KEY_IN_SV_DETECTIONS] = simplified_polygon[
                    np.newaxis, :
                ]

                # Scale polygon
                scaled_polygon = scale_polygon(
                    simplified_polygon,
                    scale=scale_ratio,
                )
                simplified_polygons.append(scaled_polygon)

                # Mask of scaled polygon
                updated_detection.mask = np.array(
                    [
                        sv.polygon_to_mask(
                            polygon=scaled_polygon,
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
