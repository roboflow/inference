from typing import List, Literal, Optional, Type, Union

import cv2 as cv
import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
)
from inference.core.workflows.execution_engine.entities.types import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    StepOutputSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

OUTPUT_KEY: str = "zones"
TYPE: str = "roboflow_core/dynamic_zone@v1"
SHORT_DESCRIPTION = (
    "Simplify polygons so they are geometrically convex "
    "and simplify them to contain only requested amount of vertices"
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
        }
    )
    type: Literal[f"{TYPE}", "DynamicZone"]
    predictions: StepOutputSelector(
        kind=[
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
        ]
    ) = Field(  # type: ignore
        description="",
        examples=["$segmentation.predictions"],
    )
    required_number_of_vertices: Union[int, WorkflowParameterSelector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Keep simplifying polygon until number of vertices matches this number",
        examples=[4, "$inputs.vertices"],
    )

    @classmethod
    def accepts_batch_input(cls) -> bool:
        return True

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name=OUTPUT_KEY, kind=[LIST_OF_VALUES_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


def calculate_simplified_polygon(
    mask: np.ndarray, required_number_of_vertices: int, max_steps: int = 1000
) -> np.array:
    contours = sv.mask_to_polygons(mask)
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
    return simplified_polygon


class DynamicZonesBlockV1(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return DynamicZonesManifest

    def run(
        self,
        predictions: Batch[sv.Detections],
        required_number_of_vertices: int,
    ) -> BlockResult:
        result = []
        for detections in predictions:
            if detections is None:
                result.append({OUTPUT_KEY: None})
                continue
            simplified_polygons = []
            if detections.mask is None:
                result.append({OUTPUT_KEY: []})
                continue
            for mask in detections.mask:
                simplified_polygon = calculate_simplified_polygon(
                    mask=mask,
                    required_number_of_vertices=required_number_of_vertices,
                )
                if len(simplified_polygon) != required_number_of_vertices:
                    continue
                simplified_polygons.append(simplified_polygon)
            result.append({OUTPUT_KEY: simplified_polygons})
        return result
