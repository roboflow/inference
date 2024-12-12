from typing import List, Literal, Optional, Tuple, Type, Union

import cv2 as cv
import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
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

OUTPUT_KEY = "dimensions"
SHORT_DESCRIPTION = (
    "Measure the dimensions of objects in relation to a reference object."
)
LONG_DESCRIPTION = """
The `SizeMeasurementBlock` is a transformer block designed to measure the dimensions of objects
in relation to a reference object. The reference object is detected using one model,
and the object to be measured is detected using another model. The block outputs the dimensions of the
objects to be measured in terms of the reference object.
Note: if reference_predictions provides multiple boxes, the most confident one will be selected.
In order to achieve different behavior you can use Detection Transformation block with custom filter
and also continue_if block if no reference detection meets expectations.
"""


class SizeMeasurementManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Size Measurement",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
            "ui_manifest": {
                "section": "classical_cv",
                "icon": "far fa-ruler",
                "opencv": True,
            },
        }
    )
    type: Literal["roboflow_core/size_measurement@v1"]

    reference_predictions: Selector(
        kind=[
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
            OBJECT_DETECTION_PREDICTION_KIND,
        ]
    ) = Field(
        description="Predictions from the reference object model",
        examples=["$segmentation.reference_predictions"],
    )
    object_predictions: Selector(
        kind=[
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
            OBJECT_DETECTION_PREDICTION_KIND,
        ]
    ) = Field(
        description="Predictions from the model that detects the object to measure",
        examples=["$segmentation.object_predictions"],
    )
    reference_dimensions: Union[
        str,
        Tuple[float, float],
        List[float],
        Selector(
            kind=[STRING_KIND, LIST_OF_VALUES_KIND],
        ),
    ] = Field(
        description="Dimensions of the reference object (width, height) in desired units (e.g., inches) as a string in the format 'width,height' or as a tuple (width, height)",
        examples=["5.0,5.0", (5.0, 5.0), "$inputs.reference_dimensions"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name=OUTPUT_KEY, kind=[LIST_OF_VALUES_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


def horizontal_score(angle: float) -> float:
    """
    Determine how close an angle is to horizontal (0 or 180 degrees).
    Lower score means more horizontal.
    """
    mod_angle = abs(angle % 180)
    return min(mod_angle, 180 - mod_angle)


def compute_aligned_dimensions(contour: np.ndarray) -> Tuple[float, float]:
    """
    Compute the width and height of an object based on its contour, ensuring proper orientation.

    This function:
    1. Finds the minimum area rectangle that encloses the contour
    2. Determines which edges correspond to width and height by analyzing their angles
    3. Returns dimensions where width is the more horizontal edge and height is the more vertical edge

    Args:
        contour (np.ndarray): Array of points representing the object's contour

    Returns:
        Tuple[float, float]: A tuple of (width_pixels, height_pixels) where:
            - width_pixels: Length of the more horizontal edge
            - height_pixels: Length of the more vertical edge

    Note:
        The function uses angle analysis to ensure consistent width/height assignment
        regardless of the object's rotation. The edge closer to horizontal (0° or 180°)
        is always considered the width.
    """
    rect = cv.minAreaRect(contour)
    box = cv.boxPoints(rect)
    box = np.array(box, dtype=np.float32)

    edge1 = box[1] - box[0]
    edge2 = box[2] - box[1]

    len_edge1 = np.linalg.norm(edge1)
    len_edge2 = np.linalg.norm(edge2)

    angle1 = np.degrees(np.arctan2(edge1[1], edge1[0]))
    angle2 = np.degrees(np.arctan2(edge2[1], edge2[0]))

    h_score1 = horizontal_score(angle1)
    h_score2 = horizontal_score(angle2)

    if h_score1 < h_score2:
        width_pixels = len_edge1
        height_pixels = len_edge2
    else:
        width_pixels = len_edge2
        height_pixels = len_edge1

    return float(width_pixels), float(height_pixels)


def get_detection_dimensions(
    detection: sv.Detections, index: int
) -> Tuple[float, float]:
    """
    Retrieve the width and height dimensions of a detected object in pixels.

    Args:
        detection (sv.Detections): Detection object containing masks and/or bounding boxes
        index (int): Index of the specific detection to analyze

    Returns:
        Tuple[float, float]: A tuple of (width_pixels, height_pixels) where:
            - width_pixels: Width of the object in pixels
            - height_pixels: Height of the object in pixels

    Notes:
        The function uses two methods to compute dimensions:
        1. If a segmentation mask is available:
           - Extracts the largest contour from the mask
           - Uses compute_aligned_dimensions() to get orientation-aware measurements
        2. If no mask is available:
           - Falls back to using the bounding box dimensions
           - Simply computes width and height as box edges
    """
    if detection.mask is not None:
        mask = detection.mask[index].astype(np.uint8)
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv.contourArea)
            if cv.contourArea(largest_contour) > 0:
                return compute_aligned_dimensions(largest_contour)

    else:
        bbox = detection.xyxy[index]
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        return float(w), float(h)


def parse_reference_dimensions(
    reference_dimensions: Union[str, Tuple[float, float], List[float]]
) -> Tuple[float, float]:
    """Parse reference dimensions from various input formats."""
    if isinstance(reference_dimensions, str):
        parts = reference_dimensions.split(",")
        if len(parts) != 2:
            raise ValueError(
                "reference_dimensions must be a string in the format 'width,height'"
            )
        try:
            reference_dimensions = [float(p.strip()) for p in parts]
        except ValueError:
            raise ValueError("Invalid format for reference_dimensions")

    if len(reference_dimensions) != 2:
        raise ValueError("reference_dimensions must have two values (width, height)")

    return tuple(reference_dimensions)


class SizeMeasurementBlockV1(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return SizeMeasurementManifest

    def run(
        self,
        reference_predictions: sv.Detections,
        object_predictions: sv.Detections,
        reference_dimensions: Union[str, Tuple[float, float], List[float]],
    ) -> BlockResult:
        ref_width_actual, ref_height_actual = parse_reference_dimensions(
            reference_dimensions
        )

        if (
            reference_predictions.confidence is None
            or len(reference_predictions.confidence) == 0
        ):
            return {OUTPUT_KEY: None}

        ref_index = int(np.argmax(reference_predictions.confidence))
        ref_width_pixels, ref_height_pixels = get_detection_dimensions(
            reference_predictions, ref_index
        )

        if ref_width_pixels <= 0 or ref_height_pixels <= 0:
            return {OUTPUT_KEY: None}

        width_scale = ref_width_actual / ref_width_pixels
        height_scale = ref_height_actual / ref_height_pixels

        dimensions = []
        for i in range(len(object_predictions)):
            obj_w_pixels, obj_h_pixels = get_detection_dimensions(object_predictions, i)
            if obj_w_pixels > 0 and obj_h_pixels > 0:
                obj_w_actual = obj_w_pixels * width_scale
                obj_h_actual = obj_h_pixels * height_scale
                dimensions.append({"width": obj_w_actual, "height": obj_h_actual})
            else:
                dimensions.append(None)

        return {OUTPUT_KEY: dimensions}
