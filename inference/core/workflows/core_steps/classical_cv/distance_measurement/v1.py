from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INTEGER_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

SHORT_DESCRIPTION = "Calculate the distance between two bounding boxes on a 2D plane."

LONG_DESCRIPTION = """
Calculate the distance between two bounding boxes on a 2D plane, leveraging a perpendicular camera view and either a reference object or a pixel-to-unit scaling ratio for precise measurements."""

OUTPUT_KEY_CENTIMETER = "distance_cm"
OUTPUT_KEY_PIXEL = "distance_pixel"


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Distance Measurement",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "classical_computer_vision",
            "ui_manifest": {
                "section": "classical_cv",
                "icon": "far fa-ruler-triangle",
                "blockPriority": 11,
                "opencv": True,
            },
        }
    )

    type: Literal["roboflow_core/distance_measurement@v1"]

    predictions: Selector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
        ]
    ) = Field(
        title="Object Detections",
        description="The output of a detection model describing the bounding boxes that will be used to measure the objects.",
        examples=["$steps.model.predictions"],
    )

    object_1_class_name: str = Field(
        title="First Object Class Name",
        description="The class name of the first object.",
        examples=["car"],
    )

    object_2_class_name: str = Field(
        title="Second Object Class Name",
        description="The class name of the second object.",
        examples=["person"],
    )

    reference_axis: Literal["horizontal", "vertical"] = Field(
        title="Reference Axis",
        description="The axis along which the distance will be measured.",
        examples=["vertical", "horizontal", "$inputs.reference_axis"],
    )

    calibration_method: Literal["reference object", "pixel to centimeter"] = Field(
        title="Calibration Method",
        description="Select how to calibrate the measurement of distance between objects.",
    )

    reference_object_class_name: Union[str, Selector(kind=[STRING_KIND])] = Field(
        title="Reference Object Class Name",
        description="The class name of the reference object.",
        default="reference-object",
        examples=["reference-object", "$inputs.reference_object_class_name"],
        json_schema_extra={
            "relevant_for": {
                "calibration_method": {
                    "values": ["reference object"],
                    "required": True,
                },
            },
        },
    )

    reference_width: Union[float, Selector(kind=[FLOAT_KIND])] = Field(
        title="Width",
        default=2.5,
        description="Width of the reference object in centimeters",
        examples=[2.5, "$inputs.reference_width"],
        gt=0,
        json_schema_extra={
            "relevant_for": {
                "calibration_method": {
                    "values": ["reference object"],
                    "required": True,
                },
            },
        },
    )

    reference_height: Union[float, Selector(kind=[FLOAT_KIND])] = Field(  # type: ignore
        title="Height",
        default=2.5,
        description="Height of the reference object in centimeters",
        examples=[2.5, "$inputs.reference_height"],
        gt=0,
        json_schema_extra={
            "relevant_for": {
                "calibration_method": {
                    "values": ["reference object"],
                    "required": True,
                },
            },
        },
    )

    pixel_ratio: Union[float, Selector(kind=[FLOAT_KIND])] = Field(
        title="Reference Pixel-to-Centimeter Ratio",
        description="The pixel-to-centimeter ratio of the input image, i.e. 1 centimeter = 100 pixels.",
        default=100,
        examples=[100, "$inputs.pixel_ratio"],
        gt=0,
        json_schema_extra={
            "relevant_for": {
                "calibration_method": {
                    "values": ["pixel to centimeter"],
                    "required": True,
                },
            },
        },
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name=OUTPUT_KEY_CENTIMETER,
                kind=[INTEGER_KIND],
            ),
            OutputDefinition(
                name=OUTPUT_KEY_PIXEL,
                kind=[INTEGER_KIND],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class DistanceMeasurementBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        predictions: sv.Detections,
        object_1_class_name: str,
        object_2_class_name: str,
        reference_axis: Literal["horizontal", "vertical"],
        calibration_method: Literal["reference object", "pixel to centimeter"],
        reference_object_class_name: str,
        reference_width: float,
        reference_height: float,
        pixel_ratio: float,
    ) -> BlockResult:
        if calibration_method == "reference object":
            reference_predictions = predictions
            distances = measure_distance_with_reference_object(
                detections=predictions,
                object_1_class_name=object_1_class_name,
                object_2_class_name=object_2_class_name,
                reference_predictions=reference_predictions,
                reference_object_class_name=reference_object_class_name,
                reference_width=reference_width,
                reference_height=reference_height,
                reference_axis=reference_axis,
            )
        elif calibration_method == "pixel to centimeter":
            distances = measure_distance_with_pixel_ratio(
                detections=predictions,
                pixel_ratio=pixel_ratio,
                object_1_class_name=object_1_class_name,
                object_2_class_name=object_2_class_name,
                reference_axis=reference_axis,
            )
        else:
            raise ValueError(f"Invalid calibration type: {calibration_method}")

        return distances


def measure_distance_with_reference_object(
    detections: sv.Detections,
    object_1_class_name: str,
    object_2_class_name: str,
    reference_predictions: sv.Detections,
    reference_object_class_name: str,
    reference_width: float,
    reference_height: float,
    reference_axis: Literal["horizontal", "vertical"],
):
    reference_bbox_1 = None
    reference_bbox_2 = None

    reference_bbox_1, reference_bbox_2 = find_reference_bboxes(
        detections, object_1_class_name, object_2_class_name
    )

    if not reference_bbox_1 or not reference_bbox_2:
        raise ValueError(
            f"Reference class '{object_1_class_name}' or '{object_2_class_name}' not found in predictions."
        )

    if has_overlap(reference_bbox_1, reference_bbox_2) or not has_axis_gap(
        reference_bbox_1, reference_bbox_2, reference_axis
    ):
        return {OUTPUT_KEY_CENTIMETER: 0, OUTPUT_KEY_PIXEL: 0}

    # get the reference object bounding box
    reference_bbox = None
    for (x_min, y_min, x_max, y_max), class_name in zip(
        reference_predictions.xyxy.round().astype(dtype=int),
        reference_predictions.data["class_name"],
    ):
        if class_name == reference_object_class_name:
            reference_bbox = (x_min, y_min, x_max, y_max)
            break

    if not reference_bbox:
        raise ValueError(
            f"Reference class '{reference_object_class_name}' not found in predictions."
        )

    # calculate the pixel-to-centimeter ratio
    reference_width_pixels = abs(reference_bbox[2] - reference_bbox[0])
    reference_height_pixels = abs(reference_bbox[3] - reference_bbox[1])

    # Ensure the reference dimensions are positive and non-zero
    if reference_width <= 0 or reference_height <= 0:
        raise ValueError("Reference object dimensions must be greater than zero.")

    pixel_ratio_width = reference_width_pixels / reference_width
    pixel_ratio_height = reference_height_pixels / reference_height

    # get the average pixel ratio
    pixel_ratio = (pixel_ratio_width + pixel_ratio_height) / 2

    distance_pixels = measure_distance_pixels(
        reference_axis, reference_bbox_1, reference_bbox_2
    )

    distance_cm = distance_pixels / pixel_ratio

    return {OUTPUT_KEY_CENTIMETER: distance_cm, OUTPUT_KEY_PIXEL: distance_pixels}


def measure_distance_with_pixel_ratio(
    detections: sv.Detections,
    pixel_ratio: float,
    object_1_class_name: str,
    object_2_class_name: str,
    reference_axis: Literal["horizontal", "vertical"],
) -> List[Dict[str, Union[str, float]]]:
    reference_bbox_1 = None
    reference_bbox_2 = None

    reference_bbox_1, reference_bbox_2 = find_reference_bboxes(
        detections, object_1_class_name, object_2_class_name
    )

    if not reference_bbox_1 or not reference_bbox_2:
        raise ValueError(
            f"Reference class '{object_1_class_name}' or '{object_2_class_name}' not found in predictions."
        )

    if has_overlap(reference_bbox_1, reference_bbox_2) or not has_axis_gap(
        reference_bbox_1, reference_bbox_2, reference_axis
    ):
        return {OUTPUT_KEY_CENTIMETER: 0, OUTPUT_KEY_PIXEL: 0}

    if pixel_ratio is None:
        raise ValueError("Pixel-to-centimeter ratio must be provided.")

    if not isinstance(pixel_ratio, (int, float)):
        raise ValueError("Pixel-to-centimeter ratio must be a number.")

    if pixel_ratio <= 0:
        raise ValueError("Pixel-to-centimeter ratio must be greater than zero.")

    distance_pixels = measure_distance_pixels(
        reference_axis, reference_bbox_1, reference_bbox_2
    )

    distance_cm = distance_pixels / pixel_ratio

    return {OUTPUT_KEY_CENTIMETER: distance_cm, OUTPUT_KEY_PIXEL: distance_pixels}


def has_overlap(
    bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]
) -> bool:
    """
    Check if two bounding boxes overlap.

    Args:
        bbox1: A tuple of (x_min, y_min, x_max, y_max) for the first bounding box.
        bbox2: A tuple of (x_min, y_min, x_max, y_max) for the second bounding box.

    Returns:
        True if the bounding boxes overlap, False otherwise.
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    if x1_max < x2_min or x2_max < x1_min:
        return False
    if y1_max < y2_min or y2_max < y1_min:
        return False

    return True


def has_axis_gap(
    reference_bbox_1: Tuple[int, int, int, int],
    reference_bbox_2: Tuple[int, int, int, int],
    reference_axis: str,
) -> bool:
    if reference_axis == "horizontal":
        if (
            reference_bbox_1[0] < reference_bbox_2[2]
            and reference_bbox_1[2] > reference_bbox_2[0]
        ):
            return False
    else:
        if (
            reference_bbox_1[1] < reference_bbox_2[3]
            and reference_bbox_1[3] > reference_bbox_2[1]
        ):
            return False

    return True


def find_reference_bboxes(
    detections: sv.Detections, object_1_class_name: str, object_2_class_name: str
):
    reference_bbox_1 = None
    reference_bbox_2 = None

    for (x_min, y_min, x_max, y_max), class_name in zip(
        detections.xyxy.round().astype(dtype=int), detections.data["class_name"]
    ):
        if class_name == object_1_class_name:
            reference_bbox_1 = (x_min, y_min, x_max, y_max)
        elif class_name == object_2_class_name:
            reference_bbox_2 = (x_min, y_min, x_max, y_max)

        if reference_bbox_1 and reference_bbox_2:
            break

    return reference_bbox_1, reference_bbox_2


def measure_distance_pixels(
    reference_axis: str,
    reference_bbox_1: Tuple[int, int, int, int],
    reference_bbox_2: Tuple[int, int, int, int],
):
    if reference_axis == "vertical":
        distance_pixels = (
            abs(reference_bbox_2[1] - reference_bbox_1[3])
            if reference_bbox_2[1] > reference_bbox_1[3]
            else abs(reference_bbox_1[1] - reference_bbox_2[3])
        )
    else:
        distance_pixels = (
            abs(reference_bbox_2[0] - reference_bbox_1[2])
            if reference_bbox_2[0] > reference_bbox_1[2]
            else abs(reference_bbox_1[0] - reference_bbox_2[2])
        )
    return distance_pixels
