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
Calculate the distance between two detected objects on a 2D plane using bounding box coordinates, supporting horizontal or vertical distance measurement along a specified axis, with two calibration methods (reference object with known dimensions or pixel-to-centimeter ratio) to convert pixel distances to real-world measurements for spatial analysis, object spacing assessment, safety monitoring, and measurement workflows.

## How This Block Works

This block measures the distance between two detected objects by analyzing their bounding box positions and converting pixel distances to real-world units (centimeters). The block:

1. Receives detection predictions containing bounding boxes and class names for objects in the image
2. Identifies the two target objects using their class names (object_1_class_name and object_2_class_name):
   - Searches through all detections to find bounding boxes matching the specified class names
   - Extracts bounding box coordinates (x_min, y_min, x_max, y_max) for both objects
   - Validates that both objects are found in the detections
3. Validates object positioning for distance measurement:
   - Checks if bounding boxes overlap (if they overlap, distance is set to 0)
   - Verifies objects have a gap along the specified reference axis (horizontal or vertical)
   - Returns 0 distance if objects overlap or are positioned incorrectly for the selected axis
4. Determines the calibration method and performs calibration:

   **For Reference Object Calibration:**
   - Searches detections for a reference object with known real-world dimensions (reference_object_class_name)
   - Extracts the reference object's bounding box coordinates
   - Measures reference object dimensions in pixels (width and height)
   - Calculates pixel-to-centimeter ratios:
     - Width ratio: reference_width_pixels / reference_width (cm)
     - Height ratio: reference_height_pixels / reference_height (cm)
   - Computes average pixel ratio from width and height ratios for more accurate scaling
   - Uses the average ratio to convert all pixel measurements to centimeters

   **For Pixel-to-Centimeter Ratio Calibration:**
   - Uses the provided pixel_ratio directly (e.g., 100 pixels = 1 centimeter)
   - Applies the ratio to convert pixel distances to centimeter distances
   - Suitable when the pixel-to-real-world scale is already known or calibrated

5. Measures pixel distance between the two objects along the specified axis:
   - **For Vertical Distance**: Calculates distance along the Y-axis (vertical separation)
     - Finds the gap between bounding boxes vertically
     - Measures distance from bottom of upper object to top of lower object (or vice versa)
     - Accounts for bounding box positions to find the actual gap distance
   - **For Horizontal Distance**: Calculates distance along the X-axis (horizontal separation)
     - Finds the gap between bounding boxes horizontally
     - Measures distance from right edge of left object to left edge of right object (or vice versa)
     - Accounts for bounding box positions to find the actual gap distance
6. Converts pixel distance to centimeter distance:
   - Divides pixel distance by the pixel-to-centimeter ratio (from calibration)
   - Produces real-world distance measurement in centimeters
7. Returns both pixel distance and centimeter distance values

The block assumes a perpendicular camera view (top-down or frontal view) where perspective distortion is minimal, ensuring accurate 2D distance measurements. Distance is measured as the gap between bounding boxes along the specified axis (horizontal or vertical), not the diagonal distance between object centers. The calibration process converts pixel measurements to real-world units using either a reference object with known dimensions (more flexible, works with different scales) or a direct pixel ratio (simpler, requires pre-calibration). This enables accurate spatial measurements for monitoring, analysis, and control applications.

## Common Use Cases

- **Safety Monitoring**: Measure distances between objects to ensure safe spacing (e.g., measure distance between people for social distancing, monitor spacing between vehicles, ensure safe gaps in industrial settings), enabling safety monitoring workflows
- **Warehouse Management**: Measure spacing between items or objects in storage and logistics (e.g., measure gaps between packages, assess shelf spacing, monitor object placement), enabling warehouse management workflows
- **Quality Control**: Verify spacing and positioning of objects in manufacturing and assembly (e.g., measure gaps between components, verify spacing in assembly lines, check positioning accuracy), enabling quality control workflows
- **Traffic Analysis**: Measure distances between vehicles or objects in traffic monitoring (e.g., measure vehicle spacing, assess safe following distances, monitor traffic gaps), enabling traffic analysis workflows
- **Retail Analytics**: Measure spacing between products or customers in retail environments (e.g., measure product spacing on shelves, assess customer spacing, monitor display arrangements), enabling retail analytics workflows
- **Agricultural Monitoring**: Measure spacing between crops, plants, or agricultural objects (e.g., measure crop spacing, assess plant gaps, monitor field arrangements), enabling agricultural monitoring workflows

## Connecting to Other Blocks

This block receives detection predictions and produces distance_cm and distance_pixel values:

- **After object detection or instance segmentation blocks** to measure distances between detected objects (e.g., measure distance between detected objects, calculate spacing from detections, analyze object relationships), enabling detection-to-measurement workflows
- **Before logic blocks** like Continue If to make decisions based on distance measurements (e.g., continue if distance is safe, filter based on spacing requirements, make decisions using distance thresholds), enabling distance-based decision workflows
- **Before analysis blocks** to analyze spatial relationships between objects (e.g., analyze object spacing, process distance measurements, work with spatial data), enabling spatial analysis workflows
- **Before notification blocks** to alert when distances violate thresholds (e.g., send alerts when spacing is too close, notify on distance violations, trigger actions based on measurements), enabling distance-based notification workflows
- **Before data storage blocks** to record distance measurements (e.g., store distance measurements, log spacing data, record spatial metrics), enabling distance measurement logging workflows
- **In measurement pipelines** where distance calculation is part of a larger spatial analysis workflow (e.g., measure distances in analysis pipelines, calculate spacing in monitoring systems, process spatial measurements in chains), enabling spatial measurement pipeline workflows

## Requirements

This block requires detection predictions with bounding boxes and class names. The image should be captured from a perpendicular camera view (top-down or frontal) to minimize perspective distortion and ensure accurate 2D distance measurements. For reference object calibration, a reference object with known dimensions must be present in the detections. For pixel-to-centimeter ratio calibration, the pixel ratio must be pre-calibrated or known for the camera setup. Objects must not overlap and must have a gap along the specified measurement axis (horizontal or vertical). The block assumes objects are on the same plane for accurate 2D measurement.
"""

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
        description="Detection predictions containing bounding boxes and class names for objects in the image. Must include detections for the two objects to measure (object_1_class_name and object_2_class_name) and optionally a reference object (if using reference object calibration method). The bounding boxes will be used to calculate distances between objects. Both object detection and instance segmentation predictions are supported. The detections must contain class_name information to identify objects.",
        examples=["$steps.model.predictions"],
    )

    object_1_class_name: str = Field(
        title="First Object Class Name",
        description="Class name of the first object to measure distance from. Must match exactly the class name in the detection predictions. The block searches for this class name in the detections and uses its bounding box for distance calculation. Example: if detections contain objects labeled 'person', 'car', 'bicycle', use 'person' to measure distance from a person to another object. The class name is case-sensitive and must match exactly.",
        examples=["car"],
    )

    object_2_class_name: str = Field(
        title="Second Object Class Name",
        description="Class name of the second object to measure distance to. Must match exactly the class name in the detection predictions. The block searches for this class name in the detections and uses its bounding box for distance calculation. Example: if detections contain objects labeled 'person', 'car', 'bicycle', use 'person' to measure distance to a person from another object. The class name is case-sensitive and must match exactly. The block measures the gap between object_1 and object_2 along the specified reference_axis.",
        examples=["person"],
    )

    reference_axis: Literal["horizontal", "vertical"] = Field(
        title="Reference Axis",
        description="Axis along which to measure the distance between the two objects. Options: 'horizontal' measures distance along the X-axis (left-right gap between objects, useful when objects are side-by-side), or 'vertical' measures distance along the Y-axis (top-bottom gap between objects, useful when objects are stacked vertically). The distance is measured as the gap between bounding boxes along the selected axis. Objects must have a gap along this axis (not overlap) for accurate measurement. Choose based on object orientation: horizontal for side-by-side objects, vertical for stacked objects.",
        examples=["vertical", "horizontal", "$inputs.reference_axis"],
    )

    calibration_method: Literal["reference object", "pixel to centimeter"] = Field(
        title="Calibration Method",
        description="Method to calibrate pixel measurements to real-world units (centimeters). Options: 'reference object' (uses a reference object with known dimensions in the image to calculate pixel-to-centimeter ratio automatically, more flexible for different scales), or 'pixel to centimeter' (uses a pre-calibrated pixel ratio directly, simpler but requires known scale). For reference object method, a reference object must be present in detections with known width and height. For pixel ratio method, the pixel_ratio must be pre-calibrated for your camera setup.",
    )

    reference_object_class_name: Union[str, Selector(kind=[STRING_KIND])] = Field(
        title="Reference Object Class Name",
        description="Class name of the reference object used for calibration (only used when calibration_method is 'reference object'). Must match exactly the class name in the detection predictions. The reference object must have known real-world dimensions (reference_width and reference_height). The block measures the reference object's pixel dimensions and calculates a pixel-to-centimeter ratio to convert all distance measurements. Default is 'reference-object'. The reference object must be present in the detections and should be clearly visible and correctly detected.",
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
        description="Real-world width of the reference object in centimeters (only used when calibration_method is 'reference object'). Must be greater than 0. This is the actual physical width of the reference object. The block measures the reference object's width in pixels and divides by this value to calculate the pixel-to-centimeter ratio. Use accurate measurements for best results. Example: if your reference object is a 2.5cm wide card, use 2.5. The reference_width and reference_height are used to calculate separate width and height ratios, then averaged for more accurate scaling.",
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
        description="Real-world height of the reference object in centimeters (only used when calibration_method is 'reference object'). Must be greater than 0. This is the actual physical height of the reference object. The block measures the reference object's height in pixels and divides by this value to calculate the pixel-to-centimeter ratio. Use accurate measurements for best results. Example: if your reference object is a 2.5cm tall card, use 2.5. The reference_width and reference_height are used to calculate separate width and height ratios, then averaged for more accurate scaling.",
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
        description="Pixel-to-centimeter conversion ratio for the image (only used when calibration_method is 'pixel to centimeter'). Must be greater than 0. This value represents how many pixels equal 1 centimeter. Example: if 100 pixels = 1 centimeter, use 100. The block divides pixel distances by this ratio to convert to centimeters. This ratio must be pre-calibrated for your specific camera setup, viewing distance, and image resolution. Typical values range from 10-500 depending on camera distance and resolution. A higher ratio means more pixels per centimeter (objects appear larger, camera is closer), a lower ratio means fewer pixels per centimeter (objects appear smaller, camera is farther).",
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
