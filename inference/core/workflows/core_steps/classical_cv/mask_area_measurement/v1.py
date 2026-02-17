from typing import List, Literal, Optional, Type

import cv2 as cv
import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    LIST_OF_VALUES_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

OUTPUT_KEY = "areas"
SHORT_DESCRIPTION = "Measure the mask area of detected objects in square pixels."
LONG_DESCRIPTION = """
Measure the area of detected objects in square pixels. For detections with segmentation masks, the area is computed from the largest contour using `cv2.contourArea`. For bounding-box-only detections, the area is the bounding box width multiplied by height.

## How This Block Works

This block calculates the area of each detected object in an image, returning a list of area values in square pixels. The block operates in two modes depending on the type of predictions it receives:

1. **Polygon/Mask Area (Instance Segmentation)**: When the input detections include segmentation masks, the block extracts the mask for each detection, finds the largest external contour using OpenCV's `findContours`, and computes its area with `cv2.contourArea`. This provides a precise measurement that follows the actual shape of the object rather than its bounding rectangle.

2. **Bounding Box Area (Object Detection)**: When no segmentation mask is available, the block falls back to computing the area as the bounding box width multiplied by height (`w * h`). This is less precise but works with any object detection model output.

The block processes all detections in the input and returns a list of area values (one per detection) under the `areas` output key.

## Common Use Cases

- **Size-Based Filtering**: Use area measurements to filter out small noise detections or irrelevant objects. Chain the output with a filtering block to keep only detections above a minimum area threshold, reducing false positives from small artifacts.
- **Agricultural Analysis**: Measure leaf area, crop coverage, or canopy extent from aerial or close-up imagery. Combine with segmentation models trained on plant structures to quantify growth or health metrics.
- **Quality Control**: Verify that manufactured components meet size specifications by measuring their area in pixels and comparing against expected ranges. Flag parts that fall outside acceptable tolerances.
- **Medical Imaging**: Quantify the area of wounds, lesions, skin conditions, or anatomical structures in clinical images. When paired with a calibration reference, pixel areas can be converted to real-world units for clinical documentation.
- **Inventory and Logistics**: Estimate the footprint of packages, pallets, or items on conveyor belts. Area measurements help classify objects by size category or detect misplaced items that differ from expected dimensions.

## Connecting to Other Blocks

This block receives detection predictions and produces a list of area values:

- **Upstream -- Detection and Segmentation Models**: Connect the output of an object detection or instance segmentation model to the `predictions` input. Instance segmentation models (which produce masks) yield more accurate area measurements than bounding-box-only detections.
- **Upstream -- Camera Calibration Block**: If your camera introduces lens distortion, run images through the Camera Calibration block before detection to correct geometric distortions. This ensures that area measurements are not skewed by barrel or pincushion distortion.
- **Upstream -- Perspective Correction Block**: For images captured at an angle (e.g., overhead cameras that are not perfectly perpendicular), apply the Perspective Correction block before detection. This transforms the image to a top-down view so that area measurements reflect true object footprints rather than perspective-distorted projections.
- **Downstream -- Filtering Blocks**: Pass the `areas` output to a filtering or condition block (e.g., Continue If, Detection Filter) to keep only detections whose area meets a threshold, enabling size-based filtering workflows.
- **Downstream -- Analytics and Visualization**: Feed area values into analytics blocks for aggregation, charting, or overlay display. Combine with other measurement blocks (Size Measurement, Distance Measurement) for comprehensive spatial analysis.

## Converting Pixel Areas to Real-World Units

This block returns areas in **square pixels**. To convert to real-world units (e.g., cm², in², mm²), you need a calibration value: the number of pixels per unit of length along one axis (pixels_per_unit).

**Important:** Because area is two-dimensional, you must **square** the pixels_per_unit ratio:

```
area_real = area_px / (pixels_per_unit ** 2)
```

For example, if your calibration is 130 pixels/cm:

```
area_cm² = area_px / (130 ** 2) = area_px / 16900
```

A common mistake is dividing by `pixels_per_unit` instead of `pixels_per_unit²`, which produces values that are off by a factor of the ratio itself.

**How to determine pixels_per_unit:** Place an object of known size in the camera's field of view (e.g., a ruler or calibration target). Measure its length in pixels in the image and divide by its real-world length. For instance, if a 10 cm reference object spans 1300 pixels, then `pixels_per_cm = 1300 / 10 = 130`.

You can perform this conversion downstream using a Dynamic Python Block or a simple post-processing step in your application code.

## Handling Camera Distortion

Lens distortion (barrel or pincushion) warps the image so that objects near the edges appear stretched or compressed compared to objects near the center. This directly affects area measurements — the same real-world object will report different pixel areas depending on where it appears in the frame.

**When distortion matters:**
- Wide-angle or fisheye lenses with visible barrel distortion
- Applications requiring consistent area measurements across the entire frame
- When objects of interest appear near the edges of the image

**When distortion can be ignored:**
- Objects are always near the center of the frame
- The lens has minimal distortion (e.g., telecentric or narrow field-of-view lenses)
- Relative area comparisons (larger vs. smaller) are sufficient and absolute accuracy is not required

**How to correct for it:**
1. **Camera Calibration Block**: Use the `roboflow_core/camera_calibration@v1` block upstream of your detection model. This block applies OpenCV's `undistort` to remove lens distortion before the image is processed, so that mask areas reflect the true object shape.
2. **Perspective Correction Block**: If the camera views the scene at an angle (not perpendicular), use the `roboflow_core/perspective_correction@v1` block to warp the image to a top-down view. Without this, objects farther from the camera appear smaller and their measured areas will be understated.

Both corrections should be applied **before** the detection model runs, so that the segmentation masks and bounding boxes are computed on the corrected image.

## Requirements

This block requires detection predictions from an object detection or instance segmentation model. No additional environment variables, API keys, or external dependencies are needed beyond the standard OpenCV and NumPy libraries included with inference. For the most accurate area measurements, use instance segmentation models that produce per-object masks. Bounding-box-only detections will yield rectangular area approximations.
"""


class MaskAreaMeasurementManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Mask Area Measurement",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
            "ui_manifest": {
                "section": "classical_cv",
                "icon": "far fa-ruler-combined",
                "blockPriority": 12,
                "opencv": True,
            },
        }
    )
    type: Literal["roboflow_core/mask_area_measurement@v1"]

    predictions: Selector(
        kind=[
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
            OBJECT_DETECTION_PREDICTION_KIND,
        ]
    ) = Field(
        description="Detection predictions to measure areas for.",
        examples=["$steps.model.predictions"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name=OUTPUT_KEY, kind=[LIST_OF_VALUES_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


def get_detection_area(detection: sv.Detections, index: int) -> float:
    """Compute the area of a single detection in square pixels.

    If a segmentation mask is available, the area is computed from the largest
    contour found in the mask using ``cv2.contourArea``.  Otherwise the area
    falls back to bounding-box width * height.

    Args:
        detection: A supervision Detections object.
        index: Index of the detection to measure.

    Returns:
        Area in square pixels.
    """
    if detection.mask is not None:
        mask = detection.mask[index].astype(np.uint8)
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv.contourArea)
            area = cv.contourArea(largest_contour)
            if area > 0:
                return float(area)

    bbox = detection.xyxy[index]
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return float(w * h)


class MaskAreaMeasurementBlockV1(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return MaskAreaMeasurementManifest

    def run(
        self,
        predictions: sv.Detections,
    ) -> BlockResult:
        areas = []
        for i in range(len(predictions)):
            areas.append(get_detection_area(predictions, i))
        return {OUTPUT_KEY: areas}
