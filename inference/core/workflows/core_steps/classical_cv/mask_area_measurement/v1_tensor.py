from typing import List, Literal, Optional, Type, Union

import cv2 as cv
import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.workflows.execution_engine.constants import (
    AREA_CONVERTED_KEY_IN_SV_DETECTIONS,
    AREA_KEY_IN_SV_DETECTIONS,
)
from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

OUTPUT_KEY = "predictions"
SHORT_DESCRIPTION = (
    "Measure the area of detected objects and optionally convert to real-world units."
)
LONG_DESCRIPTION = """
Measure the area of detected objects. For instance segmentation masks, the area is computed by counting non-zero mask pixels (correctly handling holes). For bounding-box-only detections, the area is width multiplied by height. Optionally converts pixel areas to real-world units using a `pixels_per_unit` calibration value.

## How This Block Works

This block calculates the area of each detected object and stores two values per detection:

- **`area_px`** — area in square pixels (always computed)
- **`area_converted`** — area in real-world units: `area_px / (pixels_per_unit ** 2)` (equals `area_px` when `pixels_per_unit` is 1.0)

Both values are attached to each detection and included in the serialized JSON output. The block returns the input detections with these fields added, so downstream blocks (e.g., label visualization) can display the area values.

### Area Computation

The block operates in two modes depending on the type of predictions it receives:

1. **Mask Pixel Area (Instance Segmentation)**: When the input detections include segmentation masks, the block counts the non-zero pixels in each mask using `cv2.countNonZero`. This correctly handles masks with holes — hole pixels are zero and are excluded from the count.

2. **Bounding Box Area (Object Detection)**: When no segmentation mask is available, the block falls back to computing the area as the bounding box width multiplied by height (`w * h`).

### Unit Conversion

Set the `pixels_per_unit` input to convert pixel areas to real-world units (e.g., cm², in², mm²). Because area is two-dimensional, the conversion squares the ratio:

```
area_converted = area_px / (pixels_per_unit ** 2)
```

For example, if your calibration is 130 pixels/cm, a detection with `area_px = 16900` would have `area_converted = 16900 / 16900 = 1.0 cm²`.

**How to determine pixels_per_unit:** Place an object of known size in the camera's field of view (e.g., a ruler or calibration target). Measure its length in pixels in the image and divide by its real-world length. For instance, if a 10 cm reference object spans 1300 pixels, then `pixels_per_cm = 1300 / 10 = 130`. If you are using perspective correction, the calibration object must be placed on the same plane from which the perspective correction was calculated.

## Common Use Cases

- **Size-Based Filtering**: Filter out small noise detections by chaining with a filtering block to keep only detections above a minimum area threshold.
- **Quality Control**: Verify that manufactured components meet size specifications by comparing measured areas against expected ranges.
- **Agricultural Analysis**: Measure leaf area, crop coverage, or canopy extent from aerial or close-up imagery.
- **Medical Imaging**: Quantify the area of wounds, lesions, or anatomical structures. Use `pixels_per_unit` to get real-world measurements for clinical documentation.

## Connecting to Other Blocks

- **Upstream -- Detection and Segmentation Models**: Connect the output of an object detection or instance segmentation model to the `predictions` input. Instance segmentation models (which produce masks) yield more accurate area measurements than bounding-box-only detections.
- **Upstream -- Camera Calibration Block**: Use `roboflow_core/camera_calibration@v1` upstream to correct lens distortion before detection.
- **Upstream -- Perspective Correction Block**: Use `roboflow_core/perspective_correction@v1` upstream to transform angled images to a top-down view so that area measurements reflect true object footprints.
- **Downstream -- Visualization**: Pass the output `predictions` to label or polygon visualization blocks. The `area_px` and `area_converted` fields are available for display as labels.
- **Downstream -- Filtering Blocks**: Use the enriched detections with a filtering block to keep only detections whose area meets a threshold.

## Requirements

This block requires detection predictions from an object detection or instance segmentation model. No additional environment variables, API keys, or external dependencies are needed beyond OpenCV and NumPy (included with inference). For the most accurate area measurements, use instance segmentation models that produce per-object masks.
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

    pixels_per_unit: Union[float, Selector(kind=[FLOAT_KIND])] = Field(  # type: ignore
        default=1.0,
        description="Number of pixels per real-world unit of length (e.g., pixels per cm). "
        "The converted area is computed as area_px / (pixels_per_unit ** 2). "
        "Default 1.0 means no conversion (area_converted equals area_px).",
        examples=[1.0, 130.0],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name=OUTPUT_KEY,
                kind=[
                    OBJECT_DETECTION_PREDICTION_KIND,
                    INSTANCE_SEGMENTATION_PREDICTION_KIND,
                ],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


def compute_detection_areas(detections: sv.Detections) -> List[float]:
    """Compute the area of all detections in square pixels.

    For bounding-box-only detections, areas are computed in a single vectorized
    operation. For detections with segmentation masks, the area is the count of
    non-zero mask pixels (via ``cv2.countNonZero``). This correctly handles masks
    with holes — hole pixels are zero and are not counted. Falls back to the
    bounding box area when the mask pixel count is zero.

    Args:
        detections: A supervision Detections object.

    Returns:
        List of areas in square pixels, one per detection.
    """
    n = len(detections)
    if n == 0:
        return []

    areas = []
    for i in range(n):
        if detections.mask is not None:
            count = cv.countNonZero(detections.mask[i].astype(np.uint8))
            if count > 0:
                areas.append(float(count))
                continue
        x1, y1, x2, y2 = detections.xyxy[i]
        areas.append(float((x2 - x1) * (y2 - y1)))

    return areas


class MaskAreaMeasurementBlockV1(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return MaskAreaMeasurementManifest

    def run(
        self,
        predictions: sv.Detections,
        pixels_per_unit: float = 1.0,
    ) -> BlockResult:
        areas_px = np.array(compute_detection_areas(predictions))
        predictions.data[AREA_KEY_IN_SV_DETECTIONS] = areas_px
        scale = pixels_per_unit**2 if pixels_per_unit > 0 else 1.0
        predictions.data[AREA_CONVERTED_KEY_IN_SV_DETECTIONS] = areas_px / scale
        return {OUTPUT_KEY: predictions}
