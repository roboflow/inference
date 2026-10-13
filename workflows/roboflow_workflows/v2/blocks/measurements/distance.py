"""Gap between two detected boxes along one axis: ``v2/distance_measurement@v1``.

Port of V1 ``roboflow_core/distance_measurement@v1`` (latest tensor version).
Boxes are rounded to whole pixels first, as in V1. Per call::

    names    = row "class" string, else class_names[class_id], else "class_<id>"
    box_1, box_2 = scan rows in order; keep the latest row of each class;
                   stop as soon as both classes have a row
    overlap(box_1, box_2) or no gap on the axis  ->  0 cm, 0 px
    gap_px   = far edge of the nearer box to near edge of the other, on the axis
    ratio    = pixel_ratio                                     "pixel to centimeter"
             = mean(ref_w_px / reference_width,
                    ref_h_px / reference_height)                "reference object"
    gap_cm   = gap_px / ratio

The reference object is the first row of ``reference_object_class_name``.
Boxes touching on the axis count as overlapping when the other axis overlaps
too, so they measure 0. Pixels are pixels of the image the prediction
describes: a crop keeps the scale of its source, a resized image does not.
A ratio calibrated on one resolution does not hold on another.

Physical meaning: centimetres are only as good as the calibration. The block
assumes a camera looking straight at one plane holding both objects (and the
reference). It cannot detect perspective, lens distortion or objects at
different depths; those give wrong centimetres without an error.

CPU boundary: two copies to the host per call, the ``(N, 4)`` boxes and the
``N`` class ids; the rest is scalar Python arithmetic.

Differences from V1:

* ``distance_cm`` is declared ``float`` (V1 declared ``integer`` for both).
  The zero result is ``0.0`` cm and ``0`` px.
* The same class for both objects is rejected with its own message; V1 could
  never find the second object and reported it as missing.
* A reference box of zero width and height raises ``ValueError``; V1 divided
  by zero.
* An infinite ``pixel_ratio``, ``reference_width`` or ``reference_height``
  raises ``ValueError``; V1 returned meaningless centimetres (0 cm for an
  infinite ``pixel_ratio``) or divided by zero.
* Row class names follow the shared V2 rule (``row_class_name``): a non-string
  ``class`` value is ignored; V1 converted it with ``str``.
"""

import math
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple

from pydantic import Field
from roboflow_workflows.v2.blocks.geometry.mask_inputs import (
    INSTANCE_SEGMENTATION_VIEW_KIND,
    boxes_of_view,
)
from roboflow_workflows.v2.data.kinds import FLOAT_KIND, INTEGER_KIND, STRING_KIND
from roboflow_workflows.v2.data.prediction_ops import class_names_by_id, row_class_names
from roboflow_workflows.v2.data.predictions import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
)
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref

__all__ = ["DistanceMeasurementBlock"]

DETECTION_KINDS = (
    OBJECT_DETECTION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_VIEW_KIND,
)
PositiveFloat = Annotated[float, Field(gt=0)]
Box = Tuple[int, int, int, int]
Axis = Literal["horizontal", "vertical"]
AXES = ("horizontal", "vertical")


class DistanceMeasurementBlock(Block):
    """Measure the gap between two detected objects along one axis.

    Example: a ``cup`` box ending at x = 100 and a ``plate`` box starting at
    x = 160 are 60 px apart horizontally; with ``pixel_ratio`` 20 px/cm that
    is 3 cm.
    """

    type = "v2/distance_measurement@v1"
    prunable = True
    metadata = {
        "name": "Distance Measurement",
        "port_of": "roboflow_core/distance_measurement@v1",
        "section": "classical_cv",
    }
    outputs = {
        "distance_cm": Output(
            FLOAT_KIND,
            source="predictions",
            description="Gap in centimetres under the chosen calibration.",
        ),
        "distance_pixel": Output(
            INTEGER_KIND,
            source="predictions",
            description="Gap in whole pixels of the prediction's image.",
        ),
    }

    class Params(BlockParams):
        predictions: Ref(*DETECTION_KINDS) = Field(
            description="Detections holding both objects and, for 'reference "
            "object' calibration, the reference object.",
            examples=["$steps.model.predictions"],
        )
        object_1_class_name: str = Field(
            description="Exact, case-sensitive class name of the first object.",
            examples=["car"],
        )
        object_2_class_name: str = Field(
            description="Exact, case-sensitive class name of the second object.",
            examples=["person"],
        )
        reference_axis: Axis | Ref(STRING_KIND) = Field(
            description="'horizontal' measures the left-right gap, 'vertical' "
            "the top-bottom gap.",
            examples=["vertical", "$inputs.reference_axis"],
        )
        calibration_method: Literal["reference object", "pixel to centimeter"] = Field(
            description="'reference object' derives pixels per cm from a "
            "detected object of known size; 'pixel to centimeter' uses "
            "pixel_ratio.",
        )
        reference_object_class_name: str | Ref(STRING_KIND) = Field(
            default="reference-object",
            description="Class name of the reference object; 'reference object' "
            "calibration only. The first such row is used.",
            examples=["reference-object", "$inputs.reference_object_class_name"],
        )
        reference_width: PositiveFloat | Ref(FLOAT_KIND) = Field(
            default=2.5,
            description="Real width of the reference object in centimetres.",
            examples=[2.5, "$inputs.reference_width"],
        )
        reference_height: PositiveFloat | Ref(FLOAT_KIND) = Field(
            default=2.5,
            description="Real height of the reference object in centimetres.",
            examples=[2.5, "$inputs.reference_height"],
        )
        pixel_ratio: PositiveFloat | Ref(FLOAT_KIND) = Field(
            default=100,
            description="Pixels per centimetre in the prediction's image; "
            "'pixel to centimeter' calibration only.",
            examples=[100, "$inputs.pixel_ratio"],
        )

    def run(
        self,
        *,
        predictions: Any,
        object_1_class_name: str,
        object_2_class_name: str,
        reference_axis: Axis,
        calibration_method: str,
        reference_object_class_name: str,
        reference_width: float,
        reference_height: float,
        pixel_ratio: float,
    ) -> Dict[str, Any]:
        """Measure the gap between the two objects.

        Args:
            predictions: Native detections, or a ``SegmentationView`` (only
                its boxes are read; no mask is materialized).
            object_1_class_name: Class of the first object.
            object_2_class_name: Class of the second object.
            reference_axis: ``horizontal`` or ``vertical``.
            calibration_method: ``reference object`` or ``pixel to centimeter``.
            reference_object_class_name: Class of the reference object.
            reference_width: Real reference width in centimetres.
            reference_height: Real reference height in centimetres.
            pixel_ratio: Pixels per centimetre.

        Returns:
            ``distance_cm`` (float) and ``distance_pixel`` (int).

        Raises:
            ValueError: When both class names are equal, the axis is unknown,
                an object or the reference object is missing, the reference
                box has no size, or the used calibration values are not
                positive and finite (selector values bypass the parameter
                check). As in
                V1, calibration is checked only when the objects are apart.
        """
        if object_1_class_name == object_2_class_name:
            raise ValueError(
                f"object_1_class_name and object_2_class_name are both "
                f"{object_1_class_name!r}; the block measures two different "
                "classes"
            )

        if reference_axis not in AXES:
            raise ValueError(
                f"reference_axis must be one of {list(AXES)}, got {reference_axis!r}"
            )

        rows = _named_boxes(boxes_of_view(predictions))
        box_1, box_2 = _find_object_boxes(
            rows, first=object_1_class_name, second=object_2_class_name
        )
        if box_1 is None or box_2 is None:
            raise ValueError(
                f"Reference class '{object_1_class_name}' or "
                f"'{object_2_class_name}' not found in predictions."
            )

        if _boxes_overlap(box_1, box_2) or not _has_axis_gap(
            box_1, box_2, axis=reference_axis
        ):
            return {"distance_cm": 0.0, "distance_pixel": 0}

        if calibration_method == "reference object":
            pixels_per_cm = _reference_pixels_per_cm(
                rows,
                class_name=reference_object_class_name,
                size_cm=(reference_width, reference_height),
            )
        elif math.isfinite(pixel_ratio) and pixel_ratio > 0:
            pixels_per_cm = pixel_ratio
        else:
            raise ValueError(
                "Pixel-to-centimeter ratio must be greater than zero and finite, "
                f"got {pixel_ratio!r}."
            )

        distance_pixel = _gap_pixels(box_1, box_2, axis=reference_axis)
        result = {
            "distance_cm": distance_pixel / pixels_per_cm,
            "distance_pixel": distance_pixel,
        }

        return result


def _named_boxes(predictions: Any) -> List[Tuple[Box, Optional[str]]]:
    # V1 rounds the float boxes (half to even, as numpy and torch do) and
    # truncates to int.
    xyxy = predictions.xyxy.detach().round().to(device="cpu").tolist()
    class_ids = predictions.class_id.detach().to(device="cpu").tolist()
    row_metadata = predictions.bboxes_metadata or [None] * len(class_ids)
    names = row_class_names(
        class_ids,
        row_metadata=[row or {} for row in row_metadata],
        names_by_id=class_names_by_id(predictions.image_metadata),
    )
    rows = [
        (
            tuple(int(value) for value in box),
            name if name is not None else f"class_{class_id}",
        )
        for box, class_id, name in zip(xyxy, class_ids, names)
    ]

    return rows


def _find_object_boxes(
    rows: List[Tuple[Box, Optional[str]]], *, first: str, second: str
) -> Tuple[Optional[Box], Optional[Box]]:
    # V1 scan: the latest row of each class wins until both have been seen.
    box_1 = box_2 = None
    for box, name in rows:
        if name == first:
            box_1 = box
        elif name == second:
            box_2 = box
        if box_1 is not None and box_2 is not None:
            break

    return box_1, box_2


def _reference_pixels_per_cm(
    rows: List[Tuple[Box, Optional[str]]],
    *,
    class_name: str,
    size_cm: Tuple[float, float],
) -> float:
    reference = next((box for box, name in rows if name == class_name), None)
    if reference is None:
        raise ValueError(f"Reference class '{class_name}' not found in predictions.")

    if not all(math.isfinite(value) and value > 0 for value in size_cm):
        raise ValueError(
            "Reference object dimensions must be greater than zero and finite, "
            f"got {size_cm!r}."
        )

    width_cm, height_cm = size_cm

    width_px = abs(reference[2] - reference[0])
    height_px = abs(reference[3] - reference[1])
    pixels_per_cm = (width_px / width_cm + height_px / height_cm) / 2
    if pixels_per_cm == 0:
        raise ValueError(
            f"Reference object '{class_name}' has a box of zero width and height; "
            "it cannot calibrate pixels per centimetre"
        )

    return pixels_per_cm


def _boxes_overlap(box_1: Box, box_2: Box) -> bool:
    # Inclusive: touching edges overlap.
    x1_min, y1_min, x1_max, y1_max = box_1
    x2_min, y2_min, x2_max, y2_max = box_2
    apart_on_x = x1_max < x2_min or x2_max < x1_min
    apart_on_y = y1_max < y2_min or y2_max < y1_min

    return not (apart_on_x or apart_on_y)


def _has_axis_gap(box_1: Box, box_2: Box, *, axis: Axis) -> bool:
    low, high = (0, 2) if axis == "horizontal" else (1, 3)
    intervals_overlap = box_1[low] < box_2[high] and box_1[high] > box_2[low]

    return not intervals_overlap


def _gap_pixels(box_1: Box, box_2: Box, *, axis: Axis) -> int:
    low, high = (0, 2) if axis == "horizontal" else (1, 3)
    if box_2[low] > box_1[high]:
        return abs(box_2[low] - box_1[high])

    gap = abs(box_1[low] - box_2[high])

    return gap
