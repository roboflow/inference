"""Area of every detection: ``roboflow_core/mask_area_measurement@v1``.

Port of V1 ``roboflow_core/mask_area_measurement@v1`` (latest tensor version).
Each row gains two metadata values::

    area_px        = nonzero pixels of the row's reference mask
                     (holes are not counted; a 0/255 mask counts 1 per pixel)
                     box width * box height    no mask, or an empty mask
    area_converted = area_px / pixels_per_unit ** 2

Areas are square pixels of the image the prediction describes. A crop keeps
its source's pixel size, so the area of an object is the same in the crop and
in the full image. A resized image does not: calibrate ``pixels_per_unit`` on
the image the model actually saw.

Mask semantics per input representation; every count is exact for its
representation, never a low-resolution estimate::

    input                         counted pixels                 host work
    Detections (boxes only)       box area                       N floats
    InstanceDetections, dense     nonzero count per row, on      N floats
                                  device
    InstanceDetections, RLE       COCO RLE area of the counts    run lengths only;
                                  (pycocotools), no decode       no mask decoded
    SegmentationView              reference masks of the view    N floats
                                  (mask_provider_of(view)
                                  .binary(fidelity="reference"))

A ``SegmentationView`` needs its reference masks, the model's own full
resolution post-processing, to be counted: one ``(N, H, W)`` materialization
per view, shared with every other consumer of the same view. The result is a
new view over the same scores with the new row metadata; it starts with an
empty cache, so a later consumer of its masks materializes them again.

When the prediction names its frame, dense and RLE masks must have the
frame's size; a mismatch raises rather than counting pixels of another
raster.

The input is not modified: the result shares tensors and masks with it and
holds fresh row dictionaries.

Differences from V1:

* ``pixels_per_unit`` must be positive and finite. V1 silently used 1.0 for
  zero or negative values, reporting pixels as if they were calibrated units,
  and reported 0 for infinity.
* V1 wrote the areas into the input prediction; V2 returns a new one.
"""

import math
from dataclasses import replace
from typing import Annotated, Any, Dict, List, Optional, Tuple
import torch
from pycocotools import mask as mask_utils
from pydantic import Field
from roboflow_workflows.v2.blocks.geometry.mask_inputs import (
    INSTANCE_SEGMENTATION_VIEW_KIND,
    boxes_of_view,
)
from roboflow_workflows.v2.data.errors import ContractError
from roboflow_workflows.v2.data.frames import frame_of
from roboflow_workflows.v2.data.kinds import FLOAT_KIND
from roboflow_workflows.v2.data.masks import SegmentationView
from roboflow_workflows.v2.data.masks.providers import mask_provider_of
from roboflow_workflows.v2.data.predictions import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
)
from roboflow_workflows.v2.data.rle import check_rle_masks
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.types import InstancesRLEMasks


__all__ = ["AREA_CONVERTED_KEY", "AREA_KEY", "MaskAreaMeasurementBlock"]

AREA_KEY = "area_px"

AREA_CONVERTED_KEY = "area_converted"

PREDICTION_KINDS = (
    OBJECT_DETECTION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_VIEW_KIND,
)

PositiveFloat = Annotated[float, Field(gt=0)]

class MaskAreaMeasurementBlock(Block):
    """Add each detection's area, in pixels and in calibrated units.

    Example: a ring-shaped mask of 400 foreground pixels with a 100-pixel
    hole gets ``area_px`` 400. With ``pixels_per_unit`` 10 px/cm,
    ``area_converted`` is 4 cm².
    """

    type = "roboflow_core/mask_area_measurement@v1"
    prunable = True
    metadata = {
        "name": "Mask Area Measurement",
        "port_of": "roboflow_core/mask_area_measurement@v1",
        "section": "classical_cv",
    }
    outputs = {
        "predictions": Output(
            *PREDICTION_KINDS,
            source="predictions",
            description="Same rows and representation with area_px and "
            "area_converted in each row's metadata.",
        ),
    }

    class Params(BlockParams):
        predictions: Ref(*PREDICTION_KINDS) = Field(
            description="Detections to measure; masks give the most accurate " "areas.",
            examples=["$steps.model.predictions"],
        )
        pixels_per_unit: PositiveFloat | Ref(FLOAT_KIND) = Field(
            default=1.0,
            description="Pixels per real-world unit of length in the "
            "prediction's image, e.g. pixels per cm. 1.0 leaves area_converted "
            "equal to area_px.",
            examples=[1.0, 130.0, "$inputs.pixels_per_unit"],
        )

    def run(self, *, predictions: Any, pixels_per_unit: float) -> Dict[str, Any]:
        """Measure every row's area.

        Args:
            predictions: ``Detections``, ``InstanceDetections`` (dense or RLE
                masks) or a ``SegmentationView``.
            pixels_per_unit: Pixels per unit of length.

        Returns:
            ``predictions``: a new prediction of the same type and rows, with
            ``area_px`` and ``area_converted`` in every row's metadata.

        Raises:
            ValueError: When ``pixels_per_unit`` is not positive and finite (a
                selector value bypasses the parameter check), or the mask raster
                differs from the frame size.
            ContractError: On malformed RLE counts.
        """
        if not (math.isfinite(pixels_per_unit) and pixels_per_unit > 0):
            raise ValueError(
                "pixels_per_unit must be greater than zero and finite, got "
                f"{pixels_per_unit!r}"
            )

        areas = _areas_px(predictions)
        boxes = boxes_of_view(predictions)
        rows = boxes.bboxes_metadata or [None] * len(areas)
        unit_area = pixels_per_unit**2
        measured_rows = [
            {**(row or {}), AREA_KEY: area, AREA_CONVERTED_KEY: area / unit_area}
            for row, area in zip(rows, areas)
        ]

        if isinstance(predictions, SegmentationView):
            measured = predictions.with_row_metadata(measured_rows)
        else:
            measured = replace(predictions, bboxes_metadata=measured_rows)
        result = {"predictions": measured}

        return result

def _areas_px(predictions: Any) -> List[float]:
    if isinstance(predictions, SegmentationView):
        reference = mask_provider_of(predictions).binary(fidelity="reference")
        predictions = reference.predictions

    if not isinstance(predictions, InstanceDetections):
        areas = _box_areas(predictions.xyxy).tolist()
        return areas

    if isinstance(predictions.mask, InstancesRLEMasks):
        areas = _rle_areas(predictions)
        return areas

    mask = predictions.mask
    _check_mask_raster(predictions, mask_size_hw=tuple(mask.shape[1:]))
    # Count, not sum: a uint8 mask may hold 255 for foreground, as V1's
    # cv2.countNonZero accepts.
    counts = torch.count_nonzero(mask, dim=(1, 2))
    # float64 holds every int64 count below 2**53 and every float32 box area
    # exactly, so one transfer carries both.
    areas = torch.where(
        counts > 0,
        counts.to(dtype=torch.float64),
        _box_areas(predictions.xyxy).to(dtype=torch.float64),
    )
    host_areas = areas.tolist()

    return host_areas

def _rle_areas(predictions: InstanceDetections) -> List[float]:
    rle = predictions.mask
    if not rle.masks:
        return []

    check_rle_masks(rle)
    height, width = (int(value) for value in rle.image_size)
    _check_mask_raster(predictions, mask_size_hw=(height, width))
    counts = mask_utils.area(
        [{"size": [height, width], "counts": counts} for counts in rle.masks]
    ).tolist()
    box_areas = _box_areas(predictions.xyxy).tolist()
    areas = [
        float(count) if count > 0 else box_area
        for count, box_area in zip(counts, box_areas)
    ]

    return areas

def _box_areas(xyxy: torch.Tensor) -> torch.Tensor:
    # V1 arithmetic: (x2 - x1) * (y2 - y1) in the boxes' own dtype.
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])

    return areas

def _check_mask_raster(
    predictions: InstanceDetections, *, mask_size_hw: Tuple[int, int]
) -> None:
    frame_size = _frame_size_hw(predictions)
    mask_size = tuple(int(value) for value in mask_size_hw)
    if frame_size is not None and mask_size != frame_size:
        raise ValueError(
            f"Masks are {mask_size[0]}x{mask_size[1]} but the prediction's "
            f"image is {frame_size[0]}x{frame_size[1]}; mask areas need masks "
            "at the image's resolution"
        )

def _frame_size_hw(predictions: Any) -> Optional[Tuple[int, int]]:
    # Unknown when the prediction does not name its frame, e.g. hand-built
    # predictions without image metadata.
    try:
        frame = frame_of(predictions)
    except ContractError:
        return None

    size_hw = tuple(frame.size_hw)

    return size_hw
