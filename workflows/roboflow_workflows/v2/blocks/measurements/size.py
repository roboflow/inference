"""Object sizes relative to a reference object: ``v2/size_measurement@v1``.

Port of V1 ``roboflow_core/size_measurement@v1`` (latest tensor version)::

    reference      = highest-confidence reference row (first on ties),
                     or a reference polygon given as a list of [x, y] points
    size_px(row)   = mask: minAreaRect of the largest external contour;
                           width = the more horizontal edge
                     boxes only: (x2 - x1, y2 - y1)
    scale          = (reference_width / ref_w_px, reference_height / ref_h_px)
    dimensions[i]  = {width, height, longer, shorter} in reference units,
                     or None when row i has no measurable size

``dimensions`` is ``None`` when there is no reference row, or the reference
has no measurable size. Otherwise it has one entry per object row, in row
order. Widths and heights are scaled separately, so a reference that is
wider in pixels than in reality changes widths, not heights.

Mask sizes follow OpenCV: contour points are pixel centres, so a filled
w x h pixel rectangle measures (w - 1) x (h - 1). A box measures its own
width and height. Mixing a mask reference with box objects (or the reverse)
therefore differs by about one pixel per edge, as in V1.

Pixels are pixels of the image each prediction describes. When both
predictions name their frames, their pixels must have the same size in the
workflow input (a crop keeps it, a resize does not); otherwise the block
raises instead of mixing pixel sizes. A reference polygon names no frame and
is not checked. When a prediction names its frame, its dense and RLE masks
must have the frame's size; a mismatch raises rather than measuring pixels of
another raster.

Physical meaning: the result assumes the reference and the objects lie in one
plane facing the camera. Perspective, lens distortion and depth differences
give wrong sizes without an error. Units are those of ``reference_dimensions``.

CPU boundary: contours and rectangles are OpenCV host algorithms. For a dense
mask, two small transfers (per-row and per-column "any pixel" flags) find
every row's pixel window, then each non-empty window is copied to the host:
2 + one copy per non-empty row. RLE masks are decoded on the host one row at a
time. A ``SegmentationView`` gives its reference masks
(``mask_provider_of(view).binary(fidelity="reference")``); for the reference
prediction only the chosen row is materialized.

Differences from V1:

* Reference dimensions must be two positive, finite numbers; V1 accepted
  zero or negative values and returned meaningless sizes.
* Predictions describing differently scaled images raise ``ValueError``.
* Masks of another size than the prediction's named frame raise
  ``ValueError``; V1 measured them as frame pixels.
* A reference polygon may have float vertices; V1 failed in OpenCV for them.
* A reference polygon that is not a list of finite ``[x, y]`` points raises
  a ``ValueError`` naming ``reference_predictions``; V1 failed in OpenCV or
  NumPy.
"""

import math
from typing import Annotated, Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from pydantic import Field
from roboflow_workflows.v2.blocks.geometry.mask_inputs import (
    INSTANCE_SEGMENTATION_VIEW_KIND,
    boxes_of_view,
)
from roboflow_workflows.v2.data.errors import ContractError
from roboflow_workflows.v2.data.frames import frame_of, root_placement_from_metadata
from roboflow_workflows.v2.data.kinds import LIST_OF_VALUES_KIND, STRING_KIND
from roboflow_workflows.v2.data.masks import SegmentationView
from roboflow_workflows.v2.data.masks.providers import mask_provider_of
from roboflow_workflows.v2.data.predictions import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
)
from roboflow_workflows.v2.data.rle import decode_rle_masks
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.types import InstancesRLEMasks

__all__ = ["SizeMeasurementBlock", "parse_reference_dimensions"]

DETECTION_KINDS = (
    OBJECT_DETECTION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_VIEW_KIND,
)
Size = Tuple[float, float]
# (x0, y0) of a host mask window and its uint8 pixels.
Window = Tuple[Tuple[int, int], np.ndarray]
Point = Annotated[List[float], Field(min_length=2, max_length=2)]


class SizeMeasurementBlock(Block):
    """Measure objects in the units of a reference object of known size.

    Example: a 5 x 3 cm card is 50 x 30 px, so 0.1 cm per pixel each way; a
    100 x 60 px package measures 10 x 6 cm.
    """

    type = "v2/size_measurement@v1"
    prunable = True
    metadata = {
        "name": "Size Measurement",
        "port_of": "roboflow_core/size_measurement@v1",
        "section": "classical_cv",
    }
    outputs = {
        "dimensions": Output(
            LIST_OF_VALUES_KIND,
            source="object_predictions",
            description="Per object row {width, height, longer, shorter} in "
            "reference units, or None; None as a whole without a reference.",
        ),
    }

    class Params(BlockParams):
        object_predictions: Ref(*DETECTION_KINDS) = Field(
            description="Detections to measure; masks give rotation-aware sizes.",
            examples=["$steps.objects.predictions"],
        )
        reference_predictions: List[Point] | Ref(
            *DETECTION_KINDS, LIST_OF_VALUES_KIND
        ) = Field(
            description="Detections of the reference object (the most confident "
            "row is used), or its polygon as [x, y] points.",
            examples=["$steps.reference.predictions"],
        )
        reference_dimensions: (
            str
            | Tuple[float, float]
            | List[float]
            | Ref(STRING_KIND, LIST_OF_VALUES_KIND)
        ) = Field(
            description="Real reference width and height, e.g. '4.5,3.0' or "
            "[4.5, 3.0]; results use the same unit.",
            examples=[[4.5, 3.0], "5.0,5.0", "$inputs.reference_dimensions"],
        )

    def run(
        self,
        *,
        object_predictions: Any,
        reference_predictions: Any,
        reference_dimensions: Union[str, Sequence[float]],
    ) -> Dict[str, Any]:
        """Measure every object row.

        Args:
            object_predictions: Detections, instance segmentation or a view.
            reference_predictions: Reference detections or a polygon.
            reference_dimensions: Real reference width and height.

        Returns:
            ``dimensions``: a list aligned with the object rows, or ``None``.

        Raises:
            ValueError: On invalid reference dimensions, a malformed reference
                polygon, masks of another size than their named frame, or
                predictions whose pixels have different sizes.
        """
        reference_width, reference_height = parse_reference_dimensions(
            reference_dimensions
        )
        reference_px = _reference_size_px(reference_predictions)
        if reference_px is None:
            return {"dimensions": None}

        if not isinstance(reference_predictions, list):
            _check_same_pixel_size(object_predictions, reference_predictions)

        width_scale = reference_width / reference_px[0]
        height_scale = reference_height / reference_px[1]
        dimensions = [
            _scaled(size, scale=(width_scale, height_scale))
            for size in _sizes_px(object_predictions, name="object_predictions")
        ]
        result = {"dimensions": dimensions}

        return result


def parse_reference_dimensions(dimensions: Any) -> Size:
    """Read the reference width and height.

    Args:
        dimensions: ``"width,height"`` text, or a tuple or list of two numbers.

    Returns:
        ``(width, height)`` as floats.

    Raises:
        ValueError: When the value has another form, or a number is not
            positive and finite.
    """
    if isinstance(dimensions, str):
        parts = dimensions.split(",")
        if len(parts) != 2:
            raise ValueError(
                "reference_dimensions must be a string in the format 'width,height'"
            )
        try:
            dimensions = [float(part.strip()) for part in parts]
        except ValueError:
            raise ValueError("Invalid format for reference_dimensions") from None

    if not isinstance(dimensions, (list, tuple)) or len(dimensions) != 2:
        raise ValueError("reference_dimensions must have two values (width, height)")

    width, height = (float(value) for value in dimensions)
    if not all(math.isfinite(value) and value > 0 for value in (width, height)):
        raise ValueError(
            f"reference_dimensions must be positive and finite, got {dimensions!r}"
        )

    return width, height


def _reference_size_px(reference: Any) -> Optional[Size]:
    if isinstance(reference, list):
        if len(reference) < 2:
            return None
        size = _measurable(_polygon_size(_reference_polygon(reference)))
        return size

    confidence = boxes_of_view(reference).confidence
    if len(confidence) == 0:
        return None

    best_row = int(torch.argmax(confidence).item())
    size = _sizes_px(reference, name="reference_predictions", rows=[best_row])[0]

    return size


def _reference_polygon(points: List[Any]) -> np.ndarray:
    # The [x, y] bound of the parameter covers literals only; a selector
    # value arrives unchecked.
    try:
        polygon = np.asarray(points, dtype=np.float32)
    except (TypeError, ValueError):
        polygon = None

    is_points = polygon is not None and polygon.ndim == 2 and polygon.shape[1] == 2
    if not (is_points and np.isfinite(polygon).all()):
        raise ValueError(
            "reference_predictions as a polygon must be a list of [x, y] points "
            f"with finite numbers, got {str(points)[:80]}"
        )

    return polygon


def _sizes_px(
    predictions: Any, *, name: str, rows: Optional[List[int]] = None
) -> List[Optional[Size]]:
    # Size of every row (or of the given rows) in pixels; None if unmeasurable.
    # ``name`` is the parameter the predictions came from, for errors.
    if isinstance(predictions, SegmentationView):
        view = predictions
        if rows is not None:
            # Rows come from argmax, so they are in range on any device.
            indices = torch.tensor(rows, device=predictions.scores.device)
            view = predictions.select(indices)
        reference = mask_provider_of(view).binary(fidelity="reference")
        sizes = _sizes_px(reference.predictions, name=name)
        return sizes

    if rows is None:
        rows = list(range(len(predictions.xyxy)))

    if not isinstance(predictions, InstanceDetections):
        # Widths and heights in the boxes' own dtype, as V1 computes them.
        boxes = predictions.xyxy[rows].detach()
        sides = torch.stack([boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]], 1)
        sizes = [_measurable(tuple(side)) for side in sides.to(device="cpu").tolist()]
        return sizes

    _check_mask_raster(predictions, name=name)
    windows = _mask_windows(predictions.mask, rows=rows)
    sizes = [_measurable(_mask_size(window)) for window in windows]

    return sizes


def _mask_windows(mask: Any, *, rows: List[int]) -> List[Optional[Window]]:
    # Host uint8 pixels of the smallest window holding each row's mask.
    if isinstance(mask, InstancesRLEMasks):
        windows = []
        for row in rows:
            pixels = decode_rle_masks(mask, rows=[row])[0]
            bounds = _bounds(pixels.any(axis=1), pixels.any(axis=0))
            windows.append(_cut(pixels, bounds=bounds))
        return windows

    # Callers pass every row in order, or one chosen row; copy only the latter.
    selected = mask if len(rows) == mask.shape[0] else mask[rows]
    any_in_row = selected.any(dim=2).to(device="cpu").numpy()
    any_in_column = selected.any(dim=1).to(device="cpu").numpy()
    windows = [
        _cut(pixels, bounds=_bounds(in_row, in_column))
        for pixels, in_row, in_column in zip(selected, any_in_row, any_in_column)
    ]

    return windows


def _bounds(
    in_row: np.ndarray, in_column: np.ndarray
) -> Optional[Tuple[int, int, int, int]]:
    # (x0, y0, x1, y1), exclusive ends, of the foreground; None when empty.
    ys, xs = np.flatnonzero(in_row), np.flatnonzero(in_column)
    if ys.size == 0:
        return None

    bounds = (int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1)

    return bounds


def _cut(pixels: Any, *, bounds: Optional[Tuple[int, ...]]) -> Optional[Window]:
    # Only the window crosses to the host for a device tensor.
    if bounds is None:
        return None

    x0, y0, x1, y1 = bounds
    window = pixels[y0:y1, x0:x1]
    if isinstance(window, torch.Tensor):
        window = window.to(device="cpu").numpy()

    cut = ((x0, y0), window.astype(np.uint8))

    return cut


def _mask_size(window: Optional[Window]) -> Optional[Size]:
    # V1: largest external contour by area. One zero pixel of margin keeps
    # edge pixels inside; contour points are moved back to image coordinates
    # because OpenCV fits the rectangle in float32, which rounds differently
    # for shifted points.
    if window is None:
        return None

    (x0, y0), pixels = window
    contours, _ = cv2.findContours(
        np.pad(pixels, 1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) <= 0:
        return None

    in_image = largest + np.array([x0 - 1, y0 - 1], dtype=largest.dtype)
    size = _aligned_rectangle_size(in_image)

    return size


def _polygon_size(polygon: np.ndarray) -> Optional[Size]:
    if len(polygon) < 3 or cv2.contourArea(polygon) <= 0:
        return None

    size = _aligned_rectangle_size(polygon)

    return size


def _aligned_rectangle_size(points: np.ndarray) -> Size:
    # Minimum-area rectangle; its more horizontal edge is the width. float32
    # angles as in V1, so near-45-degree ties resolve the same way.
    corners = cv2.boxPoints(cv2.minAreaRect(points)).astype(np.float32)
    edge_1 = corners[1] - corners[0]
    edge_2 = corners[2] - corners[1]
    length_1 = float(np.linalg.norm(edge_1))
    length_2 = float(np.linalg.norm(edge_2))
    if _tilt_from_horizontal(edge_1) < _tilt_from_horizontal(edge_2):
        size = (length_1, length_2)
    else:
        size = (length_2, length_1)

    return size


def _tilt_from_horizontal(edge: np.ndarray) -> float:
    angle = abs(np.degrees(np.arctan2(edge[1], edge[0])) % 180)
    tilt = min(angle, 180 - angle)

    return float(tilt)


def _measurable(size: Optional[Size]) -> Optional[Size]:
    if size is None or not (size[0] > 0 and size[1] > 0):
        return None

    return size


def _scaled(size: Optional[Size], *, scale: Size) -> Optional[Dict[str, float]]:
    if size is None:
        return None

    width = size[0] * scale[0]
    height = size[1] * scale[1]
    dimensions = {
        "width": width,
        "height": height,
        "longer": max(width, height),
        "shorter": min(width, height),
    }

    return dimensions


def _check_mask_raster(predictions: InstanceDetections, *, name: str) -> None:
    mask = predictions.mask
    if isinstance(mask, InstancesRLEMasks):
        mask_size = tuple(int(value) for value in mask.image_size)
    else:
        mask_size = tuple(mask.shape[1:])

    # Unknown when the prediction does not name its frame, e.g. hand-built
    # predictions without image metadata.
    try:
        frame_size = tuple(frame_of(predictions).size_hw)
    except ContractError:
        return

    if mask_size != frame_size:
        raise ValueError(
            f"{name} masks are {mask_size[0]}x{mask_size[1]} but the "
            f"prediction's image is {frame_size[0]}x{frame_size[1]}; sizes need "
            "masks at the image's resolution"
        )


def _check_same_pixel_size(objects: Any, reference: Any) -> None:
    object_scale = _root_scale(objects)
    reference_scale = _root_scale(reference)
    if object_scale is None or reference_scale is None:
        return

    if not all(math.isclose(a, b) for a, b in zip(object_scale, reference_scale)):
        raise ValueError(
            "object_predictions and reference_predictions describe images with "
            f"different pixel sizes (input pixels per pixel {object_scale} and "
            f"{reference_scale}); measure both on images of the same scale"
        )


def _root_scale(predictions: Any) -> Optional[Tuple[float, float]]:
    # Input pixels per prediction pixel; None when the prediction does not
    # say, or its root is a composite canvas.
    metadata = boxes_of_view(predictions).image_metadata or {}
    try:
        placement = root_placement_from_metadata(metadata)
    except ContractError:
        return None

    if placement is None:
        return None

    scale = tuple(placement.root.scale_xy)

    return scale
