"""Minimum-area rotated rectangle of each instance mask: ``v2/bounding_rect@v1``.

Port of V1 ``roboflow_core/bounding_rect@v1``. Per instance, as V1::

    contours = OpenCV findContours(mask, RETR_TREE, CHAIN_APPROX_SIMPLE), >= 3 points
    rect     = minAreaRect(contour with the most points)
    corners  = boxPoints(rect), truncated to int
    mask    <- filled corner polygon;  box <- min/max of the corners

Row metadata gains ``rect`` (four ``[x, y]`` integer corners), ``width``,
``height`` and ``angle`` (OpenCV's ``minAreaRect`` values). Rows stay one to
one, so they keep their row keys. A ``SegmentationView`` input gives its
reference masks first (``geometry.mask_inputs``); the result is
``InstanceDetections``.

CPU boundary: contours and rectangles are OpenCV host algorithms. Dense masks
stay on their device except the window around each instance: one small
transfer finds the windows, then only the window pixels go to the host and
the filled rectangles come back. RLE masks are host data; they are decoded and
re-encoded on the host.

Differences from V1: corners are exact integers (V1 stored them as float16,
which rounds values above 2048); an instance whose mask is empty keeps its box
and mask and gets ``None`` metadata, where V1 failed the whole step.
"""

from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils
from pydantic import Field
from roboflow_workflows.v2.blocks.geometry._rows import without_host_mirrors
from roboflow_workflows.v2.blocks.geometry.mask_inputs import (
    INSTANCE_SEGMENTATION_VIEW_KIND,
    instance_predictions_of,
)
from roboflow_workflows.v2.data.predictions import INSTANCE_SEGMENTATION_PREDICTION_KIND
from roboflow_workflows.v2.data.rle import decode_rle_masks
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref

from inference_models.models.base.types import InstancesRLEMasks

RECT_KEYS = ("rect", "width", "height", "angle")
MIN_POLYGON_POINTS = 3

# (x0, y0) of a host mask window and the window itself.
Window = Tuple[Tuple[int, int], np.ndarray]


class BoundingRectBlock(Block):
    """Replace each instance's mask and box by its minimum-area rectangle.

    Example: an L-shaped mask gets the smallest rotated rectangle around its
    outline; its ``angle`` says how far that rectangle is turned.
    """

    type = "v2/bounding_rect@v1"
    outputs = {
        "predictions": Output(
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
            source="predictions",
            description="Same rows; rectangle masks, boxes and rect metadata.",
        ),
    }

    class Params(BlockParams):
        predictions: Ref(
            INSTANCE_SEGMENTATION_PREDICTION_KIND, INSTANCE_SEGMENTATION_VIEW_KIND
        ) = Field(description="Instance segmentation prediction with masks.")

    def run(self, *, predictions: Any) -> Dict[str, Any]:
        """Fit a rotated rectangle to every instance mask.

        Args:
            predictions: ``InstanceDetections`` or a ``SegmentationView``.

        Returns:
            ``predictions``: same rows with rectangle masks, boxes and metadata.

        Raises:
            ValueError: When the prediction has no masks.
        """
        predictions = instance_predictions_of(predictions)
        if predictions.mask is None:
            raise ValueError(
                "v2/bounding_rect@v1 needs instance masks; bind an instance "
                "segmentation prediction"
            )
        if len(predictions.xyxy) == 0:
            return {"predictions": predictions}

        windows = _mask_windows(predictions.mask)
        rectangles = [_min_area_rectangle(window) for window in windows]
        rows = predictions.bboxes_metadata or [None] * len(windows)
        fitted = replace(
            predictions,
            xyxy=_rectangle_boxes(predictions.xyxy, rectangles=rectangles),
            mask=_rectangle_masks(predictions.mask, rectangles=rectangles),
            bboxes_metadata=[
                {**without_host_mirrors(row), **_rect_metadata(rectangle)}
                for row, rectangle in zip(rows, rectangles)
            ],
        )
        result = {"predictions": fitted}

        return result


def _mask_windows(mask: Any) -> List[Optional[Window]]:
    # The smallest window holding each instance, on the host; None when empty.
    if isinstance(mask, InstancesRLEMasks):
        host_masks = decode_rle_masks(mask)
        any_in_row = host_masks.any(axis=2)
        any_in_column = host_masks.any(axis=1)
    else:
        host_masks = None
        any_in_row = mask.any(dim=2).cpu().numpy()
        any_in_column = mask.any(dim=1).cpu().numpy()

    windows: List[Optional[Window]] = []
    for instance, (rows, columns) in enumerate(zip(any_in_row, any_in_column)):
        ys, xs = np.flatnonzero(rows), np.flatnonzero(columns)
        if ys.size == 0:
            windows.append(None)
            continue

        y0, y1, x0, x1 = int(ys[0]), int(ys[-1]) + 1, int(xs[0]), int(xs[-1]) + 1
        if host_masks is None:
            pixels = mask[instance, y0:y1, x0:x1].cpu().numpy()
        else:
            pixels = host_masks[instance, y0:y1, x0:x1]
        windows.append(((x0, y0), pixels.astype(np.uint8)))

    return windows


def _min_area_rectangle(window: Optional[Window]) -> Optional[Dict[str, Any]]:
    if window is None:
        return None

    (x0, y0), pixels = window
    # One zero pixel of margin: contours of a window see the same edges as
    # contours of the full mask.
    padded = np.pad(pixels, 1)
    contours, _ = cv2.findContours(padded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if c.shape[0] >= MIN_POLYGON_POINTS]
    if not contours:
        return None

    largest = max(contours, key=len).reshape(-1, 2) + np.array([x0 - 1, y0 - 1])
    center, (width, height), angle = cv2.minAreaRect(largest.astype(np.int32))
    corners = cv2.boxPoints((center, (width, height), angle)).astype(np.int64)
    rectangle = {
        "corners": corners,
        "width": float(width),
        "height": float(height),
        "angle": float(angle),
    }

    return rectangle


def _rect_metadata(rectangle: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if rectangle is None:
        return dict.fromkeys(RECT_KEYS)

    metadata = {
        "rect": rectangle["corners"].tolist(),
        "width": rectangle["width"],
        "height": rectangle["height"],
        "angle": rectangle["angle"],
    }

    return metadata


def _rectangle_boxes(
    xyxy: torch.Tensor, *, rectangles: List[Optional[Dict[str, Any]]]
) -> torch.Tensor:
    boxes = xyxy.clone()
    for row, rectangle in enumerate(rectangles):
        if rectangle is None:
            continue
        corners = rectangle["corners"]
        box = [*corners.min(axis=0).tolist(), *corners.max(axis=0).tolist()]
        boxes[row] = torch.tensor(box, dtype=boxes.dtype)

    return boxes


def _rectangle_masks(mask: Any, *, rectangles: List[Optional[Dict[str, Any]]]) -> Any:
    if isinstance(mask, InstancesRLEMasks):
        return _rectangle_rle_masks(mask, rectangles=rectangles)

    height, width = int(mask.shape[1]), int(mask.shape[2])
    filled = mask.clone()
    for row, rectangle in enumerate(rectangles):
        if rectangle is None:
            continue
        (x0, y0), window = _filled_window(rectangle["corners"], size_hw=(height, width))
        filled[row].zero_()
        filled[row, y0 : y0 + window.shape[0], x0 : x0 + window.shape[1]] = (
            torch.from_numpy(window).to(device=mask.device, dtype=mask.dtype)
        )

    return filled


def _rectangle_rle_masks(
    mask: InstancesRLEMasks, *, rectangles: List[Optional[Dict[str, Any]]]
) -> InstancesRLEMasks:
    height, width = mask.image_size
    encoded = []
    for row, rectangle in enumerate(rectangles):
        if rectangle is None:
            encoded.append(mask.masks[row])
            continue
        canvas = np.zeros((height, width), dtype=np.uint8)
        (x0, y0), window = _filled_window(rectangle["corners"], size_hw=(height, width))
        canvas[y0 : y0 + window.shape[0], x0 : x0 + window.shape[1]] = window
        encoded.append(mask_utils.encode(np.asfortranarray(canvas))["counts"])

    converted = InstancesRLEMasks(image_size=mask.image_size, masks=encoded)

    return converted


def _filled_window(corners: np.ndarray, *, size_hw: Tuple[int, int]) -> Window:
    # Fill the corner polygon inside its own clipped bounding window; filling a
    # shifted polygon on a window equals filling it on the full canvas.
    height, width = size_hw
    x0, y0 = (int(v) for v in np.clip(corners.min(axis=0), 0, [width, height]))
    x1, y1 = (int(v) for v in np.clip(corners.max(axis=0) + 1, 0, [width, height]))
    window = np.zeros((max(y1 - y0, 0), max(x1 - x0, 0)), dtype=np.uint8)
    if window.size:
        shifted = (corners - np.array([x0, y0])).astype(np.int32)
        cv2.fillPoly(window, [shifted], color=(1,))

    return (x0, y0), window
