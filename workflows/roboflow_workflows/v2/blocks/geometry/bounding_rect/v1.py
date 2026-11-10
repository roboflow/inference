"""Minimum-area rotated rectangle of each instance mask: ``roboflow_core/bounding_rect@v1``.

Port of V1 ``roboflow_core/bounding_rect@v1``. Per instance::

    contours = OpenCV findContours(mask, RETR_TREE, CHAIN_APPROX_SIMPLE)
    eligible = outer contours (parent -1) with >= 3 points
    rect     = minAreaRect(eligible contour with the most points)
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
and mask and gets ``None`` metadata, where V1 failed the whole step. Hole
contours cannot replace the outer outline: this deliberately corrects an
inherited V1 defect while retaining its most-points choice among outer contours.
"""

from dataclasses import replace
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
import torch
from pydantic import Field
from roboflow_workflows.v2.blocks.geometry._masks import (
    Window,
    mask_windows,
    polygon_masks,
)
from roboflow_workflows.v2.blocks.geometry._rows import without_host_mirrors
from roboflow_workflows.v2.blocks.geometry.mask_inputs import (
    INSTANCE_SEGMENTATION_VIEW_KIND,
    instance_predictions_of,
)
from roboflow_workflows.v2.data.predictions import INSTANCE_SEGMENTATION_PREDICTION_KIND
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref


RECT_KEYS = ("rect", "width", "height", "angle")

MIN_POLYGON_POINTS = 3

PARENT = 3

class BoundingRectBlock(Block):
    """Replace each instance's mask and box by its minimum-area rectangle.

    Example: an L-shaped mask gets the smallest rotated rectangle around its
    outline; its ``angle`` says how far that rectangle is turned.
    """

    type = "roboflow_core/bounding_rect@v1"
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
                "roboflow_core/bounding_rect@v1 needs instance masks; bind an instance "
                "segmentation prediction"
            )
        if len(predictions.xyxy) == 0:
            return {"predictions": predictions}

        windows = mask_windows(predictions.mask)
        rectangles = [_min_area_rectangle(window) for window in windows]
        corners = [None if r is None else r["corners"] for r in rectangles]
        rows = predictions.bboxes_metadata or [None] * len(windows)
        fitted = replace(
            predictions,
            xyxy=_rectangle_boxes(predictions.xyxy, rectangles=rectangles),
            mask=polygon_masks(predictions.mask, polygons=corners),
            bboxes_metadata=[
                {**without_host_mirrors(row), **_rect_metadata(rectangle)}
                for row, rectangle in zip(rows, rectangles)
            ],
        )
        result = {"predictions": fitted}

        return result

def _min_area_rectangle(window: Optional[Window]) -> Optional[Dict[str, Any]]:
    if window is None:
        return None

    (x0, y0), pixels = window
    # One zero pixel of margin: contours of a window see the same edges as
    # contours of the full mask.
    padded = np.pad(pixels, 1)
    contours, hierarchy = cv2.findContours(
        padded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    # A round hole can have more points than its straight outer outline.
    contours = [
        contour
        for contour, links in zip(contours, hierarchy[0] if contours else [])
        if links[PARENT] == -1 and contour.shape[0] >= MIN_POLYGON_POINTS
    ]
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
