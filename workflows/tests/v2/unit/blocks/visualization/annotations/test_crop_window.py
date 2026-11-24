"""Crop visualization of parent boxes that cross or miss a child image.

A parent prediction drawn on a crop of the parent is mapped into the crop and
may get negative coordinates. Only the part inside the crop is copied, and
the copy keeps the anchor of the whole mapped box::

    parent 64x48 ─ crop (16, 12, 48, 40) ─▶ child 32x28
    parent box (6, 18, 26, 30) ─▶ child box (-10, 6, 10, 18)
                                  copied:   (0, 6, 10, 18)
    BOTTOM_RIGHT anchor (10, 18): copy pasted at (10, 18)..(20, 30)

Boxes entirely outside the child draw nothing; Python slicing must not wrap
negative bounds to the opposite edge. Expected pixels are built here from the
child's pixels and ``cv2.rectangle``, independent of the block.
"""

from typing import Tuple

import cv2
import numpy as np
import pytest
import torch
from roboflow_workflows.v2.blocks.visualization.annotations import (
    CropVisualizationBlock,
)
from roboflow_workflows.v2.data.image import ImageData

from .support import detections, image_of, pixels_of, run_v2, scene

CHILD_XYXY = (16, 12, 48, 40)
BORDER = (255, 0, 255)
STYLE = {
    "color_palette": "CUSTOM",
    "custom_colors": ["#FF00FF"],
    "border_thickness": 1,
}
# Parent box -> (box mapped into the child, its part inside the child).
CROSSING = {
    "left": ((6, 18, 26, 30), (-10, 6, 10, 18), (0, 6, 10, 18)),
    "top": ((24, 2, 36, 20), (8, -10, 20, 8), (8, 0, 20, 8)),
    "top_left": ((8, 4, 24, 20), (-8, -8, 8, 8), (0, 0, 8, 8)),
}
# Parent boxes whose mapped box lies entirely left of, above, or both.
OUTSIDE = {
    "left": (0, 16, 12, 36),
    "top": (20, 0, 40, 10),
    "top_left": (0, 0, 10, 8),
}


@pytest.fixture
def parent() -> ImageData:
    image = image_of(scene(), image_id="parent")

    return image


@pytest.fixture
def child(parent: ImageData) -> ImageData:
    image = parent.crop(CHILD_XYXY, image_id="child")

    return image


def parent_row(parent: ImageData, box: Tuple[int, int, int, int]):
    predictions = detections(parent, [(*map(float, box), 0, 0.9)])

    return predictions


def pasted(
    child: ImageData,
    source: Tuple[int, int, int, int],
    *,
    at: Tuple[int, int],
) -> np.ndarray:
    """The child with ``source`` pasted at ``at`` (clipped) and its border."""
    pixels = pixels_of(child).copy()
    height, width = pixels.shape[:2]
    sx1, sy1, sx2, sy2 = source
    copy = pixels_of(child)[sy1:sy2, sx1:sx2]
    x, y = at
    for row in range(copy.shape[0]):
        for column in range(copy.shape[1]):
            if 0 <= y + row < height and 0 <= x + column < width:
                pixels[y + row, x + column] = copy[row, column]
    corner = (x + copy.shape[1], y + copy.shape[0])
    cv2.rectangle(pixels, (x, y), corner, BORDER, 1)

    return pixels


@pytest.mark.parametrize("side", sorted(CROSSING))
def test_a_box_crossing_the_left_or_top_edge_copies_its_visible_part(
    parent: ImageData, child: ImageData, side: str
) -> None:
    box, mapped, visible = CROSSING[side]
    parent_before = pixels_of(parent).copy()
    child_before = pixels_of(child).copy()
    predictions = parent_row(parent, box)
    boxes_before = predictions.xyxy.clone()

    output = run_v2(
        CropVisualizationBlock,
        image=child,
        predictions=predictions,
        position="BOTTOM_RIGHT",
        scale_factor=1.0,
        **STYLE,
    )

    # The BOTTOM_RIGHT anchor is the mapped box's corner (x2, y2).
    expected = pasted(child, visible, at=(mapped[2], mapped[3]))
    np.testing.assert_array_equal(pixels_of(output), expected)
    # Source pixels are the parent's pixels under the visible part.
    x1, y1, x2, y2 = visible
    dx, dy = CHILD_XYXY[:2]
    np.testing.assert_array_equal(
        child_before[y1:y2, x1:x2], parent_before[y1 + dy : y2 + dy, x1 + dx : x2 + dx]
    )
    np.testing.assert_array_equal(pixels_of(child), child_before)
    np.testing.assert_array_equal(pixels_of(parent), parent_before)
    assert torch.equal(predictions.xyxy, boxes_before)


def test_the_copy_keeps_the_anchor_of_the_whole_mapped_box(
    parent: ImageData, child: ImageData
) -> None:
    # Mapped (-8, -8, 8, 8): its centre (0, 0) anchors the 8x8 visible part,
    # so the copy spans (-4, -4)..(4, 4). The visible part's own centre
    # (4, 4) would place it at (0, 0)..(8, 8) instead.
    box, _, visible = CROSSING["top_left"]

    output = run_v2(
        CropVisualizationBlock,
        image=child,
        predictions=parent_row(parent, box),
        position="CENTER",
        scale_factor=1.0,
        **STYLE,
    )

    expected = pasted(child, visible, at=(-4, -4))
    np.testing.assert_array_equal(pixels_of(output), expected)


def test_a_scaled_copy_is_sized_from_the_visible_part(
    parent: ImageData, child: ImageData
) -> None:
    # Visible 10x12 at scale 2 -> 20x24, pasted at the anchor (10, 18).
    box, mapped, _ = CROSSING["left"]

    output = run_v2(
        CropVisualizationBlock,
        image=child,
        predictions=parent_row(parent, box),
        position="BOTTOM_RIGHT",
        scale_factor=2.0,
        **STYLE,
    )

    changed = (pixels_of(output) != pixels_of(child)).any(axis=2)
    ys, xs = np.nonzero(changed)
    height, width = child.size_hw
    assert (xs.min(), ys.min()) == (mapped[2], mapped[3])
    assert (xs.max() + 1, ys.max() + 1) == (min(10 + 20 + 1, width), height)
    border_row = pixels_of(output)[mapped[3], mapped[2] : mapped[2] + 21]
    assert (border_row == BORDER).all()


@pytest.mark.parametrize("side", sorted(OUTSIDE))
# Both anchors put a wrapped copy inside the child, so a wrap would show.
@pytest.mark.parametrize("position", ["BOTTOM_RIGHT", "CENTER"])
def test_a_box_entirely_outside_the_left_or_top_edge_draws_nothing(
    parent: ImageData, child: ImageData, side: str, position: str
) -> None:
    child_before = pixels_of(child).copy()

    output = run_v2(
        CropVisualizationBlock,
        image=child,
        predictions=parent_row(parent, OUTSIDE[side]),
        position=position,
        scale_factor=3.0,
        **STYLE,
    )

    np.testing.assert_array_equal(pixels_of(output), child_before)
    assert output.tensor_image is not child.tensor_image


def test_outside_rows_add_nothing_to_a_crossing_row(
    parent: ImageData, child: ImageData
) -> None:
    # One call with the crossing row and all three outside rows.
    box, mapped, visible = CROSSING["left"]
    rows = [box, *OUTSIDE.values()]
    predictions = detections(parent, [(*map(float, row), 0, 0.9) for row in rows])

    output = run_v2(
        CropVisualizationBlock,
        image=child,
        predictions=predictions,
        position="BOTTOM_RIGHT",
        scale_factor=1.0,
        **STYLE,
    )

    expected = pasted(child, visible, at=(mapped[2], mapped[3]))
    np.testing.assert_array_equal(pixels_of(output), expected)
