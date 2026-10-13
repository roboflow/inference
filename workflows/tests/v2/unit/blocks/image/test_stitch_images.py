"""Unit tests of ``v2/stitch_images@v1``, called directly.

Inputs are two overlapping crops of one generated, textured 160 x 320 scene:
``left`` covers columns 0-199 and ``right`` columns 120-319. Which crop is
the anchor depends on the keypoints, so each case states the seed it uses:
seed 6 anchors ``right`` (the moving crop lands left of it), seed 8 anchors
``left`` (the moving crop lands right of it, the case V1 cropped). Warped
corners truncate to integers as in V1, so other seeds may place the anchor
one pixel off; pixel checks therefore align the scene through the anchor tile.
"""

from typing import Tuple

import cv2 as cv
import numpy as np
import pytest
import torch
from roboflow_workflows.v2.blocks.image import stitch_images
from roboflow_workflows.v2.blocks.image.stitch_images import (
    DEFAULT_MAX_CANVAS_PIXELS,
    MAX_CANVAS_SIDE,
    MAX_WARP_SOURCE_SIDE,
    StitchImagesBlock,
)
from roboflow_workflows.v2.data.image import ImageData

SCENE_HW = (160, 320)
LEFT_COLUMNS = (0, 200)
RIGHT_COLUMNS = (120, 320)
RIGHT_ANCHORED_SEED = 6
LEFT_ANCHORED_SEED = 8


def scene(seed: int) -> torch.Tensor:
    # (3, H, W) uint8: smooth noise plus filled circles, rich in SIFT corners.
    height, width = SCENE_HW
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 256, size=(height // 4, width // 4, 3), dtype=np.uint8)
    pixels = cv.resize(noise, (width, height), interpolation=cv.INTER_CUBIC)
    for _ in range(25):
        x, y = int(rng.integers(0, width - 10)), int(rng.integers(0, height - 10))
        color = tuple(int(value) for value in rng.integers(0, 256, size=3))
        cv.circle(pixels, (x, y), int(rng.integers(3, 12)), color, -1)
    tensor = torch.from_numpy(pixels).permute(2, 0, 1).contiguous()

    return tensor


def crops(seed: int) -> Tuple[torch.Tensor, ImageData, ImageData]:
    pixels = scene(seed)
    left = ImageData.from_tensor(pixels[:, :, slice(*LEFT_COLUMNS)].contiguous())
    right = ImageData.from_tensor(pixels[:, :, slice(*RIGHT_COLUMNS)].contiguous())

    return pixels, left, right


def stitch(image1: ImageData, image2: ImageData, **overrides) -> ImageData:
    params = {
        "max_allowed_reprojection_error": 3.0,
        "count_of_best_matches_per_query_descriptor": 2,
        "max_canvas_pixels": DEFAULT_MAX_CANVAS_PIXELS,
        **overrides,
    }
    stitched = StitchImagesBlock().run(image1=image1, image2=image2, **params)

    return stitched["stitched_image"]


def mean_scene_difference(
    stitched: ImageData, pixels: torch.Tensor, *, anchor_columns: Tuple[int, int]
) -> float:
    # Mean absolute difference over the canvas columns that show the scene,
    # aligned so that the anchor tile sits on its own scene columns.
    (tile,) = stitched.composite_sources
    shift = anchor_columns[0] - tile.canvas_xyxy[0]
    first = max(0, -shift)
    last = min(stitched.width, SCENE_HW[1] - shift)
    canvas = stitched.tensor_image[:, :, first:last].int()
    expected = pixels[:, :, first + shift : last + shift].int()
    difference = (canvas - expected).abs().float().mean().item()

    return difference


def test_anchor_tile_is_pasted_exactly_and_mapped_back() -> None:
    # given: the right crop, image2, anchors
    _, left, right = crops(RIGHT_ANCHORED_SEED)

    # when
    stitched = stitch(left, right)

    # then: one source, the anchor, at its exact rectangle with scale 1
    assert stitched.is_composite
    (tile,) = stitched.composite_sources
    assert tile.index == (1,)
    assert tile.image_id == right.image_id
    x0, y0, x1, y1 = tile.canvas_xyxy
    assert (x1 - x0, y1 - y0) == (right.width, right.height)
    assert torch.equal(stitched.tensor_image[:, y0:y1, x0:x1], right.tensor_image)
    assert tile.parent.map_xy(x0 + 5, y0 + 7) == (5.0, 7.0)


def test_canvas_has_its_own_stitched_id() -> None:
    # given
    _, left, right = crops(RIGHT_ANCHORED_SEED)

    # when
    first = stitch(left, right)
    second = stitch(left, right)

    # then: errors and lineage name a stitched image, not a generic mosaic;
    # every canvas is a new identity
    assert first.image_id.startswith("stitched-")
    assert first.image_id != second.image_id
    assert first.image_id not in (left.image_id, right.image_id)
    assert first.root.frame_id == first.image_id


def test_canvas_reproduces_the_scene() -> None:
    # given
    pixels, left, right = crops(RIGHT_ANCHORED_SEED)

    # when
    stitched = stitch(left, right)

    # then: canvas covers the scene; warped pixels agree up to resampling
    assert stitched.size_hw == SCENE_HW
    assert mean_scene_difference(stitched, pixels, anchor_columns=RIGHT_COLUMNS) < 3.0


def test_moving_image_right_of_anchor_keeps_full_width() -> None:
    # given: the left crop anchors at the canvas origin, so the moving crop
    # warps to positive x. V1 shrank the canvas here and lost columns.
    pixels, left, right = crops(LEFT_ANCHORED_SEED)

    # when
    stitched = stitch(left, right)

    # then
    (tile,) = stitched.composite_sources
    assert tile.index == (0,)
    assert tile.canvas_xyxy == (0, 0, left.width, left.height)
    assert stitched.size_hw == SCENE_HW
    assert mean_scene_difference(stitched, pixels, anchor_columns=LEFT_COLUMNS) < 3.0


def test_image_order_keeps_the_anchor_crop() -> None:
    # given
    pixels, left, right = crops(RIGHT_ANCHORED_SEED)

    # when
    forward = stitch(left, right)
    backward = stitch(right, left)

    # then: the same crop anchors and its index follows its parameter. Swapped
    # matching roles fit a slightly different homography, so the canvases
    # are not bit-identical.
    assert forward.composite_sources[0].image_id == right.image_id
    assert backward.composite_sources[0].image_id == right.image_id
    assert forward.composite_sources[0].index == (1,)
    assert backward.composite_sources[0].index == (0,)
    assert mean_scene_difference(backward, pixels, anchor_columns=RIGHT_COLUMNS) < 3.0


def test_untextured_images_give_none() -> None:
    # given: flat images have no SIFT keypoints
    flat = ImageData.from_tensor(torch.full((3, 64, 64), 128, dtype=torch.uint8))

    # when
    stitched = stitch(flat, flat)

    # then
    assert stitched is None


def test_unrelated_images_give_none() -> None:
    # given: textures from different seeds share no matching keypoints
    _, left, _ = crops(RIGHT_ANCHORED_SEED)
    _, _, unrelated = crops(LEFT_ANCHORED_SEED)

    # when
    stitched = stitch(left, unrelated)

    # then
    assert stitched is None


def test_grayscale_and_rgb_give_rgb_canvas() -> None:
    # given
    _, left, right = crops(RIGHT_ANCHORED_SEED)
    rgb = right.tensor_image.permute(1, 2, 0).numpy()
    gray = torch.from_numpy(cv.cvtColor(rgb, cv.COLOR_RGB2GRAY))[None]
    gray_right = ImageData.from_tensor(gray)

    # when
    stitched = stitch(left, gray_right)

    # then: the gray anchor is repeated into all three channels
    assert stitched.channels == 3
    (tile,) = stitched.composite_sources
    assert tile.index == (1,)
    x0, y0, x1, y1 = tile.canvas_xyxy
    for channel in range(3):
        assert torch.equal(stitched.tensor_image[channel, y0:y1, x0:x1], gray[0])


def test_noncontiguous_inputs_match_contiguous_inputs() -> None:
    # given: the same crops as strided views of the scene
    pixels, left, right = crops(RIGHT_ANCHORED_SEED)
    left_view = ImageData.from_tensor(pixels[:, :, slice(*LEFT_COLUMNS)])
    right_view = ImageData.from_tensor(pixels[:, :, slice(*RIGHT_COLUMNS)])
    assert not left_view.tensor_image.is_contiguous()

    # when
    from_views = stitch(left_view, right_view)
    from_copies = stitch(left, right)

    # then
    assert torch.equal(from_views.tensor_image, from_copies.tensor_image)


def test_inputs_are_not_modified() -> None:
    # given
    _, left, right = crops(RIGHT_ANCHORED_SEED)
    left_before = left.tensor_image.clone()
    right_before = right.tensor_image.clone()

    # when
    stitch(left, right)

    # then
    assert torch.equal(left.tensor_image, left_before)
    assert torch.equal(right.tensor_image, right_before)


def test_images_on_different_devices_raise() -> None:
    # given: a meta tensor has a device but no data; the check runs first
    _, left, _ = crops(RIGHT_ANCHORED_SEED)
    elsewhere = ImageData.from_tensor(
        torch.zeros((3, 8, 8), dtype=torch.uint8, device="meta")
    )

    # when / then
    with pytest.raises(ValueError, match="both images on one device"):
        stitch(left, elsewhere)


def test_canvas_above_max_canvas_pixels_raises_before_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # given: a limit one pixel below the 160 x 320 canvas; warping must not
    # start, so the error comes before any canvas allocation
    _, left, right = crops(RIGHT_ANCHORED_SEED)

    def refuse_warp(*args, **kwargs):
        raise AssertionError("canvas allocated before the size check")

    monkeypatch.setattr(stitch_images.cv, "warpPerspective", refuse_warp)

    # when / then
    with pytest.raises(ValueError, match="320x160 = 51200 pixels.*=51199"):
        stitch(left, right, max_canvas_pixels=51_199)


def test_canvas_at_max_canvas_pixels_is_allowed() -> None:
    # given
    _, left, right = crops(RIGHT_ANCHORED_SEED)

    # when
    stitched = stitch(left, right, max_canvas_pixels=320 * 160)

    # then
    assert stitched.size_hw == SCENE_HW


def test_default_max_canvas_pixels_is_4096_squared() -> None:
    # then: about 48 MiB for an RGB uint8 canvas
    assert DEFAULT_MAX_CANVAS_PIXELS == 4096 * 4096
    params = StitchImagesBlock.Params.model_fields["max_canvas_pixels"]
    assert params.default == DEFAULT_MAX_CANVAS_PIXELS


@pytest.mark.parametrize("homography_scale_xy", [(200.0, 1.0), (1.0, 200.0)])
def test_canvas_side_above_opencv_limit_raises_regardless_of_area(
    homography_scale_xy: Tuple[float, float],
) -> None:
    # given: a 200x stretch of a 200-pixel image along one axis
    moving = np.zeros((200, 200, 3), dtype=np.uint8)
    anchor = np.zeros((10, 10, 3), dtype=np.uint8)
    homography = np.diag([*homography_scale_xy, 1.0])

    # when / then: the side limit holds even with no area limit
    with pytest.raises(ValueError, match=f"side above {MAX_CANVAS_SIDE}"):
        stitch_images._place(
            moving, anchor, homography=homography, max_canvas_pixels=10**12
        )


@pytest.mark.parametrize(
    "moving_hw, homography_diagonal",
    [
        ((2, MAX_WARP_SOURCE_SIDE + 1), (2 / (MAX_WARP_SOURCE_SIDE + 1), 1.0)),
        ((MAX_WARP_SOURCE_SIDE + 1, 2), (1.0, 2 / (MAX_WARP_SOURCE_SIDE + 1))),
    ],
)
def test_moving_side_above_opencv_source_limit_raises_before_warping(
    moving_hw: Tuple[int, int],
    homography_diagonal: Tuple[float, float],
) -> None:
    # given: the warp shrinks the long side to a 2 x 2 canvas, well inside the
    # canvas limits; OpenCV still asserts on the source side
    moving = np.zeros((*moving_hw, 3), dtype=np.uint8)
    anchor = np.zeros((2, 2, 3), dtype=np.uint8)
    homography = np.diag([*homography_diagonal, 1.0])

    # when / then
    with pytest.raises(ValueError, match=f"up to {MAX_WARP_SOURCE_SIDE} pixels"):
        stitch_images._place(
            moving, anchor, homography=homography, max_canvas_pixels=16
        )


@pytest.mark.parametrize(
    "moving_hw, homography_diagonal",
    [
        ((2, MAX_WARP_SOURCE_SIDE), (2 / MAX_WARP_SOURCE_SIDE, 1.0)),
        ((MAX_WARP_SOURCE_SIDE, 2), (1.0, 2 / MAX_WARP_SOURCE_SIDE)),
    ],
)
def test_moving_side_at_opencv_source_limit_is_warped(
    moving_hw: Tuple[int, int],
    homography_diagonal: Tuple[float, float],
) -> None:
    # given: the largest source side OpenCV accepts
    moving = np.full((*moving_hw, 3), 200, dtype=np.uint8)
    anchor = np.full((2, 2, 3), 50, dtype=np.uint8)
    homography = np.diag([*homography_diagonal, 1.0])

    # when
    canvas, anchor_xy = stitch_images._place(
        moving, anchor, homography=homography, max_canvas_pixels=16
    )

    # then: the anchor covers the whole 2 x 2 canvas
    assert anchor_xy == (0, 0)
    assert canvas.shape == (2, 2, 3)
    assert (canvas == 50).all()


def test_translation_right_of_anchor_sizes_canvas_to_cover_both() -> None:
    # given: the moving image shifted 50 pixels right of the anchor origin
    moving = np.full((10, 20, 3), 200, dtype=np.uint8)
    anchor = np.full((10, 30, 3), 50, dtype=np.uint8)
    homography = np.array([[1.0, 0.0, 50.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    # when
    canvas, anchor_xy = stitch_images._place(
        moving, anchor, homography=homography, max_canvas_pixels=10_000
    )

    # then: anchor at the origin, a black gap, the moving image at x 50-69
    assert anchor_xy == (0, 0)
    assert canvas.shape == (10, 70, 3)
    assert (canvas[:, :30] == 50).all()
    assert (canvas[:, 30:50] == 0).all()
    assert (canvas[:, 50:] == 200).all()


def test_translation_left_of_anchor_offsets_the_anchor() -> None:
    # given: the moving image ends where the anchor starts
    moving = np.full((10, 20, 3), 200, dtype=np.uint8)
    anchor = np.full((10, 30, 3), 50, dtype=np.uint8)
    homography = np.array([[1.0, 0.0, -20.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    # when
    canvas, anchor_xy = stitch_images._place(
        moving, anchor, homography=homography, max_canvas_pixels=10_000
    )

    # then
    assert anchor_xy == (20, 0)
    assert canvas.shape == (10, 50, 3)
    assert (canvas[:, :20] == 200).all()
    assert (canvas[:, 20:] == 50).all()
