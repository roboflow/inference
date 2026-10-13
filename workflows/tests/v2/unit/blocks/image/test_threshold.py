"""Unit tests of ``v2/threshold@v1``, called directly.

Expected values are worked out by hand from OpenCV's documented rules, come
from an independent Otsu search written here, or from the OpenCV call V1
made on the same image. All comparisons are exact.
"""

from datetime import datetime, timezone

import cv2
import numpy as np
import pytest
import torch
from roboflow_workflows.execution_engine.entities.base import VideoMetadata
from roboflow_workflows.v2.blocks.image.threshold import (
    ImageThresholdBlock,
    otsu_threshold,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.row_identity import RowRef

VIDEO = VideoMetadata(
    video_identifier="cam-1",
    frame_number=7,
    frame_timestamp=datetime(2026, 10, 12, tzinfo=timezone.utc),
)
RAMP = torch.tensor([0, 126, 127, 128, 255], dtype=torch.uint8).reshape(1, 1, 5)
OPENCV_FIXED_TYPES = {
    "binary": cv2.THRESH_BINARY,
    "binary_inv": cv2.THRESH_BINARY_INV,
    "trunc": cv2.THRESH_TRUNC,
    "tozero": cv2.THRESH_TOZERO,
    "tozero_inv": cv2.THRESH_TOZERO_INV,
}


def run_threshold(
    pixels: torch.Tensor, threshold_type: str, thresh_value: int = 127, max_value=255
) -> torch.Tensor:
    result = ImageThresholdBlock().run(
        image=ImageData.from_tensor(pixels),
        threshold_type=threshold_type,
        thresh_value=thresh_value,
        max_value=max_value,
    )

    return result["image"].tensor_image


def opencv_threshold(
    pixels: torch.Tensor, threshold_type: str, thresh_value: int, max_value: int
) -> np.ndarray:
    # V1's calls; per-channel rules give the same result in RGB or BGR order.
    array = np.ascontiguousarray(pixels.permute(1, 2, 0).numpy())
    if array.shape[2] == 1:
        array = array[:, :, 0]
    if threshold_type in OPENCV_FIXED_TYPES:
        flag = OPENCV_FIXED_TYPES[threshold_type]
        _, result = cv2.threshold(array, thresh_value, max_value, flag)
    elif threshold_type == "otsu":
        flags = cv2.THRESH_BINARY + cv2.THRESH_OTSU
        _, result = cv2.threshold(array, 0, max_value, flags)
    else:
        method = (
            cv2.ADAPTIVE_THRESH_MEAN_C
            if threshold_type == "adaptive_mean"
            else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        )
        result = cv2.adaptiveThreshold(
            array, max_value, method, cv2.THRESH_BINARY, 11, 2
        )
    expected = result.reshape(pixels.shape[1], pixels.shape[2], -1).transpose(2, 0, 1)

    return expected


def random_pixels(channels: int, height: int, width: int, seed: int) -> torch.Tensor:
    generator = np.random.default_rng(seed)
    pixels = generator.integers(0, 256, (channels, height, width), dtype=np.uint8)
    tensor = torch.from_numpy(pixels)

    return tensor


@pytest.mark.parametrize(
    "threshold_type, expected",
    [
        ("binary", [0, 0, 0, 200, 200]),
        ("binary_inv", [200, 200, 200, 0, 0]),
        ("trunc", [0, 126, 127, 127, 127]),
        ("tozero", [0, 0, 0, 128, 255]),
        ("tozero_inv", [0, 126, 127, 0, 0]),
    ],
)
def test_fixed_types_compare_strictly_above_the_threshold(
    threshold_type: str, expected: list
) -> None:
    thresholded = run_threshold(RAMP, threshold_type, 127, 200)

    assert thresholded.flatten().tolist() == expected


@pytest.mark.parametrize("max_value, written", [(300, 255), (-5, 0), (128, 128)])
def test_max_value_is_clamped_to_the_pixel_range(max_value: int, written: int) -> None:
    thresholded = run_threshold(RAMP, "binary", 127, max_value)

    assert thresholded.flatten().tolist() == [0, 0, 0, written, written]


@pytest.mark.parametrize(
    "threshold_type, thresh_value, expected",
    [
        ("binary", -1, [9] * 5),
        ("binary", 255, [0] * 5),
        ("binary_inv", 255, [9] * 5),
        ("trunc", -1, [0] * 5),
        ("tozero", 255, [0] * 5),
        ("tozero", -1, [0, 126, 127, 128, 255]),
        ("tozero_inv", -1, [0] * 5),
        ("tozero_inv", 300, [0, 126, 127, 128, 255]),
    ],
)
def test_thresholds_outside_the_pixel_range_follow_opencv(
    threshold_type: str, thresh_value: int, expected: list
) -> None:
    thresholded = run_threshold(RAMP, threshold_type, thresh_value, 9)

    assert thresholded.flatten().tolist() == expected


def test_otsu_separates_two_populations() -> None:
    # given: 50s on the left, 200s on the right; thresh_value is ignored
    pixels = torch.full((1, 4, 8), 50, dtype=torch.uint8)
    pixels[:, :, 4:] = 200

    # when
    thresholded = run_threshold(pixels, "otsu", thresh_value=250, max_value=255)

    # then
    assert thresholded[:, :, :4].unique().tolist() == [0]
    assert thresholded[:, :, 4:].unique().tolist() == [255]


def brute_force_otsu(counts: list) -> int:
    # Independent: the first threshold t maximising w0 * w1 * (m0 - m1)^2,
    # with each class mean computed directly.
    total = sum(counts)
    best, best_value = -1.0, 0
    for t in range(255):
        lower = counts[: t + 1]
        upper = counts[t + 1 :]
        w0, w1 = sum(lower) / total, sum(upper) / total
        if w0 == 0 or w1 == 0:
            continue
        m0 = sum(i * c for i, c in enumerate(lower)) / sum(lower)
        m1 = sum((i + t + 1) * c for i, c in enumerate(upper)) / sum(upper)
        spread = w0 * w1 * (m0 - m1) ** 2
        if spread > best * (1 + 1e-12):
            best, best_value = spread, t

    return best_value


@pytest.mark.parametrize("seed", range(5))
def test_otsu_threshold_agrees_with_a_direct_search(seed: int) -> None:
    generator = np.random.default_rng(seed)
    values = np.concatenate(
        [generator.normal(70, 15, 500), generator.normal(180, 20, 300)]
    )
    counts = np.bincount(np.clip(values, 0, 255).astype(int), minlength=256)

    assert otsu_threshold(counts.tolist()) == brute_force_otsu(counts.tolist())


def test_adaptive_mean_marks_a_dark_dot_in_a_bright_field() -> None:
    # given: 200 everywhere except one 0. Around the dot the 11 x 11 mean is
    # about 198, and 0 > 198 - 2 fails; elsewhere 200 > mean - 2 holds.
    pixels = torch.full((1, 15, 15), 200, dtype=torch.uint8)
    pixels[0, 7, 7] = 0

    # when
    thresholded = run_threshold(pixels, "adaptive_mean", max_value=255)[0]

    # then
    assert thresholded[7, 7].item() == 0
    assert (thresholded == 255).sum().item() == 15 * 15 - 1


@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("threshold_type", list(OPENCV_FIXED_TYPES))
@pytest.mark.parametrize("thresh_value, max_value", [(127, 255), (40, 90), (254, 7)])
def test_fixed_types_equal_opencv_per_channel(
    channels: int, threshold_type: str, thresh_value: int, max_value: int
) -> None:
    pixels = random_pixels(channels, 13, 17, seed=channels)

    thresholded = run_threshold(pixels, threshold_type, thresh_value, max_value)

    expected = opencv_threshold(pixels, threshold_type, thresh_value, max_value)
    assert np.array_equal(thresholded.numpy(), expected)


@pytest.mark.parametrize(
    "threshold_type", ["otsu", "adaptive_mean", "adaptive_gaussian"]
)
@pytest.mark.parametrize("size_hw", [(1, 1), (3, 2), (23, 31), (64, 48)])
def test_one_channel_types_equal_opencv(threshold_type: str, size_hw: tuple) -> None:
    pixels = random_pixels(1, size_hw[0], size_hw[1], seed=size_hw[0])
    pixels = (pixels // 2 + torch.arange(size_hw[1], dtype=torch.uint8)).clamp(0, 255)

    thresholded = run_threshold(pixels, threshold_type, 0, 200)

    expected = opencv_threshold(pixels, threshold_type, 0, 200)
    assert np.array_equal(thresholded.numpy(), expected)


@pytest.mark.parametrize(
    "threshold_type", ["otsu", "adaptive_mean", "adaptive_gaussian"]
)
def test_one_channel_types_reject_rgb_and_name_the_fix(threshold_type: str) -> None:
    with pytest.raises(ValueError, match="v2/convert_grayscale@v1"):
        run_threshold(random_pixels(3, 5, 5, seed=0), threshold_type)


def test_unknown_threshold_type_is_rejected() -> None:
    with pytest.raises(ValueError, match="threshold_type must be one of"):
        run_threshold(RAMP, "triangle")


def test_keeps_identity_geometry_and_metadata_and_leaves_the_input() -> None:
    # given
    canvas = random_pixels(1, 20, 24, seed=3)
    root = ImageData.from_tensor(canvas, image_id="root", video_metadata=VIDEO)
    crop = root.crop((2, 2, 20, 18)).crop((1, 3, 15, 13), image_id="inner")
    crop = crop.with_parent_row(RowRef("detections", 0))
    before = crop.tensor_image.clone()

    # when
    image = ImageThresholdBlock().run(
        image=crop, threshold_type="tozero", thresh_value=300, max_value=255
    )["image"]

    # then: tozero with T >= 255 is all zeros; identity and geometry are kept
    assert image.tensor_image.max().item() == 0
    assert (image.image_id, image.parent, image.root) == (
        "inner",
        crop.parent,
        crop.root,
    )
    assert image.root.offset_xy == (3, 5)
    assert image.video_metadata is VIDEO
    assert image.parent_row == RowRef("detections", 0)
    assert torch.equal(crop.tensor_image, before)


def test_copying_branch_returns_new_pixels() -> None:
    # tozero_inv with T >= 255 keeps every value but must not alias the input
    pixels = random_pixels(1, 4, 4, seed=1)

    thresholded = run_threshold(pixels, "tozero_inv", 255)

    assert torch.equal(thresholded, pixels)
    assert thresholded.data_ptr() != pixels.data_ptr()


def test_channels_last_view_equals_opencv() -> None:
    view = random_pixels(3, 10, 12, seed=8).permute(1, 2, 0).contiguous()
    view = view.permute(2, 0, 1)

    thresholded = run_threshold(view, "binary", 100, 255)

    assert np.array_equal(
        thresholded.numpy(), opencv_threshold(view, "binary", 100, 255)
    )


def test_declares_a_pure_prunable_block() -> None:
    assert ImageThresholdBlock.type == "v2/threshold@v1"
    assert ImageThresholdBlock.prunable is True
    assert ImageThresholdBlock.mutates == ()
