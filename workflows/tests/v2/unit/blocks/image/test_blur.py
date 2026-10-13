"""Unit tests of ``v2/image_blur@v1``, called directly.

Small cases are worked out by hand. Every other case is compared with the
OpenCV call V1 made, on the same image in V1's BGR layout: all results are
expected to be exact, also where the block itself calls OpenCV on the CPU.
"""

from datetime import datetime, timezone

import cv2
import numpy as np
import pytest
import torch
from roboflow_workflows.execution_engine.entities.base import VideoMetadata
from roboflow_workflows.v2.blocks.image import blur as blur_module
from roboflow_workflows.v2.blocks.image.blur import ImageBlurBlock
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.row_identity import RowRef

VIDEO = VideoMetadata(
    video_identifier="cam-1",
    frame_number=7,
    frame_timestamp=datetime(2026, 10, 12, tzinfo=timezone.utc),
)


def run_blur(pixels: torch.Tensor, blur_type: str, kernel_size: int) -> torch.Tensor:
    result = ImageBlurBlock().run(
        image=ImageData.from_tensor(pixels),
        blur_type=blur_type,
        kernel_size=kernel_size,
    )

    return result["image"].tensor_image


def opencv_blur(pixels: torch.Tensor, blur_type: str, kernel_size: int) -> np.ndarray:
    # V1's calls, on V1's layout; kernel sizes as V1 coerced them.
    array = pixels.permute(1, 2, 0).numpy()
    array = np.ascontiguousarray(array[:, :, ::-1] if array.shape[2] == 3 else array)
    odd = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
    odd = max(1, odd)
    if blur_type == "average":
        blurred = cv2.blur(array, (kernel_size, kernel_size))
    elif blur_type == "gaussian":
        blurred = cv2.GaussianBlur(array, (odd, odd), 0)
    elif blur_type == "median":
        blurred = cv2.medianBlur(array, odd)
    else:
        blurred = cv2.bilateralFilter(array, kernel_size, 75, 75)
    blurred = blurred.reshape(array.shape[0], array.shape[1], -1)
    if blurred.shape[2] == 3:
        blurred = blurred[:, :, ::-1]
    expected = np.ascontiguousarray(blurred.transpose(2, 0, 1))

    return expected


def random_pixels(channels: int, height: int, width: int, seed: int) -> torch.Tensor:
    generator = np.random.default_rng(seed)
    pixels = generator.integers(0, 256, (channels, height, width), dtype=np.uint8)
    tensor = torch.from_numpy(pixels)

    return tensor


def single_dot(size: int = 9) -> torch.Tensor:
    pixels = torch.zeros((1, size, size), dtype=torch.uint8)
    pixels[0, size // 2, size // 2] = 255

    return pixels


def test_average_of_a_single_dot_spreads_its_value_evenly() -> None:
    # 255 / 25 = 10.2 -> 10 over the 5 x 5 window, 0 outside it
    blurred = run_blur(single_dot(), "average", 5)[0]

    assert blurred[2:7, 2:7].unique().tolist() == [10]
    assert blurred.sum().item() == 10 * 25


def test_gaussian_three_by_three_kernel_weighs_the_centre_by_a_quarter() -> None:
    # OpenCV's 3-tap kernel is (1, 2, 1) / 4: centre 255 / 4 = 63.75 -> 64,
    # edges 255 / 8 = 31.9 -> 32, corners 255 / 16 = 15.9 -> 16
    blurred = run_blur(single_dot(), "gaussian", 3)[0]

    assert blurred[3:6, 3:6].tolist() == [[16, 32, 16], [32, 64, 32], [16, 32, 16]]


def test_median_removes_an_isolated_dot() -> None:
    blurred = run_blur(single_dot(), "median", 3)

    assert blurred.max().item() == 0


@pytest.mark.parametrize("blur_type", ["average", "gaussian", "median", "bilateral"])
def test_uniform_image_stays_uniform(blur_type: str) -> None:
    pixels = torch.full((3, 12, 10), 77, dtype=torch.uint8)

    assert torch.equal(run_blur(pixels, blur_type, 5), pixels)


@pytest.mark.parametrize(
    "blur_type, kernel_size, coerced",
    [("gaussian", 4, 5), ("gaussian", 0, 1), ("median", -3, 1), ("median", 6, 7)],
)
def test_gaussian_and_median_round_kernel_size_up_to_odd(
    blur_type: str, kernel_size: int, coerced: int
) -> None:
    pixels = random_pixels(3, 20, 17, seed=1)

    assert torch.equal(
        run_blur(pixels, blur_type, kernel_size), run_blur(pixels, blur_type, coerced)
    )


@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize(
    "blur_type, kernel_size",
    [
        ("average", 1),
        ("average", 3),
        ("average", 15),
        ("average", 4),  # even: CPU
        ("average", 17),  # large: CPU
        ("gaussian", 1),
        ("gaussian", 5),
        ("gaussian", 7),
        ("gaussian", 9),  # CPU
        ("gaussian", 15),  # CPU
        ("median", 3),
        ("median", 15),
        ("median", 17),  # CPU
        ("bilateral", 5),  # always CPU
        ("bilateral", 0),  # V1: OpenCV derives the diameter from sigma
    ],
)
def test_equals_opencv(channels: int, blur_type: str, kernel_size: int) -> None:
    pixels = random_pixels(channels, 31, 27, seed=kernel_size)

    blurred = run_blur(pixels, blur_type, kernel_size)

    expected = opencv_blur(pixels, blur_type, kernel_size)
    assert np.array_equal(blurred.numpy(), expected)


@pytest.mark.parametrize("blur_type", ["average", "gaussian", "median"])
@pytest.mark.parametrize("size_hw", [(1, 1), (2, 3), (4, 2)])
def test_images_smaller_than_the_kernel_equal_opencv(
    blur_type: str, size_hw: tuple
) -> None:
    pixels = random_pixels(3, size_hw[0], size_hw[1], seed=7)

    blurred = run_blur(pixels, blur_type, 5)

    assert np.array_equal(blurred.numpy(), opencv_blur(pixels, blur_type, 5))


@pytest.mark.parametrize(
    "blur_type, kernel_size", [("average", 5), ("gaussian", 7), ("median", 15)]
)
def test_common_settings_stay_on_the_image_device(
    monkeypatch: pytest.MonkeyPatch, blur_type: str, kernel_size: int
) -> None:
    def refuse(*args, **kwargs):
        raise AssertionError("crossed to the CPU")

    monkeypatch.setattr(blur_module, "opencv_on_host", refuse)

    blurred = run_blur(random_pixels(3, 20, 20, seed=2), blur_type, kernel_size)

    assert blurred.shape == (3, 20, 20)


def test_keeps_identity_geometry_and_metadata_and_leaves_the_input() -> None:
    # given: a crop of a crop, referring to a detection row
    canvas = random_pixels(3, 30, 40, seed=5)
    root = ImageData.from_tensor(canvas, image_id="root", video_metadata=VIDEO)
    crop = root.crop((3, 2, 33, 25), image_id="outer").crop((4, 1, 24, 21))
    crop = crop.with_parent_row(RowRef("detections", 1))
    before = crop.tensor_image.clone()

    # when
    blurred = ImageBlurBlock().run(image=crop, blur_type="gaussian", kernel_size=5)

    # then
    image = blurred["image"]
    assert (image.image_id, image.parent, image.root) == (
        crop.image_id,
        crop.parent,
        crop.root,
    )
    assert image.root.offset_xy == (7, 3)
    assert image.video_metadata is VIDEO
    assert image.parent_row == RowRef("detections", 1)
    assert torch.equal(crop.tensor_image, before)
    assert image.tensor_image.data_ptr() != crop.tensor_image.data_ptr()


def test_channels_last_view_equals_opencv() -> None:
    view = random_pixels(3, 16, 18, seed=9).permute(1, 2, 0).contiguous()
    view = view.permute(2, 0, 1)
    assert not view.is_contiguous()

    for blur_type in ("average", "bilateral"):
        blurred = run_blur(view, blur_type, 3)
        assert np.array_equal(blurred.numpy(), opencv_blur(view, blur_type, 3))


def test_average_needs_a_positive_kernel() -> None:
    with pytest.raises(ValueError, match="kernel_size >= 1"):
        run_blur(single_dot(), "average", 0)


def test_unknown_blur_type_is_rejected() -> None:
    with pytest.raises(ValueError, match="blur_type must be one of"):
        run_blur(single_dot(), "motion", 3)


def test_declares_a_pure_prunable_block() -> None:
    assert ImageBlurBlock.type == "v2/image_blur@v1"
    assert ImageBlurBlock.prunable is True
    assert ImageBlurBlock.mutates == ()
