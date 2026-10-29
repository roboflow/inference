"""Unit tests of ``roboflow_core/image_blur@v1``, called directly.

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
from roboflow_workflows.v2.blocks.image.image_blur import v1 as blur_module
from roboflow_workflows.v2.blocks.image.image_blur.v1 import ImageBlurBlock, torch_blur, torch_blur_is_exact
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


DEVICE_SETTINGS = [
    ("average", 1),
    ("average", 5),
    ("average", 15),
    ("gaussian", 1),
    ("gaussian", 3),
    ("gaussian", 7),
    ("median", 1),
    ("median", 3),
    ("median", 15),
]


@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("blur_type, kernel_size", DEVICE_SETTINGS)
@pytest.mark.parametrize("size_hw", [(31, 27), (16, 16), (8, 9)])
def test_torch_blur_equals_opencv_where_exact(
    channels: int, blur_type: str, kernel_size: int, size_hw: tuple
) -> None:
    # The off-CPU kernels, checked here on CPU tensors.
    pixels = random_pixels(channels, size_hw[0], size_hw[1], seed=kernel_size)
    if not torch_blur_is_exact(pixels, blur_type=blur_type, kernel_size=kernel_size):
        pytest.skip("larger than the image: OpenCV handles it")

    blurred = torch_blur(pixels, blur_type=blur_type, kernel_size=kernel_size)

    expected = opencv_blur(pixels, blur_type, kernel_size)
    assert np.array_equal(blurred.numpy(), expected)


@pytest.mark.parametrize("blur_type, kernel_size", DEVICE_SETTINGS)
def test_cpu_tensors_always_use_opencv(
    monkeypatch: pytest.MonkeyPatch, blur_type: str, kernel_size: int
) -> None:
    def refuse(*args, **kwargs):
        raise AssertionError("used the PyTorch emulation on the CPU")

    monkeypatch.setattr(blur_module, "torch_blur", refuse)
    pixels = random_pixels(3, 20, 20, seed=2)

    blurred = run_blur(pixels, blur_type, kernel_size)

    assert np.array_equal(blurred.numpy(), opencv_blur(pixels, blur_type, kernel_size))


@pytest.mark.parametrize(
    "blur_type, kernel_size, route",
    [
        ("average", 5, "torch"),
        ("gaussian", 7, "torch"),
        ("median", 15, "torch"),
        ("median", 4, "torch"),  # rounded up to 5
        ("average", 4, "opencv"),
        ("gaussian", 9, "opencv"),
        ("median", 17, "opencv"),
        ("bilateral", 5, "opencv"),
    ],
)
def test_off_cpu_tensors_blur_common_settings_on_their_device(
    monkeypatch: pytest.MonkeyPatch, blur_type: str, kernel_size: int, route: str
) -> None:
    # A shape-only "meta" tensor stands in for a CUDA image: no device needed.
    routes = []
    monkeypatch.setattr(
        blur_module, "torch_blur", lambda pixels, **kwargs: routes.append("torch")
    )
    monkeypatch.setattr(
        blur_module, "call_on_host", lambda pixels, **kwargs: routes.append("opencv")
    )
    pixels = torch.empty((3, 20, 20), dtype=torch.uint8, device="meta")

    blur_module.blur(pixels, blur_type=blur_type, kernel_size=kernel_size)

    assert routes == [route]


def test_off_cpu_kernel_wider_than_the_image_uses_opencv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    routes = []
    monkeypatch.setattr(
        blur_module, "call_on_host", lambda pixels, **kwargs: routes.append("opencv")
    )
    pixels = torch.empty((3, 4, 20), dtype=torch.uint8, device="meta")

    blur_module.blur(pixels, blur_type="median", kernel_size=9)

    assert routes == ["opencv"]


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
    assert ImageBlurBlock.type == "roboflow_core/image_blur@v1"
    assert ImageBlurBlock.prunable is True
    assert ImageBlurBlock.mutates == ()
