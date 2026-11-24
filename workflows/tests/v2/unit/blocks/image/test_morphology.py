"""Unit tests of ``roboflow_core/morphological_transformation@v2``, called directly.

Small shapes are worked out by hand. Every other case is compared with the
OpenCV call V1 made, with a ``kernel_size`` square of ones, on the same image
in V1's BGR layout. All comparisons are exact.
"""

from datetime import datetime, timezone

import cv2
import numpy as np
import pytest
import torch
from roboflow_workflows.execution_engine.entities.base import VideoMetadata
from roboflow_workflows.v2.blocks.image.morphological_transformation import v2 as morphology_module
from roboflow_workflows.v2.blocks.image.morphological_transformation.v2 import MorphologicalTransformationBlock
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.row_identity import RowRef

VIDEO = VideoMetadata(
    video_identifier="cam-1",
    frame_number=7,
    frame_timestamp=datetime(2026, 10, 12, tzinfo=timezone.utc),
)
OPERATIONS = [
    "Erosion",
    "Dilation",
    "Opening",
    "Closing",
    "Opening then Closing",
    "Gradient",
    "Top Hat",
    "Black Hat",
]


def transform(pixels: torch.Tensor, operation: str, kernel_size: int) -> torch.Tensor:
    result = MorphologicalTransformationBlock().run(
        image=ImageData.from_tensor(pixels),
        kernel_size=kernel_size,
        operation=operation,
    )

    return result["image"].tensor_image


def opencv_transform(pixels: torch.Tensor, operation: str, kernel_size: int):
    # V1's calls: gray is first copied to three channels (GRAY2BGR).
    array = pixels.permute(1, 2, 0).numpy()
    if array.shape[2] == 1:
        array = cv2.cvtColor(np.ascontiguousarray(array[:, :, 0]), cv2.COLOR_GRAY2BGR)
    else:
        array = np.ascontiguousarray(array[:, :, ::-1])
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if operation == "Erosion":
        result = cv2.erode(array, kernel, iterations=1)
    elif operation == "Dilation":
        result = cv2.dilate(array, kernel, iterations=1)
    elif operation == "Opening then Closing":
        opened = cv2.morphologyEx(array, cv2.MORPH_OPEN, kernel)
        result = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    else:
        flag = {
            "Opening": cv2.MORPH_OPEN,
            "Closing": cv2.MORPH_CLOSE,
            "Gradient": cv2.MORPH_GRADIENT,
            "Top Hat": cv2.MORPH_TOPHAT,
            "Black Hat": cv2.MORPH_BLACKHAT,
        }[operation]
        result = cv2.morphologyEx(array, flag, kernel)
    expected = np.ascontiguousarray(result[:, :, ::-1].transpose(2, 0, 1))

    return expected


def dot(size: int = 7, *, at=(3, 3), value: int = 255, background: int = 0):
    pixels = torch.full((1, size, size), background, dtype=torch.uint8)
    pixels[0, at[0], at[1]] = value

    return pixels


def random_pixels(channels: int, height: int, width: int, seed: int) -> torch.Tensor:
    generator = np.random.default_rng(seed)
    pixels = generator.integers(0, 256, (channels, height, width), dtype=np.uint8)
    tensor = torch.from_numpy(pixels)

    return tensor


def test_dilation_grows_a_dot_into_a_square() -> None:
    dilated = transform(dot(), "Dilation", 3)[0]

    expected = torch.zeros((7, 7), dtype=torch.uint8)
    expected[2:5, 2:5] = 255
    assert torch.equal(dilated, expected)


def test_even_kernel_reaches_further_up_and_left() -> None:
    # A 2 x 2 window covers its pixel and the pixels up and left of it, so
    # a dot at (3, 3) lights (3..4, 3..4).
    dilated = transform(dot(), "Dilation", 2)[0]

    assert torch.nonzero(dilated).tolist() == [[3, 3], [3, 4], [4, 3], [4, 4]]


@pytest.mark.parametrize("operation", ["Erosion", "Opening"])
def test_erosion_and_opening_remove_a_bright_dot(operation: str) -> None:
    assert transform(dot(), operation, 3).max().item() == 0


def test_closing_fills_a_dark_hole() -> None:
    hole = dot(value=0, background=255)

    assert transform(hole, "Closing", 3).min().item() == 255


def test_outside_the_image_does_not_count() -> None:
    # A white image stays white under erosion: the border is not black.
    white = torch.full((3, 5, 6), 255, dtype=torch.uint8)

    assert torch.equal(transform(white, "Erosion", 5), white)


def test_gradient_outlines_a_square_and_top_hat_keeps_a_small_dot() -> None:
    square = torch.zeros((1, 9, 9), dtype=torch.uint8)
    square[0, 3:6, 3:6] = 200

    outline = transform(square, "Gradient", 3)[0]
    details = transform(dot(), "Top Hat", 3)[0]

    assert outline[4, 4].item() == 0
    assert outline[2, 2].item() == 200 and outline[3, 3].item() == 200
    assert torch.equal(details, dot()[0])


def test_one_channel_input_becomes_three_equal_channels() -> None:
    result = transform(dot(), "Dilation", 3)

    assert result.shape == (3, 7, 7)
    assert torch.equal(result[0], result[1]) and torch.equal(result[1], result[2])


@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("kernel_size", [1, 2, 3, 4, 5, 8])
def test_equals_opencv(channels: int, operation: str, kernel_size: int) -> None:
    pixels = random_pixels(channels, 19, 23, seed=kernel_size)

    transformed = transform(pixels, operation, kernel_size)

    expected = opencv_transform(pixels, operation, kernel_size)
    assert np.array_equal(transformed.numpy(), expected)


@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("kernel_size", [1, 2, 3, 4, 5, 8, 9])
def test_torch_operations_equal_opencv(
    channels: int, operation: str, kernel_size: int
) -> None:
    # The off-CPU versions, checked here on CPU tensors; 9 exceeds 3 x 4.
    size_hw = (3, 4) if kernel_size == 9 else (19, 23)
    pixels = random_pixels(channels, size_hw[0], size_hw[1], seed=kernel_size)

    transformed = morphology_module.OPERATIONS[operation](
        pixels.expand(3, -1, -1), kernel_size
    )

    expected = opencv_transform(pixels, operation, kernel_size)
    assert np.array_equal(transformed.numpy(), expected)


def test_cpu_tensors_use_opencv(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args, **kwargs):
        raise AssertionError("used the PyTorch emulation on the CPU")

    monkeypatch.setattr(morphology_module, "_window_extreme", refuse)
    pixels = random_pixels(3, 19, 23, seed=6)

    for operation in OPERATIONS:
        transformed = transform(pixels, operation, 5)
        expected = opencv_transform(pixels, operation, 5)
        assert np.array_equal(transformed.numpy(), expected), operation


def test_off_cpu_tensors_stay_on_their_device(monkeypatch: pytest.MonkeyPatch) -> None:
    # A shape-only "meta" tensor stands in for a CUDA image: no device needed.
    def refuse(*args, **kwargs):
        raise AssertionError("crossed to the CPU")

    monkeypatch.setattr(morphology_module, "call_on_host", refuse)
    pixels = torch.empty((3, 20, 20), dtype=torch.uint8, device="meta")

    for operation in OPERATIONS:
        transformed = morphology_module.morphology(
            pixels, operation=operation, kernel_size=5
        )
        assert transformed.device.type == "meta"
        assert transformed.shape == (3, 20, 20)


@pytest.mark.parametrize("operation", ["Opening then Closing", "Black Hat"])
def test_kernel_larger_than_the_image_equals_opencv(operation: str) -> None:
    pixels = random_pixels(3, 3, 4, seed=11)

    transformed = transform(pixels, operation, 9)

    assert np.array_equal(transformed.numpy(), opencv_transform(pixels, operation, 9))


def test_channels_last_view_equals_opencv() -> None:
    view = random_pixels(3, 12, 10, seed=4).permute(1, 2, 0).contiguous()
    view = view.permute(2, 0, 1)

    transformed = transform(view, "Gradient", 3)

    assert np.array_equal(transformed.numpy(), opencv_transform(view, "Gradient", 3))


def test_keeps_identity_geometry_and_metadata_and_leaves_the_input() -> None:
    # given: a gray crop of a crop
    canvas = random_pixels(1, 30, 30, seed=6)
    root = ImageData.from_tensor(canvas, image_id="root", video_metadata=VIDEO)
    crop = root.crop((5, 5, 25, 25)).crop((2, 0, 12, 10), image_id="inner")
    crop = crop.with_parent_row(RowRef("detections", 4))
    before = crop.tensor_image.clone()

    # when
    image = MorphologicalTransformationBlock().run(
        image=crop, kernel_size=3, operation="Opening"
    )["image"]

    # then
    assert (image.image_id, image.parent, image.root) == (
        "inner",
        crop.parent,
        crop.root,
    )
    assert image.root.offset_xy == (7, 5)
    assert image.video_metadata is VIDEO
    assert image.parent_row == RowRef("detections", 4)
    assert torch.equal(crop.tensor_image, before)


def test_kernel_size_below_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        transform(dot(), "Erosion", 0)


def test_unknown_operation_is_rejected() -> None:
    with pytest.raises(ValueError, match="operation must be one of"):
        transform(dot(), "Skeleton", 3)


def test_declares_a_pure_prunable_block() -> None:
    assert MorphologicalTransformationBlock.type == "roboflow_core/morphological_transformation@v2"
    assert MorphologicalTransformationBlock.prunable is True
    assert MorphologicalTransformationBlock.mutates == ()
