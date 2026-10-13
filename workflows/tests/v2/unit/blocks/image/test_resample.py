"""Unit tests of the shared resampling helpers in ``blocks/image/_resample.py``.

Expected pixels are worked out by hand: integer positions copy pixels,
half-pixel positions outside the image blend with black, and positions are
rounded to 1/32 pixel before blending. Device cases run where the device is
present; CPU is the only device every run has.
"""

from dataclasses import replace
from datetime import datetime, timezone

import pytest
import torch
from roboflow_workflows.execution_engine.entities.base import VideoMetadata
from roboflow_workflows.v2.blocks.image._resample import (
    compute_dtype,
    new_root_image,
    remap_bilinear,
    require_same_image,
)
from roboflow_workflows.v2.data.errors import ContractError
from roboflow_workflows.v2.data.frames import frame_of
from roboflow_workflows.v2.data.image import FrameMapping, ImageData
from roboflow_workflows.v2.data.row_identity import RowRef

from inference_models.models.base.object_detection import Detections

VIDEO = VideoMetadata(
    video_identifier="cam-1",
    frame_number=7,
    frame_timestamp=datetime(2026, 10, 12, tzinfo=timezone.utc),
)


def random_pixels(channels: int, height: int, width: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(channels * 1000 + height + width)
    pixels = torch.randint(
        0, 256, (channels, height, width), dtype=torch.uint8, generator=generator
    )

    return pixels


def integer_grid(height: int, width: int, *, dtype: torch.dtype = torch.float64):
    ys, xs = torch.meshgrid(
        torch.arange(height, dtype=dtype),
        torch.arange(width, dtype=dtype),
        indexing="ij",
    )

    return xs, ys


def one_box(metadata) -> Detections:
    detections = Detections(
        xyxy=torch.tensor([[1.0, 1.0, 3.0, 3.0]]),
        class_id=torch.zeros(1, dtype=torch.int64),
        confidence=torch.full((1,), 0.9),
        image_metadata=metadata,
    )

    return detections


@pytest.mark.parametrize("channels", [1, 3])
def test_integer_positions_copy_pixels_exactly(channels: int) -> None:
    pixels = random_pixels(channels, 5, 7)
    source_x, source_y = integer_grid(5, 7)

    remapped = remap_bilinear(pixels, source_x=source_x, source_y=source_y)

    assert remapped.dtype == torch.uint8
    assert torch.equal(remapped, pixels)


def test_output_takes_the_shape_of_the_positions() -> None:
    pixels = random_pixels(3, 6, 8)
    source_x, source_y = integer_grid(2, 3)

    remapped = remap_bilinear(pixels, source_x=source_x + 4, source_y=source_y + 1)

    assert remapped.shape == (3, 2, 3)
    assert torch.equal(remapped, pixels[:, 1:3, 4:7])


def test_positions_past_the_border_blend_with_black() -> None:
    pixels = torch.full((1, 4, 4), 200, dtype=torch.uint8)
    source_x = torch.tensor([[-0.5, 3.5, -1.0, 4.0, 0.0]])
    source_y = torch.tensor([[1.0, 1.0, 1.0, 1.0, -0.5]])

    remapped = remap_bilinear(pixels, source_x=source_x, source_y=source_y)

    assert remapped.tolist() == [[[100, 100, 0, 0, 100]]]


def test_outside_and_non_finite_positions_give_black() -> None:
    pixels = torch.full((1, 4, 4), 200, dtype=torch.uint8)
    source_x = torch.tensor([[1.0, -5.0, float("nan"), 1.5, float("-inf")]])
    source_y = torch.tensor([[1.0, 1.0, 1.0, float("inf"), 1.0]])

    remapped = remap_bilinear(pixels, source_x=source_x, source_y=source_y)

    assert remapped.tolist() == [[[200, 0, 0, 0, 0]]]


def test_positions_are_rounded_to_one_32nd_pixel() -> None:
    # 0.52 rounds to 17/32 = 0.53125: 255 * 0.53125 = 135.47 -> 135.
    # Without rounding the level would be 255 * 0.52 = 132.6 -> 133.
    pixels = torch.tensor([[[0, 255]]], dtype=torch.uint8)
    source_x = torch.tensor([[0.52, 0.5]])
    source_y = torch.zeros((1, 2))

    remapped = remap_bilinear(pixels, source_x=source_x, source_y=source_y)

    assert remapped.tolist() == [[[135, 128]]]


def test_float32_and_float64_positions_give_the_same_pixels() -> None:
    pixels = random_pixels(3, 9, 11)
    generator = torch.Generator().manual_seed(5)
    source_x = torch.rand((6, 8), generator=generator, dtype=torch.float64) * 12 - 1
    source_y = torch.rand((6, 8), generator=generator, dtype=torch.float64) * 10 - 1
    # Positions on the 1/32 lattice are exact in both float types.
    source_x = torch.round(source_x * 32) / 32
    source_y = torch.round(source_y * 32) / 32

    wide = remap_bilinear(pixels, source_x=source_x, source_y=source_y)
    narrow = remap_bilinear(
        pixels, source_x=source_x.float(), source_y=source_y.float()
    )

    assert torch.equal(wide, narrow)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="needs a non-CPU torch device"
)
def test_remap_stays_on_its_device() -> None:
    pixels = random_pixels(3, 9, 11)
    source_x, source_y = integer_grid(9, 11, dtype=torch.float32)
    on_cpu = remap_bilinear(pixels, source_x=source_x + 0.25, source_y=source_y)

    on_device = remap_bilinear(
        pixels.to("mps"),
        source_x=(source_x + 0.25).to("mps"),
        source_y=source_y.to("mps"),
    )

    assert on_device.device.type == "mps"
    assert on_device.dtype == torch.uint8
    assert torch.equal(on_device.cpu(), on_cpu)


@pytest.mark.parametrize(
    ("device", "expected"),
    [("cpu", torch.float64), ("cuda", torch.float64), ("mps", torch.float32)],
)
def test_compute_dtype_is_float64_except_on_mps(device: str, expected) -> None:
    assert compute_dtype(torch.device(device)) == expected


def test_new_root_image_keeps_source_time_and_parent_row() -> None:
    like = ImageData.from_tensor(
        random_pixels(3, 6, 8), image_id="camera", video_metadata=VIDEO
    ).with_parent_row(RowRef("plates", 2))
    pixels = random_pixels(3, 4, 5)

    image = new_root_image(pixels, like=like, image_id="warped-1")

    assert image.image_id == "warped-1"
    assert image.size_hw == (4, 5)
    assert image.root == FrameMapping.identity("warped-1", (4, 5))
    assert image.parent == image.root
    assert image.video_metadata == VIDEO
    assert image.parent_row == RowRef("plates", 2)
    assert torch.equal(image.tensor_image, pixels)


def test_predictions_of_the_same_image_are_accepted() -> None:
    image = ImageData.from_tensor(random_pixels(3, 6, 8), image_id="camera")

    require_same_image(
        one_box(image.prediction_metadata()), image=image, consumer="v2/test@v1"
    )


def test_predictions_of_another_image_are_refused() -> None:
    image = ImageData.from_tensor(random_pixels(3, 6, 8), image_id="camera")
    other = ImageData.from_tensor(random_pixels(3, 6, 8), image_id="other")

    with pytest.raises(ContractError, match="v2/test@v1: .* image 'other'"):
        require_same_image(
            one_box(other.prediction_metadata()), image=image, consumer="v2/test@v1"
        )


def test_a_frame_without_an_image_id_is_not_compared() -> None:
    image = ImageData.from_tensor(random_pixels(3, 6, 8), image_id="camera")
    frame = replace(frame_of(one_box(image.prediction_metadata())), image_id=None)

    require_same_image(
        one_box(frame.prediction_metadata()), image=image, consumer="v2/test@v1"
    )


@pytest.mark.parametrize(
    "metadata", [None, {"class_names": {0: "car"}}], ids=["none", "no-geometry"]
)
def test_predictions_without_a_readable_frame_are_refused(metadata) -> None:
    image = ImageData.from_tensor(random_pixels(3, 6, 8), image_id="camera")

    with pytest.raises(ContractError):
        require_same_image(one_box(metadata), image=image, consumer="v2/test@v1")
