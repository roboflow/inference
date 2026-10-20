"""Unit tests of ``roboflow_core/contrast_enhancement@v1``, called directly.

Simple ramps are worked out by hand. Random images are compared with V1's
own NumPy implementation (``core_steps/classical_cv/contrast_enhancement``),
the accepted oracle, on the same pixels in V1's BGR layout. All comparisons
are exact.
"""

from datetime import datetime, timezone

import numpy as np
import pytest
import torch
from roboflow_workflows.core_steps.classical_cv.contrast_enhancement.v1 import (
    enhance_contrast,
)
from roboflow_workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)
from roboflow_workflows.v2.blocks.image.contrast_enhancement.v1 import ContrastEnhancementBlock
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.row_identity import RowRef

VIDEO = VideoMetadata(
    video_identifier="cam-1",
    frame_number=7,
    frame_timestamp=datetime(2026, 10, 12, tzinfo=timezone.utc),
)
RAMP = torch.arange(100, 151, dtype=torch.uint8).reshape(1, 1, 51)


def enhance(
    image: ImageData,
    clip_limit: int = 0,
    contrast_multiplier: float = 1.0,
    normalize_brightness: bool = False,
) -> ImageData:
    result = ContrastEnhancementBlock().run(
        image=image,
        clip_limit=clip_limit,
        contrast_multiplier=contrast_multiplier,
        normalize_brightness=normalize_brightness,
    )

    return result["image"]


def v1_enhance(pixels: torch.Tensor, **settings) -> np.ndarray:
    # V1's function on its NumPy layout: 2-D gray or (H, W, 3) BGR.
    array = pixels.permute(1, 2, 0).numpy()
    array = array[:, :, 0] if array.shape[2] == 1 else array[:, :, ::-1]
    v1_image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="v1"),
        numpy_image=np.ascontiguousarray(array),
    )
    enhanced = enhance_contrast(image=v1_image, **settings).numpy_image
    enhanced = enhanced.reshape(enhanced.shape[0], enhanced.shape[1], -1)
    if enhanced.shape[2] == 3:
        enhanced = enhanced[:, :, ::-1]
    expected = np.ascontiguousarray(enhanced.transpose(2, 0, 1))

    return expected


def random_pixels(channels: int, seed: int, *, low: int = 0, high: int = 256):
    generator = np.random.default_rng(seed)
    pixels = generator.integers(low, high, (channels, 29, 33), dtype=np.uint8)
    tensor = torch.from_numpy(pixels)

    return tensor


def test_stretches_the_range_to_full_scale() -> None:
    # (v - 100) / 50 * 255, truncated: 100 -> 0, 125 -> 127.5 -> 127, 150 -> 255
    enhanced = enhance(ImageData.from_tensor(RAMP)).tensor_image[0, 0]

    assert enhanced[[0, 25, 50]].tolist() == [0, 127, 255]


def test_contrast_multiplier_scales_around_mid_gray() -> None:
    # 110 -> 51 -> 128 + (51 - 128) * 2 < 0 -> 0; 125 -> 127; 140 -> 204 -> 255
    enhanced = enhance(ImageData.from_tensor(RAMP), contrast_multiplier=2.0)

    assert enhanced.tensor_image[0, 0, [10, 25, 40]].tolist() == [0, 127, 255]


def test_normalize_brightness_lifts_midtones() -> None:
    enhanced = enhance(ImageData.from_tensor(RAMP), normalize_brightness=True)

    expected = int(255 * (127.5 / 255) ** (1 / 1.3))
    assert expected == 149
    assert enhanced.tensor_image[0, 0, 25].item() == expected


def test_flat_one_channel_image_is_returned_unchanged() -> None:
    image = ImageData.from_tensor(torch.full((1, 4, 4), 90, dtype=torch.uint8))

    assert enhance(image, clip_limit=5) is image


def test_flat_channel_of_a_color_image_is_left_unchanged() -> None:
    # given: red spans 100..150, green is constant 77, blue is 0 or 255
    blue = (torch.arange(51, dtype=torch.uint8) % 2 * 255).reshape(1, 51)
    pixels = torch.stack([RAMP[0], torch.full_like(RAMP[0], 77), blue])

    # when
    enhanced = enhance(ImageData.from_tensor(pixels)).tensor_image

    # then
    assert enhanced[0, 0, [0, 25, 50]].tolist() == [0, 127, 255]
    assert enhanced[1].unique().tolist() == [77]
    assert torch.equal(enhanced[2], pixels[2])


@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("clip_limit", [0, 1, 7, 50])
@pytest.mark.parametrize("contrast_multiplier", [0.1, 1.0, 1.7, 5.0])
@pytest.mark.parametrize("normalize_brightness", [False, True])
def test_equals_v1(
    channels: int,
    clip_limit: int,
    contrast_multiplier: float,
    normalize_brightness: bool,
) -> None:
    pixels = random_pixels(channels, seed=clip_limit, low=40, high=200)
    settings = dict(
        clip_limit=clip_limit,
        contrast_multiplier=contrast_multiplier,
        normalize_brightness=normalize_brightness,
    )

    enhanced = enhance(ImageData.from_tensor(pixels), **settings)

    assert np.array_equal(enhanced.tensor_image.numpy(), v1_enhance(pixels, **settings))


@pytest.mark.parametrize("seed", range(20))
def test_equals_v1_on_many_value_distributions(seed: int) -> None:
    # Narrow, skewed and sparse histograms stress percentiles and truncation.
    generator = np.random.default_rng(seed)
    channels = 1 if seed % 2 else 3
    center, spread = generator.integers(20, 236), generator.integers(1, 60)
    values = generator.normal(center, spread, (channels, 17, 19))
    pixels = torch.from_numpy(np.clip(values, 0, 255).astype(np.uint8))
    settings = dict(
        clip_limit=int(generator.integers(0, 51)),
        contrast_multiplier=float(generator.uniform(0.1, 5.0)),
        normalize_brightness=bool(seed % 3 == 0),
    )

    enhanced = enhance(ImageData.from_tensor(pixels), **settings)

    assert np.array_equal(enhanced.tensor_image.numpy(), v1_enhance(pixels, **settings))


def test_keeps_identity_geometry_and_metadata_and_leaves_the_input() -> None:
    # given
    canvas = random_pixels(3, seed=1)
    root = ImageData.from_tensor(canvas, image_id="root", video_metadata=VIDEO)
    crop = root.crop((4, 3, 30, 27)).crop((1, 2, 21, 18), image_id="inner")
    crop = crop.with_parent_row(RowRef("detections", 2))
    before = crop.tensor_image.clone()

    # when
    image = enhance(crop, clip_limit=2)

    # then
    assert (image.image_id, image.parent, image.root) == (
        "inner",
        crop.parent,
        crop.root,
    )
    assert image.root.offset_xy == (5, 5)
    assert image.video_metadata is VIDEO
    assert image.parent_row == RowRef("detections", 2)
    assert torch.equal(crop.tensor_image, before)


def test_channels_last_view_equals_v1() -> None:
    view = random_pixels(3, seed=5).permute(1, 2, 0).contiguous().permute(2, 0, 1)
    assert not view.is_contiguous()

    enhanced = enhance(ImageData.from_tensor(view), clip_limit=3)

    assert np.array_equal(enhanced.tensor_image.numpy(), v1_enhance(view, clip_limit=3))


def test_declares_a_pure_prunable_block() -> None:
    assert ContrastEnhancementBlock.type == "roboflow_core/contrast_enhancement@v1"
    assert ContrastEnhancementBlock.prunable is True
    assert ContrastEnhancementBlock.mutates == ()
