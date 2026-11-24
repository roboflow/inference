"""Unit tests of ``roboflow_core/contrast_equalization@v1``, called directly.

Small histograms are worked out by hand. Other cases call scikit-image's
functions directly with V1's arguments, dtypes and BGR layout, as the
reference. All comparisons are exact.
"""

from datetime import datetime, timezone

import numpy as np
import pytest
import torch
from roboflow_workflows.execution_engine.entities.base import VideoMetadata
from roboflow_workflows.v2.blocks.image.contrast_equalization.v1 import ContrastEqualizationBlock
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.row_identity import RowRef
from skimage import exposure

VIDEO = VideoMetadata(
    video_identifier="cam-1",
    frame_number=7,
    frame_timestamp=datetime(2026, 10, 12, tzinfo=timezone.utc),
)
# The reference equalize_hist warns that it flattens a color image, as V1 did.
pytestmark = pytest.mark.filterwarnings("ignore:This might be a color image")
TYPES = ["Contrast Stretching", "Histogram Equalization", "Adaptive Equalization"]


def equalize(pixels: torch.Tensor, equalization_type: str) -> torch.Tensor:
    result = ContrastEqualizationBlock().run(
        image=ImageData.from_tensor(pixels), equalization_type=equalization_type
    )

    return result["image"].tensor_image


def skimage_reference(pixels: torch.Tensor, equalization_type: str) -> np.ndarray:
    # V1's calls on V1's layout: 2-D gray or (H, W, 3) BGR.
    array = pixels.permute(1, 2, 0).numpy()
    array = np.ascontiguousarray(
        array[:, :, 0] if array.shape[2] == 1 else array[:, :, ::-1]
    )
    if equalization_type == "Contrast Stretching":
        low, high = np.percentile(array, 2), np.percentile(array, 98)
        result = exposure.rescale_intensity(array, in_range=(low, high))
    elif equalization_type == "Histogram Equalization":
        scaled = array.astype(np.float32) / 255
        result = (exposure.equalize_hist(scaled) * 255).astype(np.uint8)
    else:
        scaled = array.astype(np.float32) / 255
        equalized = exposure.equalize_adapthist(scaled, clip_limit=0.03)
        result = (equalized * 255).astype(np.uint8)
    result = result.reshape(array.shape[0], array.shape[1], -1)
    if result.shape[2] == 3:
        result = result[:, :, ::-1]
    expected = np.ascontiguousarray(result.transpose(2, 0, 1))

    return expected


def random_pixels(channels: int, height: int, width: int, seed: int) -> torch.Tensor:
    generator = np.random.default_rng(seed)
    values = generator.normal(120, 25, (channels, height, width))
    tensor = torch.from_numpy(np.clip(values, 0, 255).astype(np.uint8))

    return tensor


def test_histogram_equalization_of_two_equal_populations() -> None:
    # Half 100, half 150: the cumulative share is 0.5 at 100 and 1.0 at 150,
    # so 100 -> 0.5 * 255 = 127.5 -> 127 and 150 -> 255.
    pixels = torch.full((1, 4, 4), 100, dtype=torch.uint8)
    pixels[:, 2:] = 150

    equalized = equalize(pixels, "Histogram Equalization")

    assert equalized[:, :2].unique().tolist() == [127]
    assert equalized[:, 2:].unique().tolist() == [255]


def test_docstring_example_one_pixel_of_each_value_100_to_150() -> None:
    # The darkest of 51 equally common values keeps its own share: 255 / 51.
    pixels = torch.arange(100, 151, dtype=torch.uint8).reshape(1, 3, 17)

    equalized = equalize(pixels, "Histogram Equalization")
    stretched = equalize(pixels, "Contrast Stretching")

    assert (int(equalized.min()), int(equalized.max())) == (5, 255)
    assert np.array_equal(
        equalized.numpy(), skimage_reference(pixels, "Histogram Equalization")
    )
    assert (int(stretched.min()), int(stretched.max())) == (0, 255)


def test_contrast_stretching_maps_percentiles_to_the_full_range() -> None:
    # 0..99, one pixel each: p2 = 1.98, p98 = 97.02; values at or below p2
    # become 0, at or above p98 become 255.
    pixels = torch.arange(100, dtype=torch.uint8).reshape(1, 10, 10)

    stretched = equalize(pixels, "Contrast Stretching").flatten()

    assert stretched[:2].tolist() == [0, 0]
    assert stretched[98:].tolist() == [255, 255]
    assert torch.all(stretched[1:] >= stretched[:-1])


@pytest.mark.parametrize("equalization_type", TYPES)
@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("seed", range(3))
def test_equals_scikit_image(equalization_type: str, channels: int, seed: int) -> None:
    pixels = random_pixels(channels, 40, 52, seed=seed)

    equalized = equalize(pixels, equalization_type)

    expected = skimage_reference(pixels, equalization_type)
    assert np.array_equal(equalized.numpy(), expected)


@pytest.mark.parametrize("equalization_type", TYPES[:2])
def test_uniform_image_equals_scikit_image(equalization_type: str) -> None:
    pixels = torch.full((3, 6, 5), 42, dtype=torch.uint8)

    equalized = equalize(pixels, equalization_type)

    assert np.array_equal(
        equalized.numpy(), skimage_reference(pixels, equalization_type)
    )


def test_keeps_identity_geometry_and_metadata_and_leaves_the_input() -> None:
    # given
    canvas = random_pixels(3, 40, 40, seed=9)
    root = ImageData.from_tensor(canvas, image_id="root", video_metadata=VIDEO)
    crop = root.crop((6, 6, 36, 36)).crop((3, 2, 23, 22), image_id="inner")
    crop = crop.with_parent_row(RowRef("detections", 5))
    before = crop.tensor_image.clone()

    for equalization_type in TYPES:
        # when
        image = ContrastEqualizationBlock().run(
            image=crop, equalization_type=equalization_type
        )["image"]

        # then
        assert (image.image_id, image.parent, image.root) == (
            "inner",
            crop.parent,
            crop.root,
        )
        assert image.root.offset_xy == (9, 8)
        assert image.video_metadata is VIDEO
        assert image.parent_row == RowRef("detections", 5)
        assert torch.equal(crop.tensor_image, before)


def test_channels_last_view_equals_scikit_image() -> None:
    view = random_pixels(3, 20, 24, seed=1).permute(1, 2, 0).contiguous()
    view = view.permute(2, 0, 1)

    for equalization_type in TYPES:
        equalized = equalize(view, equalization_type)
        expected = skimage_reference(view, equalization_type)
        assert np.array_equal(equalized.numpy(), expected)


def test_unknown_equalization_type_is_rejected() -> None:
    with pytest.raises(ValueError, match="equalization_type must be one of"):
        equalize(torch.zeros((1, 2, 2), dtype=torch.uint8), "Gamma")


def test_declares_a_pure_prunable_block() -> None:
    assert ContrastEqualizationBlock.type == "roboflow_core/contrast_equalization@v1"
    assert ContrastEqualizationBlock.prunable is True
    assert ContrastEqualizationBlock.mutates == ()
