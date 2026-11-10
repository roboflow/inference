"""``roboflow_core/pixel_color_count@v1``: inclusive per-channel ranges and colour forms."""

import numpy as np
import pytest
import torch
from roboflow_workflows.v2.blocks.measurements import PixelColorCountBlock
from roboflow_workflows.v2.blocks.measurements.pixel_color_count.v1 import parse_rgb_color
from roboflow_workflows.v2.data.image import ImageData


def image_with_patches():
    # 10 x 20 black image: a 3 x 4 patch of (200, 30, 40) and a 2 x 2 patch of
    # (205, 30, 40), 5 above the first in red only.
    pixels = np.zeros((10, 20, 3), dtype=np.uint8)
    pixels[1:4, 1:5] = (200, 30, 40)
    pixels[6:8, 10:12] = (205, 30, 40)
    image = ImageData.from_numpy_rgb(pixels)

    return image


def count(image, target_color, tolerance=0):
    result = PixelColorCountBlock().run(
        image=image, target_color=target_color, tolerance=tolerance
    )

    return result["matching_pixels_count"]


def test_exact_colour_counts_its_patch() -> None:
    assert count(image_with_patches(), (200, 30, 40)) == 12


@pytest.mark.parametrize(
    "tolerance, expected",
    [(4, 12), (5, 16), (-1, 0)],
    ids=["below-second-patch", "bound-is-inclusive", "negative-matches-nothing"],
)
def test_tolerance_is_an_inclusive_per_channel_range(tolerance, expected) -> None:
    assert count(image_with_patches(), (200, 30, 40), tolerance) == expected


def test_bounds_outside_0_to_255_are_no_limit() -> None:
    # Black: 0 - 10 < 0 is no lower limit; every pixel outside the patches.
    assert count(image_with_patches(), (0, 0, 0), 10) == 200 - 12 - 4
    # 250 + 10 > 255: no upper limit in red, but green/blue must be 0..20.
    assert count(image_with_patches(), (250, 0, 0), 10) == 0
    assert count(image_with_patches(), (250, 30, 40), 50) == 16


@pytest.mark.parametrize(
    "color",
    ["#C81E28", "#c81e28", "(200, 30, 40)", (200, 30, 40), [200, 30, 40]],
)
def test_colour_forms_are_rgb(color) -> None:
    assert count(image_with_patches(), color) == 12


def test_channel_order_is_rgb_not_bgr() -> None:
    assert count(image_with_patches(), (40, 30, 200)) == 0


def test_short_hex_repeats_each_digit() -> None:
    assert parse_rgb_color("#C1E") == (0xCC, 0x11, 0xEE)


@pytest.mark.parametrize(
    "color, message",
    [
        ("#GG0000", "Invalid hex color format"),
        ("#12345", "Invalid color format"),
        ("red", "Invalid color format"),
        ("(1, 2, x)", "Invalid tuple color format"),
        ("(1, 2)", "expected three integers"),
        ((1, 2), "Invalid color format"),
        ([1.5, 2, 3], "expected three integers"),
    ],
)
def test_invalid_colours_raise(color, message) -> None:
    with pytest.raises(ValueError, match=message.replace("(", r"\(")):
        count(image_with_patches(), color)


def test_one_channel_image_is_rejected() -> None:
    gray = ImageData.from_tensor(torch.zeros((1, 4, 4), dtype=torch.uint8))

    with pytest.raises(ValueError, match="1 channel"):
        count(gray, "#000000")


def test_count_matches_an_independent_numpy_reference() -> None:
    generator = np.random.default_rng(7)
    pixels = generator.integers(0, 256, size=(31, 17, 3), dtype=np.uint8)
    target, tolerance = np.array([120, 60, 200]), 70
    expected = int(
        np.all(np.abs(pixels.astype(int) - target) <= tolerance, axis=2).sum()
    )

    result = count(ImageData.from_numpy_rgb(pixels), tuple(target.tolist()), tolerance)

    assert expected > 0
    assert result == expected


def test_image_is_not_modified() -> None:
    image = image_with_patches()
    before = image.tensor_image.clone()

    count(image, "#C81E28", 20)

    assert torch.equal(image.tensor_image, before)
