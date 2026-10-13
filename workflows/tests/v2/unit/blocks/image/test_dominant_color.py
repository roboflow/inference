"""Unit tests of ``v2/dominant_color@v1`` on generated color images."""

import pytest
import torch
from roboflow_workflows.v2.blocks.image.dominant_color import DominantColorBlock
from roboflow_workflows.v2.data.image import ImageData

RED, BLUE = (255, 0, 0), (0, 0, 255)


def painted(height, width, color):
    pixels = torch.empty((3, height, width), dtype=torch.uint8)
    pixels[:] = torch.tensor(color, dtype=torch.uint8).reshape(3, 1, 1)

    return pixels


def run(pixels, color_clusters=4, max_iterations=100, target_size=100):
    image = ImageData.from_tensor(pixels, image_id="colors")

    return DominantColorBlock().run(
        image=image,
        color_clusters=color_clusters,
        max_iterations=max_iterations,
        target_size=target_size,
    )


def test_the_larger_area_wins_and_both_formats_agree():
    pixels = painted(40, 40, RED)
    pixels[:, :, 28:] = torch.tensor(BLUE, dtype=torch.uint8).reshape(3, 1, 1)
    before = pixels.clone()

    result = run(pixels)

    assert result == {"rgb_color": RED, "hex_color": "#FF0000"}
    assert torch.equal(pixels, before)


def test_noisy_colors_round_to_the_cluster_mean():
    # Two clusters for two colors; with more, the noisy area would split.
    generator = torch.Generator().manual_seed(3)
    pixels = painted(30, 30, (100, 150, 200)).to(torch.int16)
    pixels[:, :, :20] += torch.randint(-2, 3, (3, 30, 20), generator=generator)
    pixels[:, :, 20:] = 10

    result = run(pixels.to(torch.uint8), color_clusters=2)

    assert all(abs(a - b) <= 1 for a, b in zip(result["rgb_color"], (100, 150, 200)))


def test_sampling_takes_every_step_th_pixel_like_v1():
    # 200 x 200 with target 100: step 2 samples only even rows and columns.
    # Those are red; the other three quarters of the image are blue.
    pixels = painted(200, 200, BLUE)
    pixels[:, ::2, ::2] = torch.tensor(RED, dtype=torch.uint8).reshape(3, 1, 1)

    assert run(pixels)["rgb_color"] == RED
    assert run(pixels, target_size=250)["rgb_color"] == BLUE


def test_same_image_gives_the_same_color():
    generator = torch.Generator().manual_seed(7)
    pixels = torch.randint(0, 256, (3, 64, 48), dtype=torch.uint8, generator=generator)

    results = {run(pixels, color_clusters=6)["hex_color"] for _ in range(3)}

    assert len(results) == 1


def test_fewer_pixels_than_clusters_and_one_channel_images():
    # Three pixels, four clusters: one cluster per pixel; the two red ones join.
    three = painted(1, 3, RED)
    three[:, 0, 2] = torch.tensor(BLUE, dtype=torch.uint8)
    assert run(three, color_clusters=4)["rgb_color"] == RED
    assert run(painted(1, 1, (9, 8, 7)), color_clusters=4)["rgb_color"] == (9, 8, 7)
    gray = torch.full((1, 10, 10), 77, dtype=torch.uint8)
    assert run(gray)["hex_color"] == "#4D4D4D"


def test_noncontiguous_pixels():
    pixels = painted(30, 20, RED).transpose(1, 2)
    assert not pixels.is_contiguous()

    assert run(pixels)["rgb_color"] == RED


@pytest.mark.parametrize(
    "field, value",
    [
        ("color_clusters", 0),
        ("color_clusters", 11),
        ("max_iterations", 501),
        ("target_size", 0),
        ("target_size", 2.5),
    ],
)
def test_invalid_literals_are_rejected(field, value):
    with pytest.raises(Exception, match=field):
        DominantColorBlock.Params.model_validate(
            {"image": "$inputs.image", field: value}
        )
