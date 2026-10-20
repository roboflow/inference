"""Unit tests of ``roboflow_core/contours_detection@v1`` against OpenCV references."""

import cv2
import numpy as np
import pytest
import torch
from roboflow_workflows.v2.blocks.geometry.contours_detection.v1 import ContoursDetectionBlock
from roboflow_workflows.v2.data.image import ImageData


def two_blobs():
    # Blob A: x 5..24, y 5..14. Blob B: x 30..44, y 30..39 (gray 200).
    pixels = torch.zeros((3, 50, 50), dtype=torch.uint8)
    pixels[:, 5:15, 5:25] = 255
    pixels[:, 30:40, 30:45] = 200

    return pixels


def run(pixels, line_thickness=3, **image_options):
    image = ImageData.from_tensor(pixels, image_id="blobs", **image_options)

    return ContoursDetectionBlock().run(image=image, line_thickness=line_thickness)


def opencv_drawing(pixels, thickness):
    # V1's drawing, computed with OpenCV on an RGB copy of the input.
    rgb = np.ascontiguousarray(pixels.permute(1, 2, 0).numpy())
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    drawn = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(drawn, contours, -1, (255, 0, 255), thickness)

    return torch.from_numpy(drawn).permute(2, 0, 1)


def test_two_blobs_give_two_corner_contours_and_a_flat_hierarchy():
    result = run(two_blobs())

    assert result["number_contours"] == 2
    corners = sorted(sorted(map(tuple, c.tolist())) for c in result["contours"])
    assert corners == [
        [(5, 5), (5, 14), (24, 5), (24, 14)],
        [(30, 30), (30, 39), (44, 30), (44, 39)],
    ]
    assert all(c.dtype == torch.int32 and c.shape == (4, 2) for c in result["contours"])
    hierarchy = result["hierarchy"]
    assert hierarchy.dtype == torch.int32 and hierarchy.shape == (2, 4)
    assert hierarchy[:, 3].tolist() == [-1, -1]  # no parents


@pytest.mark.parametrize("thickness", [1, 3])
def test_drawing_matches_opencv_on_the_gray_image(thickness):
    # Random colors everywhere: the gray rounding must equal OpenCV's on every
    # pixel, not only on the flat blobs.
    generator = torch.Generator().manual_seed(1)
    pixels = torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8, generator=generator)
    pixels[:, 20:30, 20:40] = 0

    result = run(pixels, line_thickness=thickness)

    assert torch.equal(result["image"].tensor_image, opencv_drawing(pixels, thickness))


def test_input_is_not_modified_and_identity_and_transforms_are_kept():
    root = ImageData.from_tensor(two_blobs(), image_id="root")
    crop = root.crop((2, 2, 48, 48)).resize((92, 92), interpolation="nearest")
    before = crop.tensor_image.clone()

    result = ContoursDetectionBlock().run(image=crop, line_thickness=1)

    drawn = result["image"]
    assert torch.equal(crop.tensor_image, before)
    assert (drawn.image_id, drawn.parent, drawn.root) == (
        crop.image_id,
        crop.parent,
        crop.root,
    )
    # Contours are in the crop's own pixels: blob A starts at (5 - 2) * 2.
    assert min(c[:, 0].min().item() for c in result["contours"]) == 6


def test_empty_image_has_no_contours():
    result = run(torch.zeros((3, 20, 30), dtype=torch.uint8))

    assert result["number_contours"] == 0 and result["contours"] == []
    assert result["hierarchy"].shape == (0, 4)
    assert not result["image"].tensor_image.any()


def test_one_channel_and_noncontiguous_inputs():
    gray = two_blobs()[:1]
    transposed = two_blobs().transpose(1, 2)
    assert not transposed.is_contiguous()

    single = run(gray, line_thickness=1)
    swapped = run(transposed, line_thickness=1)

    assert single["image"].channels == 3
    assert torch.equal(single["image"].tensor_image, opencv_drawing(two_blobs(), 1))
    assert swapped["number_contours"] == 2


@pytest.mark.parametrize("value", [0, -1, 2.0])
def test_invalid_line_thickness_is_rejected(value):
    with pytest.raises(Exception, match="line_thickness"):
        ContoursDetectionBlock.Params.model_validate(
            {"image": "$inputs.image", "line_thickness": value}
        )
