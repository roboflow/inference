"""Unit tests of ``v2/sift@v1`` against OpenCV references."""

import cv2
import numpy as np
import torch
from roboflow_workflows.v2.blocks.geometry.sift import KEYPOINT_COLUMNS, SiftBlock
from roboflow_workflows.v2.data.image import ImageData


def texture(size=200, block=8, seed=0):
    # Random gray blocks with color tints: corners everywhere, deterministic.
    generator = torch.Generator().manual_seed(seed)
    cells = torch.randint(
        0,
        256,
        (3, size // block, size // block),
        dtype=torch.uint8,
        generator=generator,
    )

    return cells.repeat_interleave(block, dim=1).repeat_interleave(block, dim=2)


def run(pixels, image_id="texture"):
    return SiftBlock().run(image=ImageData.from_tensor(pixels, image_id=image_id))


def opencv_sift(pixels):
    # V1's computation on the same pixels: gray, then detectAndCompute.
    rgb = np.ascontiguousarray(pixels.permute(1, 2, 0).numpy())
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    return cv2.SIFT_create().detectAndCompute(gray, None)


def test_keypoints_and_descriptors_equal_opencv():
    pixels = texture()
    result = run(pixels)
    expected_keypoints, expected_descriptors = opencv_sift(pixels)

    keypoints = result["keypoints"]
    assert len(expected_keypoints) > 50
    assert keypoints.dtype == torch.float32
    assert keypoints.shape == (len(expected_keypoints), len(KEYPOINT_COLUMNS))
    assert KEYPOINT_COLUMNS == (
        "x",
        "y",
        "size",
        "angle",
        "response",
        "octave",
        "class_id",
    )
    expected_rows = [
        (*p.pt, p.size, p.angle, p.response, p.octave, p.class_id)
        for p in expected_keypoints
    ]
    assert torch.equal(keypoints, torch.tensor(expected_rows, dtype=torch.float32))
    assert torch.equal(result["descriptors"], torch.from_numpy(expected_descriptors))
    assert result["descriptors"].shape == (len(expected_keypoints), 128)


def test_shifted_image_moves_matched_keypoints_by_the_shift():
    # The same texture seen through two windows 12 px right and 7 px down.
    base, dx, dy = texture(), 12, 7
    first = run(base[:, :160, :160].contiguous())
    second = run(base[:, dy : dy + 160, dx : dx + 160].contiguous())

    matches = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True).match(
        first["descriptors"].numpy(), second["descriptors"].numpy()
    )
    shifts = torch.stack(
        [
            first["keypoints"][m.queryIdx, :2] - second["keypoints"][m.trainIdx, :2]
            for m in matches
        ]
    )

    assert len(matches) > 20
    median_shift = shifts.median(dim=0).values
    assert torch.allclose(
        median_shift, torch.tensor([dx, dy], dtype=torch.float32), atol=0.5
    )


def test_flat_image_gives_empty_tensors_and_a_gray_drawing():
    pixels = torch.full((3, 40, 30), 90, dtype=torch.uint8)

    result = run(pixels)

    assert result["keypoints"].shape == (0, 7)
    assert result["descriptors"].shape == (0, 128)
    assert result["descriptors"].dtype == torch.float32
    assert torch.equal(result["image"].tensor_image, pixels)


def test_one_channel_image_is_used_as_gray():
    rgb = texture()
    gray = torch.from_numpy(
        cv2.cvtColor(
            np.ascontiguousarray(rgb.permute(1, 2, 0).numpy()), cv2.COLOR_RGB2GRAY
        )
    ).unsqueeze(0)

    from_rgb = run(rgb)
    from_gray = run(gray)

    assert torch.equal(from_gray["keypoints"], from_rgb["keypoints"])
    assert from_gray["image"].tensor_image.shape == (3, 200, 200)


def test_drawing_keeps_the_frame_and_does_not_touch_the_input():
    pixels = texture()
    before = pixels.clone()
    image = ImageData.from_tensor(pixels, image_id="frame-7").crop((10, 20, 170, 180))

    result = SiftBlock().run(image=image)

    assert torch.equal(pixels, before)
    drawn = result["image"]
    assert drawn.image_id == image.image_id
    assert drawn.root == image.root and drawn.parent == image.parent
    assert drawn.size_hw == (160, 160) and drawn.channels == 3
    assert not torch.equal(drawn.tensor_image, image.tensor_image)
