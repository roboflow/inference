"""Unit tests of ``roboflow_core/auto_rotate_on_edges@v1``, called directly.

Synthetic ruled pages are tilted by a known angle with OpenCV. The block must
find roughly the opposite correction: V1's angle search reads these pages
up to about 6 % low (6.58 degrees for a 7 degree tilt, 29.29 for 30), so
the analytic bound is 1 degree. V1's block on V1's BGR layout is the exact
oracle: angles and canvas sizes are identical, and pixels may differ by 1
level (float bilinear blending, measured on CPU only).
"""

from datetime import datetime, timezone

import cv2
import numpy as np
import pytest
import torch
from roboflow_workflows.core_steps.classical_cv.auto_rotate_on_edges.v1 import (
    AutoRotateOnEdgesBlockV1,
)
from roboflow_workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)
from roboflow_workflows.v2.blocks.image.auto_rotate_on_edges.v1 import AutoRotateOnEdgesBlock, canvas_size, correction_angle, rotation_matrix
from roboflow_workflows.v2.blocks.image.camera_focus.v2 import gray_plane
from roboflow_workflows.v2.data.image import FrameMapping, ImageData
from roboflow_workflows.v2.data.row_identity import RowRef

VIDEO = VideoMetadata(
    video_identifier="scanner",
    frame_number=1,
    frame_timestamp=datetime(2026, 10, 12, tzinfo=timezone.utc),
)


def ruled_page(tilt: float, height: int = 400, width: int = 600) -> torch.Tensor:
    """Horizontal ruled lines rotated counter-clockwise by ``tilt`` degrees."""
    canvas = np.full((height + 400, width + 400, 3), 235, np.uint8)
    for y in range(40, height + 400, 24):
        cv2.line(canvas, (0, y), (width + 400, y), (30, 60, 90), 2)
    matrix = cv2.getRotationMatrix2D(((width + 400) / 2, (height + 400) / 2), tilt, 1)
    canvas = cv2.warpAffine(
        canvas, matrix, (width + 400, height + 400), borderValue=(235, 235, 235)
    )
    page = canvas[200 : 200 + height, 200 : 200 + width]
    pixels = torch.from_numpy(np.ascontiguousarray(page.transpose(2, 0, 1)))

    return pixels


def straighten(image: ImageData, **settings) -> dict:
    values = dict(
        target_orientation="horizontal",
        skip_below_degrees=0.4,
        max_correction_degrees=90.0,
        internal_resolution=1000,
    )
    values.update(settings)
    result = AutoRotateOnEdgesBlock().run(image=image, **values)

    return result


@pytest.mark.parametrize(
    ("tilt", "orientation", "resolution"),
    [
        (7, "horizontal", 1000),
        (-12.5, "horizontal", 1000),
        (3, "vertical", 1000),
        (20, "either", 300),
        (0.2, "horizontal", 1000),
    ],
)
def test_angle_canvas_and_pixels_match_v1(tilt, orientation, resolution) -> None:
    pixels = ruled_page(tilt)
    bgr = np.ascontiguousarray(pixels.permute(1, 2, 0).numpy()[..., ::-1])
    settings = dict(
        target_orientation=orientation,
        skip_below_degrees=0.4,
        max_correction_degrees=90.0,
        internal_resolution=resolution,
    )
    v1 = AutoRotateOnEdgesBlockV1().run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="v1"), numpy_image=bgr
        ),
        **settings,
    )

    result = straighten(ImageData.from_tensor(pixels), **settings)

    assert result["angle"] == v1["angle"]
    expected = v1["image"].numpy_image[..., ::-1].transpose(2, 0, 1)
    rotated = result["image"].tensor_image.numpy()
    assert rotated.shape == expected.shape
    assert np.abs(rotated.astype(int) - expected.astype(int)).max() <= 1


@pytest.mark.parametrize("tilt", [7.0, -12.5, 30.0])
def test_tilted_lines_come_back_horizontal(tilt: float) -> None:
    result = straighten(ImageData.from_tensor(ruled_page(tilt)))

    assert result["angle"] == pytest.approx(-tilt, abs=1.0)
    remaining = correction_angle(
        gray_plane(result["image"].tensor_image),
        target_orientation="horizontal",
        internal_resolution=1000,
    )
    assert abs(remaining) < 1.0


def test_canvas_holds_the_whole_rotated_image() -> None:
    matrix = rotation_matrix(300, 200, 7.0)
    width, height = canvas_size(300, 200, matrix=matrix)
    corners = np.array([[0, 0, 1], [299, 0, 1], [299, 199, 1], [0, 199, 1]])

    mapped = corners @ matrix.T

    assert (width, height) == (322, 235)
    assert mapped[:, 0].min() >= -1 and mapped[:, 0].max() <= width
    assert mapped[:, 1].min() >= -1 and mapped[:, 1].max() <= height
    assert np.allclose(matrix @ [150, 100, 1], [width / 2, height / 2])


def test_rotated_image_is_a_new_root_that_keeps_source_time_and_parent_row() -> None:
    image = ImageData.from_tensor(
        ruled_page(7), image_id="page", video_metadata=VIDEO
    ).with_parent_row(RowRef("pages", 3))
    before = image.tensor_image.clone()

    rotated = straighten(image)["image"]

    assert rotated.image_id != image.image_id
    assert rotated.root == FrameMapping.identity(rotated.image_id, rotated.size_hw)
    assert rotated.parent == rotated.root
    assert rotated.size_hw != image.size_hw
    assert rotated.video_metadata == VIDEO
    assert rotated.parent_row == RowRef("pages", 3)
    assert torch.equal(image.tensor_image, before)


@pytest.mark.parametrize(
    ("pixels", "settings"),
    [
        (torch.full((3, 200, 300), 128, dtype=torch.uint8), {}),
        (ruled_page(0.2), {}),
        (ruled_page(7), {"max_correction_degrees": 5.0}),
        (ruled_page(7), {"skip_below_degrees": 10.0}),
    ],
    ids=["flat", "below-skip", "above-max", "raised-skip"],
)
def test_skipped_corrections_return_the_input_with_angle_zero(pixels, settings) -> None:
    image = ImageData.from_tensor(pixels)

    result = straighten(image, **settings)

    assert result == {"image": image, "angle": 0.0}


def test_gray_images_rotate_on_one_channel() -> None:
    gray = ruled_page(7)[:1].contiguous()

    result = straighten(ImageData.from_tensor(gray))

    assert result["angle"] == pytest.approx(-7, abs=1.0)
    assert result["image"].tensor_image.shape[0] == 1


def test_unknown_orientation_is_refused() -> None:
    with pytest.raises(ValueError, match="target_orientation"):
        straighten(ImageData.from_tensor(ruled_page(7)), target_orientation="diagonal")
