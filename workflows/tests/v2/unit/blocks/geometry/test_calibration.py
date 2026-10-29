"""Unit tests of ``roboflow_core/camera-calibration@v1``, called directly.

Two independent oracles: V1's ``remove_distortions`` on V1's BGR layout
(pixels may differ by 1 level: float bilinear blending, measured on CPU only),
and the lens model written out by hand: a dot placed where the lens images a
point must come back where the undistorted camera images it.
"""

from datetime import datetime, timezone

import cv2
import numpy as np
import pytest
import torch
from roboflow_workflows.core_steps.transformations.camera_calibration.v1 import (
    remove_distortions,
)
from roboflow_workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)
from roboflow_workflows.v2.blocks.geometry.camera_calibration.v1 import CameraCalibrationBlock
from roboflow_workflows.v2.data.image import FrameMapping, ImageData
from roboflow_workflows.v2.data.row_identity import RowRef

VIDEO = VideoMetadata(
    video_identifier="cam-1",
    frame_number=3,
    frame_timestamp=datetime(2026, 10, 12, tzinfo=timezone.utc),
)
INTRINSICS = dict(fx=300.0, fy=310.0, cx=161.3, cy=118.7)
ZERO_DISTORTION = dict(k1=0.0, k2=0.0, k3=0.0, p1=0.0, p2=0.0)
STANDARD = dict(k1=-0.3, k2=0.08, k3=-0.01, p1=0.002, p2=-0.001)


def checkerboard(height: int = 240, width: int = 320) -> torch.Tensor:
    ys, xs = np.mgrid[0:height, 0:width]
    board = (((xs // 20) + (ys // 20)) % 2 * 200 + 30).astype(np.uint8)
    pixels = torch.from_numpy(np.stack([board, 255 - board, board // 2]))

    return pixels


def calibrate(image: ImageData, **settings) -> ImageData:
    values = {**INTRINSICS, **ZERO_DISTORTION, "use_fisheye_model": False}
    values.update(settings)
    result = CameraCalibrationBlock().run(image=image, **values)

    return result["calibrated_image"]


def v1_calibrate(pixels: torch.Tensor, **settings) -> np.ndarray:
    values = {**INTRINSICS, **ZERO_DISTORTION, "use_fisheye_model": False}
    values.update(settings)
    bgr = np.ascontiguousarray(pixels.permute(1, 2, 0).numpy()[..., ::-1])
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="v1"), numpy_image=bgr
    )
    undistorted = remove_distortions(image=image, **values).numpy_image[..., ::-1]
    expected = np.ascontiguousarray(undistorted.transpose(2, 0, 1))

    return expected


@pytest.mark.parametrize(
    "settings",
    [
        STANDARD,
        {"k1": 0.15, "use_fisheye_model": True},
        {"k1": -0.2, "use_fisheye_model": True},
        {"k1": 0.05, "p2": 0.01},
    ],
)
def test_undistortion_matches_v1_within_one_level(settings) -> None:
    pixels = checkerboard()

    calibrated = calibrate(ImageData.from_tensor(pixels), **settings)

    expected = v1_calibrate(pixels, **settings)
    difference = np.abs(calibrated.tensor_image.numpy().astype(int) - expected)
    assert difference.max() <= 1
    assert (difference > 0).mean() < 0.02


def test_a_lens_imaged_dot_returns_to_its_undistorted_position() -> None:
    # Forward lens model by hand: undistorted normalized point -> distorted pixel.
    k1, k2, k3, p1, p2 = (STANDARD[name] for name in ("k1", "k2", "k3", "p1", "p2"))
    fx, fy, cx, cy = (INTRINSICS[name] for name in ("fx", "fy", "cx", "cy"))
    camera = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    distortion = np.array([k1, k2, p1, p2, k3], dtype=np.float32)
    new_camera, _ = cv2.getOptimalNewCameraMatrix(
        camera, distortion, (320, 240), 1, (320, 240)
    )
    for x, y in [(-0.3, -0.2), (0.25, 0.1), (0.0, 0.0), (0.35, -0.25)]:
        r2 = x * x + y * y
        radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
        x_d = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        y_d = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        column, row = round(fx * x_d + cx), round(fy * y_d + cy)
        pixels = torch.zeros((1, 240, 320), dtype=torch.uint8)
        pixels[0, row - 1 : row + 2, column - 1 : column + 2] = 255
        expected_x = new_camera[0, 0] * x + new_camera[0, 2] + (column - fx * x_d - cx)
        expected_y = new_camera[1, 1] * y + new_camera[1, 2] + (row - fy * y_d - cy)

        calibrated = calibrate(ImageData.from_tensor(pixels), **STANDARD)

        weights = calibrated.tensor_image[0].double()
        ys, xs = torch.meshgrid(
            torch.arange(240, dtype=torch.float64),
            torch.arange(320, dtype=torch.float64),
            indexing="ij",
        )
        found_x = float((weights * xs).sum() / weights.sum())
        found_y = float((weights * ys).sum() / weights.sum())
        assert abs(found_x - expected_x) < 1.0
        assert abs(found_y - expected_y) < 1.0


def test_default_coefficients_are_zero_distortion() -> None:
    fields = CameraCalibrationBlock.Params.model_fields
    defaults = {name: fields[name].default for name in ZERO_DISTORTION}
    pixels = checkerboard()

    calibrated = calibrate(ImageData.from_tensor(pixels), **defaults)

    assert defaults == ZERO_DISTORTION
    difference = np.abs(
        calibrated.tensor_image.numpy().astype(int) - v1_calibrate(pixels)
    )
    assert difference.max() <= 1
    near_identity = np.abs(calibrated.tensor_image.numpy().astype(int) - pixels.numpy())
    assert (near_identity > 0).mean() < 0.05


def test_fisheye_without_k1_is_the_identity() -> None:
    pixels = checkerboard()

    calibrated = calibrate(ImageData.from_tensor(pixels), use_fisheye_model=True)

    assert torch.equal(calibrated.tensor_image, pixels)


def test_result_is_a_new_root_that_keeps_source_time_and_parent_row() -> None:
    gray = checkerboard()[:1]
    image = ImageData.from_tensor(
        gray, image_id="camera", video_metadata=VIDEO
    ).with_parent_row(RowRef("cameras", 0))
    before = image.tensor_image.clone()

    calibrated = calibrate(image, **STANDARD)

    assert calibrated.image_id != image.image_id
    assert calibrated.size_hw == image.size_hw
    assert calibrated.tensor_image.shape == (1, 240, 320)
    assert calibrated.tensor_image.dtype == torch.uint8
    assert calibrated.tensor_image.device == image.tensor_image.device
    assert calibrated.root == FrameMapping.identity(calibrated.image_id, (240, 320))
    assert calibrated.parent == calibrated.root
    assert calibrated.video_metadata == VIDEO
    assert calibrated.parent_row == RowRef("cameras", 0)
    assert torch.equal(image.tensor_image, before)


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"fx": 0.0}, "focal length fx must be positive"),
        ({"fy": -2.0}, "focal length fy must be positive"),
        ({"k1": float("nan")}, "k1 must be a finite number"),
        ({"cx": float("inf")}, "cx must be a finite number"),
    ],
)
def test_invalid_selected_values_are_refused(settings, message) -> None:
    with pytest.raises(ValueError, match=message):
        calibrate(ImageData.from_tensor(checkerboard()), **settings)
