"""Remove lens distortion with known intrinsics: ``v2/camera_calibration@v1``.

Port of V1 ``roboflow_core/camera-calibration@v1``. Each output pixel samples
the input where the lens model says that pixel was imaged::

    standard (OpenCV)  K' = getOptimalNewCameraMatrix(K, d, alpha=1)
                       x, y = K'^-1 (u, v);  r2 = x^2 + y^2
                       radial = 1 + k1 r2 + k2 r2^2 + k3 r2^3
                       x_d = x radial + 2 p1 x y + p2 (r2 + 2 x^2)
                       y_d = y radial + p1 (r2 + 2 y^2) + 2 p2 x y
                       source = K (x_d, y_d)
    fisheye (division) x, y = K^-1 (u, v)
                       scale = (1 - sqrt(max(0, 1 - 4 k1 r2))) / (2 k1 r2)
                       source = K (x scale, y scale)

``K`` holds ``fx``, ``fy``, ``cx``, ``cy``. ``alpha=1`` keeps every input
pixel, so curved borders appear black. The fisheye model uses only ``k1``.

The calibration belongs to one camera: bind each source's own values through
workflow inputs. The block keeps no state between calls. Only the 3x3 ``K'``
is computed on the host (``cv2.getOptimalNewCameraMatrix``); the sampling map
and the remap run on the image's device.

The undistorted image is a new root frame: no scale-and-offset mapping can
describe the lens model, so predictions on it are not mapped back to the
distorted input. It keeps source, time and ``parent_row``. V1 kept the
input's provenance instead, so mapped-back predictions landed in wrong
places.

Other differences from V1: distortion coefficients default to 0 (V1 required
all of them), non-finite or non-positive focal lengths are rejected, and
float bilinear blending may differ from OpenCV's fixed-point weights by 1
level.
"""

import math
import uuid
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import torch
from pydantic import Field, StrictBool
from roboflow_workflows.v2.blocks.image._resample import (
    compute_dtype,
    new_root_image,
    remap_bilinear,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import BOOLEAN_KIND, FLOAT_KIND
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref

# Below this |k1| the fisheye model is the identity; below this r2 a pixel
# sits on the optical axis (V1 thresholds).
FISHEYE_EPSILON = 1e-8


class CameraCalibrationBlock(Block):
    """Undistort an image with known camera intrinsics; see the module docstring.

    Example: with ``fx = fy = 800``, ``cx = 320``, ``cy = 240`` and
    ``k1 = -0.3`` a barrel-distorted 640x480 checkerboard comes back with
    straight lines.
    """

    type = "v2/camera_calibration@v1"
    outputs = {
        "calibrated_image": Output(
            IMAGE_KIND,
            source="image",
            description="The undistorted image, same size, as a new root frame.",
        )
    }
    prunable = True

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="Image taken by the camera.")
        fx: float | Ref(FLOAT_KIND) = Field(
            gt=0, description="Focal length along x, in pixels."
        )
        fy: float | Ref(FLOAT_KIND) = Field(
            gt=0, description="Focal length along y, in pixels."
        )
        cx: float | Ref(FLOAT_KIND) = Field(description="Principal point x, pixels.")
        cy: float | Ref(FLOAT_KIND) = Field(description="Principal point y, pixels.")
        k1: float | Ref(FLOAT_KIND) = Field(
            default=0.0, description="First radial coefficient (fisheye: the only one)."
        )
        k2: float | Ref(FLOAT_KIND) = Field(
            default=0.0, description="Second radial coefficient."
        )
        k3: float | Ref(FLOAT_KIND) = Field(
            default=0.0, description="Third radial coefficient."
        )
        p1: float | Ref(FLOAT_KIND) = Field(
            default=0.0, description="First tangential coefficient."
        )
        p2: float | Ref(FLOAT_KIND) = Field(
            default=0.0, description="Second tangential coefficient."
        )
        use_fisheye_model: StrictBool | Ref(BOOLEAN_KIND) = Field(
            default=False,
            description="Use the one-parameter division (fisheye) model with k1.",
        )

    def run(
        self,
        *,
        image: ImageData,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        k1: float,
        k2: float,
        k3: float,
        p1: float,
        p2: float,
        use_fisheye_model: bool,
    ) -> Dict[str, Any]:
        """Undistort ``image``.

        Args:
            image: Distorted image.
            fx: Focal length along x.
            fy: Focal length along y.
            cx: Principal point x.
            cy: Principal point y.
            k1: First radial coefficient.
            k2: Second radial coefficient.
            k3: Third radial coefficient.
            p1: First tangential coefficient.
            p2: Second tangential coefficient.
            use_fisheye_model: Division model instead of OpenCV's.

        Returns:
            ``calibrated_image``: the undistorted image.

        Raises:
            ValueError: On a non-finite value or a non-positive focal length.
        """
        intrinsics = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
        coefficients = {"k1": k1, "k2": k2, "k3": k3, "p1": p1, "p2": p2}
        _check_values(intrinsics, coefficients)

        if use_fisheye_model:
            source_x, source_y = fisheye_source_map(
                image.size_hw, intrinsics=intrinsics, k1=k1, device=image.device
            )
        else:
            source_x, source_y = undistortion_source_map(
                image.size_hw,
                intrinsics=intrinsics,
                coefficients=coefficients,
                device=image.device,
            )
        pixels = remap_bilinear(
            image.tensor_image, source_x=source_x, source_y=source_y
        )
        calibrated = new_root_image(
            pixels, like=image, image_id=f"calibrated-{uuid.uuid4().hex}"
        )
        result = {"calibrated_image": calibrated}

        return result


def undistortion_source_map(
    size_hw: Tuple[int, int],
    *,
    intrinsics: Dict[str, float],
    coefficients: Dict[str, float],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Input position of every output pixel for OpenCV's standard model.

    Matches ``cv2.undistort`` with ``getOptimalNewCameraMatrix(alpha=1)``,
    as V1 called it (float32 camera matrix and coefficients).

    Args:
        size_hw: Image size ``(height, width)``.
        intrinsics: ``fx``, ``fy``, ``cx``, ``cy``.
        coefficients: ``k1``, ``k2``, ``k3``, ``p1``, ``p2``.
        device: Device of the returned maps.

    Returns:
        ``(source_x, source_y)``, each ``(height, width)``.
    """
    height, width = size_hw
    camera = _camera_matrix(intrinsics, dtype=np.float32)
    distortion = np.array(
        [coefficients[name] for name in ("k1", "k2", "p1", "p2", "k3")],
        dtype=np.float32,
    )
    new_camera, _ = cv2.getOptimalNewCameraMatrix(
        camera, distortion, (width, height), 1, (width, height)
    )

    # OpenCV builds this map in float64 (float32 where the device lacks it).
    x, y = _normalized_grid(
        size_hw, camera=new_camera, device=device, dtype=compute_dtype(device)
    )
    k1, k2, k3, p1, p2 = (float(value) for value in distortion[[0, 1, 4, 2, 3]])
    r2 = x * x + y * y
    radial = 1 + ((k3 * r2 + k2) * r2 + k1) * r2
    x_distorted = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    y_distorted = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    source = _to_pixels(x_distorted, y_distorted, camera=camera)

    return source


def fisheye_source_map(
    size_hw: Tuple[int, int],
    *,
    intrinsics: Dict[str, float],
    k1: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Input position of every output pixel for V1's division (fisheye) model.

    Args:
        size_hw: Image size ``(height, width)``.
        intrinsics: ``fx``, ``fy``, ``cx``, ``cy``.
        k1: Division coefficient.
        device: Device of the returned maps.

    Returns:
        ``(source_x, source_y)``, each ``(height, width)``.
    """
    # V1 built this map in float32 numpy; the same float32 arithmetic keeps
    # every position on the same 1/32-pixel step.
    camera = _camera_matrix(intrinsics, dtype=np.float64)
    x, y = _normalized_grid(size_hw, camera=camera, device=device, dtype=torch.float32)
    if abs(k1) < FISHEYE_EPSILON:
        source = _to_pixels(x, y, camera=camera)
        return source

    r2 = x * x + y * y
    discriminant = (1 - 4 * k1 * r2).clamp(min=0)
    on_axis = r2.abs() < FISHEYE_EPSILON
    scale = (1 - discriminant.sqrt()) / (2 * k1 * torch.where(on_axis, 1, r2))
    scale = torch.where(on_axis, torch.ones_like(scale), scale)
    source = _to_pixels(x * scale, y * scale, camera=camera)

    return source


def _check_values(intrinsics: Dict[str, float], coefficients: Dict[str, float]) -> None:
    for name, value in {**intrinsics, **coefficients}.items():
        if not math.isfinite(value):
            raise ValueError(
                f"v2/camera_calibration@v1: {name} must be a finite number, got "
                f"{value!r}"
            )
    for name in ("fx", "fy"):
        if intrinsics[name] <= 0:
            raise ValueError(
                f"v2/camera_calibration@v1: focal length {name} must be positive, "
                f"got {intrinsics[name]!r}"
            )


def _camera_matrix(intrinsics: Dict[str, float], *, dtype: type) -> np.ndarray:
    camera = np.array(
        [
            [intrinsics["fx"], 0, intrinsics["cx"]],
            [0, intrinsics["fy"], intrinsics["cy"]],
            [0, 0, 1],
        ],
        dtype=dtype,
    )

    return camera


def _normalized_grid(
    size_hw: Tuple[int, int],
    *,
    camera: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    height, width = size_hw
    ys, xs = torch.meshgrid(
        torch.arange(height, dtype=dtype, device=device),
        torch.arange(width, dtype=dtype, device=device),
        indexing="ij",
    )
    fx, fy = float(camera[0, 0]), float(camera[1, 1])
    cx, cy = float(camera[0, 2]), float(camera[1, 2])
    normalized = ((xs - cx) / fx, (ys - cy) / fy)

    return normalized


def _to_pixels(
    x: torch.Tensor, y: torch.Tensor, *, camera: np.ndarray
) -> Tuple[torch.Tensor, torch.Tensor]:
    fx, fy = float(camera[0, 0]), float(camera[1, 1])
    cx, cy = float(camera[0, 2]), float(camera[1, 2])
    pixels = (x * fx + cx, y * fy + cy)

    return pixels
