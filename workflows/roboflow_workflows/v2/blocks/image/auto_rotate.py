"""Straighten an image by its dominant edges: ``v2/auto_rotate_on_edges@v1``.

Port of V1 ``roboflow_core/auto_rotate_on_edges@v1``::

    gray (device) ─▶ host: shrink to internal_resolution (INTER_AREA)
                    ─▶ flat image (std < 3)?            ─▶ unchanged, angle 0
                    ─▶ CLAHE, 5x5 blur, 3x3 Sobel
                    ─▶ histogram of gradient direction (mod 180, weighted by
                       magnitude, strongest 20 % of pixels)
                    ─▶ no clear peak?                   ─▶ unchanged, angle 0
                    ─▶ peak refined by a weighted mean within 3 degrees
    angle  = correction in (-90, 90], or (-45, 45] for "either"
    |angle| < skip_below_degrees or > max_correction_degrees ─▶ unchanged, 0
    else   rotate counter-clockwise by angle on a canvas grown to hold the
           whole image (OpenCV's getRotationMatrix2D about (W // 2, H // 2)),
           black corners, on the image's device

The angle search is OpenCV CPU work on one small grayscale plane: the full
plane is copied to the host once and shrunk there, which keeps the angle
identical to V1. The full-size rotation stays on the device.

A rotated image is a new root frame: a rotation cannot be written as an
image provenance mapping (scale and offset). It keeps source, time and
``parent_row``; predictions made on it stay in its frame. An unchanged image
keeps its provenance. ``rotation_matrix`` rebuilds the transform from the
input size and ``angle`` for callers that map points back themselves.

Differences from V1: V1 kept the input's provenance on the rotated image,
so predictions mapped back to wrong places. Rotated pixels use float
bilinear blending at OpenCV's 1/32-pixel positions; a pixel may differ by 1
level.
"""

import uuid
from typing import Any, Dict, Literal, Optional

import cv2
import numpy as np
import torch
from pydantic import AliasChoices, Field, StrictInt
from roboflow_workflows.v2.blocks.image._resample import (
    compute_dtype,
    new_root_image,
    remap_bilinear,
)
from roboflow_workflows.v2.blocks.image.camera_focus import gray_plane
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import FLOAT_KIND, INTEGER_KIND, STRING_KIND
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref

ORIENTATIONS = ("vertical", "horizontal", "either")
# V1 constants of the angle search.
FLAT_IMAGE_STD = 3.0
STRONG_GRADIENT_PERCENTILE = 80.0
MIN_PEAK_RATIO = 1.5
REFINE_WINDOW_DEGREES = 3.0
# cv2.warpAffine positions: 1/1024-pixel terms, rounded to 1/32 pixel.
WARP_SCALE = 1024
WARP_ROUND = WARP_SCALE // 32 // 2


class AutoRotateOnEdgesBlock(Block):
    """Deskew an image so its dominant lines are axis-aligned.

    Example: a 600x400 page whose ruled lines are tilted 7 degrees
    clockwise comes back rotated counter-clockwise by ``angle`` of about
    6.6 onto a larger canvas, with near-horizontal lines. The search reads
    steep tilts a few percent low, exactly as V1's does.
    """

    type = "v2/auto_rotate_on_edges@v1"
    outputs = {
        "image": Output(
            IMAGE_KIND,
            source="image",
            description=(
                "The deskewed image (a new root frame), or the input when unchanged."
            ),
        ),
        "angle": Output(
            FLOAT_KIND,
            description=(
                "Applied counter-clockwise rotation in degrees; 0.0 when unchanged."
            ),
        ),
    }
    prunable = True

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(
            validation_alias=AliasChoices("image", "images"),
            description="Image to straighten.",
        )
        target_orientation: Literal["vertical", "horizontal", "either"] | Ref(
            STRING_KIND
        ) = Field(
            default="vertical",
            description=(
                "Align the dominant lines vertically, horizontally, or to the "
                "nearer of the two ('either', at most 45 degrees)."
            ),
        )
        skip_below_degrees: float | Ref(FLOAT_KIND) = Field(
            default=0.4,
            ge=0,
            description="Leave the image unchanged below this correction.",
        )
        max_correction_degrees: float | Ref(FLOAT_KIND) = Field(
            default=90.0,
            ge=0,
            description=(
                "Leave the image unchanged above this correction; 90 disables "
                "the limit."
            ),
        )
        internal_resolution: StrictInt | Ref(INTEGER_KIND) = Field(
            default=1000,
            ge=1,
            description=(
                "Longest side, in pixels, of the copy the angle is searched on."
            ),
        )

    def run(
        self,
        *,
        image: ImageData,
        target_orientation: str,
        skip_below_degrees: float,
        max_correction_degrees: float,
        internal_resolution: int,
    ) -> Dict[str, Any]:
        """Find the correction angle of ``image`` and apply it.

        Args:
            image: Image to straighten.
            target_orientation: ``"vertical"``, ``"horizontal"`` or ``"either"``.
            skip_below_degrees: Smallest correction applied.
            max_correction_degrees: Largest correction applied.
            internal_resolution: Longest side of the search copy.

        Returns:
            ``image`` and ``angle``.

        Raises:
            ValueError: On an unknown orientation.
        """
        if target_orientation not in ORIENTATIONS:
            raise ValueError(
                f"target_orientation must be one of {list(ORIENTATIONS)}, got "
                f"{target_orientation!r}"
            )

        angle = correction_angle(
            gray_plane(image.tensor_image),
            target_orientation=target_orientation,
            internal_resolution=internal_resolution,
        )
        if (
            angle is None
            or abs(angle) < skip_below_degrees
            or abs(angle) > max_correction_degrees
        ):
            return {"image": image, "angle": 0.0}

        pixels = rotate_pixels(image.tensor_image, degrees=angle)
        rotated = new_root_image(
            pixels, like=image, image_id=f"auto-rotate-{uuid.uuid4().hex}"
        )
        result = {"image": rotated, "angle": float(angle)}

        return result


def correction_angle(
    gray: torch.Tensor, *, target_orientation: str, internal_resolution: int
) -> Optional[float]:
    """Return V1's correction angle for a grayscale image.

    Args:
        gray: ``(H, W)`` uint8 grayscale, on any device.
        target_orientation: ``"vertical"``, ``"horizontal"`` or ``"either"``.
        internal_resolution: Longest side of the search copy.

    Returns:
        Counter-clockwise degrees in ``(-90, 90]`` (``(-45, 45]`` for
        ``"either"``), or ``None`` for a flat image or no dominant direction.
    """
    small = _shrink(gray.cpu().numpy(), longest_side=internal_resolution)
    if float(small.std()) < FLAT_IMAGE_STD:
        return None

    enhanced = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(small)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0).astype(np.float32)
    gradient_angle = _dominant_gradient_angle(blurred)
    if gradient_angle is None:
        return None

    # Vertical lines have horizontal gradients (0 degrees); horizontal lines
    # have vertical ones. "either" folds onto the nearer axis below.
    if target_orientation == "horizontal":
        gradient_angle -= 90.0
    period = 90.0 if target_orientation == "either" else 180.0
    angle = gradient_angle % period
    if angle > period / 2:
        angle -= period

    return angle


def rotation_matrix(width: int, height: int, degrees: float) -> np.ndarray:
    """Build V1's canvas-growing rotation, from input to output pixels.

    Args:
        width: Input width.
        height: Input height.
        degrees: Counter-clockwise angle.

    Returns:
        ``(2, 3)`` affine matrix; its output canvas is ``canvas_size``.
    """
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, degrees, 1.0)
    new_width, new_height = canvas_size(width, height, matrix=matrix)
    matrix[0, 2] += new_width / 2 - center[0]
    matrix[1, 2] += new_height / 2 - center[1]

    return matrix


def canvas_size(width: int, height: int, *, matrix: np.ndarray) -> tuple:
    """Return the ``(width, height)`` of the canvas holding the rotated image.

    Args:
        width: Input width.
        height: Input height.
        matrix: Rotation matrix; only its rotation part is read.

    Returns:
        ``int`` truncations of the rotated bounding box, as V1.
    """
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    size = (int(height * sin + width * cos), int(height * cos + width * sin))

    return size


def rotate_pixels(pixels: torch.Tensor, *, degrees: float) -> torch.Tensor:
    """Rotate counter-clockwise onto a grown canvas, on the image's device.

    Args:
        pixels: ``(C, H, W)`` uint8 image.
        degrees: Counter-clockwise angle.

    Returns:
        ``(C, H', W')`` uint8 image with black corners.
    """
    height, width = int(pixels.shape[-2]), int(pixels.shape[-1])
    matrix = rotation_matrix(width, height, degrees)
    new_width, new_height = canvas_size(width, height, matrix=matrix)
    inverse = cv2.invertAffineTransform(matrix)
    device = pixels.device
    # Positions on MPS (float32) may move by one 1/32 step.
    dtype = compute_dtype(device)
    xs = torch.arange(new_width, dtype=dtype, device=device).view(1, -1)
    ys = torch.arange(new_height, dtype=dtype, device=device).view(-1, 1)
    source_x = _warp_affine_position(inverse[0], xs=xs, ys=ys)
    source_y = _warp_affine_position(inverse[1], xs=xs, ys=ys)
    rotated = remap_bilinear(pixels, source_x=source_x, source_y=source_y)

    return rotated


def _warp_affine_position(
    row: np.ndarray, *, xs: torch.Tensor, ys: torch.Tensor
) -> torch.Tensor:
    # cv2.warpAffine's fixed point: terms rounded to 1/1024 pixel, the sum
    # rounded to 1/32 (AB_BITS = 10, INTER_BITS = 5).
    per_column = torch.round(row[0] * xs * WARP_SCALE)
    per_row = torch.round((row[1] * ys + row[2]) * WARP_SCALE)
    steps = torch.div(
        per_column + per_row + WARP_ROUND, WARP_SCALE // 32, rounding_mode="floor"
    )
    position = steps / 32

    return position


def _shrink(gray: np.ndarray, *, longest_side: int) -> np.ndarray:
    height, width = gray.shape
    largest = max(height, width)
    if largest <= longest_side:
        return gray

    scale = longest_side / float(largest)
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    shrunk = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)

    return shrunk


def _dominant_gradient_angle(prepared: np.ndarray) -> Optional[float]:
    """Gradient direction (degrees mod 180) of the strongest edges, or None."""
    gradient_x = cv2.Sobel(prepared, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(prepared, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gradient_x, gradient_y)
    threshold = max(float(np.percentile(magnitude, STRONG_GRADIENT_PERCENTILE)), 1e-6)
    strong = magnitude >= threshold
    if not np.any(strong):
        return None

    theta = np.mod(np.degrees(np.arctan2(gradient_y[strong], gradient_x[strong])), 180)
    weights = magnitude[strong].astype(np.float64)
    histogram, _ = np.histogram(theta, bins=180, range=(0.0, 180.0), weights=weights)
    # Circular 5-bin moving average.
    wrapped = np.concatenate([histogram[-2:], histogram, histogram[:2]])
    smoothed = np.convolve(wrapped, np.ones(5) / 5.0, mode="same")[2:-2]
    peak_ratio = (float(smoothed.max()) - float(np.median(smoothed))) / (
        float(smoothed.std()) + 1e-9
    )
    if peak_ratio < MIN_PEAK_RATIO:
        return None

    peak = float(np.argmax(smoothed)) + 0.5
    distance = np.abs(theta - peak)
    near = np.minimum(distance, 180.0 - distance) <= REFINE_WINDOW_DEGREES
    if not np.any(near):
        return peak

    # Weighted circular mean of doubled angles, so 179 and 1 average to 0.
    doubled = np.radians(2.0 * theta[near])
    near_weights = weights[near]
    refined = 0.5 * np.degrees(
        np.arctan2(
            float((near_weights * np.sin(doubled)).sum()),
            float((near_weights * np.cos(doubled)).sum()),
        )
    )

    return refined
