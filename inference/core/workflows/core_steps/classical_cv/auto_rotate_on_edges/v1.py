from typing import List, Literal, Optional, Tuple, Type, Union

import cv2
import numpy as np
from pydantic import AliasChoices, ConfigDict, Field, PositiveInt

from inference.core.workflows.core_steps.visualizations.common.base import (
    OUTPUT_IMAGE_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_KIND,
    IMAGE_KIND,
    INTEGER_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

SHORT_DESCRIPTION: str = (
    "Automatically rotate an image so its dominant straight edges become vertical "
    "and/or horizontal."
)
LONG_DESCRIPTION = """
Automatically deskew an image by finding the rotation angle that best aligns the
image's dominant straight edges (e.g. the sides of a document, package, panel, or
shelf) to the vertical axis, the horizontal axis, or whichever of the two is
closest, then rotating the full-resolution image by that angle with an
automatically expanded canvas so nothing is cropped.

## How This Block Works

This block estimates the rotation angle from the image's dominant gradient
orientation, then applies that rotation to the image. The block:

1. Converts the image to grayscale (color images are converted with `BGR2GRAY`)
2. Downscales a working copy so its largest dimension is at most `internal_resolution` pixels (default 1000, for speed;
   the detected angle is scale-invariant and is later applied to the full-resolution
   original)
3. Checks for a flat/low-texture image (very low pixel standard deviation) and, if
   found, skips processing and returns the image unchanged with `angle = 0.0`
4. Enhances local contrast with CLAHE and applies a light Gaussian blur to reduce
   noise sensitivity, producing a float32 image used only for scoring candidate
   angles
5. Computes Sobel gradients once and builds a magnitude-weighted histogram of
   gradient orientations (mod 180 degrees): gradients of the dominant lines
   point perpendicular to the lines themselves, so vertical lines concentrate
   the histogram around 0 degrees and horizontal lines around 90 degrees
6. Rejects indistinct histograms (no clearly dominant orientation) and returns
   the image unchanged
7. Refines the histogram's dominant mode with a doubled-angle weighted
   circular mean over nearby orientations for sub-degree precision, and
   normalizes the correction into `(-90, 90]` degrees for
   `vertical`/`horizontal` or `(-45, 45]` degrees for `either`
8. If the final angle is smaller in magnitude than `skip_below_degrees`, returns the
   image unchanged with `angle = 0.0` (avoids unnecessary re-encoding/resampling for
   already-straight images). Likewise, if the final angle is larger in magnitude
   than `max_correction_degrees`, the image is returned unchanged with
   `angle = 0.0` - a correction beyond the plausible skew range usually means the
   search aligned to a different dominant structure (e.g. a long object
   silhouette) rather than the lines of interest
9. Otherwise rotates the full-resolution original image by the final angle around
    its center, automatically expanding the output canvas so the entire rotated
    image is preserved (matching the canvas-expansion behavior of
    `roboflow_core/image_preprocessing@v1`'s rotate task)

The block outputs both the deskewed image (with expanded canvas) and the applied
rotation angle in degrees. Angles follow OpenCV's convention: positive angles
rotate counter-clockwise.

## Common Use Cases

- **Document and Label Scanning**: Straighten photographed documents, labels, or
  packaging before OCR or downstream detection, improving text and barcode
  reading accuracy
- **Fixed-Camera Inspection**: Correct small mounting-induced tilts in
  fixed-position industrial cameras so downstream measurement blocks (e.g. size or
  distance measurement) operate on axis-aligned imagery
- **Conveyor and Shelf Alignment**: Align images of conveyors, shelving, or panels
  whose edges should be vertical or horizontal, improving the reliability of
  downstream line/edge-based analysis
- **Pre-processing for Detection Models**: Straighten input images before running
  object detection or classification models that are sensitive to rotation

## Connecting to Other Blocks

This block receives an image and produces a deskewed image plus the applied angle:

- **Before detection or classification models** to straighten images prior to
  inference, enabling more reliable model outputs on tilted source imagery
- **Before measurement blocks** (e.g. distance or size measurement) that assume an
  axis-aligned scene
- **With downstream coordinate mapping** - the `angle` output (together with the
  original image's dimensions) fully determines the applied transform, so
  detections made on the rotated image can be mapped back into the original
  image's coordinate space by inverting the rotation matrix built by
  `build_auto_rotate_matrix`
"""


class AutoRotateOnEdgesManifest(WorkflowBlockManifest):
    type: Literal["roboflow_core/auto_rotate_on_edges@v1"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Auto Rotate on Edges",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "classical_computer_vision",
            "search_keywords": [
                "deskew",
                "rotate",
                "orient",
                "straighten",
                "skew",
                "alignment",
            ],
            "ui_manifest": {
                "section": "classical_cv",
                "icon": "far fa-compass",
                "blockPriority": 8,
                "opencv": True,
            },
        }
    )

    image: Selector(kind=[IMAGE_KIND]) = Field(
        title="Input Image",
        description="Input image (color or grayscale) to deskew. Color images are "
        "automatically converted to grayscale for angle detection; the detected "
        "rotation angle is applied to the full-resolution original image. The "
        "output includes both the rotated image (with an automatically expanded "
        "canvas) and the applied `angle` in degrees.",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("image", "images"),
    )
    target_orientation: Union[
        Selector(kind=[STRING_KIND]), Literal["vertical", "horizontal", "either"]
    ] = Field(  # type: ignore
        title="Target Orientation",
        default="vertical",
        description="Which edge direction the image's dominant straight lines "
        "should be aligned to. 'vertical' searches for the rotation that makes "
        "the dominant lines vertical, 'horizontal' searches for the rotation that "
        "makes the dominant lines horizontal, and 'either' aligns to whichever of "
        "the two is closest (search is limited to a +/-45 degree range around the "
        "nearest right angle).",
        examples=["vertical", "horizontal", "either", "$inputs.target_orientation"],
    )
    skip_below_degrees: Union[float, Selector(kind=[FLOAT_KIND])] = Field(  # type: ignore
        title="Skip Below Degrees",
        default=0.4,
        description="If the detected correction angle's absolute value is smaller "
        "than this threshold (in degrees), the block returns the input image "
        "unchanged (identity passthrough) with `angle = 0.0`, avoiding "
        "unnecessary re-encoding/resampling of images that are already "
        "sufficiently straight.",
        examples=[0.4, "$inputs.skip_below_degrees"],
    )
    max_correction_degrees: Union[float, Selector(kind=[FLOAT_KIND])] = Field(  # type: ignore
        title="Max Correction Degrees",
        default=90.0,
        description="If the best correction angle found exceeds this cap (in "
        "absolute degrees), the block returns the input image unchanged "
        "(identity passthrough) with `angle = 0.0`. Set this when the plausible "
        "skew range of your imagery is known (e.g. parts photographed at most "
        "~40 degrees off-axis) so that a different dominant structure in the "
        "image (e.g. a long object silhouette instead of the lines of interest) "
        "cannot cause a wild, incorrect rotation. The default of 90.0 disables "
        "the cap.",
        examples=[45.0, "$inputs.max_correction_degrees"],
    )
    internal_resolution: Union[PositiveInt, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        title="Internal Resolution",
        default=1000,
        description="The rotation-angle search runs on a working copy of the "
        "image downscaled so that its longest side is at most this many "
        "pixels; the full-resolution image is only used for the final "
        "rotation. Lower values make the search faster but can blur very "
        "thin lines; higher values preserve fine-line detail at the cost of "
        "search time.",
        examples=[1000, "$inputs.internal_resolution"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name=OUTPUT_IMAGE_KEY,
                kind=[IMAGE_KIND],
            ),
            OutputDefinition(
                name="angle",
                kind=[FLOAT_KIND],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class AutoRotateOnEdgesBlockV1(WorkflowBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_manifest(cls) -> Type[AutoRotateOnEdgesManifest]:
        return AutoRotateOnEdgesManifest

    def run(
        self,
        image: WorkflowImageData,
        target_orientation: str,
        skip_below_degrees: float,
        max_correction_degrees: float = 90.0,
        internal_resolution: int = 1000,
        *args,
        **kwargs,
    ) -> BlockResult:
        if target_orientation not in ("vertical", "horizontal", "either"):
            raise ValueError(
                "target_orientation must be one of 'vertical', 'horizontal', "
                f"'either', got: {target_orientation!r}"
            )
        input_image = image.numpy_image
        gray_small = _to_analysis_gray(input_image, max_dim=int(internal_resolution))

        if float(gray_small.std()) < 3.0:
            # Flat / low-texture image: nothing meaningful to align to.
            return {OUTPUT_IMAGE_KEY: image, "angle": 0.0}

        prepped = _prepare_for_scoring(gray_small)

        period = 90.0 if target_orientation == "either" else 180.0

        best_angle = _gradient_histogram_angle(prepped, target_orientation)
        if best_angle is None:
            # Indistinct edge signal: nothing trustworthy to align to.
            return {OUTPUT_IMAGE_KEY: image, "angle": 0.0}

        best_angle = _normalize_angle(best_angle, period=period)

        if abs(best_angle) < skip_below_degrees:
            return {OUTPUT_IMAGE_KEY: image, "angle": 0.0}

        if abs(best_angle) > max_correction_degrees:
            # The dominant structure wants a rotation beyond the plausible skew
            # range - most likely the search latched onto something other than
            # the lines of interest. Declining is safer than a wild rotation.
            return {OUTPUT_IMAGE_KEY: image, "angle": 0.0}

        rotated = _rotate_full_size(input_image, best_angle)
        output_image = WorkflowImageData.copy_and_replace(
            origin_image_data=image,
            numpy_image=rotated,
        )
        return {OUTPUT_IMAGE_KEY: output_image, "angle": float(best_angle)}


def build_auto_rotate_matrix(
    width: int, height: int, angle_degrees: float
) -> np.ndarray:
    """
    Build the canvas-expanding rotation matrix used to deskew an image.

    This mirrors `apply_rotate_image` in
    `inference/core/workflows/core_steps/classical_cv/image_preprocessing/v1.py`
    EXACTLY (same center computed via integer `//` division, same
    `cv2.getRotationMatrix2D` call, same canvas-expansion math using `int()`
    truncation). This function is the single source of truth for that math:
    downstream consumers rebuild this matrix from (width, height, angle) and
    invert it to map detections made on the rotated image back into the
    original image's coordinate space. If you change this function, any such
    consumer MUST be re-verified against it.

    Parameters
    ----------
    width : int
        Width (in pixels) of the ORIGINAL (pre-rotation) image.
    height : int
        Height (in pixels) of the ORIGINAL (pre-rotation) image.
    angle_degrees : float
        Rotation angle in degrees (OpenCV convention: positive is
        counter-clockwise).

    Returns
    -------
    np.ndarray
        The adjusted 2x3 affine rotation matrix, already translated to rotate
        about the original image's center and re-centered into the expanded
        output canvas.
    """
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)

    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))

    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]

    return rotation_matrix


def _rotate_full_size(np_image: np.ndarray, angle_degrees: float) -> np.ndarray:
    height, width = np_image.shape[:2]
    rotation_matrix = build_auto_rotate_matrix(width, height, angle_degrees)

    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))

    return cv2.warpAffine(
        np_image,
        rotation_matrix,
        (new_width, new_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _to_analysis_gray(np_image: np.ndarray, max_dim: int) -> np.ndarray:
    """Grayscale (or copy) the input and downscale it to the analysis size."""
    if np_image.ndim == 3 and np_image.shape[2] == 3:
        gray = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = np_image.copy()
    return _downscale_max_dim(gray, max_dim=max_dim)


def _downscale_max_dim(gray: np.ndarray, max_dim: int) -> np.ndarray:
    height, width = gray.shape[:2]
    largest = max(height, width)
    if largest <= max_dim:
        return gray
    scale = max_dim / float(largest)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_AREA)


def _prepare_for_scoring(gray_small: np.ndarray) -> np.ndarray:
    gray_uint8 = gray_small
    if gray_uint8.dtype != np.uint8:
        gray_uint8 = np.clip(gray_uint8, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_uint8)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    return blurred.astype(np.float32)


def _gradient_histogram_angle(
    prepped: np.ndarray, target_orientation: str
) -> Optional[float]:
    """Single-pass dominant-gradient-orientation estimate.

    Gradients of the dominant lines point perpendicular to them: vertical
    lines produce horizontal gradients (0 degrees mod 180), horizontal lines
    produce vertical gradients (90 degrees mod 180). The magnitude-weighted
    orientation histogram's dominant mode is refined with a doubled-angle
    weighted circular mean for sub-degree precision. Returns None when the
    histogram has no distinct peak.
    """
    gradient_x = cv2.Sobel(prepped, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(prepped, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gradient_x, gradient_y)
    threshold = max(float(np.percentile(magnitude, 80.0)), 1e-6)
    keep = magnitude >= threshold
    if not np.any(keep):
        return None
    theta = np.degrees(np.arctan2(gradient_y[keep], gradient_x[keep]))
    weights = magnitude[keep].astype(np.float64)
    theta_mod = np.mod(theta, 180.0)
    histogram, _ = np.histogram(
        theta_mod, bins=180, range=(0.0, 180.0), weights=weights
    )
    extended = np.concatenate([histogram[-2:], histogram, histogram[:2]])
    smoothed = np.convolve(extended, np.ones(5) / 5.0, mode="same")[2:-2]
    peak_ratio = (float(smoothed.max()) - float(np.median(smoothed))) / (
        float(smoothed.std()) + 1e-9
    )
    if peak_ratio < 1.5:
        return None
    peak = float(np.argmax(smoothed)) + 0.5
    distance = np.abs(theta_mod - peak)
    distance = np.minimum(distance, 180.0 - distance)
    selected = distance <= 3.0
    if np.any(selected):
        doubled = np.radians(2.0 * theta_mod[selected])
        selected_weights = weights[selected]
        gradient_angle = 0.5 * np.degrees(
            np.arctan2(
                float((selected_weights * np.sin(doubled)).sum()),
                float((selected_weights * np.cos(doubled)).sum()),
            )
        )
    else:
        gradient_angle = peak
    if target_orientation == "horizontal":
        return gradient_angle - 90.0
    # 'vertical' uses the gradient angle directly; 'either' relies on the
    # caller's mod-90 normalization to fold onto the nearest axis.
    return gradient_angle


def _normalize_angle(angle: float, period: float) -> float:
    half_period = period / 2.0
    normalized = angle % period
    if normalized > half_period:
        normalized -= period
    return normalized
