"""Threshold an image: ``v2/threshold@v1``.

Port of V1 ``roboflow_core/threshold@v1``. Every type equals the OpenCV call
V1 made, pixel for pixel. Here ``T`` is ``thresh_value`` and ``M`` is
``max_value`` clamped to ``0..255``:

==================  ========================================  ==============  ========
threshold_type      pixel ``v`` becomes                       channels        runs on
==================  ========================================  ==============  ========
binary              ``M`` if ``v > T`` else 0                 each channel    device
binary_inv          0 if ``v > T`` else ``M``                 each channel    device
trunc               ``min(v, T)``; ``M`` is unused            each channel    device
tozero              ``v`` if ``v > T`` else 0                 each channel    device
tozero_inv          0 if ``v > T`` else ``v``                 each channel    device
otsu                binary with ``T`` chosen by Otsu's        one only        device
                    method; ``thresh_value`` is unused
adaptive_mean       ``M`` if ``v > mean - 2`` else 0, over    one only        device
                    the 11 x 11 neighbourhood (edge
                    pixels repeated)
adaptive_gaussian   as adaptive_mean, Gaussian-weighted mean  one only        CPU
==================  ========================================  ==============  ========

A CPU tensor is the exception for ``adaptive_mean``: it gets OpenCV's own
``adaptiveThreshold``, as V1 called it. Off the CPU a PyTorch version equal
to it runs on the image's device.

Convert an RGB image with ``v2/convert_grayscale@v1`` first for the three
one-channel types. ``T`` and ``M`` outside ``0..255`` follow OpenCV: a ``T``
below 0 or from 255 up fills or copies the image (``binary`` with ``T = 255``
is all zeros).

Otsu's method reads one 256-bin histogram back to the host and scans it as
OpenCV does. ``adaptive_gaussian`` crosses to the CPU and calls OpenCV
there (``host.call_on_host``), because OpenCV's float32 Gaussian filter
rounds differently on different CPUs and cannot be matched exactly.

The result is the same image with new pixels: identity, mappings, source and
time are kept, and the input tensor is not modified.

Difference from V1: a three-channel image with a one-channel type raises a
``ValueError`` that names the fix, instead of an OpenCV assertion.
"""

import math
from typing import Any, Dict, List, Literal

import cv2
import torch
import torch.nn.functional as F
from pydantic import Field, StrictInt
from roboflow_workflows.v2.blocks.image.host import call_on_host
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import INTEGER_KIND, STRING_KIND
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref

ThresholdType = Literal[
    "binary",
    "binary_inv",
    "trunc",
    "tozero",
    "tozero_inv",
    "adaptive_mean",
    "adaptive_gaussian",
    "otsu",
]

FIXED_TYPES = ("binary", "binary_inv", "trunc", "tozero", "tozero_inv")
ONE_CHANNEL_TYPES = ("otsu", "adaptive_mean", "adaptive_gaussian")
THRESHOLD_TYPES = FIXED_TYPES + ONE_CHANNEL_TYPES
ADAPTIVE_BLOCK_SIZE = 11
ADAPTIVE_OFFSET = 2
ADAPTIVE_METHODS = {
    "adaptive_mean": cv2.ADAPTIVE_THRESH_MEAN_C,
    "adaptive_gaussian": cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
}
# OpenCV's Otsu scan guards with the C float epsilon (2**-23).
FLOAT_EPSILON = 2.0**-23


class ImageThresholdBlock(Block):
    """Threshold one image; see the module docstring for every type.

    Example: with ``threshold_type="binary"``, ``thresh_value=127`` and
    ``max_value=255``, the pixel values 127 and 128 become 0 and 255.
    """

    type = "v2/threshold@v1"
    outputs = {
        "image": Output(
            IMAGE_KIND,
            source="image",
            description="Thresholded image with the input's identity and geometry.",
        )
    }
    prunable = True

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(
            description=(
                "Image to threshold; one channel for 'otsu', 'adaptive_mean' "
                "and 'adaptive_gaussian'."
            )
        )
        threshold_type: ThresholdType | Ref(STRING_KIND) = Field(
            default="binary",
            description=(
                "Rule applied to each pixel: binary, binary_inv, trunc, tozero, "
                "tozero_inv, otsu, adaptive_mean or adaptive_gaussian."
            ),
            examples=["binary", "$inputs.threshold_type"],
        )
        thresh_value: StrictInt | Ref(INTEGER_KIND) = Field(
            description=(
                "Threshold T; a pixel counts as above it when strictly greater. "
                "Unused by 'otsu' and the adaptive types."
            ),
            examples=[127, "$inputs.thresh_value"],
        )
        max_value: StrictInt | Ref(INTEGER_KIND) = Field(
            default=255,
            description=(
                "Value written for pixels that pass (clamped to 0..255); unused "
                "by 'trunc', 'tozero' and 'tozero_inv'."
            ),
            examples=[255, "$inputs.max_value"],
        )

    def run(
        self,
        *,
        image: ImageData,
        threshold_type: str,
        thresh_value: int,
        max_value: int,
    ) -> Dict[str, Any]:
        """Threshold ``image``.

        Args:
            image: Image to threshold.
            threshold_type: One of the types in the module docstring.
            thresh_value: Threshold for the fixed types.
            max_value: Value of passing pixels.

        Returns:
            ``image``: the thresholded image.

        Raises:
            ValueError: On an unknown type, or a three-channel image with a
                one-channel type.
        """
        thresholded = threshold(
            image.tensor_image,
            threshold_type=threshold_type,
            thresh_value=thresh_value,
            max_value=max_value,
        )
        result = {"image": image.with_pixels(thresholded)}

        return result


def threshold(
    pixels: torch.Tensor, *, threshold_type: str, thresh_value: int, max_value: int
) -> torch.Tensor:
    """Threshold ``(channels, height, width)`` ``uint8`` pixels like V1.

    Args:
        pixels: Image tensor on any device; not modified.
        threshold_type: One of the types in the module docstring.
        thresh_value: Threshold for the fixed types.
        max_value: Value of passing pixels.

    Returns:
        A new tensor of the same shape on the same device.

    Raises:
        ValueError: On an unknown type, or a three-channel image with a
            one-channel type.
    """
    if threshold_type not in THRESHOLD_TYPES:
        raise ValueError(
            f"threshold_type must be one of {THRESHOLD_TYPES}, got {threshold_type!r}"
        )
    if threshold_type in ONE_CHANNEL_TYPES and pixels.shape[0] != 1:
        raise ValueError(
            f"threshold_type {threshold_type!r} needs a one-channel image, got "
            f"{pixels.shape[0]} channels; convert it with v2/convert_grayscale@v1 "
            "first"
        )

    if threshold_type in FIXED_TYPES:
        thresholded = fixed_threshold(
            pixels,
            threshold_type=threshold_type,
            thresh_value=thresh_value,
            max_value=max_value,
        )
    elif threshold_type == "otsu":
        counts = torch.bincount(pixels.reshape(-1).long(), minlength=256)
        thresholded = fixed_threshold(
            pixels,
            threshold_type="binary",
            thresh_value=otsu_threshold(counts.cpu().tolist()),
            max_value=max_value,
        )
    elif threshold_type == "adaptive_mean" and pixels.device.type != "cpu":
        thresholded = torch_adaptive_mean(pixels, max_value=max_value)
    else:
        method = ADAPTIVE_METHODS[threshold_type]
        thresholded = call_on_host(
            pixels,
            function=lambda plane: cv2.adaptiveThreshold(
                plane,
                max_value,
                method,
                cv2.THRESH_BINARY,
                ADAPTIVE_BLOCK_SIZE,
                ADAPTIVE_OFFSET,
            ),
        )

    return thresholded


def fixed_threshold(
    pixels: torch.Tensor, *, threshold_type: str, thresh_value: int, max_value: int
) -> torch.Tensor:
    """Apply one of OpenCV's fixed threshold rules to every channel.

    Args:
        pixels: ``uint8`` tensor; not modified.
        threshold_type: ``"binary"``, ``"binary_inv"``, ``"trunc"``,
            ``"tozero"`` or ``"tozero_inv"``.
        thresh_value: Threshold ``T``; pixels strictly above it pass.
        max_value: Value of passing pixels, clamped to ``0..255``.

    Returns:
        A new tensor of the same shape and device.
    """
    level = math.floor(thresh_value)
    fill = level if threshold_type == "trunc" else round(max_value)
    fill = min(max(fill, 0), 255)
    if level < 0 or level >= 255:
        # OpenCV decides these without comparing pixels.
        degenerate = _degenerate_threshold(
            pixels, threshold_type=threshold_type, level=level, fill=fill
        )
        return degenerate

    above = pixels > level
    zero = pixels.new_zeros(())
    fill_value = pixels.new_full((), fill)
    if threshold_type == "binary":
        thresholded = torch.where(above, fill_value, zero)
    elif threshold_type == "binary_inv":
        thresholded = torch.where(above, zero, fill_value)
    elif threshold_type == "trunc":
        thresholded = torch.clamp(pixels, max=fill)
    elif threshold_type == "tozero":
        thresholded = torch.where(above, pixels, zero)
    else:
        thresholded = torch.where(above, zero, pixels)

    return thresholded


def otsu_threshold(counts: List[int]) -> int:
    """Choose a threshold from a 256-bin histogram, exactly as OpenCV does.

    Follows OpenCV's ``getThreshVal_Otsu_8u`` in double precision: the same
    order of operations and validity guards, and the first of equally good
    thresholds wins.

    Args:
        counts: Number of pixels of each value ``0..255``; not all zero.

    Returns:
        The threshold in ``0..254``; pixels strictly above it are foreground.
    """
    scale = 1.0 / sum(counts)
    mean = sum(value * float(count) for value, count in enumerate(counts)) * scale
    lower_mean, lower_weight, best_spread, best_value = 0.0, 0.0, 0.0, 0
    for value, count in enumerate(counts):
        probability = count * scale
        lower_mean *= lower_weight
        lower_weight += probability
        upper_weight = 1.0 - lower_weight
        if (
            min(lower_weight, upper_weight) < FLOAT_EPSILON
            or max(lower_weight, upper_weight) > 1.0 - FLOAT_EPSILON
        ):
            continue

        lower_mean = (lower_mean + value * probability) / lower_weight
        upper_mean = (mean - lower_weight * lower_mean) / upper_weight
        spread = lower_weight * upper_weight * (lower_mean - upper_mean) ** 2
        if spread > best_spread:
            best_spread, best_value = spread, value

    return best_value


def _degenerate_threshold(
    pixels: torch.Tensor, *, threshold_type: str, level: int, fill: int
) -> torch.Tensor:
    # T < 0: every pixel is above; T >= 255: none is.
    everything_above = level < 0
    if threshold_type == "binary":
        value = fill if everything_above else 0
    elif threshold_type == "binary_inv":
        value = 0 if everything_above else fill
    elif threshold_type in ("trunc", "tozero_inv"):
        value = 0 if everything_above else None
    else:
        value = None if everything_above else 0

    filled = pixels.clone() if value is None else torch.full_like(pixels, value)

    return filled


def torch_adaptive_mean(pixels: torch.Tensor, *, max_value: int) -> torch.Tensor:
    """Equal OpenCV's 11 x 11 adaptive mean threshold in PyTorch on the device.

    ``threshold`` uses this only off the CPU. OpenCV rounds the mean to
    ``uint8`` first. Sums of 121 values are exact in float32 and sum / 121
    never ties, so the rounding matches.

    Args:
        pixels: ``(1, height, width)`` ``uint8`` tensor; not modified.
        max_value: Value of passing pixels, clamped to ``0..255``.

    Returns:
        A new tensor of the same shape on the same device.
    """
    fill = min(max(round(max_value), 0), 255)
    pad = ADAPTIVE_BLOCK_SIZE // 2
    padded = _replicate_pad(pixels.to(torch.float32).unsqueeze(0), pad=pad)
    window = torch.ones(
        (1, 1, ADAPTIVE_BLOCK_SIZE, ADAPTIVE_BLOCK_SIZE),
        dtype=torch.float32,
        device=pixels.device,
    )
    sums = F.conv2d(padded, window)
    means = torch.round(sums / (ADAPTIVE_BLOCK_SIZE**2)).to(torch.int16).squeeze(0)
    above = pixels.to(torch.int16) > means - ADAPTIVE_OFFSET
    thresholded = torch.where(above, pixels.new_full((), fill), pixels.new_zeros(()))

    return thresholded


def _replicate_pad(planes: torch.Tensor, *, pad: int) -> torch.Tensor:
    # Clamped indices repeat the edge pixel, also for images smaller than pad.
    height, width = int(planes.shape[-2]), int(planes.shape[-1])
    device = planes.device
    rows = torch.arange(-pad, height + pad, device=device).clamp_(0, height - 1)
    columns = torch.arange(-pad, width + pad, device=device).clamp_(0, width - 1)
    padded = planes.index_select(-2, rows).index_select(-1, columns)

    return padded
