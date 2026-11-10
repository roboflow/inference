"""Stretch image contrast: ``roboflow_core/contrast_enhancement@v1``.

Port of V1 ``roboflow_core/contrast_enhancement@v1``. Each channel is
stretched on its own, from its range ``[low, high]`` to ``0..255``::

    low, high  = channel percentiles clip_limit and 100 - clip_limit
                 (minimum and maximum when clip_limit is 0)
    v' = (v - low) / (high - low) * 255
    v' = 128 + (v' - 128) * contrast_multiplier       when it is not 1
    v' = 255 * (v' / 255) ** (1 / 1.3)                with normalize_brightness
    out = v' clamped to 0..255, then truncated to an integer

A channel whose ``high`` is not above ``low`` (for example a flat one) is
left unchanged; a flat one-channel image is returned as it is, as in V1.

How it runs: the result depends only on a channel's value, so the block
counts the 256 values of each channel on the image's device, reads those
counts back to the host (one wait on a GPU), computes a 256-entry table per
channel with NumPy in exactly V1's arithmetic, and looks every pixel up on the
device. Results equal V1 exactly, including V1's float64 arithmetic for
three-channel images and float32 for one channel.

The result is the same image with new pixels: identity, mappings, source and
time are kept, and the input tensor is not modified.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
from pydantic import Field, StrictBool, StrictFloat, StrictInt
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref


BRIGHTNESS_GAMMA = 1.0 / 1.3

CONTRAST_CENTER = 128.0

class ContrastEnhancementBlock(Block):
    """Stretch each channel's range to ``0..255``, optionally clipping outliers.

    Example: a gray image whose values span ``100..150`` spans ``0..255``
    afterwards (``100 -> 0``, ``125 -> 127``, ``150 -> 255``).
    """

    type = "roboflow_core/contrast_enhancement@v1"
    outputs = {
        "image": Output(
            IMAGE_KIND,
            source="image",
            description="Enhanced image with the input's identity and geometry.",
        )
    }
    prunable = True

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="Image to enhance.")
        clip_limit: StrictInt = Field(
            default=0,
            ge=0,
            le=50,
            description=(
                "Percent of the darkest and of the brightest values per channel "
                "to ignore when finding its range; 0 uses minimum and maximum."
            ),
        )
        contrast_multiplier: StrictFloat = Field(
            default=1.0,
            ge=0.1,
            le=5.0,
            description=(
                "Contrast around mid-gray 128 after stretching: above 1 "
                "increases it, below 1 reduces it."
            ),
        )
        normalize_brightness: StrictBool = Field(
            default=False,
            description="Brighten midtones with gamma 1/1.3 after stretching.",
        )

    def run(
        self,
        *,
        image: ImageData,
        clip_limit: int,
        contrast_multiplier: float,
        normalize_brightness: bool,
    ) -> Dict[str, Any]:
        """Enhance the contrast of ``image``.

        Args:
            image: Image to enhance.
            clip_limit: Percent of extreme values ignored per side, 0..50.
            contrast_multiplier: Contrast factor around 128.
            normalize_brightness: Whether to brighten midtones.

        Returns:
            ``image``: the enhanced image.
        """
        pixels = image.tensor_image
        channels = pixels.shape[0]
        counts = torch.bincount(
            (pixels.reshape(channels, -1).long() + _channel_offsets(pixels)).reshape(
                -1
            ),
            minlength=256 * channels,
        )
        tables = contrast_tables(
            counts.reshape(channels, 256).cpu().numpy(),
            clip_limit=clip_limit,
            contrast_multiplier=contrast_multiplier,
            normalize_brightness=normalize_brightness,
        )
        if tables is None:
            return {"image": image}

        table = torch.from_numpy(tables).to(pixels.device)
        enhanced = torch.gather(table, 1, pixels.reshape(channels, -1).long())
        result = {"image": image.with_pixels(enhanced.reshape(pixels.shape))}

        return result

def contrast_tables(
    counts: np.ndarray,
    *,
    clip_limit: int,
    contrast_multiplier: float,
    normalize_brightness: bool,
) -> Optional[np.ndarray]:
    """Compute V1's per-channel value mapping from 256-bin histograms.

    Args:
        counts: ``(channels, 256)`` pixel counts of each value per channel.
        clip_limit: Percent of extreme values ignored per side, 0..50.
        contrast_multiplier: Contrast factor around 128.
        normalize_brightness: Whether to brighten midtones.

    Returns:
        ``(channels, 256)`` ``uint8`` tables, where ``table[c, v]`` replaces
        value ``v`` in channel ``c``; ``None`` for a one-channel image that
        V1 returned unchanged.
    """
    one_channel = counts.shape[0] == 1
    tables = np.empty(counts.shape, dtype=np.uint8)
    for channel, channel_counts in enumerate(counts):
        low, high = _channel_range(
            channel_counts, clip_limit=clip_limit, one_channel=one_channel
        )
        if high <= low:
            if one_channel:
                return None
            tables[channel] = np.arange(256, dtype=np.uint8)
            continue

        tables[channel] = _stretch(
            np.arange(256, dtype=np.float32),
            low=low,
            high=high,
            contrast_multiplier=float(contrast_multiplier),
            gamma=BRIGHTNESS_GAMMA if normalize_brightness else 1.0,
            via_float32=not one_channel,
        )

    return tables

def _channel_offsets(pixels: torch.Tensor) -> torch.Tensor:
    # Channel c counts into bins 256 * c .. 256 * c + 255 of one bincount.
    offsets = torch.arange(pixels.shape[0], device=pixels.device).mul_(256)
    column = offsets.unsqueeze(1)

    return column

def _channel_range(
    counts: np.ndarray, *, clip_limit: int, one_channel: bool
) -> Tuple[Any, Any]:
    # V1 used np.percentile with a float32 q for one channel (float32 range)
    # and np.quantile with a float64 q for three (float64 range); the
    # unclipped range is a float32 minimum and maximum in both.
    present = np.flatnonzero(counts)
    if clip_limit == 0:
        value_range = (np.float32(present[0]), np.float32(present[-1]))
        return value_range

    # Not a no-op: V1 round-trips through a fraction, and the float error
    # (7 -> 7.000000000000001) changes the quantile. Keep it for exact parity.
    lower_percent = float(clip_limit) / 100.0 * 100
    upper_percent = 100 - lower_percent
    if one_channel:
        quantiles = [
            np.true_divide(percent, np.float32(100))
            for percent in (lower_percent, upper_percent)
        ]
    else:
        quantiles = list(np.asarray([lower_percent / 100, upper_percent / 100]))
    value_range = tuple(_percentile_from_counts(counts, quantile=q) for q in quantiles)

    return value_range

def _percentile_from_counts(counts: np.ndarray, *, quantile: Any) -> Any:
    # NumPy's default "linear" quantile, operation for operation, using the
    # two order statistics read from the histogram instead of a sorted array.
    # quantile is a NumPy scalar whose dtype NumPy would have used.
    total = int(counts.sum())
    cumulative = np.cumsum(counts)
    position = np.asanyarray((total - 1) * quantile)
    previous = np.floor(position)
    previous_rank = int(min(max(previous, 0), total - 1))
    next_rank = int(min(previous_rank + 1, total - 1))
    weight = np.asanyarray(position - previous, dtype=position.dtype)
    lower = np.asarray(np.float32(np.searchsorted(cumulative, previous_rank + 1)))
    upper = np.asarray(np.float32(np.searchsorted(cumulative, next_rank + 1)))

    difference = np.subtract(upper, lower)
    value = np.asanyarray(np.add(lower, difference * weight))
    np.subtract(
        upper,
        difference * (1 - weight),
        out=value,
        where=weight >= 0.5,
        casting="unsafe",
        dtype=type(value.dtype),
    )
    percentile = value[()]

    return percentile

def _stretch(
    values: np.ndarray,
    *,
    low: Any,
    high: Any,
    contrast_multiplier: float,
    gamma: float,
    via_float32: bool,
) -> np.ndarray:
    # V1's expression in V1's order and dtypes; values are float32 0..255.
    # V1 stored three-channel results in a float32 array before truncating.
    stretched = (values - low) / (high - low) * 255.0
    if contrast_multiplier != 1.0:
        stretched = (
            CONTRAST_CENTER + (stretched - CONTRAST_CENTER) * contrast_multiplier
        )
    if gamma != 1.0:
        stretched = np.clip(stretched, 0, 255)
        stretched = np.power(stretched / 255.0, gamma) * 255.0
    clipped = np.clip(stretched, 0, 255)
    if via_float32:
        clipped = clipped.astype(np.float32)
    table = clipped.astype(np.uint8)

    return table
