"""OpenCV ``INTER_AREA`` resampling of ``uint8`` image tensors, on their device.

V1 ``image_preprocessing`` resizes with ``cv2.resize(..., cv2.INTER_AREA)``.
``resize_area`` reproduces that function for ``(channels, height, width)``
tensors without a host copy of the pixels. OpenCV picks one of three laws from
the per-axis scale ``source / target``:

==========================  ===========================================  ==================
case                        output pixel                                 rounding
==========================  ===========================================  ==================
any axis enlarged           2 taps per axis: the source pixel under the  11-bit fixed point
                            output start, plus the next one weighted by
                            the part of it the output covers
both axes shrunk by         mean of each ``factor_y x factor_x`` block   2x2: half up;
integer factors                                                          else half to even
both axes shrunk,           mean weighted by the exact overlap of the    half to even
otherwise                   output and source pixel intervals
==========================  ===========================================  ==================

The first two laws repeat OpenCV's integer arithmetic exactly, with one
deliberate exception. OpenCV sums an integer-shrink block in ``int32``, which
overflows once a block holds more than 8,421,504 pixels of value 255: an
all-255 ``2160 x 4096`` image shrunk to ``1 x 1`` gives 0. Such blocks are
summed here in ``int64`` and their exact mean is rounded half to even, so that
image gives 255. The overlap law sums in ``float32`` in a different order than
OpenCV, so a value within float error of ``.5`` can round 1 intensity
differently.
"""

import sys
from typing import Tuple

import torch

SizeHW = Tuple[int, int]

# OpenCV fixed-point linear coefficients: weight 1.0 is 1 << 11.
_COEFFICIENT_SCALE = 2048
# OpenCV skips an overlap of at most this fraction of one source pixel.
_MIN_PARTIAL_OVERLAP = 1e-3
# Most uint8 values whose sum always fits int32: 255 * 8_421_504 < 2**31.
_INT32_SAFE_PIXEL_COUNT = (2**31 - 1) // 255


def resize_area(pixels: torch.Tensor, size_hw: SizeHW) -> torch.Tensor:
    """Resample ``uint8`` pixels like ``cv2.resize`` with ``cv2.INTER_AREA``.

    Args:
        pixels: ``(channels, height, width)`` ``uint8`` tensor on any device;
            any strides.
        size_hw: Target ``(height, width)``, positive integers.

    Returns:
        A new ``(channels, *size_hw)`` ``uint8`` tensor on the device of
        ``pixels``.
    """
    height, width = size_hw
    source_height, source_width = pixels.shape[1], pixels.shape[2]
    if height > source_height or width > source_width:
        resized = _resize_linear(pixels, size_hw)
    elif _is_integer_shrink((source_height, source_width), size_hw):
        resized = _resize_blocks(pixels, size_hw)
    else:
        resized = _resize_overlap(pixels, size_hw)

    return resized


def _is_integer_shrink(source_hw: SizeHW, size_hw: SizeHW) -> bool:
    # OpenCV's own test, in its double arithmetic.
    for source, target in zip(source_hw, size_hw):
        scale = 1.0 / (target / source)
        if abs(scale - round(scale)) >= sys.float_info.epsilon:
            return False

    return True


def _resize_blocks(pixels: torch.Tensor, size_hw: SizeHW) -> torch.Tensor:
    height, width = size_hw
    channels, source_height, source_width = pixels.shape
    factors = (source_height // height, source_width // width)
    area = factors[0] * factors[1]

    blocks = pixels.int().reshape(channels, height, factors[0], width, factors[1])
    # Two single-axis sums: one sum over dims (2, 4) is ~70x slower on MPS.
    # Each stays int32 unless the shape lets its own uint8 total overflow.
    sums = blocks.sum(dim=4, dtype=_sum_dtype(factors[1]))
    sums = sums.sum(dim=2, dtype=_sum_dtype(area))

    if area > _INT32_SAFE_PIXEL_COUNT:
        # OpenCV overflows here; take the exact mean instead.
        means = _round_half_to_even(sums, area)
    elif factors == (2, 2):
        # OpenCV's vectorised 2x2 path rounds half up.
        means = (sums + 2) // 4
    else:
        # OpenCV multiplies by the float32 reciprocal, then rounds half to even.
        reciprocal = torch.tensor(1.0 / area, dtype=torch.float32)
        means = (sums.float() * reciprocal).round()

    resized = means.to(torch.uint8)

    return resized


def _sum_dtype(pixel_count: int) -> torch.dtype:
    # Decided from the shape alone: no device value is read on the host.
    if pixel_count > _INT32_SAFE_PIXEL_COUNT:
        return torch.int64

    return torch.int32


def _round_half_to_even(sums: torch.Tensor, area: int) -> torch.Tensor:
    # Exact integer division of non-negative sums; MPS has no float64.
    quotients = torch.div(sums, area, rounding_mode="floor")
    twice_remainders = 2 * (sums - quotients * area)
    round_up = (twice_remainders > area) | (
        (twice_remainders == area) & (quotients % 2 == 1)
    )
    means = quotients + round_up

    return means


def _resize_overlap(pixels: torch.Tensor, size_hw: SizeHW) -> torch.Tensor:
    height, width = size_hw
    columns = _overlap_taps(pixels.shape[2], width, device=pixels.device)
    rows = _overlap_taps(pixels.shape[1], height, device=pixels.device)

    resampled = _apply_taps(pixels.float(), *columns, dim=2)
    resampled = _apply_taps(resampled, *rows, dim=1)

    resized = resampled.round().clamp(0, 255).to(torch.uint8)

    return resized


def _overlap_taps(
    source: int, target: int, *, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Exact integer geometry in units of 1 / target source pixels: source
    # pixel i spans [i * target, (i + 1) * target), output pixel k spans
    # [k * source, (k + 1) * source). Weight = overlap / output width.
    tap_count = -(-source // target) + 1
    starts = torch.arange(target, device=device)[:, None] * source
    indices = starts // target + torch.arange(tap_count, device=device)[None, :]
    overlap = torch.minimum((indices + 1) * target, starts + source) - torch.maximum(
        indices * target, starts
    )
    overlap = overlap.clamp(min=0)
    overlap = torch.where(overlap <= _MIN_PARTIAL_OVERLAP * target, 0, overlap)

    weights = overlap.float() / source
    indices = indices.clamp(max=source - 1)

    return indices, weights


def _resize_linear(pixels: torch.Tensor, size_hw: SizeHW) -> torch.Tensor:
    height, width = size_hw
    column_indices, column_weights = _linear_taps(pixels.shape[2], width)
    row_indices, row_weights = _linear_taps(pixels.shape[1], height)
    column_indices = column_indices.to(pixels.device)
    column_weights = column_weights.to(pixels.device)
    row_indices = row_indices.to(pixels.device)
    row_weights = row_weights.to(pixels.device)

    # Horizontal pass: exact integer sums at scale 2048.
    rows = _apply_taps(pixels.int(), column_indices, column_weights, dim=2)

    # Vertical pass: OpenCV's vectorised fixed-point rounding, bit for bit.
    top = rows.index_select(1, row_indices[:, 0]) >> 4
    bottom = rows.index_select(1, row_indices[:, 1]) >> 4
    top_weight = row_weights[:, 0].view(1, -1, 1)
    bottom_weight = row_weights[:, 1].view(1, -1, 1)
    combined = ((top_weight * top) >> 16) + ((bottom_weight * bottom) >> 16)
    resized = ((combined + 2) >> 2).clamp(0, 255).to(torch.uint8)

    return resized


def _linear_taps(source: int, target: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # OpenCV's area-mode linear coefficients in its float64-then-float32
    # arithmetic. V1 parity keeps its rounding: an exact integer position
    # such as 9 * 14 / 18 = 7 evaluates to 6.999... and takes source pixel 6.
    # Built on the host because MPS has no float64; the tables are O(target).
    inverse_scale = target / source
    scale = 1.0 / inverse_scale
    destination = torch.arange(target, dtype=torch.float64)
    first = torch.floor(destination * scale)
    fraction = ((destination + 1) - (first + 1) * inverse_scale).float()
    fraction = torch.where(fraction <= 0, 0.0, fraction - torch.floor(fraction))

    first = first.long()
    fraction = torch.where(first >= source - 1, 0.0, fraction)
    first = first.clamp(max=source - 1)

    indices = torch.stack([first, (first + 1).clamp(max=source - 1)], dim=1)
    weights = torch.stack([1 - fraction, fraction], dim=1)
    fixed_weights = (weights * _COEFFICIENT_SCALE).round().int()

    return indices, fixed_weights


def _apply_taps(
    pixels: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor, *, dim: int
) -> torch.Tensor:
    # Output j along `dim` = sum over taps t of weights[j, t] * pixels[indices[j, t]].
    target, tap_count = indices.shape
    gathered = pixels.index_select(dim, indices.flatten())
    gathered = gathered.unflatten(dim, (target, tap_count))
    weight_shape = [1] * gathered.ndim
    weight_shape[dim : dim + 2] = [target, tap_count]

    resampled = (gathered * weights.view(weight_shape)).sum(
        dim=dim + 1, dtype=pixels.dtype
    )

    return resampled
