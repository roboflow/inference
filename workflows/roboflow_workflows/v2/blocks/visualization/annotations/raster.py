"""Device pixel operations of the region blocks, following OpenCV's rules.

=====================  ==================================  =================
function               OpenCV call it follows              equality
=====================  ==================================  =================
box_blur               cv2.blur (BORDER_REFLECT_101)       exact (tested)
grow_mask              cv2.dilate with an elliptic kernel  exact
resize_nearest         cv2.resize INTER_NEAREST            exact
resize_linear          cv2.resize INTER_LINEAR             sample positions
                                                           exact; float32
                                                           weights, so pixels
                                                           may differ by 1
                                                           from OpenCV's
                                                           fixed point
=====================  ==================================  =================

All inputs are ``(C, H, W)`` uint8 (or bool for masks) tensors on any device;
outputs stay there. Nothing reads tensor values on the host.

Resize index and weight tables are computed on the host (OpenCV's double
arithmetic). A block resizing many rows builds every row's tables first and
uploads them together, one copy per call::

    host  = [linear_tables(...), nearest_tables(...), ...]   # NumPy
    tables = upload_tables(host, device=...)                 # one copy
    for each row:  apply_linear(region, tables[i]) ...       # device only

Mask growth likewise uploads ``elliptic_kernel`` once per call.
"""

from typing import List, NamedTuple, Optional, Sequence, Tuple, TypeVar, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Resize tables hold NumPy arrays on the host and tensors after upload.
Table = Union[np.ndarray, torch.Tensor]

_ALIGNMENT = 8
# cv2.blur keeps uint8 window sums of up to 256 pixels in 16 bits and divides
# them in fixed point; larger windows divide in float32.
_SHORT_SUM_AREA = 256
_TORCH_DTYPES = {np.dtype(np.int64): torch.int64, np.dtype(np.float32): torch.float32}


def box_blur(roi: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Average each pixel over a ``kernel_size`` square, as ``cv2.blur``.

    The window of pixel ``i`` is ``[i - k // 2, i - k // 2 + k)``; pixels
    beyond the edge are mirrored without repeating it (reflect 101), as often
    as needed for small regions. Each window sum ``s`` over ``area = k * k``
    pixels becomes the value ``cv2.blur`` gives (OpenCV 4.12; every sum of
    every kernel 1 to 64 compared)::

        area <= 256, a power of two   (s + area // 2 + 1) // area
        area <= 256, otherwise        (s + area // 2) // area    # half up
        area > 256                    round_half_even(float32(s) * float32(1 / area))

    So ``.5`` means round up for small kernels, and a small mean such as
    ``1 / 4`` rounds up for kernel 2: a dim glow keeps OpenCV's 1-level
    pixels.

    Args:
        roi: ``(C, H, W)`` uint8 region.
        kernel_size: Window side, >= 1.

    Returns:
        New ``(C, H, W)`` uint8 blurred region.
    """
    channels, height, width = roi.shape
    if kernel_size == 1:
        return roi.clone()

    before = kernel_size // 2
    after = kernel_size - 1 - before
    rows = _reflect_101(height, before=before, after=after, device=roi.device)
    columns = _reflect_101(width, before=before, after=after, device=roi.device)
    padded = roi.to(torch.int32)[:, rows][:, :, columns]
    sums = _window_sums(_window_sums(padded, kernel_size, dim=1), kernel_size, dim=2)
    area = kernel_size * kernel_size
    if area <= _SHORT_SUM_AREA:
        is_power_of_two = area & (area - 1) == 0
        bias = area // 2 + 1 if is_power_of_two else area // 2
        means = (sums + bias) // area
    else:
        # Both factors are float32 values, so their product rounds to float32
        # the same way on every device (also when computed in float64).
        scale = float(np.float32(1 / area))
        means = torch.round(sums.to(torch.float32) * scale)
    blurred = means.to(torch.uint8)

    return blurred


def elliptic_kernel(padding: int, *, device: torch.device) -> torch.Tensor:
    """Return OpenCV's elliptic kernel for ``padding`` as float32 on ``device``.

    Build it once per block call and pass it to every ``grow_mask`` call.

    Args:
        padding: Growth radius, >= 1.
        device: Device of the masks to grow.

    Returns:
        ``(2 * padding + 1, 2 * padding + 1)`` float32 tensor of 0 and 1.
    """
    size = 2 * padding + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    weights = _to_device(kernel.astype(np.float32), device=device)

    return weights


def grow_mask(
    mask: torch.Tensor, padding: int, *, kernel: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Grow a ``(H, W)`` bool mask by ``padding`` with OpenCV's elliptic kernel.

    Args:
        mask: ``(H, W)`` bool mask.
        padding: Growth radius, >= 1.
        kernel: ``elliptic_kernel(padding)`` on the mask's device; built here
            (one upload) when omitted.

    Returns:
        New ``(H, W)`` bool mask; pixels outside the mask's raster do not
        contribute, as with ``cv2.dilate``'s default border.
    """
    if kernel is None:
        kernel = elliptic_kernel(padding, device=mask.device)
    grown = F.conv2d(
        mask.to(torch.float32)[None, None], kernel[None, None], padding=padding
    )[0, 0]
    grown = grown > 0

    return grown


class NearestTables(NamedTuple):
    """Source rows and columns (int64) of one ``cv2.INTER_NEAREST`` resize."""

    rows: Table
    columns: Table


class LinearTables(NamedTuple):
    """Neighbours (int64) and weights (float32) of one ``INTER_LINEAR`` resize."""

    top: Table
    bottom: Table
    row_weight: Table
    left: Table
    right: Table
    column_weight: Table


Tables = TypeVar("Tables", NearestTables, LinearTables)


def nearest_tables(
    source_hw: Tuple[int, int], size_hw: Tuple[int, int]
) -> NearestTables:
    """Build on the host the tables of a nearest resize; see ``resize_nearest``.

    Args:
        source_hw: Input ``(height, width)``.
        size_hw: Output ``(height, width)``.

    Returns:
        NumPy tables; pass them through ``upload_tables``.
    """
    tables = NearestTables(
        rows=_nearest_sources(source_hw[0], size_hw[0]),
        columns=_nearest_sources(source_hw[1], size_hw[1]),
    )

    return tables


def linear_tables(
    source_hw: Tuple[int, int],
    size_hw: Tuple[int, int],
    *,
    scale_xy: Tuple[float, float],
) -> LinearTables:
    """Build on the host the tables of a bilinear resize; see ``resize_linear``.

    Args:
        source_hw: Input ``(height, width)``.
        size_hw: Output ``(height, width)``.
        scale_xy: Source pixels per output pixel, as OpenCV computes it.

    Returns:
        NumPy tables; pass them through ``upload_tables``.
    """
    top, bottom, row_weight = _linear_sources(
        source_hw[0], size_hw[0], scale=scale_xy[1], clamp_weight=False
    )
    left, right, column_weight = _linear_sources(
        source_hw[1], size_hw[1], scale=scale_xy[0], clamp_weight=True
    )
    tables = LinearTables(top, bottom, row_weight, left, right, column_weight)

    return tables


def upload_tables(tables: Sequence[Tables], *, device: torch.device) -> List[Tables]:
    """Copy every array of ``tables`` to ``device`` in one transfer.

    The arrays are packed into one byte buffer and come back as views of its
    device copy, with their own dtypes. The buffer holds the tables' bytes and
    at most 7 alignment bytes per array: per resize, 8 bytes per output row
    and column (nearest) or 20 (linear).

    Args:
        tables: Host tables from ``nearest_tables`` or ``linear_tables``.
        device: Target device.

    Returns:
        The same tables in order, holding device tensors. No transfer when
        ``tables`` is empty.
    """
    arrays = [np.ascontiguousarray(array) for table in tables for array in table]
    if not arrays:
        return []

    starts, end = [], 0
    for array in arrays:
        # A dtype view of the buffer must start at a multiple of its item size.
        start = -(-end // _ALIGNMENT) * _ALIGNMENT
        starts.append(start)
        end = start + array.nbytes
    packed = np.zeros(end, dtype=np.uint8)
    for array, start in zip(arrays, starts):
        packed[start : start + array.nbytes] = array.view(np.uint8)

    buffer = _to_device(packed, device=device)
    views = iter(
        buffer[start : start + array.nbytes].view(_TORCH_DTYPES[array.dtype])
        for array, start in zip(arrays, starts)
    )
    uploaded = [type(table)(*(next(views) for _ in table)) for table in tables]

    return uploaded


def apply_nearest(roi: torch.Tensor, tables: NearestTables) -> torch.Tensor:
    """Resize ``roi`` with uploaded nearest tables; see ``resize_nearest``.

    Args:
        roi: ``(C, H, W)`` region of the tables' source size.
        tables: ``nearest_tables`` output on the roi's device.

    Returns:
        New ``(C, height, width)`` region.
    """
    resized = roi[:, tables.rows][:, :, tables.columns]

    return resized


def apply_linear(roi: torch.Tensor, tables: LinearTables) -> torch.Tensor:
    """Resize ``roi`` with uploaded bilinear tables; see ``resize_linear``.

    Args:
        roi: ``(C, H, W)`` uint8 region of the tables' source size.
        tables: ``linear_tables`` output on the roi's device.

    Returns:
        New ``(C, height, width)`` uint8 region.
    """
    values = roi.to(torch.float32)
    rows = (
        values[:, tables.top] * (1 - tables.row_weight)[:, None]
        + values[:, tables.bottom] * tables.row_weight[:, None]
    )
    mixed = (
        rows[:, :, tables.left] * (1 - tables.column_weight)
        + rows[:, :, tables.right] * tables.column_weight
    )
    resized = mixed.round().clamp(0, 255).to(torch.uint8)

    return resized


def resize_nearest(roi: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
    """Resize with ``cv2.INTER_NEAREST``: source ``floor(i * in / out)``.

    One upload per call; batch many resizes with ``nearest_tables``,
    ``upload_tables`` and ``apply_nearest``.

    Args:
        roi: ``(C, H, W)`` region.
        size_hw: Output ``(height, width)``.

    Returns:
        New ``(C, height, width)`` region.
    """
    host = nearest_tables((roi.shape[1], roi.shape[2]), size_hw)
    (tables,) = upload_tables([host], device=roi.device)
    resized = apply_nearest(roi, tables)

    return resized


def resize_linear(
    roi: torch.Tensor, size_hw: Tuple[int, int], *, scale_xy: Tuple[float, float]
) -> torch.Tensor:
    """Resize with ``cv2.INTER_LINEAR`` sample positions and clamping.

    Output pixel ``d`` samples source ``(d + 0.5) * scale - 0.5``; columns
    outside the source take its edge column, rows are clamped to the edge
    rows, as OpenCV does. One upload per call; batch many resizes with
    ``linear_tables``, ``upload_tables`` and ``apply_linear``.

    Args:
        roi: ``(C, H, W)`` uint8 region.
        size_hw: Output ``(height, width)``.
        scale_xy: Source pixels per output pixel, as OpenCV computes it
            (``1 / fx`` for a factor, ``in / out`` for a size).

    Returns:
        New ``(C, height, width)`` uint8 region.
    """
    host = linear_tables((roi.shape[1], roi.shape[2]), size_hw, scale_xy=scale_xy)
    (tables,) = upload_tables([host], device=roi.device)
    resized = apply_linear(roi, tables)

    return resized


def _reflect_101(
    size: int, *, before: int, after: int, device: torch.device
) -> torch.Tensor:
    # cv::borderInterpolate(BORDER_REFLECT_101) for indices -before .. size+after-1.
    indices = torch.arange(-before, size + after, device=device)
    if size == 1:
        return torch.zeros_like(indices)

    period = 2 * (size - 1)
    folded = torch.remainder(indices, period)
    reflected = torch.where(folded >= size, period - folded, folded)

    return reflected


def _window_sums(values: torch.Tensor, kernel_size: int, *, dim: int) -> torch.Tensor:
    cumulative = torch.cumsum(values, dim=dim)
    zero = torch.zeros_like(cumulative.narrow(dim, 0, 1))
    cumulative = torch.cat([zero, cumulative], dim=dim)
    length = values.shape[dim] - kernel_size + 1
    sums = cumulative.narrow(dim, kernel_size, length) - cumulative.narrow(
        dim, 0, length
    )

    return sums


def _nearest_sources(source: int, target: int) -> np.ndarray:
    # cv::resizeNN: min(floor(i * (1 / (out / in))), in - 1), in double.
    scale = 1.0 / (target / source)
    indices = np.minimum(
        np.floor(np.arange(target) * scale).astype(np.int64), source - 1
    )

    return indices


def _linear_sources(
    source: int, target: int, *, scale: float, clamp_weight: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # cv::resize INTER_LINEAR: position in double, stored as float32. Columns
    # outside the source take the edge with weight 0; rows keep their weight
    # and clamp both neighbours.
    positions = ((np.arange(target) + 0.5) * scale - 0.5).astype(np.float32)
    lows = np.floor(positions).astype(np.int64)
    weights = positions - lows.astype(np.float32)
    if clamp_weight:
        below, beyond = lows < 0, lows >= source - 1
        lows = np.where(below, 0, np.where(beyond, source - 1, lows))
        weights = np.where(below | beyond, np.float32(0.0), weights)
    highs = np.clip(lows + 1, 0, source - 1)
    lows = np.clip(lows, 0, source - 1)

    sources = (lows, highs, weights.astype(np.float32))

    return sources


def _to_device(host: np.ndarray, *, device: torch.device) -> torch.Tensor:
    # The only host-to-device copy in this module.
    on_device = torch.from_numpy(host).to(device)

    return on_device
