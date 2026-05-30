"""COCO RLE to OpenCV-style polygon conversion."""

from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

_EMPTY_POLYGON = np.zeros((0, 2), dtype=np.float32)
_ColumnIntervals = Dict[int, List[Tuple[int, int]]]


def rle_masks_to_polygons(masks: object) -> List[np.ndarray]:
    """Convert COCO RLE masks into the legacy largest external polygon.

    The old adapter path decoded every RLE into a full-frame dense mask and then
    called ``cv2.findContours(mask, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)``. This
    path keeps the RLE sparse until the final contour step, where it materializes
    only the foreground bounding crop needed by OpenCV.
    """

    height, width = masks.image_size
    sparse_counts = _get_lazy_uncompressed_counts(masks=masks)
    if sparse_counts is not None:
        counts, lengths = sparse_counts
        return [
            polygon_from_uncompressed_counts(
                counts=counts[i, : int(lengths[i])],
                height=height,
                width=width,
            )
            for i in range(lengths.shape[0])
        ]
    return [
        polygon_from_coco_counts(counts=counts, height=height, width=width)
        for counts in masks.masks
    ]


def polygon_from_coco_counts(
    counts: object,
    height: int,
    width: int,
) -> np.ndarray:
    columns = _coco_counts_to_column_intervals(
        counts=counts,
        height=height,
        width=width,
    )
    return _polygon_from_column_intervals(columns=columns)


def polygon_from_uncompressed_counts(
    counts: Iterable[int],
    height: int,
    width: int,
) -> np.ndarray:
    columns = _counts_to_column_intervals(
        counts=counts,
        height=height,
        width=width,
    )
    return _polygon_from_column_intervals(columns=columns)


def _get_lazy_uncompressed_counts(
    masks: object,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    ensure_rle_cpu = getattr(masks, "_ensure_rle_cpu", None)
    if callable(ensure_rle_cpu):
        ensure_rle_cpu()
    counts = getattr(masks, "_rle_counts_cpu", None)
    lengths = getattr(masks, "_rle_lengths_cpu", None)
    if counts is None or lengths is None:
        return None
    return counts, lengths


def _coco_counts_to_column_intervals(
    counts: object,
    height: int,
    width: int,
) -> _ColumnIntervals:
    if isinstance(counts, str):
        encoded = counts.encode("ascii")
    elif isinstance(counts, bytes):
        encoded = counts
    elif isinstance(counts, bytearray):
        encoded = bytes(counts)
    else:
        return _counts_to_column_intervals(
            counts=counts,
            height=height,
            width=width,
        )

    total_size = height * width
    columns: _ColumnIntervals = {}
    cursor = 0
    value = 0
    previous_two = 0
    previous_one = 0
    count_index = 0
    index = 0
    encoded_len = len(encoded)
    while index < encoded_len:
        run_length = 0
        shift = 0
        while True:
            char = encoded[index] - 48
            index += 1
            run_length |= (char & 0x1F) << shift
            shift += 5
            if char & 0x20:
                continue
            if char & 0x10:
                run_length |= -1 << shift
            break
        # Compressed COCO RLE stores deltas from count index 3 onward.
        if count_index > 2:
            run_length += previous_two
        if run_length < 0:
            raise ValueError("COCO RLE counts must be non-negative")
        next_cursor = cursor + run_length
        if next_cursor > total_size:
            raise ValueError("COCO RLE counts exceed the mask size")
        if value and run_length:
            _append_foreground_run(
                columns=columns,
                cursor=cursor,
                run_length=run_length,
                height=height,
            )
        cursor = next_cursor
        previous_two, previous_one = previous_one, run_length
        count_index += 1
        value ^= 1
    return columns


def _counts_to_column_intervals(
    counts: Iterable[int],
    height: int,
    width: int,
) -> _ColumnIntervals:
    total_size = height * width
    columns: _ColumnIntervals = {}
    cursor = 0
    value = 0
    for raw_count in counts:
        count = int(raw_count)
        if count < 0:
            raise ValueError("COCO RLE counts must be non-negative")
        next_cursor = cursor + count
        if next_cursor > total_size:
            raise ValueError("COCO RLE counts exceed the mask size")
        if value and count:
            _append_foreground_run(
                columns=columns,
                cursor=cursor,
                run_length=count,
                height=height,
            )
        cursor = next_cursor
        value ^= 1
    return columns


def _append_foreground_run(
    columns: _ColumnIntervals,
    cursor: int,
    run_length: int,
    height: int,
) -> None:
    end = cursor + run_length
    run_cursor = cursor
    while run_cursor < end:
        x = run_cursor // height
        y = run_cursor - x * height
        column_run_length = min(end - run_cursor, height - y)
        columns.setdefault(x, []).append((y, y + column_run_length))
        run_cursor += column_run_length


def _polygon_from_column_intervals(columns: _ColumnIntervals) -> np.ndarray:
    if not columns:
        return _EMPTY_POLYGON.copy()

    x_min = min(columns)
    x_max = max(columns)
    y_min = min(y0 for intervals in columns.values() for y0, _ in intervals)
    y_max = max(y1 for intervals in columns.values() for _, y1 in intervals)
    crop = np.zeros((y_max - y_min, x_max - x_min + 1), dtype=np.uint8)
    for x, intervals in columns.items():
        crop_x = x - x_min
        for y0, y1 in intervals:
            crop[y0 - y_min : y1 - y_min, crop_x] = 1

    contours = cv2.findContours(
        crop,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
        offset=(x_min, y_min),
    )[0]
    if not contours:
        return _EMPTY_POLYGON.copy()

    contour_lengths = np.fromiter((len(c) for c in contours), dtype=np.intp)
    selected_contour = contours[int(contour_lengths.argmax())]
    return np.asarray(selected_contour, dtype=np.float32).reshape(-1, 2)
