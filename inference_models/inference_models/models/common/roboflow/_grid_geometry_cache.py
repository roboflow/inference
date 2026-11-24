"""Bounded reuse of the per-image geometry constants of grid boxes.

``build_instance_segmentation_grid`` maps boxes with three small constants
built by ``boxes.new_tensor`` on every image. A stream of frames repeats the
same geometry, so CUDA constants are kept in a small LRU keyed by device and
by the exact values (type and ``float.hex``, so ``-0.0`` differs from ``0.0``).

An entry is published only after construction, with an event on its producer
stream. Each use makes the consuming stream wait for the event and records the
stream on the tensors, so eviction never frees memory that queued work still
reads. CPU tensors and CUDA graph capture build the constants uncached.
"""

import threading
from collections import OrderedDict
from typing import Sequence, Tuple

import torch

MAX_ENTRIES = 64

_lock = threading.Lock()
_entries: "OrderedDict[tuple, Tuple[Tuple[torch.Tensor, ...], torch.cuda.Event]]" = (
    OrderedDict()
)


def geometry_constants(
    boxes: torch.Tensor,
    padding: Sequence[float],
    scale: Sequence[float],
    offset: Sequence[float],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``boxes.new_tensor`` of each value list, reused on CUDA.

    Args:
        boxes: Tensor whose dtype and device the constants take.
        padding: Values of the padding constant.
        scale: Values of the scale constant.
        offset: Values of the offset constant.

    Returns:
        Read-only-by-contract ``(padding, scale, offset)`` tensors.
    """
    if boxes.device.type != "cuda":
        return _build(boxes, padding, scale, offset)

    # Capture is a property of the target device's current stream, not the
    # caller's ambient device. Capture-owned tensors must never be published.
    with torch.cuda.device(boxes.device):
        capturing = torch.cuda.is_current_stream_capturing()
    if capturing:
        return _build(boxes, padding, scale, offset)

    key = (str(boxes.device), boxes.dtype, _values_key(padding, scale, offset))
    with _lock:
        entry = _entries.get(key)
        if entry is not None:
            _entries.move_to_end(key)
    if entry is not None:
        tensors, ready = entry
        stream = torch.cuda.current_stream(boxes.device)
        stream.wait_event(ready)
        for tensor in tensors:
            tensor.record_stream(stream)
        return tensors

    tensors = _build(boxes, padding, scale, offset)
    ready = torch.cuda.Event()
    ready.record(torch.cuda.current_stream(boxes.device))
    with _lock:
        _entries[key] = (tensors, ready)
        _entries.move_to_end(key)
        while len(_entries) > MAX_ENTRIES:
            _entries.popitem(last=False)

    return tensors


def clear() -> None:
    """Drop every entry."""
    with _lock:
        _entries.clear()


def cached_entries() -> int:
    """Return the number of entries."""
    with _lock:
        return len(_entries)


def _build(boxes, padding, scale, offset):
    return (
        boxes.new_tensor(list(padding)),
        boxes.new_tensor(list(scale)),
        boxes.new_tensor(list(offset)),
    )


def _values_key(*groups):
    return tuple(
        tuple(
            (
                type(value).__name__,
                value.hex() if isinstance(value, float) else repr(value),
            )
            for value in group
        )
        for group in groups
    )
