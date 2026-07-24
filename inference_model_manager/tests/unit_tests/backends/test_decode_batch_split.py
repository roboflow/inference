"""Decoded-bytes batch cap: estimate_decoded_bytes + _split_batch_by_decoded_bytes.

Regression: 32 x 21MB JPEGs (74MP each, ~223MB decoded) formed one decode
batch (~7GB VRAM) and OOM-killed the GPU. Batches must also be capped by
estimated decoded size, not image count alone.
"""

from __future__ import annotations

import struct

import pytest

from inference_model_manager.backends.decode import (
    FALLBACK_DECODED_BYTES,
    estimate_decoded_bytes,
    jpeg_sof_dims,
)
from inference_model_manager.backends.subproc import (
    _decode_bytes_budget,
    _split_batch_by_decoded_bytes,
)
from inference_model_manager.backends.utils.shm_pool import SHMPool


def _jpeg(h: int, w: int) -> bytes:
    return (
        b"\xff\xd8"
        + b"\xff\xc0"
        + struct.pack(">H", 17)
        + b"\x08"
        + struct.pack(">HH", h, w)
        + b"\x03"
    )


def _png(h: int, w: int) -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", 13)
        + b"IHDR"
        + struct.pack(">II", w, h)
    )


def test_jpeg_sof_dims():
    assert jpeg_sof_dims(_jpeg(6459, 11483)) == (6459, 11483, 3)


def test_estimate_jpeg():
    est = estimate_decoded_bytes(_jpeg(1000, 2000), 4096)
    assert est == 1000 * 2000 * 3


def test_estimate_png():
    est = estimate_decoded_bytes(_png(500, 800), 4096)
    assert est == 500 * 800 * 3


def test_estimate_npy_uses_input_size():
    assert estimate_decoded_bytes(b"\x93NUMPY rest", 12345) == 12345


def test_estimate_unknown_format_pessimistic():
    assert estimate_decoded_bytes(b"GIF89a....", 4096) == FALLBACK_DECODED_BYTES


def test_estimate_jpeg_without_sof_pessimistic():
    assert estimate_decoded_bytes(b"\xff\xd8\xff\xd9", 4096) == FALLBACK_DECODED_BYTES


def test_budget_from_free_vram():
    # (8GB free - 1GB headroom) / scratch factor 2 = 3.5GB
    assert _decode_bytes_budget(8 * 1024**3, 1024.0, 2.0) == int(3.5 * 1024**3)


def test_budget_floors_at_one_byte_when_vram_starved():
    # Starved GPU: budget must stay positive so every image still gets its
    # own chunk (max_bytes=0 would disable splitting entirely).
    assert _decode_bytes_budget(100 * 1024**2, 1024.0, 2.0) == 1


@pytest.fixture()
def pool():
    p = SHMPool.create(n_slots=4, input_mb=0.5)
    yield p
    p.close()


def _fill_slot(pool: SHMPool, req_id: int, payload: bytes) -> int:
    slot_id = pool.alloc_slot()
    pool.mark_allocated(slot_id, request_id=req_id)
    mv = pool.data_memoryview(slot_id)
    mv[: len(payload)] = payload
    mv.release()
    pool.mark_written(slot_id, len(payload))
    return slot_id


def test_split_respects_decoded_bytes_cap(pool):
    # 4 x 1000x1000 JPEG, ~3MB decoded each; cap 7MB -> chunks of 2.
    batch = [
        (_fill_slot(pool, 100 + i, _jpeg(1000, 1000)), 100 + i, b"{}") for i in range(4)
    ]
    chunks = _split_batch_by_decoded_bytes(pool, batch, 7_000_000)
    assert [len(c) for c in chunks] == [2, 2]
    assert [item for c in chunks for item in c] == batch


def test_split_oversized_image_gets_own_chunk(pool):
    batch = [
        (_fill_slot(pool, 100 + i, _jpeg(2000, 2000)), 100 + i, b"{}") for i in range(2)
    ]
    # 12MB each, cap 1MB: each image alone, never dropped.
    chunks = _split_batch_by_decoded_bytes(pool, batch, 1_000_000)
    assert [len(c) for c in chunks] == [1, 1]


def test_split_disabled_when_cap_nonpositive(pool):
    batch = [
        (_fill_slot(pool, 100 + i, _jpeg(2000, 2000)), 100 + i, b"{}") for i in range(3)
    ]
    assert _split_batch_by_decoded_bytes(pool, batch, 0) == [batch]
