"""Unit tests for SHMPool — lifecycle, header round-trip, offset math."""

import os
import struct
import time

import pytest

from inference_model_manager.backends.utils.shm_pool import (
    _HEADER_SIZE,
    _OFF_ERROR,
    _OFF_INPUT_SZ,
    _OFF_RESULT_SZ,
    _OFF_STATUS,
    SHMPool,
    SlotHeader,
    SlotStatus,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pool(n_slots=4, input_mb=1.0) -> SHMPool:
    return SHMPool.create(n_slots=n_slots, input_mb=input_mb)


# ---------------------------------------------------------------------------
# Header size
# ---------------------------------------------------------------------------


def test_header_size_is_64():
    assert _HEADER_SIZE == 64


# ---------------------------------------------------------------------------
# Create / close
# ---------------------------------------------------------------------------


def test_create_and_close():
    pool = _make_pool()
    name = pool.name
    assert name  # non-empty
    pool.close()


def test_all_slots_free_after_create():
    pool = _make_pool(n_slots=8)
    try:
        assert pool.free_count == 8
        for slot_id in range(8):
            hdr = pool.read_header(slot_id)
            assert hdr.status == SlotStatus.FREE
    finally:
        pool.close()


# ---------------------------------------------------------------------------
# Slot allocation
# ---------------------------------------------------------------------------


def test_alloc_and_free_single():
    pool = _make_pool(n_slots=2)
    try:
        slot = pool.alloc_slot()
        assert 0 <= slot < 2
        assert pool.free_count == 1
        pool.free_slot(slot)
        assert pool.free_count == 2
    finally:
        pool.close()


def test_alloc_all_then_free_all():
    n = 4
    pool = _make_pool(n_slots=n)
    try:
        slots = [pool.alloc_slot() for _ in range(n)]
        assert pool.free_count == 0
        assert sorted(slots) == list(range(n))
        for s in slots:
            pool.free_slot(s)
        assert pool.free_count == n
    finally:
        pool.close()


def test_alloc_timeout_when_exhausted():
    pool = _make_pool(n_slots=1)
    try:
        pool.alloc_slot()  # take the only slot
        with pytest.raises(TimeoutError):
            pool.alloc_slot(timeout=0.05)
    finally:
        pool.close()


def test_attach_cannot_alloc():
    pool = _make_pool(n_slots=2)
    try:
        attached = SHMPool.attach(pool.name, n_slots=2, input_mb=1.0)
        try:
            with pytest.raises(RuntimeError, match="creator"):
                attached.alloc_slot()
        finally:
            attached.close()
    finally:
        pool.close()


def test_attach_cannot_free():
    pool = _make_pool(n_slots=2)
    try:
        attached = SHMPool.attach(pool.name, n_slots=2, input_mb=1.0)
        try:
            with pytest.raises(RuntimeError, match="creator"):
                attached.free_slot(0)
        finally:
            attached.close()
    finally:
        pool.close()


# ---------------------------------------------------------------------------
# Header round-trip
# ---------------------------------------------------------------------------


def test_mark_allocated_header():
    pool = _make_pool()
    try:
        slot = pool.alloc_slot()
        req_id = 0xDEADBEEF_CAFEBABE
        pool.mark_allocated(slot, request_id=req_id)

        hdr = pool.read_header(slot)
        assert hdr.status == SlotStatus.ALLOCATED
        assert hdr.request_id == req_id
        assert hdr.error_code == 0
        assert hdr.input_size == 0
        assert hdr.result_size == 0
        assert hdr.timestamp_ns > 0
    finally:
        pool.close()


def test_mark_written_header():
    pool = _make_pool()
    try:
        slot = pool.alloc_slot()
        pool.mark_allocated(slot, request_id=1)
        pool.mark_written(slot, input_size=12345)

        hdr = pool.read_header(slot)
        assert hdr.status == SlotStatus.WRITTEN
        assert hdr.input_size == 12345
    finally:
        pool.close()


def test_mark_processing_header():
    pool = _make_pool()
    try:
        slot = pool.alloc_slot()
        pool.mark_allocated(slot, request_id=1)
        pool.mark_processing(slot, owner_pid=os.getpid())

        hdr = pool.read_header(slot)
        assert hdr.status == SlotStatus.PROCESSING
        assert hdr.owner_pid == os.getpid()
    finally:
        pool.close()


def test_mark_done_header():
    pool = _make_pool()
    try:
        slot = pool.alloc_slot()
        pool.mark_allocated(slot, request_id=1)
        pool.mark_done(slot, result_size=42)

        hdr = pool.read_header(slot)
        assert hdr.status == SlotStatus.DONE
        assert hdr.result_size == 42
    finally:
        pool.close()


def test_mark_error_header():
    pool = _make_pool()
    try:
        slot = pool.alloc_slot()
        pool.mark_error(slot, error_code=7)

        hdr = pool.read_header(slot)
        assert hdr.status == SlotStatus.ERROR
        assert hdr.error_code == 7
    finally:
        pool.close()


def test_free_slot_zeros_header():
    pool = _make_pool()
    try:
        slot = pool.alloc_slot()
        pool.mark_allocated(slot, request_id=999)
        pool.free_slot(slot)

        hdr = pool.read_header(slot)
        assert hdr.status == SlotStatus.FREE
        assert hdr.request_id == 0
    finally:
        pool.close()


# ---------------------------------------------------------------------------
# Offset math — must match app.py arithmetic
# ---------------------------------------------------------------------------


def test_data_memoryview_offset():
    """data_memoryview base == slot_id * slot_bytes + HEADER_SIZE."""
    input_mb = 1.0
    pool = _make_pool(n_slots=4, input_mb=input_mb)
    try:
        data_bytes = int(input_mb * 1024 * 1024)
        slot_bytes = _HEADER_SIZE + data_bytes

        for slot_id in range(4):
            expected_start = slot_id * slot_bytes + _HEADER_SIZE
            mv = pool.data_memoryview(slot_id)
            mv[0] = 0xAB
            actual = pool._shm.buf[expected_start]
            mv[0] = 0
            mv.release()
            assert actual == 0xAB, f"slot {slot_id}: data start offset wrong"
    finally:
        pool.close()


def test_slots_do_not_overlap():
    """Write sentinel at end of one slot's data area; neighbour is untouched."""
    pool = _make_pool(n_slots=4, input_mb=1.0)
    try:
        rv0 = pool.data_memoryview(0)
        rv1 = pool.data_memoryview(1)
        rv0[-1] = 0xFF
        neighbour_first = rv1[0]
        rv0[-1] = 0
        rv0.release()
        rv1.release()
        assert neighbour_first != 0xFF
    finally:
        pool.close()


# ---------------------------------------------------------------------------
# Data round-trip through SHM
# ---------------------------------------------------------------------------


def test_write_read_input_data():
    pool = _make_pool(n_slots=2, input_mb=1.0)
    try:
        slot = pool.alloc_slot()
        data = b"hello world" * 100
        pool.data_memoryview(slot)[: len(data)] = data
        readback = bytes(pool.data_memoryview(slot)[: len(data)])
        assert readback == data
    finally:
        pool.close()


def test_write_read_result_data():
    pool = _make_pool(n_slots=2, input_mb=1.0)
    try:
        slot = pool.alloc_slot()
        result = b"\xDE\xAD\xBE\xEF" * 50
        pool.data_memoryview(slot)[: len(result)] = result
        readback = bytes(pool.data_memoryview(slot)[: len(result)])
        assert readback == result
    finally:
        pool.close()


# ---------------------------------------------------------------------------
# Attach
# ---------------------------------------------------------------------------


def test_attach_sees_data_written_by_creator():
    pool = _make_pool(n_slots=2, input_mb=1.0)
    try:
        slot = pool.alloc_slot()
        pool.mark_allocated(slot, request_id=42)
        data = b"cross-process data"
        pool.data_memoryview(slot)[: len(data)] = data

        attached = SHMPool.attach(pool.name, n_slots=2, input_mb=1.0)
        try:
            hdr = attached.read_header(slot)
            assert hdr.status == SlotStatus.ALLOCATED
            assert hdr.request_id == 42
            readback = bytes(attached.data_memoryview(slot)[: len(data)])
            assert readback == data
        finally:
            attached.close()
    finally:
        pool.close()


# ---------------------------------------------------------------------------
# Stale slot detection
# ---------------------------------------------------------------------------


def test_stale_slots_empty_when_all_free():
    pool = _make_pool(n_slots=4)
    try:
        assert pool.stale_slots(max_age_s=0.0) == []
    finally:
        pool.close()


def test_stale_slots_detected():
    pool = _make_pool(n_slots=4)
    try:
        slot = pool.alloc_slot()
        # Backdate the timestamp so it looks old
        off = pool._slot_offset(slot)
        old_ts = time.monotonic_ns() - int(60 * 1e9)  # 60s ago
        pool.mark_allocated(slot, request_id=1)
        struct.pack_into("<Q", pool._shm.buf, off + 24, old_ts)

        stale = pool.stale_slots(max_age_s=30.0)
        assert slot in stale
    finally:
        pool.close()


def test_fresh_slots_not_stale():
    pool = _make_pool(n_slots=4)
    try:
        slot = pool.alloc_slot()
        pool.mark_allocated(slot, request_id=1)
        # Just allocated — timestamp is now, should not be stale
        assert slot not in pool.stale_slots(max_age_s=30.0)
    finally:
        pool.close()
