"""MMP-side slot ownership: submit validation, ownership-checked frees,
T_FREE wire format with req_id."""

from __future__ import annotations

import asyncio
import struct

import pytest

from inference_model_manager.backends.utils.shm_pool import SHMPool, SlotStatus
from inference_model_manager.model_manager_process import (
    _ERR_STALE,
    T_ERROR,
    ModelManagerProcess,
)


@pytest.fixture()
def pool():
    p = SHMPool.create(n_slots=2, input_mb=0.5)
    yield p
    p.close()


def _make_mmp(pool: SHMPool) -> ModelManagerProcess:
    """Bare MMP: only the state the slot-lifecycle handlers touch."""
    mmp = ModelManagerProcess.__new__(ModelManagerProcess)
    mmp._pool = pool
    mmp._pending = {}
    mmp._inflight = {}
    mmp._backends = {}
    mmp._models = {}
    mmp._model_access = {}
    mmp._model_request_times = {}
    mmp._vram_recent_window_s = 60.0
    mmp._sent = []

    async def _send(identity, msg_type, payload):
        mmp._sent.append((identity, msg_type, payload))
        return True

    mmp._send = _send
    mmp._forwarded = []
    mmp._forward_to_backend = lambda *a, **k: mmp._forwarded.append(a)
    return mmp


def _alloc(pool: SHMPool, req_id: int) -> int:
    slot_id = pool.alloc_slot()
    pool.mark_allocated(slot_id, request_id=req_id)
    return slot_id


def _submit_frame(req_id: int, slot_id: int, input_sz: int = 4) -> list[bytes]:
    mid = b"m"
    return [struct.pack(">QIIH", req_id, slot_id, input_sz, len(mid)) + mid]


def test_submit_with_matching_req_id_forwards(pool):
    mmp = _make_mmp(pool)
    s = _alloc(pool, 101)
    asyncio.run(mmp._handle_submit(b"id1", _submit_frame(101, s)))
    assert len(mmp._forwarded) == 1
    assert mmp._pending[101] == (b"id1", s, "m")
    assert pool.read_header(s).status == SlotStatus.WRITTEN


def test_submit_for_reaped_slot_rejected_with_stale(pool):
    mmp = _make_mmp(pool)
    s = _alloc(pool, 101)
    pool.alloc_slot()  # burn the other slot so re-allocation reuses s
    pool.free_slot(s, request_id=101)  # reaper got it
    s2 = pool.alloc_slot()
    pool.mark_allocated(s2, request_id=999)  # rebound to a new request
    assert s2 == s
    asyncio.run(mmp._handle_submit(b"id1", _submit_frame(101, s)))
    assert mmp._forwarded == []
    assert 101 not in mmp._pending
    _, msg_type, payload = mmp._sent[0]
    assert msg_type == T_ERROR
    assert struct.unpack(">QB", payload) == (101, _ERR_STALE)
    # New owner's slot untouched
    h = pool.read_header(s)
    assert h.status == SlotStatus.ALLOCATED and h.request_id == 999


def test_handle_free_requires_matching_req_id(pool):
    mmp = _make_mmp(pool)
    s = _alloc(pool, 101)
    mmp._handle_free([struct.pack(">QI", 999, s)])  # wrong req
    assert pool.free_count == pool.n_slots - 1
    mmp._handle_free([struct.pack(">QI", 101, s)])  # right req
    assert pool.free_count == pool.n_slots


def test_handle_free_drops_old_4_byte_format(pool):
    mmp = _make_mmp(pool)
    s = _alloc(pool, 101)
    mmp._handle_free([struct.pack(">I", s)])
    assert pool.free_count == pool.n_slots - 1


def test_late_result_free_is_ownership_checked(pool):
    mmp = _make_mmp(pool)
    s = _alloc(pool, 101)
    pool.free_slot(s, request_id=101)
    pool.alloc_slot()  # burn the other slot so re-allocation reuses s
    s2 = pool.alloc_slot()
    pool.mark_allocated(s2, request_id=999)

    # Late worker T_RESULT for req 101 (not in _pending) must not free the
    # slot now owned by req 999.
    async def _run():
        mmp._on_result_on_loop(101, s, 0)

    asyncio.run(_run())
    assert pool.read_header(s2).request_id == 999
    assert pool.free_count == 0
