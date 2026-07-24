"""Reaper vs in-flight worker tickets.

Regression: a slot whose request was cancelled is retained for the worker,
but the stale reaper freed it mid-batch anyway — the slot got re-allocated
while the worker still referenced it (torn slot / use-after-free).
"""

from __future__ import annotations

import asyncio
import struct

import pytest

from inference_model_manager.backends.utils.shm_pool import SHMPool, SlotStatus
from inference_model_manager.model_manager_process import (
    _ERR_STALE,
    T_ERROR,
    ModelManagerProcess,
    ModelState,
)


@pytest.fixture()
def pool():
    p = SHMPool.create(n_slots=2, input_mb=0.5)
    yield p
    p.close()


class _FakeBackend:
    is_healthy = True

    def __init__(self) -> None:
        self.signalled: list[tuple[int, int]] = []

    def signal_slot(self, slot_id: int, req_id: int, params_bytes: bytes) -> None:
        self.signalled.append((slot_id, req_id))


def _make_mmp(pool: SHMPool) -> ModelManagerProcess:
    mmp = ModelManagerProcess.__new__(ModelManagerProcess)
    mmp._pool = pool
    mmp._pending = {}
    mmp._inflight = {}
    mmp._backends = {}
    mmp._models = {}
    mmp._model_access = {}
    mmp._model_request_times = {}
    mmp._unloading = set()
    mmp._vram_recent_window_s = 60.0
    mmp._stale_slot_max_age_s = -1.0  # every allocated slot is instantly stale
    mmp._sent = []

    async def _send(identity, msg_type, payload):
        mmp._sent.append((identity, msg_type, payload))
        return True

    mmp._send = _send
    return mmp


def _alloc_written(pool: SHMPool, req_id: int) -> int:
    slot_id = pool.alloc_slot()
    pool.mark_allocated(slot_id, request_id=req_id)
    pool.mark_written(slot_id, 4)
    return slot_id


def test_forward_to_backend_records_inflight(pool):
    mmp = _make_mmp(pool)
    backend = _FakeBackend()
    mmp._backends["m"] = backend
    s = _alloc_written(pool, 101)
    mmp._forward_to_backend("m", s, 101)
    assert backend.signalled == [(s, 101)]
    assert mmp._inflight == {s: "m"}


def test_reaper_skips_inflight_slot(pool):
    mmp = _make_mmp(pool)
    s = _alloc_written(pool, 101)
    mmp._inflight[s] = "m"
    # Cancelled request: not in _pending, slot retained for the worker.
    mmp._reap()
    assert pool.read_header(s).request_id == 101
    assert pool.free_count == pool.n_slots - 1
    assert mmp._sent == []


def test_reaper_frees_stale_slot_without_ticket(pool):
    mmp = _make_mmp(pool)
    s = _alloc_written(pool, 101)
    mmp._pending[101] = (b"id1", s, "m")

    async def _run():
        mmp._reap()
        await asyncio.sleep(0)

    asyncio.run(_run())
    assert pool.free_count == pool.n_slots
    assert 101 not in mmp._pending
    _, msg_type, payload = mmp._sent[0]
    assert msg_type == T_ERROR
    assert struct.unpack(">QB", payload) == (101, _ERR_STALE)


def test_result_clears_inflight(pool):
    mmp = _make_mmp(pool)
    s = _alloc_written(pool, 101)
    mmp._inflight[s] = "m"
    mmp._pending[101] = (b"id1", s, "m")

    async def _run():
        mmp._on_result_on_loop(101, s, 4)
        await asyncio.sleep(0)

    asyncio.run(_run())
    assert mmp._inflight == {}


def test_late_result_clears_inflight_and_frees(pool):
    mmp = _make_mmp(pool)
    s = _alloc_written(pool, 101)
    mmp._inflight[s] = "m"
    # Request cancelled earlier: not in _pending.

    async def _run():
        mmp._on_result_on_loop(101, s, 0)

    asyncio.run(_run())
    assert mmp._inflight == {}
    assert pool.free_count == pool.n_slots


def test_schedule_reload_clears_inflight_for_dead_model(pool):
    mmp = _make_mmp(pool)
    mmp._models["m"] = ModelState(loaded=True)
    mmp._models["other"] = ModelState(loaded=True)
    mmp._inflight = {1: "m", 2: "other", 3: "m"}
    mmp._load_calls = []

    async def _load_model(model_id, api_key="", device=""):
        mmp._load_calls.append(model_id)

    mmp._load_model = _load_model

    async def _run():
        mmp._schedule_reload("m")
        await asyncio.sleep(0)

    asyncio.run(_run())
    assert mmp._inflight == {2: "other"}


def test_result_stamps_done_before_dropping_reaper_protection(pool):
    mmp = _make_mmp(pool)
    mmp._stale_slot_max_age_s = 5.0
    s = _alloc_written(pool, 101)
    mmp._inflight[s] = "m"
    mmp._pending[101] = (b"id1", s, "m")

    async def _run():
        mmp._on_result_on_loop(101, s, 4)
        # BEFORE the reply task runs: DONE must already be stamped with a
        # fresh timestamp, so a reaper tick in this gap cannot free the slot.
        hdr = pool.read_header(s)
        assert hdr.status == SlotStatus.DONE
        assert hdr.result_size == 4
        mmp._reap()
        assert pool.read_header(s).status == SlotStatus.DONE
        await asyncio.sleep(0)

    asyncio.run(_run())
    assert pool.free_count == pool.n_slots - 1  # still held for client read


def test_reply_after_ticket_void_and_rebind_does_not_stamp_foreign_slot(pool):
    mmp = _make_mmp(pool)
    s = _alloc_written(pool, 101)
    mmp._pending[101] = (b"id1", s, "m")
    # Ticket was voided (reload) and the reaper rebound the slot to a new request.
    pool.alloc_slot()  # occupy the other slot so the rebind reuses s
    pool.free_slot(s, request_id=101)
    s2 = pool.alloc_slot()
    assert s2 == s
    pool.mark_allocated(s2, request_id=202)

    async def _run():
        mmp._on_result_on_loop(101, s, 4)
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    asyncio.run(_run())
    hdr = pool.read_header(s)
    assert hdr.request_id == 202
    assert hdr.status == SlotStatus.ALLOCATED  # new owner untouched
    _, msg_type, payload = mmp._sent[0]
    assert msg_type == T_ERROR
    assert struct.unpack(">QB", payload) == (101, _ERR_STALE)


def test_schedule_reload_keeps_tickets_while_worker_alive(pool):
    from types import SimpleNamespace

    mmp = _make_mmp(pool)
    mmp._models["m"] = ModelState(loaded=True)
    mmp._backends["m"] = SimpleNamespace(worker_pid=4242)
    mmp._inflight = {1: "m", 2: "other"}
    mmp._load_calls = []

    async def _load_model(model_id, api_key="", device=""):
        mmp._load_calls.append(model_id)

    mmp._load_model = _load_model

    async def _run():
        mmp._schedule_reload("m")
        await asyncio.sleep(0)

    asyncio.run(_run())
    assert mmp._inflight == {1: "m", 2: "other"}  # live worker keeps tickets
    assert mmp._load_calls == ["m"]
