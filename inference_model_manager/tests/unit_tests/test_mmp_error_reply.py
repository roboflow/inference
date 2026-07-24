"""MMP reply path for worker error slots.

Workers report errors with result_sz=0 after writing the error message to the
slot (status=ERROR, header result_size = message bytes). MMP must point the
client at that message via T_RESULT_READY instead of collapsing every zero-size
result into a generic T_ERROR — otherwise typed input errors never reach the
client.
"""

from __future__ import annotations

import asyncio
import struct

import pytest

from inference_model_manager.backends.utils.shm_pool import SHMPool, SlotStatus
from inference_model_manager.model_manager_process import (
    T_ERROR,
    T_RESULT_READY,
    ModelManagerProcess,
)


@pytest.fixture()
def pool():
    p = SHMPool.create(n_slots=2, input_mb=0.5)
    yield p
    p.close()


def _make_mmp(pool: SHMPool) -> ModelManagerProcess:
    mmp = ModelManagerProcess.__new__(ModelManagerProcess)
    mmp._pool = pool
    mmp._pending = {}
    mmp._inflight = {}
    mmp._sent = []

    async def _send(identity, msg_type, payload):
        mmp._sent.append((identity, msg_type, payload))
        return True

    mmp._send = _send
    return mmp


def _prepare_slot(pool: SHMPool, req_id: int) -> int:
    slot_id = pool.alloc_slot()
    pool.mark_allocated(slot_id, request_id=req_id)
    pool.mark_processing(slot_id, 1234, request_id=req_id)
    return slot_id


def _run_on_loop(mmp, req_id: int, slot_id: int, result_sz: int) -> None:
    async def run():
        mmp._on_result_on_loop(req_id, slot_id, result_sz)
        await asyncio.sleep(0.01)

    asyncio.run(run())


def test_error_slot_message_forwarded_as_result(pool):
    mmp = _make_mmp(pool)
    slot = _prepare_slot(pool, req_id=201)
    msg = b"INPUT_ERROR: no embeddings were found in the cache"
    mv = pool.data_memoryview(slot)
    mv[: len(msg)] = msg
    mv.release()
    pool.mark_error(slot, error_code=1, error_size=len(msg), request_id=201)
    mmp._pending[201] = (b"client", None, None)
    mmp._inflight[slot] = "m"

    _run_on_loop(mmp, 201, slot, 0)

    assert mmp._sent
    _, msg_type, payload = mmp._sent[0]
    assert msg_type == T_RESULT_READY
    req_id, slot_id, sz = struct.unpack(">QII", payload)
    assert (req_id, slot_id, sz) == (201, slot, len(msg))
    # Slot must stay readable — the client reads the message, then frees.
    hdr = pool.read_header(slot)
    assert hdr.status == SlotStatus.ERROR
    assert bytes(pool.data_memoryview(slot)[: hdr.result_size]) == msg


def test_zero_result_without_error_message_sends_t_error(pool):
    mmp = _make_mmp(pool)
    slot = _prepare_slot(pool, req_id=202)
    mmp._pending[202] = (b"client", None, None)
    mmp._inflight[slot] = "m"

    _run_on_loop(mmp, 202, slot, 0)

    assert mmp._sent
    _, msg_type, _ = mmp._sent[0]
    assert msg_type == T_ERROR
    assert pool.read_header(slot).status == SlotStatus.FREE
