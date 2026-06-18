"""Worker-side slot ownership validation in _process_slots.

Corruption regression: a slot reaped + re-allocated between T_SLOT_READY and
batch fire must not be decoded or written by the worker under the old req_id.
"""

from __future__ import annotations

import pickle
import struct

import numpy as np
import pytest

import inference_model_manager.backends.subproc as subproc
from inference_model_manager.backends.utils.shm_pool import SHMPool, SlotStatus


class _FakeSock:
    def __init__(self) -> None:
        self.sent: list[list[bytes]] = []

    def send_multipart(self, frames: list[bytes]) -> None:
        self.sent.append(frames)


class _Log:
    def error(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass

    def exception(self, *a, **k):
        pass


def _stats() -> dict:
    import time
    from collections import deque

    return {
        "inference_count": 0,
        "error_count": 0,
        "batch_count": 0,
        "latencies": deque(maxlen=10),
        "batch_sizes": deque(maxlen=10),
        "decode_ms": deque(maxlen=10),
        "infer_ms": deque(maxlen=10),
        "write_ms": deque(maxlen=10),
        "start_ts": time.monotonic(),
    }


@pytest.fixture()
def pool():
    p = SHMPool.create(n_slots=2, input_mb=0.5)
    yield p
    p.close()


def _write_npy_input(pool: SHMPool, slot_id: int, req_id: int) -> None:
    import io

    buf = io.BytesIO()
    np.save(buf, np.zeros((2, 2, 3), dtype=np.uint8))
    data = buf.getvalue()
    pool.mark_allocated(slot_id, request_id=req_id)
    pool.data_memoryview(slot_id)[: len(data)] = data
    pool.mark_written(slot_id, len(data))


def _result_frames(sock: _FakeSock) -> list[tuple[int, int, int]]:
    out = []
    for frames in sock.sent:
        if frames[0] == subproc._MSG_RESULT:
            out.append(struct.unpack(">QII", frames[1]))
    return out


def test_mismatched_req_id_slot_is_skipped(pool, monkeypatch):
    sock = _FakeSock()
    s0 = pool.alloc_slot()
    s1 = pool.alloc_slot()
    _write_npy_input(pool, s0, req_id=101)
    _write_npy_input(pool, s1, req_id=202)

    monkeypatch.setattr(
        subproc,
        "invoke_task",
        lambda model, task, images, **kw: (
            [{"pred": True}]
            if not isinstance(images, list)
            else [{"pred": True} for _ in images]
        ),
    )

    # Signal for s1 carries a STALE req_id (999) — slot was rebound to 202.
    batch = [(s0, 101, b"{}"), (s1, 999, b"{}")]
    subproc._process_slots(
        object(), pool, batch, sock, lambda mvs: [None] * len(mvs), _Log(), _stats()
    )

    results = _result_frames(sock)
    assert [(r, s) for r, s, _ in results] == [(101, s0)]
    assert pool.read_header(s0).status == SlotStatus.DONE
    # Stale slot untouched: still WRITTEN under owner 202, input intact.
    h1 = pool.read_header(s1)
    assert h1.status == SlotStatus.WRITTEN
    assert h1.request_id == 202


def test_ownership_lost_during_inference_blocks_result_write(pool, monkeypatch):
    sock = _FakeSock()
    s0 = pool.alloc_slot()
    pool.alloc_slot()  # burn the second slot so re-allocation reuses s0
    _write_npy_input(pool, s0, req_id=101)

    def _steal_slot_mid_infer(model, task, images, **kw):
        # Simulate reaper free + re-allocation while infer is running.
        pool.free_slot(s0, request_id=101)
        s_new = pool.alloc_slot()
        assert s_new == s0
        pool.mark_allocated(s0, request_id=777)
        return [{"pred": True}]

    monkeypatch.setattr(subproc, "invoke_task", _steal_slot_mid_infer)

    subproc._process_slots(
        object(),
        pool,
        [(s0, 101, b"{}")],
        sock,
        lambda mvs: [None] * len(mvs),
        _Log(),
        _stats(),
    )

    assert _result_frames(sock) == []
    h = pool.read_header(s0)
    assert h.request_id == 777
    assert h.status == SlotStatus.ALLOCATED  # new owner's state, not DONE/ERROR


def test_valid_slot_marked_processing(pool, monkeypatch):
    sock = _FakeSock()
    s0 = pool.alloc_slot()
    _write_npy_input(pool, s0, req_id=101)
    seen_status = {}

    def _check_status(model, task, images, **kw):
        seen_status["during_infer"] = pool.read_header(s0).status
        return [{"pred": True}]

    monkeypatch.setattr(subproc, "invoke_task", _check_status)
    subproc._process_slots(
        object(),
        pool,
        [(s0, 101, b"{}")],
        sock,
        lambda mvs: [None] * len(mvs),
        _Log(),
        _stats(),
    )
    assert seen_status["during_infer"] == SlotStatus.PROCESSING
    assert pickle.loads(
        bytes(pool.data_memoryview(s0)[: pool.read_header(s0).result_size])
    ) == {"pred": True}
