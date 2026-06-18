"""SubprocessBackend failure paths: outstanding counter, liveness semantics,
idempotent death handling, drain in MMP mode, worker result-write survival."""

from __future__ import annotations

import pickle
import queue
import struct
import threading
import time
from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest

import inference_model_manager.backends.subproc as subproc
from inference_model_manager.backends.subproc import (
    _WORKER_HEARTBEAT_TIMEOUT,
    SubprocessBackend,
)
from inference_model_manager.backends.utils.shm_pool import SHMPool, SlotStatus


def _bare_backend(**attrs) -> SubprocessBackend:
    b = SubprocessBackend.__new__(SubprocessBackend)
    b._model_id = "m"
    b._state_value = "loaded"
    b._recv_dead = False
    b._death_handled = False
    b._outbound = queue.Queue()
    b._slot_futures = {}
    b._slot_lock = threading.Lock()
    b._outstanding = 0
    b._on_result_callback = None
    b._on_death_callback = None
    b._last_worker_activity = time.monotonic()
    b._worker = SimpleNamespace(is_alive=lambda: True, pid=1, exitcode=None)
    for k, v in attrs.items():
        setattr(b, k, v)
    return b


class TestOutstandingCounter:
    def test_signal_slot_increments_and_does_not_refresh_activity(self):
        b = _bare_backend()
        old_activity = b._last_worker_activity - 100
        b._last_worker_activity = old_activity
        b.signal_slot(3, 42)
        assert b._outstanding == 1
        assert b._last_worker_activity == old_activity

    def test_handle_result_decrements_not_below_zero(self):
        b = _bare_backend(_outstanding=1, _pool=None)
        b._handle_result(42, 3, 0)
        assert b._outstanding == 0
        b._handle_result(43, 4, 0)
        assert b._outstanding == 0

    def test_queue_depth_reflects_outstanding(self):
        b = _bare_backend(_outstanding=5)
        assert b.queue_depth == 5


class TestWorkerDeath:
    def test_death_is_idempotent_and_resets_state(self):
        results = []
        deaths = []
        fut: Future = Future()
        b = _bare_backend(
            _outstanding=3,
            _slot_futures={7: (42, fut)},
            _on_result_callback=lambda *a: results.append(a),
            _on_death_callback=lambda mid: deaths.append(mid),
        )
        b._handle_worker_death("test reason")
        b._handle_worker_death("second call")
        assert b._outstanding == 0
        assert b._recv_dead is True
        assert b._state_value == "unhealthy"
        assert fut.done() and isinstance(fut.exception(), RuntimeError)
        assert results == [(42, 7, 0)]
        assert deaths == ["m"]


class TestBusyLiveness:
    def test_idle_silence_past_heartbeat_timeout_is_unhealthy(self):
        b = _bare_backend(_outstanding=0)
        b._last_worker_activity = time.monotonic() - (_WORKER_HEARTBEAT_TIMEOUT + 30)
        assert b.state == "unhealthy"

    def test_busy_silence_within_busy_timeout_is_loaded(self):
        b = _bare_backend(_outstanding=1)
        b._last_worker_activity = time.monotonic() - (_WORKER_HEARTBEAT_TIMEOUT + 30)
        assert b.state == "loaded"

    def test_busy_silence_past_busy_timeout_is_unhealthy(self):
        b = _bare_backend(_outstanding=1)
        b._last_worker_activity = time.monotonic() - (subproc._WORKER_BUSY_TIMEOUT + 10)
        assert b.state == "unhealthy"


class TestDrain:
    def test_drain_returns_immediately_when_idle(self):
        b = _bare_backend(_outstanding=0)
        unloaded = []
        b.unload = lambda: unloaded.append(1)
        t0 = time.monotonic()
        b.drain_and_unload(timeout_s=5.0)
        assert time.monotonic() - t0 < 1.0
        assert unloaded == [1]

    def test_drain_waits_on_outstanding_then_force_unloads(self):
        b = _bare_backend(_outstanding=1)
        unloaded = []
        b.unload = lambda: unloaded.append(1)
        t0 = time.monotonic()
        b.drain_and_unload(timeout_s=0.3)
        assert time.monotonic() - t0 >= 0.3
        assert unloaded == [1]


# ---------------------------------------------------------------------------
# Worker survives unserializable results
# ---------------------------------------------------------------------------


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


class _Unpicklable:
    def __reduce__(self):
        raise TypeError("cannot pickle this")


def _stats() -> dict:
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


def _write_npy_input(pool: SHMPool, slot_id: int, req_id: int) -> None:
    import io

    buf = io.BytesIO()
    np.save(buf, np.zeros((2, 2, 3), dtype=np.uint8))
    data = buf.getvalue()
    pool.mark_allocated(slot_id, request_id=req_id)
    pool.data_memoryview(slot_id)[: len(data)] = data
    pool.mark_written(slot_id, len(data))


def test_unpicklable_result_errors_slot_without_killing_batch(monkeypatch):
    pool = SHMPool.create(n_slots=2, input_mb=0.5)
    try:
        sock = _FakeSock()
        s0 = pool.alloc_slot()
        s1 = pool.alloc_slot()
        _write_npy_input(pool, s0, req_id=101)
        _write_npy_input(pool, s1, req_id=202)

        monkeypatch.setattr(
            subproc,
            "invoke_task",
            lambda model, task, images, **kw: [_Unpicklable(), {"ok": True}],
        )

        subproc._process_slots(
            object(),
            pool,
            [(s0, 101, b"{}"), (s1, 202, b"{}")],
            sock,
            lambda mvs: [None] * len(mvs),
            _Log(),
            _stats(),
        )

        assert pool.read_header(s0).status == SlotStatus.ERROR
        assert pool.read_header(s1).status == SlotStatus.DONE
        results = [
            struct.unpack(">QII", f[1])
            for f in sock.sent
            if f[0] == subproc._MSG_RESULT
        ]
        assert (101, s0, 0) in results
        assert any(r == 202 and sz > 0 for r, _, sz in results)
    finally:
        pool.close()
