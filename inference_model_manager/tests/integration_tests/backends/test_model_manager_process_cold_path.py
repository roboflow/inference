"""Integration tests for ModelManagerProcess Phase 4 cold path.

Tests:
  - T_STATS returns JSON snapshot
  - T_LOAD / T_UNLOAD lifecycle messages
  - _load_flavor calls manager.load() when manager is provided
  - register_backend triggered after mock load
  - _lru_evictable_model picks LRU correctly
"""

from __future__ import annotations

import asyncio
import struct
import threading
import time
import uuid

import pytest
import zmq

from inference_model_manager.backends.utils.shm_pool import SHMPool
from inference_model_manager.model_manager_process import (
    T_ALLOC,
    T_ALLOC_OK,
    T_ENSURE_LOADED,
    T_ERROR,
    T_FREE,
    T_LOAD,
    T_MODEL_READY,
    T_OK,
    T_RESULT_READY,
    T_STATS,
    T_STATS_RESP,
    T_SUBMIT,
    T_UNLOAD,
    ModelManagerProcess,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TEST_INPUT_MB = 0.1
_TIMEOUT_S = 5.0


def _rand_addr() -> str:
    """tcp://127.0.0.1:0 — OS assigns a free port; MMP reads actual via LAST_ENDPOINT."""
    return "tcp://127.0.0.1:0"


def _req_id() -> int:
    return uuid.uuid4().int & 0xFFFF_FFFF_FFFF_FFFF


# ---------------------------------------------------------------------------
# Mock ModelManager
# ---------------------------------------------------------------------------


class _MockManager:
    """Duck-type compatible with ModelManager for Phase 4 tests."""

    def __init__(self, n_slots: int = 8, input_mb: float = _TEST_INPUT_MB):
        self.loaded: dict[str, bytes] = {}  # flavor → result_bytes

        self.calls: list[str] = []  # log of method calls
        self._mmp: ModelManagerProcess | None = None
        self._n_slots = n_slots
        self._input_mb = input_mb
        self._pool = None

    def _ensure_pool(self):
        if self._pool is None:
            from inference_model_manager.backends.utils.shm_pool import SHMPool

            self._pool = SHMPool.create(self._n_slots, self._input_mb)

    def set_mmp(self, mmp: ModelManagerProcess) -> None:
        self._mmp = mmp

    def register(self, flavor: str, result_bytes: bytes = b"ok") -> None:
        """Pre-register a flavor so load() will succeed."""
        self.loaded[flavor] = result_bytes

    # --- ModelManager interface ---

    def load(self, model_id: str, api_key: str, **kwargs) -> None:
        self.calls.append(f"load:{model_id}")
        if model_id not in self.loaded:
            raise RuntimeError(f"Mock: unknown model '{model_id}'")
        # Backend becomes available via get_backend() after this call

    def get_backend(self, model_id: str):
        if model_id not in self.loaded or self._mmp is None:
            return None
        result = self.loaded[model_id]
        mmp = self._mmp
        return _MockBackend(mmp, result_bytes=result)

    def unload(
        self, model_id: str, drain: bool = False, drain_timeout_s: float = 30.0
    ) -> None:
        self.calls.append(f"unload:{model_id}:drain={drain}")
        self.loaded.pop(model_id, None)

    def stats(self) -> dict:
        return {"gpus": [], "models_loaded": list(self.loaded.keys())}

    def shutdown(self) -> None:
        self.calls.append("shutdown")


class _MockBackend:
    """Writes fake result to SHM and calls mmp.on_result() from a thread."""

    def __init__(self, mmp: ModelManagerProcess, result_bytes: bytes = b"ok"):
        self._mmp = mmp
        self._result = result_bytes
        self._healthy = True

    @property
    def is_healthy(self) -> bool:
        return self._healthy

    def signal_slot(
        self, slot_id: int, req_id: int, params_bytes: bytes = b"{}"
    ) -> None:
        def _do() -> None:
            pool = SHMPool.attach(
                self._mmp.shm_name,
                n_slots=self._mmp._n_slots,
                input_mb=_TEST_INPUT_MB,
            )
            try:
                pool.data_memoryview(slot_id)[: len(self._result)] = self._result
            finally:
                pool.close()
            self._mmp.on_result(req_id, slot_id, len(self._result))

        threading.Thread(target=_do, daemon=True).start()


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class _MMPHarness:
    """Starts MMP in a background asyncio thread; provides sync ZMQ client."""

    def __init__(self, manager=None, n_slots: int = 8):
        self._bind_addr = _rand_addr()
        self._ready = threading.Event()
        self._mmp = ModelManagerProcess(
            n_slots=n_slots,
            input_mb=_TEST_INPUT_MB,
            stale_reap_interval_s=1.0,
            stale_slot_max_age_s=2.0,
        )
        if manager is not None:
            self._mmp._manager = manager
            if hasattr(manager, "set_mmp"):
                manager.set_mmp(self._mmp)
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="mmp-cold-test"
        )
        self._thread.start()
        assert self._ready.wait(timeout=_TIMEOUT_S), "MMP did not start"

        self.addr = self._mmp.bound_addr or self._bind_addr

        self._ctx = zmq.Context()
        self._dealer = self._ctx.socket(zmq.DEALER)
        self._dealer.setsockopt(zmq.RCVTIMEO, int(_TIMEOUT_S * 1000))
        self._dealer.setsockopt(zmq.LINGER, 0)
        self._dealer.setsockopt(
            zmq.IDENTITY,
            f"uv_{threading.get_ident()}_{uuid.uuid4().hex[:8]}".encode(),
        )
        self._dealer.connect(self.addr)

    def _run_loop(self) -> None:
        asyncio.run(self._mmp.run(addr=self._bind_addr, ready_event=self._ready))

    @property
    def mmp(self) -> ModelManagerProcess:
        return self._mmp

    # --- ZMQ helpers ---

    def send(self, msg_type: bytes, payload: bytes) -> None:
        self._dealer.send_multipart([msg_type, payload])

    def recv(self) -> tuple[bytes, bytes]:
        parts = self._dealer.recv_multipart()
        return parts[0], parts[1]

    # --- Protocol helpers ---

    def stats_req(self) -> dict:
        rid = _req_id()
        self.send(T_STATS, struct.pack(">Q", rid))
        msg_type, frame = self.recv()
        assert msg_type == T_STATS_RESP, f"expected T_STATS_RESP, got {msg_type!r}"
        import json

        _req_id_back, json_len = struct.unpack_from(">QI", frame)
        return json.loads(frame[12 : 12 + json_len])

    def load_req(self, flavor: str, api_key: str = "") -> bytes:
        rid = _req_id()
        fb = flavor.encode()
        kb = api_key.encode()
        payload = struct.pack(">QHH", rid, len(fb), len(kb)) + fb + kb
        # Reconstruct as: req_id(8) flavor_len(2) flavor(N) api_key_len(2) api_key(M)
        payload = (
            struct.pack(">QH", rid, len(fb)) + fb + struct.pack(">H", len(kb)) + kb
        )
        self.send(T_LOAD, payload)
        msg_type, _ = self.recv()
        return msg_type

    def lifecycle_req(self, msg_type_out: bytes, flavor: str) -> bytes:
        rid = _req_id()
        fb = flavor.encode()
        payload = struct.pack(">QH", rid, len(fb)) + fb
        self.send(msg_type_out, payload)
        msg_type, _ = self.recv()
        return msg_type

    def ensure_loaded(self, flavor: str, wait_ms: int = 5000) -> bytes:
        rid = _req_id()
        fb = flavor.encode()
        payload = struct.pack(">QIH", rid, wait_ms, len(fb)) + fb
        self.send(T_ENSURE_LOADED, payload)
        msg_type, _ = self.recv()
        return msg_type

    def alloc(self, flavor: str) -> tuple[int, int]:
        rid = _req_id()
        fb = flavor.encode()
        payload = struct.pack(">QH", rid, len(fb)) + fb
        self.send(T_ALLOC, payload)
        msg_type, frame = self.recv()
        assert msg_type == T_ALLOC_OK
        _, slot_id = struct.unpack(">QI", frame)
        return rid, slot_id

    def submit(self, req_id: int, slot_id: int, flavor: str, data: bytes) -> None:
        pool = SHMPool.attach(
            self.mmp.shm_name,
            n_slots=self.mmp._n_slots,
            input_mb=_TEST_INPUT_MB,
        )
        try:
            pool.data_memoryview(slot_id)[: len(data)] = data
        finally:
            pool.close()
        fb = flavor.encode()
        header = struct.pack(">QIIH", req_id, slot_id, len(data), len(fb)) + fb
        self.send(T_SUBMIT, header)

    def free(self, slot_id: int) -> None:
        self.send(T_FREE, struct.pack(">I", slot_id))

    def teardown(self) -> None:
        self._mmp.stop()
        self._dealer.close()
        self._ctx.term()
        self._thread.join(timeout=3.0)


# ---------------------------------------------------------------------------
# Tests — T_STATS
# ---------------------------------------------------------------------------


class TestStats:
    def test_stats_returns_json(self):
        h = _MMPHarness()
        snap = h.stats_req()
        h.teardown()
        assert isinstance(snap, dict)
        assert "mmp_free_slots" in snap
        assert "mmp_total_slots" in snap
        assert snap["mmp_total_slots"] == 8

    def test_stats_reflects_manager(self):
        mgr = _MockManager()
        mgr.register("yolov8n")
        h = _MMPHarness(manager=mgr)
        # Trigger load so manager.stats() has content
        h.ensure_loaded("yolov8n")
        snap = h.stats_req()
        h.teardown()
        assert "models_loaded" in snap  # from mock manager.stats()

    def test_stats_free_slots_decrements_after_alloc(self):
        mgr = _MockManager()
        mgr.register("m")
        h = _MMPHarness(manager=mgr)
        h.ensure_loaded("m")
        before = h.stats_req()["mmp_free_slots"]
        h.alloc("m")
        after = h.stats_req()["mmp_free_slots"]
        h.teardown()
        assert after == before - 1


# ---------------------------------------------------------------------------
# Tests — Lifecycle messages
# ---------------------------------------------------------------------------


class TestLifecycleMessages:
    def test_t_load_returns_ok(self):
        mgr = _MockManager()
        mgr.register("yolov8n")
        h = _MMPHarness(manager=mgr)
        msg = h.load_req("yolov8n")
        h.teardown()
        assert msg == T_OK

    def test_t_load_triggers_manager_load(self):
        mgr = _MockManager()
        mgr.register("yolov8n")
        h = _MMPHarness(manager=mgr)
        h.load_req("yolov8n")
        # Give _load_flavor task time to run in executor
        time.sleep(0.3)
        h.teardown()
        assert any(c.startswith("load:yolov8n") for c in mgr.calls)

    def test_t_load_idempotent_when_already_loaded(self):
        mgr = _MockManager()
        mgr.register("m")
        h = _MMPHarness(manager=mgr)
        h.mmp.register_backend("m", _MockBackend(h.mmp))
        msg = h.load_req("m")  # already loaded — T_OK without triggering load
        h.teardown()
        assert msg == T_OK

    def test_t_unload_returns_ok(self):
        mgr = _MockManager()
        mgr.register("m")
        h = _MMPHarness(manager=mgr)
        h.ensure_loaded("m")
        msg = h.lifecycle_req(T_UNLOAD, "m")
        h.teardown()
        assert msg == T_OK

    def test_t_unload_calls_manager(self):
        mgr = _MockManager()
        mgr.register("m")
        h = _MMPHarness(manager=mgr)
        h.ensure_loaded("m")
        h.lifecycle_req(T_UNLOAD, "m")
        h.teardown()
        assert "unload:m:drain=False" in mgr.calls


# ---------------------------------------------------------------------------
# Tests — manager.load() integration via T_ENSURE_LOADED
# ---------------------------------------------------------------------------


class TestAutoLoad:
    def test_ensure_loaded_triggers_manager_load(self):
        mgr = _MockManager()
        mgr.register("yolov8n")
        h = _MMPHarness(manager=mgr)
        msg = h.ensure_loaded("yolov8n")
        h.teardown()
        assert msg == T_MODEL_READY
        assert any(c.startswith("load:yolov8n") for c in mgr.calls)

    def test_ensure_loaded_unknown_returns_error(self):
        """Unknown model (not in mock manager) → load fails → T_ERROR."""
        mgr = _MockManager()  # no models registered
        h = _MMPHarness(manager=mgr)
        msg = h.ensure_loaded("no-such-model")
        h.teardown()
        assert msg == T_ERROR

    def test_end_to_end_with_mock_manager(self):
        """Full lifecycle: load → alloc → submit → result → free."""
        mgr = _MockManager()
        mgr.register("det", result_bytes=b"detection result")
        h = _MMPHarness(manager=mgr)

        assert h.ensure_loaded("det") == T_MODEL_READY

        req_id, slot_id = h.alloc("det")
        h.submit(req_id, slot_id, "det", b"image bytes")

        msg_type, frame = h.recv()
        assert msg_type == T_RESULT_READY
        resp_req_id, resp_slot_id, result_sz = struct.unpack(">QII", frame)
        assert resp_req_id == req_id
        assert resp_slot_id == slot_id
        assert result_sz == len(b"detection result")

        h.free(slot_id)
        h.teardown()


# ---------------------------------------------------------------------------
# Tests — LRU eviction logic (unit-level, no ZMQ)
# ---------------------------------------------------------------------------


class TestLRUEviction:
    def _make_mmp(self) -> ModelManagerProcess:
        return ModelManagerProcess(
            n_slots=4,
            input_mb=_TEST_INPUT_MB,
        )

    def test_eviction_returns_none_when_no_loaded_models(self):
        mmp = self._make_mmp()
        assert mmp._pick_eviction_candidate() is None

    def test_eviction_picks_coldest_model(self):
        mmp = self._make_mmp()
        from inference_model_manager.model_manager_process import ModelState

        mmp._models["a"] = ModelState(loaded=True)
        mmp._models["b"] = ModelState(loaded=True)
        # Both cold (access time far in the past vs idle_timeout_s=300)
        mmp._model_access["a"] = 1.0  # older = colder
        mmp._model_access["b"] = 2.0
        assert mmp._pick_eviction_candidate() == "a"

    def test_eviction_returns_none_when_all_hot(self):
        mmp = self._make_mmp()
        import time

        from inference_model_manager.model_manager_process import ModelState

        now = time.monotonic()
        mmp._models["a"] = ModelState(loaded=True)
        mmp._models["b"] = ModelState(loaded=True)
        # Both hot (accessed just now, well within idle_timeout_s=300)
        mmp._model_access["a"] = now
        mmp._model_access["b"] = now
        assert mmp._pick_eviction_candidate() is None

    def test_eviction_skips_in_flight(self):
        mmp = self._make_mmp()
        from inference_model_manager.model_manager_process import ModelState

        mmp._models["a"] = ModelState(loaded=True)
        mmp._models["b"] = ModelState(loaded=True)
        mmp._model_access["a"] = 1.0  # cold
        mmp._model_access["b"] = 1.0  # cold
        mmp._pending[42] = (b"id", 0, "a")  # "a" is in-flight
        assert mmp._pick_eviction_candidate() == "b"

    def test_check_and_evict_calls_manager_unload_drain(self):
        mgr = _MockManager()
        mmp = ModelManagerProcess(
            n_slots=4,
            input_mb=_TEST_INPUT_MB,
            evict_threshold=0.0,  # always evict
        )
        mmp._manager = mgr  # swap in mock for this test
        from inference_model_manager.model_manager_process import ModelState

        mmp._models["m"] = ModelState(loaded=True)
        mmp._model_access["m"] = 1.0
        mmp._check_and_evict()
        assert "unload:m:drain=True" in mgr.calls
        assert mmp._models["m"].loaded is False


# ---------------------------------------------------------------------------
# Worker auto-restart tests
# ---------------------------------------------------------------------------


class TestWorkerAutoRestart:
    """Tests for auto-restart when a subprocess backend worker dies."""

    def _make_mmp(self) -> ModelManagerProcess:
        return ModelManagerProcess(
            n_slots=4,
            input_mb=_TEST_INPUT_MB,
        )

    def _make_reload_mmp(self):
        """MMP with _load_model stubbed and running event loop for create_task."""
        mmp = self._make_mmp()
        mmp._reload_calls = []

        async def _fake_load(model_id, **kw):
            mmp._reload_calls.append(model_id)

        mmp._load_model = _fake_load
        return mmp

    def _run_with_loop(self, mmp, fn):
        """Run fn() inside an event loop so asyncio.create_task works."""

        async def _run():
            mmp._loop = asyncio.get_running_loop()
            fn()
            await asyncio.sleep(0)  # let tasks run

        asyncio.run(_run())
        mmp._loop = None

    def test_schedule_reload_marks_loading(self):
        mmp = self._make_reload_mmp()
        from inference_model_manager.model_manager_process import ModelState

        mmp._models["m"] = ModelState(loaded=True)

        self._run_with_loop(mmp, lambda: mmp._schedule_reload("m"))
        assert mmp._models["m"].loaded is False
        assert mmp._models["m"].loading is True
        assert "m" in mmp._reload_calls

    def test_schedule_reload_idempotent(self):
        mmp = self._make_reload_mmp()
        from inference_model_manager.model_manager_process import ModelState

        mmp._models["m"] = ModelState(loaded=False, loading=True)

        self._run_with_loop(mmp, lambda: mmp._schedule_reload("m"))
        # Already loading — should not fire _load_model again
        assert mmp._reload_calls == []

    def test_schedule_reload_noop_for_unknown_model(self):
        mmp = self._make_reload_mmp()

        self._run_with_loop(mmp, lambda: mmp._schedule_reload("nonexistent"))
        assert mmp._reload_calls == []

    def test_forward_to_unhealthy_backend_triggers_reload(self):
        mmp = self._make_reload_mmp()
        from inference_model_manager.model_manager_process import ModelState

        mmp._models["m"] = ModelState(loaded=True)
        backend = _MockBackend(mmp)
        backend._healthy = False
        mmp._backends["m"] = backend
        mmp._model_access["m"] = 1.0
        # Simulate a real pending request so _on_result_on_loop doesn't hit pool
        mmp._pending[99] = (b"test_identity", 0, "m")

        self._run_with_loop(
            mmp, lambda: mmp._forward_to_backend("m", slot_id=0, req_id=99)
        )
        assert mmp._models["m"].loaded is False
        assert mmp._models["m"].loading is True
        assert "m" in mmp._reload_calls

    def test_forward_to_healthy_backend_does_not_reload(self):
        mmp = self._make_reload_mmp()
        from inference_model_manager.model_manager_process import ModelState

        mmp._models["m"] = ModelState(loaded=True)
        mmp._model_access["m"] = 1.0

        # Lightweight mock — just records signal_slot calls, no SHM
        class _NoopBackend:
            is_healthy = True
            signals = []

            def signal_slot(self, slot_id, req_id, params_bytes=b"{}"):
                self.signals.append((slot_id, req_id))

        backend = _NoopBackend()
        mmp._backends["m"] = backend

        self._run_with_loop(
            mmp, lambda: mmp._forward_to_backend("m", slot_id=0, req_id=99)
        )
        assert mmp._models["m"].loaded is True
        assert mmp._models["m"].loading is False
        assert mmp._reload_calls == []
        assert backend.signals == [(0, 99)]
