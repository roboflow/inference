"""MMP cold-path: NVML telemetry, admission gate, eviction coordination,
non-blocking admin handlers."""

from __future__ import annotations

import asyncio
import struct
import threading
from types import SimpleNamespace

import pytest

import inference_model_manager.model_manager_process as mmp_mod
from inference_model_manager.model_manager_process import (
    _ERR_LOAD_FAILED,
    T_ERROR,
    T_OK,
    ModelManagerProcess,
    ModelState,
    _gpu_used_fraction,
)


def _fake_nvml(free_b=4 << 30, used_b=12 << 30, total_b=16 << 30):
    return SimpleNamespace(free=free_b, used=used_b, total=total_b)


class TestNvmlTelemetry:
    def test_gpu_used_fraction_is_device_level(self, monkeypatch):
        monkeypatch.setattr(mmp_mod, "_nvml_mem_info", lambda: _fake_nvml())
        assert _gpu_used_fraction() == pytest.approx(12 / 16)

    def test_gpu_used_fraction_zero_when_unavailable(self, monkeypatch):
        monkeypatch.setattr(mmp_mod, "_nvml_mem_info", lambda: None)
        assert _gpu_used_fraction() == 0.0

    def test_gpu_free_mb_none_when_unavailable(self, monkeypatch):
        monkeypatch.setattr(mmp_mod, "_nvml_mem_info", lambda: None)
        mmp = ModelManagerProcess.__new__(ModelManagerProcess)
        assert mmp._gpu_free_mb() is None

    def test_gpu_free_mb_from_nvml(self, monkeypatch):
        monkeypatch.setattr(mmp_mod, "_nvml_mem_info", lambda: _fake_nvml())
        mmp = ModelManagerProcess.__new__(ModelManagerProcess)
        assert mmp._gpu_free_mb() == pytest.approx(4 * 1024.0)


def _bare_mmp(**attrs) -> ModelManagerProcess:
    mmp = ModelManagerProcess.__new__(ModelManagerProcess)
    mmp._pending = {}
    mmp._inflight = {}
    mmp._models = {}
    mmp._backends = {}
    mmp._model_access = {}
    mmp._model_request_times = {}
    mmp._unloading = set()
    mmp._vram_window = {}
    mmp._vram_meta_cache = {}
    mmp._vram_window_size = 5
    mmp._vram_idle_cutoff_s = 300.0
    mmp._vram_headroom_mb = 0.0
    mmp._vram_recent_window_s = 60.0
    mmp._idle_timeout_s = 300.0
    mmp._rejects_pool_full = 0
    mmp._shared_heads = {}
    mmp._head_base_key = {}
    mmp._preloaded_shared_bases = {}
    for k, v in attrs.items():
        setattr(mmp, k, v)
    return mmp


class TestAdmissionGate:
    def test_admits_when_gpu_telemetry_unavailable(self):
        mmp = _bare_mmp()
        mmp._gpu_free_mb = lambda: None
        mmp._vram_meta_cache["m"] = 5000
        decision, victims, deficit = asyncio.run(mmp._vram_admission_plan("m"))
        assert decision == "admit"

    def test_need_zero_admits_when_headroom_satisfied(self):
        mmp = _bare_mmp(_vram_headroom_mb=4096.0)
        mmp._gpu_free_mb = lambda: 8192.0
        mmp._vram_meta_cache["m"] = 0
        decision, victims, deficit = asyncio.run(mmp._vram_admission_plan("m"))
        assert decision == "admit"

    def test_need_zero_still_enforces_headroom_floor(self):
        mmp = _bare_mmp(_vram_headroom_mb=4096.0)
        mmp._gpu_free_mb = lambda: 0.0
        mmp._vram_meta_cache["m"] = 0
        decision, victims, deficit = asyncio.run(mmp._vram_admission_plan("m"))
        assert decision == "no_capacity"
        assert deficit == pytest.approx(4096.0)

    def test_evict_returns_deficit(self):
        mmp = _bare_mmp()
        mmp._gpu_free_mb = lambda: 800.0
        mmp._vram_meta_cache["m"] = 1500
        mmp._models["old"] = ModelState(loaded=True)
        mmp._model_access["old"] = 1.0
        mmp._vram_meta_cache["old"] = 1000
        decision, victims, deficit = asyncio.run(mmp._vram_admission_plan("m"))
        assert decision == "evict"
        assert victims == ["old"]
        assert deficit == pytest.approx(700.0)


class TestEvictionCoordination:
    def test_pick_candidate_excludes_unloading(self):
        mmp = _bare_mmp()
        mmp._models["a"] = ModelState(loaded=True)
        mmp._model_access["a"] = 1.0
        mmp._unloading.add("a")
        assert mmp._pick_eviction_candidate() is None

    def test_evict_marks_victim_before_drain(self):
        mmp = _bare_mmp()
        mmp._models["a"] = ModelState(loaded=True)
        mmp._model_access["a"] = 1.0
        mmp._backends["a"] = object()
        seen = {}

        def _slow_unload(model_id):
            # Runs in executor while the drain is "in progress" — observe
            # loop-side state as a concurrent picker would.
            seen["loaded"] = mmp._models["a"].loaded
            seen["unloading"] = model_id in mmp._unloading
            seen["pick"] = mmp._pick_eviction_candidate()
            return True

        mmp._unload_blocking = _slow_unload
        ok = asyncio.run(mmp._evict_model("a"))
        assert ok is True
        assert seen == {"loaded": False, "unloading": True, "pick": None}
        assert "a" not in mmp._backends
        assert "a" not in mmp._unloading

    def test_failed_evict_keeps_state_cleanupable(self):
        mmp = _bare_mmp()
        mmp._models["a"] = ModelState(loaded=True)
        mmp._unload_blocking = lambda model_id: False
        ok = asyncio.run(mmp._evict_model("a"))
        assert ok is False
        assert "a" not in mmp._unloading  # always discarded


class TestEvictionReloadRace:
    def test_eviction_keeps_backend_registered_mid_drain(self):
        async def _run():
            mmp = _bare_mmp()
            mmp._loop = asyncio.get_running_loop()
            draining = threading.Event()
            release = threading.Event()

            def _slow_unload(model_id):
                draining.set()
                release.wait(2)
                return True

            mmp._unload_blocking = _slow_unload
            old = object()
            fresh = object()
            mmp._models["m"] = ModelState(loaded=True)
            mmp._backends["m"] = old

            evict = asyncio.create_task(mmp._evict_model("m"))
            await asyncio.get_running_loop().run_in_executor(None, draining.wait, 2)
            mmp.register_backend("m", fresh)
            release.set()
            ok = await evict
            assert ok is True
            fs = mmp._models["m"]
            assert fs.loaded is True
            assert mmp._backends.get("m") is fresh
            assert "m" in mmp._model_access

        asyncio.run(_run())

    def test_ensure_loaded_during_eviction_defers_load_until_drain_done(self):
        async def _run():
            mmp = _handler_mmp()
            mmp._loop = asyncio.get_running_loop()
            mmp._cache_hits = 0
            mmp._cache_misses = 0
            draining = threading.Event()
            release = threading.Event()
            loads: list = []

            def _slow_unload(model_id):
                draining.set()
                release.wait(2)
                return True

            mmp._unload_blocking = _slow_unload

            async def _fake_load(model_id, api_key="", device=""):
                loads.append(model_id)
                mmp.register_backend(model_id, object())

            mmp._load_model = _fake_load
            mmp._models["m"] = ModelState(loaded=True)
            mmp._backends["m"] = object()

            evict = asyncio.create_task(mmp._evict_model("m"))
            await asyncio.get_running_loop().run_in_executor(None, draining.wait, 2)

            frame = struct.pack(">QIH", 7, 5000, 1) + b"m"
            await mmp._handle_ensure_loaded(b"cli", [frame])
            assert loads == []  # load must not start while drain in progress

            release.set()
            ok = await evict
            assert ok is True
            await asyncio.sleep(0.05)
            assert loads == ["m"]
            assert mmp._models["m"].loaded is True
            assert mmp._backends.get("m") is not None

        asyncio.run(_run())


def _load_frame(req_id: int, model_id: bytes = b"m") -> list[bytes]:
    return [struct.pack(">QH", req_id, len(model_id)) + model_id + struct.pack(">H", 0)]


def _handler_mmp():
    mmp = _bare_mmp()
    mmp._bg_tasks = set()
    mmp._sent = []

    async def _send(identity, msg_type, payload):
        mmp._sent.append((identity, msg_type, payload))
        return True

    mmp._send = _send
    return mmp


class TestHandleLoad:
    def test_joins_in_progress_load_and_replies_ok_on_flush(self):
        async def _run():
            mmp = _handler_mmp()
            mmp._models["m"] = ModelState(loading=True)  # load in flight
            await mmp._handle_load(b"id1", _load_frame(7))
            assert mmp._sent == []  # no premature error
            fs = mmp._models["m"]
            assert fs.admin_waiters == [(b"id1", 7)]
            fs.loaded, fs.loading = True, False
            mmp._flush_load_waiters("m")
            for _ in range(3):
                await asyncio.sleep(0)
            return mmp._sent

        sent = asyncio.run(_run())
        assert sent == [(b"id1", T_OK, struct.pack(">Q", 7))]

    def test_fail_load_errors_admin_waiters_and_resets_latch(self):
        async def _run():
            mmp = _handler_mmp()

            async def _boom_load(model_id, api_key="", device=""):
                raise RuntimeError("weights download exploded")

            mmp._load_model_inner = _boom_load
            await mmp._handle_load(b"id1", _load_frame(7))
            # _handle_load spawns the load; let it crash and flush
            for _ in range(5):
                await asyncio.sleep(0)
            return mmp

        mmp = asyncio.run(_run())
        assert mmp._models["m"].loading is False  # latch reset
        assert (b"id1", T_ERROR, struct.pack(">QB", 7, _ERR_LOAD_FAILED)) in mmp._sent

    def test_handle_load_does_not_block_on_cold_load(self):
        async def _run():
            mmp = _handler_mmp()
            started = asyncio.Event()
            release = asyncio.Event()

            async def _slow_load(model_id, api_key="", device=""):
                started.set()
                await release.wait()

            mmp._load_model_inner = _slow_load
            await asyncio.wait_for(
                mmp._handle_load(b"id1", _load_frame(7)), timeout=0.5
            )  # returns immediately
            await asyncio.wait_for(started.wait(), timeout=0.5)
            release.set()

        asyncio.run(_run())


class TestHandleUnload:
    def test_unload_does_not_block_dispatch(self):
        async def _run():
            mmp = _handler_mmp()
            import threading

            release = threading.Event()
            calls = []

            class _Mgr:
                def unload(self, model_id, drain=False, **kw):
                    calls.append((model_id, drain))
                    release.wait(timeout=5)

            mmp._manager = _Mgr()
            mmp._models["m"] = ModelState(loaded=True)
            frame = [struct.pack(">QH", 9, 1) + b"m"]
            await asyncio.wait_for(mmp._handle_unload(b"id1", frame), timeout=0.5)
            assert mmp._sent == []  # reply not sent yet
            release.set()
            for _ in range(50):
                await asyncio.sleep(0.01)
                if mmp._sent:
                    break
            assert mmp._sent == [(b"id1", T_OK, struct.pack(">Q", 9))]
            assert calls == [("m", True)]  # drain=True

        asyncio.run(_run())


class TestAllocBackpressure:
    def test_alloc_rejected_when_model_at_inflight_cap(self, monkeypatch):
        from inference_model_manager import configuration as cfg
        from inference_model_manager.backends.utils.shm_pool import SHMPool
        from inference_model_manager.model_manager_process import _ERR_POOL_FULL

        monkeypatch.setattr(cfg, "INFERENCE_MAX_INFLIGHT_PER_MODEL", 2)
        pool = SHMPool.create(n_slots=8, input_mb=0.1)
        try:
            mmp = _handler_mmp()
            mmp._pool = pool
            mmp._pending = {
                1: (b"id", 0, "busy-model"),
                2: (b"id", 1, "busy-model"),
                3: (b"id", 2, "other-model"),
            }
            frame = [struct.pack(">QH", 7, 10) + b"busy-model"]
            asyncio.run(mmp._handle_alloc(b"id1", frame))
            assert mmp._sent == [
                (b"id1", T_ERROR, struct.pack(">QB", 7, _ERR_POOL_FULL))
            ]
            # Other model still allocates fine
            mmp._sent.clear()
            frame2 = [struct.pack(">QH", 8, 11) + b"other-model"]
            asyncio.run(mmp._handle_alloc(b"id1", frame2))
            assert mmp._sent[0][1] != T_ERROR
        finally:
            pool.close()


class TestRegisterBackendAccess:
    def test_register_overwrites_stale_access_timestamp(self):
        mmp = _bare_mmp()
        mmp._loop = None
        mmp._model_access["m"] = 1.0  # pre-crash stale
        mmp.register_backend("m", object())
        assert mmp._model_access["m"] > 1.0


class TestRequiredMbCaching:
    def test_transient_fetch_failure_not_cached(self):
        mmp = _bare_mmp()
        results = iter([None, 1234])
        mmp._fetch_vram_mb = lambda model_id, api_key, batch: next(results)
        assert mmp._required_mb("m") == 0  # failure → 0, NOT cached
        assert mmp._required_mb("m") == 1234  # retried, now cached
        assert mmp._required_mb("m") == 1234
        assert mmp._vram_meta_cache["m"] == 1234

    def test_no_profile_zero_is_cached(self):
        mmp = _bare_mmp()
        calls = []
        mmp._fetch_vram_mb = lambda model_id, api_key, batch: calls.append(1) or 0
        assert mmp._required_mb("m") == 0
        assert mmp._required_mb("m") == 0
        assert calls == [1]  # 0 = no-data, cached
