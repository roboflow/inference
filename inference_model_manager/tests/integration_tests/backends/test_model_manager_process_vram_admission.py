"""Unit-level tests for VRAM-aware admission control in ModelManagerProcess.

Pure logic — no GPU, no ZMQ. Construct MMP, seed state, call method, assert.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from inference_model_manager.model_manager_process import (
    _ERR_SERVER_FULL,
    ModelManagerProcess,
    ModelState,
)

_TEST_INPUT_MB = 0.1


def _mmp(**kwargs) -> ModelManagerProcess:
    kwargs.setdefault("n_slots", 4)
    kwargs.setdefault("input_mb", _TEST_INPUT_MB)
    return ModelManagerProcess(**kwargs)


def _loaded(mmp, name, access, footprint):
    mmp._models[name] = ModelState(loaded=True)
    mmp._model_access[name] = access
    mmp._vram_meta_cache[name] = footprint


def _stub_evict(mmp):
    evicted = []

    def _evict(model_id):
        evicted.append(model_id)
        fs = mmp._models.get(model_id)
        if fs:
            fs.loaded = False
        return True

    mmp._evict_model = _evict
    return evicted


# ---------------------------------------------------------------------------
# Config / feature flag
# ---------------------------------------------------------------------------


class TestConfig:
    def test_admission_off_by_default(self):
        mmp = _mmp()
        assert mmp._vram_admission is False

    def test_config_values_stored(self):
        mmp = _mmp(
            vram_admission=True,
            vram_window_size=10,
            vram_idle_cutoff_s=123.0,
            vram_headroom_mb=256.0,
            vram_recent_window_s=15.0,
        )
        assert mmp._vram_admission is True
        assert mmp._vram_window_size == 10
        assert mmp._vram_idle_cutoff_s == 123.0
        assert mmp._vram_headroom_mb == 256.0
        assert mmp._vram_recent_window_s == 15.0

    def test_idle_cutoff_defaults_to_idle_timeout(self):
        mmp = _mmp(idle_timeout_s=42.0)
        assert mmp._vram_idle_cutoff_s == 42.0


# ---------------------------------------------------------------------------
# Footprint resolution: measured-max wins, static seeds
# ---------------------------------------------------------------------------


class TestFootprint:
    def test_measured_max_wins(self):
        mmp = _mmp()
        mmp._record_vram_samples({"a": 700})
        mmp._record_vram_samples({"a": 1200})
        mmp._record_vram_samples({"a": 900})
        mmp._vram_meta_cache["a"] = 500  # static lower — should be ignored
        assert mmp._footprint_mb("a") == 1200

    def test_static_seed_when_no_samples(self):
        mmp = _mmp()
        mmp._vram_meta_cache["a"] = 500
        assert mmp._footprint_mb("a") == 500

    def test_zero_when_neither(self):
        mmp = _mmp()
        assert mmp._footprint_mb("a") == 0


# ---------------------------------------------------------------------------
# Sliding window
# ---------------------------------------------------------------------------


class TestVramWindow:
    def test_window_respects_maxlen(self):
        mmp = _mmp(vram_window_size=3)
        for v in [100, 200, 300, 400]:
            mmp._record_vram_samples({"a": v})
        assert mmp._footprint_mb("a") == 400  # 100 dropped, max of last 3
        assert len(mmp._vram_window["a"]) == 3

    def test_samples_isolated_per_model(self):
        mmp = _mmp()
        mmp._record_vram_samples({"a": 100, "b": 900})
        assert mmp._footprint_mb("a") == 100
        assert mmp._footprint_mb("b") == 900


# ---------------------------------------------------------------------------
# Required VRAM for incoming model (metadata-only, cached)
# ---------------------------------------------------------------------------


class TestRequired:
    def test_cache_hit(self):
        mmp = _mmp()
        mmp._vram_meta_cache["m"] = 1500
        assert mmp._required_mb("m") == 1500

    def test_fetch_once_then_cached(self):
        mmp = _mmp()
        calls = []

        def fake_fetch(model_id, api_key, batch):
            calls.append(model_id)
            return 800

        mmp._fetch_vram_mb = fake_fetch
        assert mmp._required_mb("m", api_key="k") == 800
        assert mmp._required_mb("m", api_key="k") == 800
        assert calls == ["m"]  # fetched once, then cache

    def test_no_data_is_zero(self):
        mmp = _mmp()
        mmp._fetch_vram_mb = lambda model_id, api_key, batch: 0
        assert mmp._required_mb("m") == 0


# ---------------------------------------------------------------------------
# Recent inference count
# ---------------------------------------------------------------------------


class TestRecentCount:
    def test_counts_within_window(self):
        mmp = _mmp()
        now = time.monotonic()
        mmp._model_request_times["a"] = [now - 50, now - 20, now - 5]
        assert mmp._recent_count("a", window_s=30.0) == 2  # -20, -5

    def test_zero_when_no_history(self):
        mmp = _mmp()
        assert mmp._recent_count("a", window_s=30.0) == 0


# ---------------------------------------------------------------------------
# Eviction planning
# ---------------------------------------------------------------------------


class TestPlanEvictions:
    def _loaded(self, mmp, name, access, footprint):
        mmp._models[name] = ModelState(loaded=True)
        mmp._model_access[name] = access
        mmp._vram_meta_cache[name] = footprint

    def test_covers_deficit_coldest_first(self):
        mmp = _mmp()
        # all cold (access far in past vs idle cutoff 300)
        self._loaded(mmp, "a", access=3.0, footprint=400)
        self._loaded(mmp, "b", access=1.0, footprint=400)  # coldest
        self._loaded(mmp, "c", access=2.0, footprint=400)
        plan = mmp._plan_evictions(deficit_mb=700)
        assert plan == ["b", "c"]  # coldest first, stop once >=700

    def test_none_when_insufficient(self):
        mmp = _mmp()
        self._loaded(mmp, "a", access=1.0, footprint=100)
        assert mmp._plan_evictions(deficit_mb=500) is None

    def test_excludes_hot_and_in_flight(self):
        mmp = _mmp()
        now = time.monotonic()
        self._loaded(mmp, "hot", access=now, footprint=1000)  # hot
        self._loaded(mmp, "busy", access=1.0, footprint=1000)
        self._loaded(mmp, "cold", access=1.0, footprint=1000)
        mmp._pending[1] = (b"id", 0, "busy")  # in-flight
        plan = mmp._plan_evictions(deficit_mb=500)
        assert plan == ["cold"]

    def test_excludes_unloading(self):
        mmp = _mmp()
        self._loaded(mmp, "a", access=1.0, footprint=1000)
        mmp._unloading.add("a")
        assert mmp._plan_evictions(deficit_mb=500) is None


# ---------------------------------------------------------------------------
# Admission decision (sync planning)
# ---------------------------------------------------------------------------


class TestAdmissionPlan:
    def test_admit_when_enough_free(self):
        mmp = _mmp(vram_headroom_mb=0.0)
        mmp._vram_meta_cache["m"] = 1000
        mmp._gpu_free_mb = lambda: 2000.0
        decision, victims = mmp._vram_admission_plan("m")
        assert decision == "admit"
        assert victims == []

    def test_evict_when_short_but_recoverable(self):
        mmp = _mmp(vram_headroom_mb=0.0)
        mmp._vram_meta_cache["m"] = 1500
        mmp._models["old"] = ModelState(loaded=True)
        mmp._model_access["old"] = 1.0
        mmp._vram_meta_cache["old"] = 1000
        mmp._gpu_free_mb = lambda: 800.0  # deficit 700, "old" covers it
        decision, victims = mmp._vram_admission_plan("m")
        assert decision == "evict"
        assert victims == ["old"]

    def test_no_capacity_when_unrecoverable(self):
        mmp = _mmp(vram_headroom_mb=0.0)
        mmp._vram_meta_cache["m"] = 5000
        mmp._gpu_free_mb = lambda: 100.0  # nothing evictable
        decision, victims = mmp._vram_admission_plan("m")
        assert decision == "no_capacity"

    def test_need_zero_always_admits(self):
        mmp = _mmp(vram_headroom_mb=0.0)
        mmp._vram_meta_cache["m"] = 0  # no data
        mmp._gpu_free_mb = lambda: 0.0
        decision, _ = mmp._vram_admission_plan("m")
        assert decision == "admit"


# ---------------------------------------------------------------------------
# Revocable eviction execution (async, no real unload)
# ---------------------------------------------------------------------------


class TestExecuteEvictionPlan:
    def test_evicts_until_deficit_met(self):
        mmp = _mmp()
        _loaded(mmp, "b", access=1.0, footprint=400)
        _loaded(mmp, "c", access=2.0, footprint=400)
        evicted = _stub_evict(mmp)
        ok = asyncio.run(mmp._execute_eviction_plan(["b", "c"], deficit_mb=700))
        assert ok is True
        assert evicted == ["b", "c"]

    def test_stops_early_once_secured(self):
        mmp = _mmp()
        _loaded(mmp, "b", access=1.0, footprint=400)
        _loaded(mmp, "c", access=2.0, footprint=400)
        evicted = _stub_evict(mmp)
        ok = asyncio.run(mmp._execute_eviction_plan(["b", "c"], deficit_mb=300))
        assert ok is True
        assert evicted == ["b"]  # c untouched

    def test_hot_victim_elects_replacement(self):
        mmp = _mmp()
        now = time.monotonic()
        _loaded(mmp, "b", access=now, footprint=400)  # turned hot
        _loaded(mmp, "c", access=1.0, footprint=400)  # cold replacement
        evicted = _stub_evict(mmp)
        ok = asyncio.run(mmp._execute_eviction_plan(["b"], deficit_mb=400))
        assert ok is True
        assert evicted == ["c"]  # b spared, replacement evicted

    def test_abort_when_no_replacement(self):
        mmp = _mmp()
        now = time.monotonic()
        _loaded(mmp, "b", access=now, footprint=400)  # hot, only model
        evicted = _stub_evict(mmp)
        ok = asyncio.run(mmp._execute_eviction_plan(["b"], deficit_mb=400))
        assert ok is False
        assert evicted == []  # nothing unloaded


# ---------------------------------------------------------------------------
# _load_model gate integration (no real load — manager stubbed)
# ---------------------------------------------------------------------------


class TestLoadModelGate:
    def test_no_capacity_fails_load(self):
        mmp = _mmp(vram_admission=True)
        mmp._manager = object()  # non-None: skip stub path, reach gate
        mmp._vram_admission_plan = lambda model_id, api_key="": ("no_capacity", [])
        failed = []
        mmp._fail_load = lambda model_id, fs, err: failed.append(err)
        asyncio.run(mmp._load_model("m"))
        assert failed == [_ERR_SERVER_FULL]

    def test_evict_abort_fails_load(self):
        mmp = _mmp(vram_admission=True)
        mmp._manager = object()
        mmp._vram_admission_plan = lambda model_id, api_key="": ("evict", ["v"])
        mmp._required_mb = lambda model_id, api_key="": 1000
        mmp._gpu_free_mb = lambda: 0.0

        async def _abort(plan, deficit_mb):
            return False

        mmp._execute_eviction_plan = _abort
        failed = []
        mmp._fail_load = lambda model_id, fs, err: failed.append(err)
        asyncio.run(mmp._load_model("m"))
        assert failed == [_ERR_SERVER_FULL]

    def test_flag_off_skips_gate(self):
        mmp = _mmp(vram_admission=False)
        called = []
        mmp._vram_admission_plan = lambda *a, **k: called.append(1) or ("no_capacity", [])

        fake_backend = type("B", (), {"signal_slot": lambda *a, **k: None})()

        class FakeMgr:
            def load(self, *a, **k):
                return None

            def get_backend(self, model_id):
                return fake_backend

        mmp._manager = FakeMgr()
        asyncio.run(mmp._load_model("m"))
        assert called == []  # gate never consulted
        assert mmp._models["m"].loaded is True
