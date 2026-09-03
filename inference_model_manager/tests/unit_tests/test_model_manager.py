"""Unit tests for ModelManager.

Uses mock backends — no real models, no GPU, no torch. Fast.
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest

from inference_model_manager.model_manager import ModelManager
from inference_model_manager.registry_defaults import registry as _registry
from inference_model_manager.serializers_typed import serialize_passthrough
from inference_model_manager.validators import validate_passthrough

# ─── Fake model + backend ──────────────────────────────────────────


class FakeModel:
    """Minimal model for unit tests. No base class needed."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._inference_count = 0

    def infer(self, images=None, **kwargs) -> Any:
        self._inference_count += 1
        return {"prediction": "fake", "model_id": self.model_id}


# Register FakeModel in registry so dispatch can find it.
_registry.register(
    FakeModel,
    "infer",
    method="infer",
    default=True,
    params=["images"],
    validator=validate_passthrough,
    serializer=serialize_passthrough,
    response_type="roboflow-generic-v1",
)


class FakeBackend:
    """Minimal Backend stand-in for unit tests."""

    def __init__(self, model_id: str, **kwargs):
        self._fake_model = FakeModel(model_id)
        self._state = "loaded"
        self._unloaded = False
        self.last_used_ts = None

    @property
    def model(self) -> FakeModel:
        return self._fake_model

    # Lifecycle
    def unload(self, drain: bool = False, drain_timeout_s: float = 30.0) -> None:
        self._state = "unhealthy"
        self._unloaded = True

    # Observability
    @property
    def device(self) -> str:
        return "cpu"

    @property
    def state(self) -> str:
        return self._state

    @property
    def is_healthy(self) -> bool:
        return self._state == "loaded"

    @property
    def is_accepting(self) -> bool:
        return self._state == "loaded"

    @property
    def queue_depth(self) -> int:
        return 0

    @property
    def max_batch_size(self) -> Optional[int]:
        return None

    def record_inference(self, t0: float, error: bool = False) -> None:
        pass

    def drain_and_unload(self, timeout_s: float = 30.0) -> None:
        self.unload()

    def stats(self) -> Dict[str, Any]:
        return {
            "backend_type": "fake",
            "state": self.state,
            "is_accepting": self.is_accepting,
            "inference_count": self._fake_model._inference_count,
            "error_count": 0,
        }

    @property
    def class_names(self) -> Optional[List[str]]:
        return ["cat", "dog"]


def _patch_create_backend(manager: ModelManager, backends: Dict[str, FakeBackend]):
    """Monkey-patch _create_backend to return FakeBackend instances."""
    original = manager._create_backend

    def fake_create(model_id, api_key, backend, **kwargs):
        fb = FakeBackend(model_id)
        backends[model_id] = fb
        return fb

    manager._create_backend = fake_create


# ─── Tests ──────────────────────────────────────────────────────────


class TestModelManagerLifecycle:

    def test_load_and_contains(self):
        mm = ModelManager()
        backends = {}
        _patch_create_backend(mm, backends)

        mm.load("model-a", api_key="")
        assert "model-a" in mm
        assert len(mm) == 1
        assert mm.loaded_models == ["model-a"]

    def test_load_duplicate_raises(self):
        mm = ModelManager()
        _patch_create_backend(mm, {})

        mm.load("model-a", api_key="")
        with pytest.raises(ValueError, match="already loaded"):
            mm.load("model-a", api_key="")

    def test_unload(self):
        mm = ModelManager()
        backends = {}
        _patch_create_backend(mm, backends)

        mm.load("model-a", api_key="")
        mm.unload("model-a")

        assert "model-a" not in mm
        assert len(mm) == 0
        assert backends["model-a"]._unloaded is True

    def test_unload_missing_raises(self):
        mm = ModelManager()
        with pytest.raises(KeyError, match="not loaded"):
            mm.unload("nonexistent")

    def test_load_multiple_models(self):
        mm = ModelManager()
        _patch_create_backend(mm, {})

        mm.load("model-a", api_key="")
        mm.load("model-b", api_key="")
        mm.load("model-c", api_key="")

        assert len(mm) == 3
        assert set(mm.loaded_models) == {"model-a", "model-b", "model-c"}

    def test_shutdown_unloads_all(self):
        mm = ModelManager()
        backends = {}
        _patch_create_backend(mm, backends)

        mm.load("model-a", api_key="")
        mm.load("model-b", api_key="")
        mm.shutdown()

        assert len(mm) == 0
        assert backends["model-a"]._unloaded is True
        assert backends["model-b"]._unloaded is True


class TestModelManagerInference:

    def test_process(self):
        mm = ModelManager()
        backends = {}
        _patch_create_backend(mm, backends)

        mm.load("model-a", api_key="")
        result = mm.process("model-a", images="some_image")

        assert result == {
            "type": "roboflow-generic-v1",
            "data": {"prediction": "fake", "model_id": "model-a"},
        }
        assert backends["model-a"]._fake_model._inference_count == 1

    def test_process_missing_model_raises(self):
        mm = ModelManager()
        with pytest.raises(KeyError, match="not loaded"):
            mm.process("nonexistent", images="image")

    def test_submit(self):
        mm = ModelManager()
        backends = {}
        _patch_create_backend(mm, backends)

        mm.load("model-a", api_key="")
        future = mm.submit("model-a", images="some_image")
        result = future.result(timeout=5)

        assert result is not None

    def test_submit_records_inference_stats(self):
        """submit() direct path must call backend.record_inference (P3 #1)."""
        mm = ModelManager()
        backends: dict = {}
        _patch_create_backend(mm, backends)

        mm.load("model-a", api_key="")
        fb = backends["model-a"]
        recorded: list[bool] = []
        original = fb.record_inference

        def _spy(t0: float, error: bool = False) -> None:
            recorded.append(error)
            original(t0, error=error)

        fb.record_inference = _spy

        mm.submit("model-a", images="img").result(timeout=5)
        assert recorded == [False]

    def test_submit_validates_task(self):
        """submit() direct path must raise on unknown task before queuing."""
        mm = ModelManager()
        _patch_create_backend(mm, {})
        mm.load("model-a", api_key="")

        with pytest.raises(ValueError):
            mm.submit("model-a", task="nonexistent-task", images="img")

    def test_process_async(self):
        mm = ModelManager()
        backends = {}
        _patch_create_backend(mm, backends)

        mm.load("model-a", api_key="")
        result = asyncio.run(mm.process_async("model-a", images="some_image"))

        assert result == {
            "type": "roboflow-generic-v1",
            "data": {"prediction": "fake", "model_id": "model-a"},
        }

    def test_infer_routes_to_correct_model(self):
        mm = ModelManager()
        backends = {}
        _patch_create_backend(mm, backends)

        mm.load("model-a", api_key="")
        mm.load("model-b", api_key="")

        r_a = mm.process("model-a", images="img")
        r_b = mm.process("model-b", images="img")

        assert r_a["data"]["model_id"] == "model-a"
        assert r_b["data"]["model_id"] == "model-b"
        assert backends["model-a"]._fake_model._inference_count == 1
        assert backends["model-b"]._fake_model._inference_count == 1


class TestModelManagerObservability:

    def test_stats_empty(self):
        mm = ModelManager()
        s = mm.stats()

        assert s["models_loaded"] == []
        assert s["models"] == []
        assert isinstance(s["gpus"], list)

    def test_stats_with_models(self):
        mm = ModelManager()
        backends = {}
        _patch_create_backend(mm, backends)

        mm.load("model-a", api_key="")
        mm.load("model-b", api_key="")
        mm.process("model-a", images="img")

        s = mm.stats()

        assert set(s["models_loaded"]) == {"model-a", "model-b"}
        assert len(s["models"]) == 2

        model_stats = {m["model_id"]: m for m in s["models"]}
        assert model_stats["model-a"]["inference_count"] == 1
        assert model_stats["model-b"]["inference_count"] == 0

    def test_model_stats(self):
        mm = ModelManager()
        backends = {}
        _patch_create_backend(mm, backends)

        mm.load("model-a", api_key="")
        s = mm.model_stats("model-a")

        assert s["model_id"] == "model-a"
        assert s["backend_type"] == "fake"
        assert s["state"] == "loaded"

    def test_model_stats_missing_raises(self):
        mm = ModelManager()
        with pytest.raises(KeyError, match="not loaded"):
            mm.model_stats("nonexistent")


class TestModelManagerThreadSafety:

    def test_concurrent_loads(self, monkeypatch):
        import inference_model_manager.configuration as cfg

        monkeypatch.setattr(cfg, "INFERENCE_MAX_ACTIVE_MODELS", 0)
        mm = ModelManager()
        _patch_create_backend(mm, {})
        errors = []

        def load_model(name):
            try:
                mm.load(name, api_key="")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=load_model, args=(f"model-{i}",)) for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(mm) == 10

    def test_concurrent_infer(self):
        mm = ModelManager()
        backends = {}
        _patch_create_backend(mm, backends)
        mm.load("model-a", api_key="")

        results = []

        def infer():
            r = mm.process("model-a", images="img")
            results.append(r)

        threads = [threading.Thread(target=infer) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 20
        assert backends["model-a"]._fake_model._inference_count == 20


class TestModelManagerBackendCreation:

    def test_unknown_backend_raises(self):
        mm = ModelManager()
        with pytest.raises(ValueError, match="Unknown backend"):
            mm.load("model-a", api_key="", backend="nonexistent")

    @patch("inference_model_manager.model_manager.ModelManager._create_backend")
    def test_load_passes_kwargs_to_backend(self, mock_create):
        fb = FakeBackend("model-a")
        mock_create.return_value = fb

        mm = ModelManager()
        mm.load(
            "model-a",
            api_key="test-key",
            backend="direct",
            device="cuda:1",
            batch_max_size=16,
            batch_max_delay_ms=50.0,
        )

        mock_create.assert_called_once_with(
            model_id="model-a",
            api_key="test-key",
            backend="direct",
            device="cuda:1",
            use_gpu=None,
            use_cuda_ipc=None,
            batch_max_size=16,
            batch_max_delay_ms=50.0,
        )

    @patch("inference_model_manager.model_manager.ModelManager._create_backend")
    def test_warmup_calls_process(self, mock_create):
        fb = FakeBackend("model-a")
        mock_create.return_value = fb

        mm = ModelManager()
        mm.load("model-a", api_key="", warmup_iters=3)

        assert fb._fake_model._inference_count == 3


class TestRawProcessContract:
    def test_process_serialize_false_returns_raw_prediction(self):
        mm = ModelManager()
        backends: dict = {}
        _patch_create_backend(mm, backends)
        mm.load("model-a", api_key="")
        raw = mm.process("model-a", serialize=False, images="img")
        assert raw == {"prediction": "fake", "model_id": "model-a"}

    def test_process_async_forwards_serialize_flag(self):
        mm = ModelManager()
        _patch_create_backend(mm, {})
        mm.load("model-a", api_key="")
        raw = asyncio.run(mm.process_async("model-a", serialize=False, images="img"))
        assert raw == {"prediction": "fake", "model_id": "model-a"}


class TestLoadLockScope:
    def test_concurrent_load_not_blocked_by_slow_backend_construction(self):
        import time as _time

        mm = ModelManager()
        started = threading.Event()
        release = threading.Event()

        def create(model_id, api_key, backend, **kw):
            if model_id == "slow":
                started.set()
                release.wait(timeout=5)
            return FakeBackend(model_id)

        mm._create_backend = create
        t = threading.Thread(target=lambda: mm.load("slow", api_key=""))
        t.start()
        try:
            assert started.wait(timeout=2)
            t0 = _time.monotonic()
            mm.load("fast", api_key="")  # must not wait on slow
            assert _time.monotonic() - t0 < 1.0
            assert "fast" in mm
        finally:
            release.set()
            t.join(timeout=5)
        assert "slow" in mm

    def test_duplicate_load_while_loading_raises(self):
        mm = ModelManager()
        started = threading.Event()
        release = threading.Event()

        def create(model_id, api_key, backend, **kw):
            started.set()
            release.wait(timeout=5)
            return FakeBackend(model_id)

        mm._create_backend = create
        t = threading.Thread(target=lambda: mm.load("dup", api_key=""))
        t.start()
        try:
            assert started.wait(timeout=2)
            with pytest.raises(ValueError, match="already loaded"):
                mm.load("dup", api_key="")
        finally:
            release.set()
            t.join(timeout=5)


class TestDirectDrain:
    def test_drain_waits_for_inflight(self):
        import time as _time

        from inference_model_manager.backends.direct import DirectBackend

        b = DirectBackend.__new__(DirectBackend)
        b._model_id = "m"
        b._state_value = "loaded"
        b._inflight = 1
        b._inflight_lock = threading.Lock()
        b._model = object()

        def _finish_soon():
            _time.sleep(0.2)
            b.inflight_end()

        threading.Thread(target=_finish_soon).start()
        t0 = _time.monotonic()
        b.drain_and_unload(timeout_s=5.0)
        assert _time.monotonic() - t0 >= 0.15
        assert b._model is None


class TestCudaReleaseOnUnload:
    def test_unload_releases_cuda_cache(self, monkeypatch):
        import inference_model_manager.model_manager as mm_mod

        calls = []
        monkeypatch.setattr(
            mm_mod, "_try_release_cuda_memory", lambda: calls.append(1)
        )
        mm = ModelManager()
        _patch_create_backend(mm, {})
        mm.load("model-a", api_key="")
        mm.unload("model-a")
        assert calls == [1]


class TestCapacityEviction:
    def _mm_at_cap(self, monkeypatch, cap):
        import inference_model_manager.configuration as cfg

        monkeypatch.setattr(cfg, "INFERENCE_MAX_ACTIVE_MODELS", cap)
        mm = ModelManager()
        backends: dict = {}
        _patch_create_backend(mm, backends)
        return mm, backends

    def test_lru_evicted_at_capacity(self, monkeypatch):
        mm, backends = self._mm_at_cap(monkeypatch, 2)
        mm.load("old", api_key="")
        mm.load("hot", api_key="")
        backends["old"].last_used_ts = 1.0
        backends["hot"].last_used_ts = 2.0
        mm.load("new", api_key="")
        assert "old" not in mm
        assert set(mm.loaded_models) == {"hot", "new"}
        assert backends["old"]._unloaded is True

    def test_pinned_model_survives_eviction(self, monkeypatch):
        mm, backends = self._mm_at_cap(monkeypatch, 2)
        mm.load("keep", api_key="", pinned=True)
        mm.load("bye", api_key="")
        backends["keep"].last_used_ts = 1.0
        backends["bye"].last_used_ts = 2.0
        mm.load("new", api_key="")
        assert "keep" in mm
        assert "bye" not in mm

    def test_all_pinned_proceeds_over_cap(self, monkeypatch):
        mm, _ = self._mm_at_cap(monkeypatch, 2)
        mm.load("a", api_key="", pinned=True)
        mm.load("b", api_key="", pinned=True)
        mm.load("c", api_key="")
        assert len(mm) == 3

    def test_pin_after_load(self, monkeypatch):
        mm, backends = self._mm_at_cap(monkeypatch, 2)
        mm.load("a", api_key="")
        mm.pin("a")
        mm.load("b", api_key="")
        backends["a"].last_used_ts = 1.0
        backends["b"].last_used_ts = 2.0
        mm.load("c", api_key="")
        assert "a" in mm
        assert "b" not in mm

    def test_pin_missing_raises(self):
        mm = ModelManager()
        with pytest.raises(KeyError):
            mm.pin("nope")

    def test_zero_cap_is_unbounded(self, monkeypatch):
        mm, _ = self._mm_at_cap(monkeypatch, 0)
        for i in range(12):
            mm.load(f"m{i}", api_key="")
        assert len(mm) == 12


class TestEvictionFailureSafety:
    def test_eviction_failure_does_not_leak_loading_id(self):
        mm = ModelManager()
        _patch_create_backend(mm, {})

        def boom(incoming):
            raise RuntimeError("eviction blew up")

        mm._evict_for_capacity = boom

        with pytest.raises(RuntimeError, match="eviction blew up"):
            mm.load("new", api_key="")

        assert "new" not in mm._loading_ids

        del mm._evict_for_capacity
        mm.load("new", api_key="")
        assert "new" in mm


class TestEvictionConcurrencySafety:
    def test_concurrent_loads_never_exceed_cap(self, monkeypatch):
        import time as _time

        import inference_model_manager.configuration as cfg

        cap = 3
        monkeypatch.setattr(cfg, "INFERENCE_MAX_ACTIVE_MODELS", cap)
        mm = ModelManager()
        backends: dict = {}
        _patch_create_backend(mm, backends)

        for i in range(cap):
            mm.load(f"old{i}", api_key="")
            backends[f"old{i}"].last_used_ts = float(i)

        def slow_create(model_id, api_key, backend, **kwargs):
            fb = FakeBackend(model_id)
            backends[model_id] = fb
            _time.sleep(0.2)
            return fb

        mm._create_backend = slow_create

        n_new = cap
        errors: list = []

        def load_model(name):
            try:
                mm.load(name, api_key="")
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=load_model, args=(f"new{i}",))
            for i in range(n_new)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0
        assert len(mm) <= cap

    def test_pin_race_during_eviction_selection(self, monkeypatch):
        import time as _time

        import inference_model_manager.configuration as cfg

        monkeypatch.setattr(cfg, "INFERENCE_MAX_ACTIVE_MODELS", 1)

        pin_result: list = []
        pin_thread_holder: list = []

        def try_pin():
            try:
                mm.pin("old")
                pin_result.append(("ok", None))
            except KeyError as exc:
                pin_result.append(("keyerror", exc))

        def rigged_getter(self):
            value = self.__dict__.get("_last_used_ts_raw")
            if not self.__dict__.get("_hook_fired"):
                self.__dict__["_hook_fired"] = True
                t = threading.Thread(target=try_pin, daemon=True)
                pin_thread_holder.append(t)
                t.start()
                _time.sleep(0.1)
            return value

        def rigged_setter(self, value):
            self.__dict__["_last_used_ts_raw"] = value

        monkeypatch.setattr(
            FakeBackend,
            "last_used_ts",
            property(rigged_getter, rigged_setter),
            raising=False,
        )

        mm = ModelManager()
        backends: dict = {}
        _patch_create_backend(mm, backends)
        mm.load("old", api_key="")
        backends["old"].last_used_ts = 1.0

        mm.load("new", api_key="")
        pin_thread_holder[0].join(timeout=5)

        assert len(pin_result) == 1
        kind, _ = pin_result[0]
        if kind == "ok":
            assert "old" in mm
            assert "old" in mm._pinned
        else:
            assert kind == "keyerror"


class TestEvictionConvergenceAndDrainSafety:
    def test_cold_burst_converges_to_cap(self, monkeypatch):
        import time as _time

        import inference_model_manager.configuration as cfg

        cap = 3
        monkeypatch.setattr(cfg, "INFERENCE_MAX_ACTIVE_MODELS", cap)
        mm = ModelManager()
        backends: dict = {}

        def slow_create(model_id, api_key, backend, **kwargs):
            fb = FakeBackend(model_id)
            backends[model_id] = fb
            _time.sleep(0.1)
            return fb

        mm._create_backend = slow_create

        n = 6
        errors: list = []

        def load_model(name):
            try:
                mm.load(name, api_key="")
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=load_model, args=(f"m{i}",)) for i in range(n)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0
        assert len(mm) <= cap

    def test_drain_failure_does_not_skip_remaining_victims(self, monkeypatch):
        import inference_model_manager.configuration as cfg
        import inference_model_manager.model_manager as mm_mod

        monkeypatch.setattr(cfg, "INFERENCE_MAX_ACTIVE_MODELS", 0)
        mm = ModelManager()
        backends: dict = {}
        _patch_create_backend(mm, backends)

        for i, name in enumerate(["v1", "v2", "v3"]):
            mm.load(name, api_key="")
            backends[name].last_used_ts = float(i)

        attempted: list = []

        def boom(timeout_s=30.0):
            attempted.append("v1")
            raise RuntimeError("drain exploded")

        backends["v1"].drain_and_unload = boom

        pressure = [True, False, False]
        monkeypatch.setattr(
            mm_mod, "_memory_pressure_detected", lambda: pressure.pop(0)
        )

        mm.load("new", api_key="")

        assert attempted == ["v1"]
        assert "new" in mm
        assert not {"v1", "v2", "v3"} & set(mm.loaded_models)
        assert backends["v2"]._unloaded is True
        assert backends["v3"]._unloaded is True


class TestMemoryPressureEviction:
    def test_pressure_evicts_up_to_three(self, monkeypatch):
        import inference_model_manager.configuration as cfg
        import inference_model_manager.model_manager as mm_mod

        monkeypatch.setattr(cfg, "INFERENCE_MAX_ACTIVE_MODELS", 0)
        pressure = [False] * 8 + [True, False, False]
        monkeypatch.setattr(
            mm_mod, "_memory_pressure_detected", lambda: pressure.pop(0)
        )
        mm = ModelManager()
        backends: dict = {}
        _patch_create_backend(mm, backends)
        for i, name in enumerate(["a", "b", "c", "d"]):
            mm.load(name, api_key="")
            backends[name].last_used_ts = float(i)
        mm.load("new", api_key="")
        assert "d" in mm and "new" in mm
        assert not {"a", "b", "c"} & set(mm.loaded_models)

    def test_no_pressure_no_eviction(self, monkeypatch):
        import inference_model_manager.configuration as cfg

        monkeypatch.setattr(cfg, "INFERENCE_MAX_ACTIVE_MODELS", 0)
        mm = ModelManager()
        _patch_create_backend(mm, {})
        mm.load("a", api_key="")
        mm.load("b", api_key="")
        assert len(mm) == 2

    def test_threshold_zero_disables_check(self, monkeypatch):
        import inference_model_manager.configuration as cfg
        import inference_model_manager.model_manager as mm_mod

        monkeypatch.setattr(cfg, "INFERENCE_MEMORY_FREE_THRESHOLD", 0.0)
        assert mm_mod._memory_pressure_detected() is False
