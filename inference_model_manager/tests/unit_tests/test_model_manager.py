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

    def test_concurrent_loads(self):
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


class _SignalSlotBackend(FakeBackend):
    """FakeBackend that also satisfies the MMP load-success path."""

    def signal_slot(self, slot_id, req_id, params_bytes=b"{}"):
        pass


class TestReloadAfterWorkerDeath:
    """Regression: reload after a worker dies must clear the stale, unhealthy
    backend left in ModelManager — otherwise the reload load() is rejected with
    'already loaded' and the model is wedged forever."""

    def test_reload_clears_stale_backend(self):
        from inference_model_manager.model_manager_process import (
            ModelManagerProcess,
            ModelState,
        )

        mmp = ModelManagerProcess(vram_admission=False)

        created: List[_SignalSlotBackend] = []

        def fake_create(model_id, api_key, backend, **kwargs):
            fb = _SignalSlotBackend(model_id)
            created.append(fb)
            return fb

        mmp._manager._create_backend = fake_create

        # Initial load — backend lands in ModelManager._backends.
        mmp._manager.load("yolov8n:0", api_key="")
        stale = created[-1]

        # Simulate worker death: entry still present but unhealthy.
        stale._state = "unhealthy"
        assert "yolov8n:0" in mmp._manager

        # Reload path (what _schedule_reload triggers).
        mmp._models["yolov8n:0"] = ModelState(loading=True, loaded=False)
        asyncio.run(mmp._load_model("yolov8n:0", api_key=""))

        # Stale entry was unloaded; a fresh healthy backend replaced it —
        # no 'already loaded' rejection.
        assert stale._unloaded is True
        assert mmp._models["yolov8n:0"].loaded is True
        assert len(created) == 2
        assert mmp._manager.get_backend("yolov8n:0") is created[-1]
        assert created[-1].is_healthy is True


class TestSubmitSlotFutureLifecycle:
    def test_cancel_refused_and_slot_freed_only_on_resolution(self):
        mm = ModelManager(n_slots=2, input_mb=0.1)
        try:
            captured: Dict[str, Any] = {}

            class _FakeSubprocBackend:
                def submit_slot(self, slot_id, req_id, future, params_bytes):
                    captured.update(slot=slot_id, req=req_id, fut=future)

            mm._backends["m"] = _FakeSubprocBackend()
            mm._ensure_pool()  # normally created on first subprocess load
            fut = mm.submit("m", raw_input=b"\xff\xd8abc")

            assert mm._pool.free_count == 1          # slot held
            assert fut.cancel() is False             # uncancellable
            assert mm._pool.free_count == 1          # cancel freed nothing

            captured["fut"].set_result("prediction")
            assert fut.result(timeout=1) == "prediction"
            assert mm._pool.free_count == 2          # freed exactly on resolution
        finally:
            mm.shutdown()

    def test_submit_slot_failure_fails_future_and_frees_slot(self):
        from inference_model_manager.model_manager import ModelManager

        mm = ModelManager(n_slots=2, input_mb=0.1)
        try:
            class _DeadBackend:
                def submit_slot(self, slot_id, req_id, future, params_bytes):
                    raise RuntimeError("recv thread is dead")

            mm._backends["m"] = _DeadBackend()
            mm._ensure_pool()
            fut = mm.submit("m", raw_input=b"\xff\xd8abc")
            with pytest.raises(RuntimeError, match="recv thread is dead"):
                fut.result(timeout=1)
            assert mm._pool.free_count == 2
        finally:
            mm.shutdown()
