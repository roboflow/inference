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

from inference_models.model_manager import ModelManager


# ─── Fake backend ───────────────────────────────────────────────────


class FakeBackend:
    """Minimal Backend stand-in for unit tests."""

    def __init__(self, model_id: str, **kwargs):
        self.model_id = model_id
        self._state = "loaded"
        self._inference_count = 0
        self._unloaded = False
        self._sleep_bytes: Optional[int] = None

    # Pipeline
    def pre_process(self, *args, **kwargs) -> Tuple[Any, Any]:
        return (args, None)

    def post_process(self, raw_output: Any, meta: Any, **kwargs) -> Any:
        return raw_output

    def infer_sync(self, *args, **kwargs) -> Any:
        if self._state != "loaded":
            raise RuntimeError(f"Cannot infer: state={self._state}")
        self._inference_count += 1
        return {"prediction": "fake", "model_id": self.model_id}

    def submit(self, raw_input: Any, *, priority: int = 0, **kwargs) -> Future:
        if self._state != "loaded":
            raise RuntimeError("not accepting")
        f: Future = Future()
        f.set_result({"prediction": "fake", "model_id": self.model_id})
        self._inference_count += 1
        return f

    async def infer_async(self, raw_input: Any, **kwargs) -> Any:
        return self.infer_sync(raw_input, **kwargs)

    # Lifecycle
    def unload(self) -> None:
        self._state = "unhealthy"
        self._unloaded = True

    def sleep(self) -> Optional[int]:
        if self._state != "loaded":
            raise RuntimeError(f"Cannot sleep: state={self._state}")
        self._state = "sleeping"
        self._sleep_bytes = 100 * 1024 * 1024  # 100MB
        return self._sleep_bytes

    def wake(self) -> None:
        if self._state != "sleeping":
            raise RuntimeError(f"Cannot wake: state={self._state}")
        self._state = "loaded"
        self._sleep_bytes = None

    # Observability
    @property
    def device(self) -> str:
        return "cpu"

    @property
    def state(self) -> str:
        return self._state

    @property
    def is_healthy(self) -> bool:
        return self._state in ("loaded", "sleeping")

    @property
    def is_accepting(self) -> bool:
        return self._state == "loaded"

    @property
    def queue_depth(self) -> int:
        return 0

    def stats(self) -> Dict[str, Any]:
        return {
            "backend_type": "fake",
            "state": self.state,
            "is_accepting": self.is_accepting,
            "inference_count": self._inference_count,
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


class TestModelManagerSleepWake:

    def test_sleep_and_wake(self):
        mm = ModelManager(max_pinned_memory_mb=500)
        backends = {}
        _patch_create_backend(mm, backends)

        mm.load("model-a", api_key="")
        mm.sleep("model-a")

        assert mm.sleeping_models == ["model-a"]
        assert mm.loaded_models == []
        assert backends["model-a"].state == "sleeping"

        mm.wake("model-a")

        assert mm.loaded_models == ["model-a"]
        assert mm.sleeping_models == []
        assert backends["model-a"].state == "loaded"

    def test_sleep_exceeds_pinned_limit_raises(self):
        # 100MB limit, FakeBackend.sleep() returns 100MB
        mm = ModelManager(max_pinned_memory_mb=150)
        backends = {}
        _patch_create_backend(mm, backends)

        mm.load("model-a", api_key="")
        mm.sleep("model-a")  # 100MB → ok

        mm.load("model-b", api_key="")
        with pytest.raises(RuntimeError, match="exceeding limit"):
            mm.sleep("model-b")  # would be 200MB > 150MB

        # model-b should be rolled back to loaded
        assert backends["model-b"].state == "loaded"

    def test_sleep_with_zero_limit_still_works(self):
        # max_pinned_memory_mb=0 means no limit
        mm = ModelManager(max_pinned_memory_mb=0)
        backends = {}
        _patch_create_backend(mm, backends)

        mm.load("model-a", api_key="")
        mm.sleep("model-a")
        assert backends["model-a"].state == "sleeping"

    def test_wake_clears_pinned_tracking(self):
        mm = ModelManager(max_pinned_memory_mb=500)
        backends = {}
        _patch_create_backend(mm, backends)

        mm.load("model-a", api_key="")
        mm.sleep("model-a")
        assert mm._pinned_bytes["model-a"] == 100 * 1024 * 1024

        mm.wake("model-a")
        assert "model-a" not in mm._pinned_bytes

    def test_unload_clears_pinned_tracking(self):
        mm = ModelManager(max_pinned_memory_mb=500)
        backends = {}
        _patch_create_backend(mm, backends)

        mm.load("model-a", api_key="")
        mm.sleep("model-a")
        mm.unload("model-a")
        assert "model-a" not in mm._pinned_bytes


class TestModelManagerInference:

    def test_infer_sync(self):
        mm = ModelManager()
        backends = {}
        _patch_create_backend(mm, backends)

        mm.load("model-a", api_key="")
        result = mm.infer_sync("model-a", "some_image")

        assert result == {"prediction": "fake", "model_id": "model-a"}
        assert backends["model-a"]._inference_count == 1

    def test_infer_sync_missing_model_raises(self):
        mm = ModelManager()
        with pytest.raises(KeyError, match="not loaded"):
            mm.infer_sync("nonexistent", "image")

    def test_submit(self):
        mm = ModelManager()
        backends = {}
        _patch_create_backend(mm, backends)

        mm.load("model-a", api_key="")
        future = mm.submit("model-a", "some_image")
        result = future.result(timeout=5)

        assert result is not None

    def test_infer_async(self):
        mm = ModelManager()
        backends = {}
        _patch_create_backend(mm, backends)

        mm.load("model-a", api_key="")
        result = asyncio.get_event_loop().run_until_complete(
            mm.infer_async("model-a", "some_image")
        )

        assert result == {"prediction": "fake", "model_id": "model-a"}

    def test_infer_routes_to_correct_model(self):
        mm = ModelManager()
        backends = {}
        _patch_create_backend(mm, backends)

        mm.load("model-a", api_key="")
        mm.load("model-b", api_key="")

        r_a = mm.infer_sync("model-a", "img")
        r_b = mm.infer_sync("model-b", "img")

        assert r_a["model_id"] == "model-a"
        assert r_b["model_id"] == "model-b"
        assert backends["model-a"]._inference_count == 1
        assert backends["model-b"]._inference_count == 1


class TestModelManagerObservability:

    def test_stats_empty(self):
        mm = ModelManager()
        s = mm.stats()

        assert s["models_loaded"] == []
        assert s["models_sleeping"] == []
        assert s["models"] == []
        assert isinstance(s["gpus"], list)

    def test_stats_with_models(self):
        mm = ModelManager()
        backends = {}
        _patch_create_backend(mm, backends)

        mm.load("model-a", api_key="")
        mm.load("model-b", api_key="")
        mm.infer_sync("model-a", "img")

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

    def test_stats_pinned_memory(self):
        mm = ModelManager(max_pinned_memory_mb=500)
        backends = {}
        _patch_create_backend(mm, backends)

        mm.load("model-a", api_key="")
        mm.sleep("model-a")

        s = mm.stats()
        assert s["ram_pinned_used_mb"] == 100.0  # FakeBackend returns 100MB


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
            threading.Thread(target=load_model, args=(f"model-{i}",))
            for i in range(10)
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
            r = mm.infer_sync("model-a", "img")
            results.append(r)

        threads = [threading.Thread(target=infer) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 20
        assert backends["model-a"]._inference_count == 20


class TestModelManagerBackendCreation:

    def test_unknown_backend_raises(self):
        mm = ModelManager()
        with pytest.raises(ValueError, match="Unknown backend"):
            mm.load("model-a", api_key="", backend="nonexistent")

    @patch("inference_models.model_manager.ModelManager._create_backend")
    def test_load_passes_kwargs_to_backend(self, mock_create):
        fb = FakeBackend("model-a")
        mock_create.return_value = fb

        mm = ModelManager()
        mm.load(
            "model-a", api_key="test-key",
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

    @patch("inference_models.model_manager.ModelManager._create_backend")
    def test_warmup_calls_infer_sync(self, mock_create):
        fb = FakeBackend("model-a")
        mock_create.return_value = fb

        mm = ModelManager()
        mm.load("model-a", api_key="", warmup_iters=3)

        assert fb._inference_count == 3
