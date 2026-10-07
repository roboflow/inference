"""Unit tests for ModelManager.

Uses mock backends — no real models, no GPU, no torch. Fast.
"""

from __future__ import annotations

import asyncio
import contextvars
import threading
from concurrent.futures import Future
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from inference_model_manager.model_manager import ModelManager
from inference_model_manager.registry_defaults import registry as _registry
from inference_model_manager.serializers_typed import serialize_passthrough
from inference_model_manager.validators import validate_passthrough

# ─── Fake model + backend ──────────────────────────────────────────


REQUEST_ID: contextvars.ContextVar = contextvars.ContextVar("request_id", default=None)


class FakeModel:
    """Minimal model for unit tests. No base class needed."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._inference_count = 0
        self.seen_request_ids: list = []

    def infer(self, images=None, **kwargs) -> Any:
        self._inference_count += 1
        self.seen_request_ids.append(REQUEST_ID.get())
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

    def test_submit_validates_action(self):
        """submit() direct path must raise on unknown action before queuing."""
        mm = ModelManager()
        _patch_create_backend(mm, {})
        mm.load("model-a", api_key="")

        with pytest.raises(ValueError):
            mm.submit("model-a", action="nonexistent-action", images="img")

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

    def test_process_async_carries_the_callers_context_into_the_model(self):
        mm = ModelManager()
        backends = {}
        _patch_create_backend(mm, backends)
        mm.load("model-a", api_key="")

        async def _call():
            REQUEST_ID.set("req-1")
            await mm.process_async("model-a", images="some_image")

        asyncio.run(_call())

        assert backends["model-a"].model.seen_request_ids == ["req-1"]

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

    def test_stats_carry_model_description_keys_on_every_entry(self):
        mm = ModelManager()
        _patch_create_backend(mm, {})

        mm.load("model-a", api_key="")
        mm.load("model-b", api_key="")

        for entry in mm.stats()["models"]:
            for key in (
                "input_height",
                "input_width",
                "vram_bytes",
                "loaded_monotonic",
            ):
                assert key in entry
                assert entry[key] is None

    def test_total_vram_bytes_sums_reported_models(self):
        mm = ModelManager()
        backends = {}
        _patch_create_backend(mm, backends)
        mm.load("model-a", api_key="")
        mm.load("model-b", api_key="")
        original_stats = backends["model-a"].stats
        backends["model-a"].stats = lambda: {**original_stats(), "vram_bytes": 250}

        s = mm.stats()

        assert s["total_vram_bytes"] == 250
        model_stats = {m["model_id"]: m for m in s["models"]}
        assert model_stats["model-a"]["vram_bytes"] == 250
        assert model_stats["model-b"]["vram_bytes"] is None

    def test_total_vram_bytes_is_none_when_no_model_reports_it(self):
        mm = ModelManager()
        _patch_create_backend(mm, {})
        mm.load("model-a", api_key="")
        mm.load("model-b", api_key="")

        assert mm.stats()["total_vram_bytes"] is None

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


class TestActionRecognitionDispatch:

    def _load_action_model(self, mm: ModelManager, model):
        backends = {}
        _patch_create_backend(mm, backends)
        mm.load("clips/1", api_key="")
        backends["clips/1"]._fake_model = model
        return backends["clips/1"]

    def test_frames_reach_infer_as_one_list(self):
        from inference_models.models.base.action_recognition import (
            ActionRecognitionModel,
            ActionRecognitionPrediction,
        )

        calls = []

        class FakeActionRecognition(ActionRecognitionModel):
            _inference_count = 0

            @classmethod
            def from_pretrained(cls, model_name_or_path, **kwargs):
                return cls()

            @property
            def class_names(self):
                return ["wave", "jump"]

            def infer(self, frames, class_names=None, fps=None, **kwargs):
                calls.append((frames, class_names, fps))
                return [ActionRecognitionPrediction(0, 1, "wave")]

        mm = ModelManager()
        self._load_action_model(mm, FakeActionRecognition())
        frames = [object(), object(), object(), object()]

        mm.process(
            "clips/1",
            action="infer",
            serialize=False,
            wire_marshalling=True,
            frames=frames,
            class_names=["wave"],
            fps=2.0,
        )

        assert calls == [(frames, ["wave"], 2.0)]
        assert calls[0][0] is frames

    def test_stats_report_video_sampling_as_plain_dict(self):
        from inference_models.models.base.action_recognition import (
            ActionRecognitionModel,
            VideoSampling,
        )

        class FakeActionRecognition(ActionRecognitionModel):
            _inference_count = 0

            @classmethod
            def from_pretrained(cls, model_name_or_path, **kwargs):
                return cls()

            @property
            def class_names(self):
                return None

            @property
            def video_sampling(self):
                return VideoSampling(window_seconds=8.0, sample_fps=2.0, max_frames=16)

            def infer(self, frames, class_names=None, fps=None, **kwargs):
                return []

        from inference_model_manager.backends.direct import DirectBackend

        backend = DirectBackend.__new__(DirectBackend)
        backend._model = FakeActionRecognition()

        assert backend.video_sampling == {
            "window_seconds": 8.0,
            "sample_fps": 2.0,
            "min_frames": 4,
            "max_frame_side": None,
            "mode": "sliding_window",
            "max_frames": 16,
        }

        mm = ModelManager()
        fake_backend = self._load_action_model(mm, FakeActionRecognition())
        fake_backend.video_sampling = backend.video_sampling

        entry = next(m for m in mm.stats()["models"] if m["model_id"] == "clips/1")

        assert entry["video_sampling"] == backend.video_sampling

    def test_stats_report_no_video_sampling_without_one(self):
        from inference_model_manager.backends.direct import DirectBackend

        backend = DirectBackend.__new__(DirectBackend)
        backend._model = FakeModel("plain")

        assert backend.video_sampling is None

        mm = ModelManager()
        _patch_create_backend(mm, {})
        mm.load("plain/1", api_key="")

        entry = next(m for m in mm.stats()["models"] if m["model_id"] == "plain/1")

        assert entry["video_sampling"] is None


class TestParamsOnlyWireMarshalling:

    def _process(self, returned, **kwargs):
        class ReturningModel(FakeModel):
            def infer(self, images=None, **infer_kwargs):
                return returned

        mm = ModelManager()
        backends = {}
        _patch_create_backend(mm, backends)
        mm.load("plain/1", api_key="")
        backends["plain/1"]._fake_model = ReturningModel("plain/1")
        return mm.process(
            "plain/1", action="infer", serialize=False, wire_marshalling=True, **kwargs
        )

    def test_params_only_list_result_is_returned_whole(self):
        assert self._process(["s1", "s2", "s3"], fps=2.0) == ["s1", "s2", "s3"]

    def test_params_only_empty_list_result_is_returned_whole(self):
        assert self._process([], fps=2.0) == []

    def test_params_only_array_result_is_returned_whole(self):
        import numpy as np

        result = self._process(np.arange(6).reshape(3, 2), fps=2.0)

        assert result.shape == (3, 2)

    def test_params_only_result_is_converted_to_numpy(self):
        import torch

        result = self._process([torch.ones(2)], fps=2.0)

        assert type(result[0]).__name__ == "ndarray"

    def test_single_image_one_element_list_is_still_unwrapped(self):
        assert self._process(["only"], images=object()) == "only"

    def test_single_image_list_of_one_is_still_unwrapped(self):
        assert self._process(["only"], images=[object()]) == "only"

    def test_batch_of_images_is_still_split_per_image(self):
        assert self._process(["a", "b"], images=[object(), object()]) == ["a", "b"]


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

    def test_process_reports_model_duration_into_the_timing_sink(self):
        mm = ModelManager()
        _patch_create_backend(mm, {})
        mm.load("model-a", api_key="")
        timing: dict = {}

        mm.process("model-a", serialize=False, timing=timing, images="img")

        assert timing["model_s"] >= 0.0

    def test_process_leaves_timing_unset_when_the_action_is_unknown(self):
        mm = ModelManager()
        _patch_create_backend(mm, {})
        mm.load("model-a", api_key="")
        timing: dict = {}

        with pytest.raises(ValueError):
            mm.process("model-a", action="nonexistent-action", timing=timing)

        assert timing == {}

    @staticmethod
    def _synthetic_clock(monkeypatch):
        import time as real_time
        from types import SimpleNamespace

        from inference_model_manager import model_manager as mm_mod

        now = [0.0]
        monkeypatch.setattr(
            mm_mod,
            "time",
            SimpleNamespace(perf_counter=lambda: now[0], monotonic=real_time.monotonic),
        )

        return now

    def test_model_duration_counts_decode_and_retries_but_not_marshalling_or_accounting(
        self, monkeypatch
    ):
        from inference_model_manager import model_manager as mm_mod

        now = self._synthetic_clock(monkeypatch)
        mm = ModelManager()
        backends: dict = {}
        _patch_create_backend(mm, backends)
        mm.load("model-a", api_key="")
        backend = backends["model-a"]

        original_decode = mm._wire_marshal_inputs

        def _decode(backend_, kwargs):
            now[0] += 2
            return original_decode(backend_, kwargs)

        def _infer(images=None, **kwargs):
            now[0] += 3
            return {"prediction": "fake"}

        def _to_numpy(result):
            now[0] += 10
            return result

        def _record(t0, error=False):
            now[0] += 40

        real_registry = mm_mod._get_registry()

        class _Registry:
            def validate(self, *args):
                return real_registry.validate(*args)

            def serialize(self, *args):
                now[0] += 20
                return real_registry.serialize(*args)

        monkeypatch.setattr(mm, "_wire_marshal_inputs", _decode)
        monkeypatch.setattr(backend.model, "infer", _infer)
        monkeypatch.setattr(mm_mod, "tensors_to_numpy", _to_numpy)
        monkeypatch.setattr(backend, "record_inference", _record)
        monkeypatch.setattr(mm_mod, "_get_registry", lambda: _Registry())
        timing: dict = {}

        mm.process(
            "model-a",
            wire_marshalling=True,
            timing=timing,
            images=["a", "b"],
        )

        assert timing["model_s"] == 2 + 3 + 3 + 3

    def test_model_duration_is_recorded_when_validation_fails_after_decode(
        self, monkeypatch
    ):
        from inference_model_manager import model_manager as mm_mod

        now = self._synthetic_clock(monkeypatch)
        mm = ModelManager()
        _patch_create_backend(mm, {})
        mm.load("model-a", api_key="")
        original_decode = mm._wire_marshal_inputs

        def _decode(backend_, kwargs):
            now[0] += 2
            return original_decode(backend_, kwargs)

        class _Registry:
            def validate(self, *args):
                raise ValueError("classes missing")

        monkeypatch.setattr(mm, "_wire_marshal_inputs", _decode)
        monkeypatch.setattr(mm_mod, "_get_registry", lambda: _Registry())
        timing: dict = {}

        with pytest.raises(ValueError):
            mm.process("model-a", wire_marshalling=True, timing=timing, images="img")

        assert timing["model_s"] == 2

    def test_model_duration_is_recorded_when_the_model_fails(self, monkeypatch):
        now = self._synthetic_clock(monkeypatch)
        mm = ModelManager()
        backends: dict = {}
        _patch_create_backend(mm, backends)
        mm.load("model-a", api_key="")

        def _infer(images=None, **kwargs):
            now[0] += 3
            raise RuntimeError("boom")

        monkeypatch.setattr(backends["model-a"].model, "infer", _infer)
        timing: dict = {}

        with pytest.raises(RuntimeError):
            mm.process("model-a", timing=timing, images="img")

        assert timing["model_s"] == 3

    def test_process_async_reports_model_duration_per_call(self):
        mm = ModelManager()
        _patch_create_backend(mm, {})
        mm.load("model-a", api_key="")
        first: dict = {}
        second: dict = {}

        async def _call():
            await asyncio.gather(
                mm.process_async("model-a", timing=first, images="img"),
                mm.process_async("model-a", timing=second, images="img"),
            )

        asyncio.run(_call())

        assert first["model_s"] >= 0.0
        assert second["model_s"] >= 0.0


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


class _FakeRemoteBackend(FakeBackend):
    _model_mro_names = ["InstanceSegmentationModel", "object"]

    def __init__(self, model_id: str, **kwargs):
        super().__init__(model_id, **kwargs)
        del self._fake_model

    @property
    def model(self):
        raise AttributeError("model")

    def submit_request(self, action=None, raw_input=None, validate=None, **kwargs):
        future: Future = Future()
        future.set_result({"prediction": "remote"})
        return future

    def stats(self) -> Dict[str, Any]:
        return {"backend_type": "remote", "state": self.state}


class _ReloadOnRelease:
    def __init__(self, lock, reload, *, when):
        self._lock = lock
        self._reload = reload
        self._when = when
        self.fired = False

    def __enter__(self):
        return self._lock.__enter__()

    def __exit__(self, *exc):
        self._lock.__exit__(*exc)
        if not self.fired and self._when():
            self.fired = True
            self._reload()


def _direct_backend(model_id: str, model: Any):
    from collections import deque

    from inference_model_manager.backends.direct import DirectBackend

    backend = DirectBackend.__new__(DirectBackend)
    backend._model_id = model_id
    backend._device_str = "cpu"
    backend._state_value = "loaded"
    backend._model = model
    backend._inflight = 0
    backend._inflight_lock = threading.Lock()
    backend._inference_count = 0
    backend._error_count = 0
    backend._last_inference_ts = 0.0
    backend._latencies = deque(maxlen=1000)
    backend._start_ts = 0.0

    return backend


class TestStreamPipeline:

    @pytest.fixture
    def segmentation_manager(self, monkeypatch):
        from inference_models.models.base.instance_segmentation import (
            InstanceSegmentationModel,
        )

        from inference_model_manager.registry_defaults import lazy_register
        from tests.unit_tests.test_stream_pipeline import FakeSegmentationModel

        lazy_register(InstanceSegmentationModel)
        monkeypatch.setenv("RFDETR_PIPELINE_DEPTH", "2")
        mm = ModelManager()
        backends: Dict[str, FakeBackend] = {}
        models: Dict[str, FakeSegmentationModel] = {}
        created: List[FakeSegmentationModel] = []

        def fake_create(model_id, api_key, backend, **kwargs):
            if model_id.startswith("remote"):
                return _FakeRemoteBackend(model_id)
            fb = FakeBackend(model_id)
            fb._fake_model = FakeSegmentationModel(
                supports_stream_pipeline=not model_id.startswith("plain")
            )
            fb._fake_model._inference_count = 0
            backends[model_id] = fb
            models[model_id] = fb._fake_model
            created.append(fb._fake_model)
            return fb

        mm._create_backend = fake_create
        yield mm, backends, models
        for model in created:
            for future in model.futures:
                future.release.set()
        mm.shutdown()

    @staticmethod
    def _frame(mm: ModelManager, model_id: str, context_id: str):
        return mm.process(
            model_id,
            serialize=False,
            wire_marshalling=True,
            images=np.zeros((4, 6, 3), dtype=np.uint8),
            stream_pipeline_context_id=context_id,
            stream_pipeline_producer_id="producer",
        )

    def test_pipelined_call_reports_no_model_duration(self, segmentation_manager):
        mm, _, _ = segmentation_manager
        mm.load("seg/1", api_key="")
        timing: dict = {}

        mm.process(
            "seg/1",
            serialize=False,
            wire_marshalling=True,
            timing=timing,
            images=np.zeros((4, 6, 3), dtype=np.uint8),
            stream_pipeline_context_id="ctx",
            stream_pipeline_producer_id="producer",
        )

        assert timing == {}

    def test_synchronous_call_through_the_pipeline_excludes_the_lock_wait(
        self, segmentation_manager
    ):
        import time as real_time

        mm, _, _ = segmentation_manager
        mm.load("seg/1", api_key="")
        pipeline = mm._stream_pipelines["seg/1"]
        timing: dict = {}
        holding = threading.Event()

        def _hold():
            with pipeline._lock:
                holding.set()
                real_time.sleep(0.3)

        holder = threading.Thread(target=_hold)
        holder.start()
        holding.wait(timeout=5)

        mm.process(
            "seg/1",
            serialize=False,
            wire_marshalling=True,
            timing=timing,
            images=np.zeros((4, 6, 3), dtype=np.uint8),
        )
        holder.join()

        assert timing["model_s"] < 0.15

    def test_load_wires_the_pipeline_for_a_supported_model(self, segmentation_manager):
        mm, _, _ = segmentation_manager

        mm.load("seg/1", api_key="")

        assert mm.model_supports_stream_pipeline("seg/1") is True
        assert mm.get_model_pipeline_depth("seg/1") == 2
        entry = next(m for m in mm.stats()["models"] if m["model_id"] == "seg/1")
        assert entry["stream_pipeline_depth"] == 2

    def test_unsupported_model_reports_depth_one(self, segmentation_manager):
        mm, _, _ = segmentation_manager

        mm.load("plain/1", api_key="")

        assert mm.model_supports_stream_pipeline("plain/1") is False
        assert mm.get_model_pipeline_depth("plain/1") == 1
        entry = next(m for m in mm.stats()["models"] if m["model_id"] == "plain/1")
        assert entry["stream_pipeline_depth"] == 1
        assert mm.flush_model_stream_pipeline("plain/1") is None
        assert mm.shutdown_model_stream_pipeline("plain/1") is None

    def test_unloaded_model_reports_legacy_defaults(self, segmentation_manager):
        mm, _, _ = segmentation_manager

        assert mm.model_supports_stream_pipeline("missing/1") is False
        assert mm.get_model_pipeline_depth("missing/1") == 1
        assert mm.flush_model_stream_pipeline("missing/1") is None
        assert mm.shutdown_model_stream_pipeline("missing/1") is None

    def test_non_direct_backend_raises_not_implemented(self, segmentation_manager):
        mm, _, _ = segmentation_manager
        mm.load("remote/1", api_key="", backend="remote")

        with pytest.raises(NotImplementedError):
            mm.model_supports_stream_pipeline("remote/1")
        with pytest.raises(NotImplementedError):
            mm.get_model_pipeline_depth("remote/1")
        with pytest.raises(NotImplementedError):
            mm.flush_model_stream_pipeline("remote/1")
        with pytest.raises(NotImplementedError):
            mm.shutdown_model_stream_pipeline("remote/1")
        entry = next(m for m in mm.stats()["models"] if m["model_id"] == "remote/1")
        assert entry["stream_pipeline_depth"] == 1

    def test_depth_one_does_not_wire_the_pipeline(
        self, segmentation_manager, monkeypatch
    ):
        mm, _, _ = segmentation_manager
        monkeypatch.setenv("RFDETR_PIPELINE_DEPTH", "1")

        mm.load("seg/1", api_key="")

        assert mm.model_supports_stream_pipeline("seg/1") is False
        assert mm.get_model_pipeline_depth("seg/1") == 1

    def test_process_routes_the_default_action_through_the_pipeline(
        self, segmentation_manager
    ):
        from inference_models.models.base.async_handoff import (
            get_async_response_context_id,
            get_async_response_future,
        )

        mm, _, models = segmentation_manager
        mm.load("seg/1", api_key="")

        first = self._frame(mm, "seg/1", "ctx-1")
        second = self._frame(mm, "seg/1", "ctx-2")

        model = models["seg/1"]
        assert model.sync_calls == []
        assert [kwargs["mask_format"] for _, kwargs in model.async_calls] == [
            "rle",
            "rle",
        ]
        assert len(first) == 0 and isinstance(first.xyxy, np.ndarray)
        assert get_async_response_future(first) is None
        assert get_async_response_context_id(second) == "ctx-1"
        model.futures[0].release.set()
        result = get_async_response_future(second).result(timeout=5)
        assert int(result[0].xyxy[0][0]) == 1
        model.futures[1].release.set()
        flushed = mm.flush_model_stream_pipeline("seg/1")
        assert [get_async_response_context_id(r) for r in flushed] == ["ctx-2"]

    def test_process_without_context_id_runs_the_model_synchronously(
        self, segmentation_manager
    ):
        mm, _, models = segmentation_manager
        mm.load("seg/1", api_key="")
        image = np.zeros((4, 6, 3), dtype=np.uint8)

        result = mm.process(
            "seg/1", serialize=False, wire_marshalling=True, images=image
        )

        assert len(result) == 1
        assert len(models["seg/1"].sync_calls) == 1
        assert models["seg/1"].async_calls == []

    def test_process_batches_stay_on_the_model(self, segmentation_manager):
        mm, _, models = segmentation_manager
        mm.load("seg/1", api_key="")
        images = [np.zeros((4, 6, 3), dtype=np.uint8)] * 2

        result = mm.process(
            "seg/1",
            serialize=False,
            wire_marshalling=True,
            images=images,
            stream_pipeline_context_id="ctx-1",
        )

        assert len(result) == 2
        assert len(models["seg/1"].sync_calls) == 1
        assert models["seg/1"].async_calls == []

    def test_unload_shuts_the_pipeline_down(self, segmentation_manager):
        mm, backends, models = segmentation_manager
        mm.load("seg/1", api_key="")
        self._frame(mm, "seg/1", "ctx-1")
        self._frame(mm, "seg/1", "ctx-2")
        pipeline = mm._stream_pipelines["seg/1"]
        executor = pipeline._response_executor
        for future in models["seg/1"].futures:
            future.release.set()

        mm.unload("seg/1")

        assert executor._shutdown is True
        assert pipeline._response_executor is None
        assert "seg/1" not in mm._stream_pipelines
        assert backends["seg/1"]._unloaded is True
        assert mm.model_supports_stream_pipeline("seg/1") is False
        assert mm.get_model_pipeline_depth("seg/1") == 1
        assert mm.flush_model_stream_pipeline("seg/1") is None

    def test_unload_resolves_an_in_flight_frame(self, segmentation_manager):
        from inference_models.models.base.async_handoff import (
            get_async_response_future,
        )

        mm, _, models = segmentation_manager
        mm.load("seg/1", api_key="")
        self._frame(mm, "seg/1", "ctx-1")
        second = self._frame(mm, "seg/1", "ctx-2")
        future = get_async_response_future(second)
        for model_future in models["seg/1"].futures:
            model_future.release.set()

        mm.unload("seg/1")

        assert future.done()
        assert len(future.result(timeout=0)) == 1
        assert "seg/1" not in mm

    def test_unload_shuts_the_captured_pipeline_down_when_a_reload_interleaves(
        self, segmentation_manager
    ):
        mm, _, models = segmentation_manager
        mm.load("seg/1", api_key="")
        self._frame(mm, "seg/1", "ctx-1")
        self._frame(mm, "seg/1", "ctx-2")
        old_model = models["seg/1"]
        old_pipeline = mm._stream_pipelines["seg/1"]
        old_executor = old_pipeline._response_executor
        for future in old_model.futures:
            future.release.set()
        mm._lifecycle_lock = _ReloadOnRelease(
            mm._lifecycle_lock,
            lambda: mm.load("seg/1", api_key=""),
            when=lambda: "seg/1" not in mm._backends,
        )

        mm.unload("seg/1")

        assert mm._lifecycle_lock.fired is True
        new_pipeline = mm._stream_pipelines["seg/1"]
        assert new_pipeline is not old_pipeline
        assert new_pipeline.model is models["seg/1"] is not old_model
        assert old_pipeline._response_executor is None
        assert old_executor._shutdown is True
        assert "seg/1" in mm

    def test_eviction_shuts_the_captured_pipeline_down_when_a_reload_interleaves(
        self, segmentation_manager, monkeypatch
    ):
        import inference_model_manager.configuration as cfg

        monkeypatch.setattr(cfg, "INFERENCE_MAX_ACTIVE_MODELS", 1)
        mm, _, models = segmentation_manager
        mm.load("seg/1", api_key="")
        self._frame(mm, "seg/1", "ctx-1")
        self._frame(mm, "seg/1", "ctx-2")
        old_model = models["seg/1"]
        old_pipeline = mm._stream_pipelines["seg/1"]
        old_executor = old_pipeline._response_executor
        for future in old_model.futures:
            future.release.set()
        mm._lifecycle_lock = _ReloadOnRelease(
            mm._lifecycle_lock,
            lambda: mm.load("seg/1", api_key="", pinned=True),
            when=lambda: "seg/1" not in mm._backends,
        )

        mm.load("seg/2", api_key="")

        assert mm._lifecycle_lock.fired is True
        new_pipeline = mm._stream_pipelines["seg/1"]
        assert new_pipeline is not old_pipeline
        assert new_pipeline.model is models["seg/1"] is not old_model
        assert old_pipeline._response_executor is None
        assert old_executor._shutdown is True
        assert "seg/1" in mm and "seg/2" in mm

    def test_admitted_request_runs_on_the_captured_wrapper_across_a_reload(
        self, segmentation_manager
    ):
        from inference_models.models.base.async_handoff import (
            get_async_response_context_id,
            get_async_response_future,
        )

        mm, backends, models = segmentation_manager
        mm.load("seg/1", api_key="")
        old_backend, old_model = backends["seg/1"], models["seg/1"]
        old_pipeline = mm._stream_pipelines["seg/1"]
        admitted, resume = threading.Event(), threading.Event()

        def inflight_begin():
            admitted.set()
            assert resume.wait(timeout=5)

        old_backend.inflight_begin = inflight_begin
        results: List[Any] = []
        worker = threading.Thread(
            target=lambda: results.append(self._frame(mm, "seg/1", "ctx-1"))
        )
        worker.start()
        assert admitted.wait(timeout=5)

        mm.unload("seg/1")
        mm.load("seg/1", api_key="")
        new_pipeline = mm._stream_pipelines["seg/1"]
        resume.set()
        worker.join(timeout=5)

        assert not worker.is_alive()
        assert new_pipeline is not old_pipeline
        assert old_pipeline._response_executor is None
        assert len(old_model.async_calls) == 1 and old_model.sync_calls == []
        assert models["seg/1"].async_calls == [] and models["seg/1"].sync_calls == []
        assert list(new_pipeline._pending_futures) == []
        assert [f for f, _, _ in old_pipeline._pending_futures] == old_model.futures
        assert get_async_response_context_id(results[0]) == "ctx-1"
        assert get_async_response_future(results[0]) is None

    @pytest.mark.parametrize("evict", [False, True])
    def test_pipeline_is_drained_after_the_last_admitted_frame(
        self, segmentation_manager, monkeypatch, evict
    ):
        import inference_model_manager.configuration as cfg
        from inference_models.models.base.async_handoff import (
            get_async_response_context_id,
        )

        from tests.unit_tests.test_stream_pipeline import (
            FakeSegmentationModel,
            queues_of,
        )

        monkeypatch.setattr(cfg, "INFERENCE_MAX_ACTIVE_MODELS", 1 if evict else 0)
        mm, _, _ = segmentation_manager
        model = FakeSegmentationModel()
        backend = _direct_backend("seg/1", model)
        fake_create = mm._create_backend
        mm._create_backend = lambda model_id, api_key, **kwargs: (
            backend if model_id == "seg/1" else fake_create(model_id, api_key, **kwargs)
        )
        mm.load("seg/1", api_key="")
        pipeline = mm._stream_pipelines["seg/1"]
        admitted, resume = threading.Event(), threading.Event()
        lease_begin = backend.inflight_begin

        def paused_begin():
            lease_begin()
            admitted.set()
            assert resume.wait(timeout=5)

        backend.inflight_begin = paused_begin
        shutdown_pipeline = pipeline.shutdown_pipeline

        def recorded_shutdown():
            model.events.append(("shutdown", len(model.async_calls)))
            shutdown_pipeline()

        pipeline.shutdown_pipeline = recorded_shutdown
        results: List[Any] = []
        worker = threading.Thread(
            target=lambda: results.append(self._frame(mm, "seg/1", "ctx-1"))
        )
        worker.start()
        assert admitted.wait(timeout=5)
        removed = threading.Event()

        def remove():
            if evict:
                mm.load("seg/2", api_key="")
            else:
                mm.unload("seg/1", drain=True, drain_timeout_s=5.0)
            removed.set()

        remover = threading.Thread(target=remove)
        remover.start()
        assert not removed.wait(timeout=0.3)
        resume.set()
        worker.join(timeout=5)
        for future in model.futures:
            future.release.set()
        remover.join(timeout=5)

        assert not worker.is_alive() and removed.is_set()
        assert model.events == [
            ("async-start", 1),
            ("async-return", 1),
            ("shutdown", 1),
        ]
        assert queues_of(pipeline) == ([], [], [])
        assert pipeline._response_executor is None
        assert backend.model is None
        assert "seg/1" not in mm
        assert get_async_response_context_id(results[0]) == "ctx-1"

    def test_shutdown_stops_every_pipeline(self, segmentation_manager):
        mm, _, models = segmentation_manager
        mm.load("seg/1", api_key="")
        mm.load("seg/2", api_key="")
        for model_id in ("seg/1", "seg/2"):
            self._frame(mm, model_id, "ctx-1")
            self._frame(mm, model_id, "ctx-2")
        pipelines = dict(mm._stream_pipelines)
        executors = {k: p._response_executor for k, p in pipelines.items()}
        for model in models.values():
            for future in model.futures:
                future.release.set()

        mm.shutdown()

        assert all(executor._shutdown for executor in executors.values())
        assert mm._stream_pipelines == {}

    def test_eviction_shuts_the_victim_pipeline_down(
        self, segmentation_manager, monkeypatch
    ):
        import inference_model_manager.configuration as cfg

        monkeypatch.setattr(cfg, "INFERENCE_MAX_ACTIVE_MODELS", 1)
        mm, _, models = segmentation_manager
        mm.load("seg/1", api_key="")
        self._frame(mm, "seg/1", "ctx-1")
        self._frame(mm, "seg/1", "ctx-2")
        pipeline = mm._stream_pipelines["seg/1"]
        executor = pipeline._response_executor
        for future in models["seg/1"].futures:
            future.release.set()

        mm.load("seg/2", api_key="")

        assert "seg/1" not in mm
        assert "seg/1" not in mm._stream_pipelines
        assert executor._shutdown is True
