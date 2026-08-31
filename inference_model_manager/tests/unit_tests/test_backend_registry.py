import importlib.metadata

import pytest

from inference_model_manager.model_manager import (
    BACKEND_FACTORIES,
    ModelManager,
    _reset_entry_point_backends_for_tests,
    register_backend_factory,
)


def test_direct_factory_registered_by_default():
    assert "direct" in BACKEND_FACTORIES


def test_register_and_resolve_custom_backend():
    created = {}

    class FakeBackend:
        def __init__(self, model_id):
            created["model_id"] = model_id

    def factory(model_id, api_key, *, manager, **kwargs):
        return FakeBackend(model_id)

    register_backend_factory("fake", factory)
    try:
        mm = ModelManager()
        backend = mm._create_backend("m1", "key", "fake")
        assert created["model_id"] == "m1"
        assert isinstance(backend, FakeBackend)
    finally:
        BACKEND_FACTORIES.pop("fake", None)


def test_unknown_backend_lists_known_names():
    mm = ModelManager()
    with pytest.raises(ValueError, match="direct"):
        mm._create_backend("m1", "key", "nope")


def test_fake_backend_end_to_end_load_process_unload():
    import concurrent.futures

    from inference_model_manager.backends.base import Backend

    class FakeBackend(Backend):
        device = "cpu"
        is_healthy = True
        is_accepting = True
        max_batch_size = None
        queue_depth = 0
        class_names = None

        def __init__(self, model_id):
            self.model_id = model_id
            self._state = "loaded"

        @property
        def state(self):
            return self._state

        def stats(self):
            return {
                "model_id": self.model_id,
                "backend_type": "fake",
                "state": self.state,
                "is_accepting": True,
                "queue_depth": 0,
                "max_batch_size": None,
                "throughput_fps": 0.0,
                "latency_p50_ms": 0.0,
                "latency_p99_ms": 0.0,
                "inference_count": 0,
                "error_count": 0,
                "last_inference_ts": None,
            }

        def submit_request(self, *, task=None, raw_input=None, validate=None, **kwargs):
            f = concurrent.futures.Future()
            f.set_result({"echo": raw_input, "task": task})
            return f

        def unload(self):
            self._state = "unloaded"

    def factory(model_id, api_key, *, manager, **kwargs):
        return FakeBackend(model_id)

    register_backend_factory("fake-e2e", factory)
    try:
        mm = ModelManager()
        mm.load("m1", "key", backend="fake-e2e")
        assert "m1" in mm.loaded_models
        assert mm.is_ready("m1")
        assert [m["model_id"] for m in mm.list_models()] == ["m1"]
        assert mm.stats()["models"][0]["model_id"] == "m1"
        result = mm.process("m1", images=b"raw", serialize=False)
        assert result == {"echo": b"raw", "task": None}
        mm.unload("m1", drain=True)
        with pytest.raises(KeyError):
            mm.unload("m1")
    finally:
        BACKEND_FACTORIES.pop("fake-e2e", None)


def test_unknown_name_triggers_entry_point_discovery(monkeypatch):
    created = {}

    class FakeBackend:
        def __init__(self, model_id):
            created["model_id"] = model_id

    def ep_factory(model_id, api_key, *, manager, **kwargs):
        return FakeBackend(model_id)

    class _FakeEntryPoint:
        name = "ep-backend"

        def load(self):
            return ep_factory

    def fake_entry_points(*, group):
        assert group == "inference_model_manager.backends"
        return [_FakeEntryPoint()]

    _reset_entry_point_backends_for_tests()
    monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)
    try:
        mm = ModelManager()
        backend = mm._create_backend("m1", "key", "ep-backend")
        assert created["model_id"] == "m1"
        assert isinstance(backend, FakeBackend)
    finally:
        BACKEND_FACTORIES.pop("ep-backend", None)
        _reset_entry_point_backends_for_tests()


def test_entry_point_discovery_runs_once(monkeypatch):
    calls = {"n": 0}

    def fake_entry_points(*, group):
        calls["n"] += 1
        return []

    _reset_entry_point_backends_for_tests()
    monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)
    try:
        mm = ModelManager()
        with pytest.raises(ValueError):
            mm._create_backend("m1", "key", "still-nope-1")
        with pytest.raises(ValueError):
            mm._create_backend("m1", "key", "still-nope-2")
        assert calls["n"] == 1
    finally:
        _reset_entry_point_backends_for_tests()


def test_entry_point_does_not_overwrite_existing_factory(monkeypatch):
    calls = []

    class FakeBackend:
        def __init__(self, model_id, source):
            self.model_id = model_id
            self.source = source

    def existing_factory(model_id, api_key, *, manager, **kwargs):
        calls.append("existing")
        return FakeBackend(model_id, "existing")

    def colliding_ep_factory(model_id, api_key, *, manager, **kwargs):
        calls.append("ep-collide")
        return FakeBackend(model_id, "ep-collide")

    def new_ep_factory(model_id, api_key, *, manager, **kwargs):
        calls.append("ep-new")
        return FakeBackend(model_id, "ep-new")

    class _CollidingEntryPoint:
        name = "collide"

        def load(self):
            return colliding_ep_factory

    class _NewEntryPoint:
        name = "brand-new"

        def load(self):
            return new_ep_factory

    def fake_entry_points(*, group):
        return [_CollidingEntryPoint(), _NewEntryPoint()]

    _reset_entry_point_backends_for_tests()
    monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)
    register_backend_factory("collide", existing_factory)
    try:
        mm = ModelManager()
        backend = mm._create_backend("m1", "key", "brand-new")
        assert backend.source == "ep-new"
        assert BACKEND_FACTORIES["collide"] is existing_factory
        assert "ep-collide" not in calls
    finally:
        BACKEND_FACTORIES.pop("collide", None)
        BACKEND_FACTORIES.pop("brand-new", None)
        _reset_entry_point_backends_for_tests()


def test_failing_entry_point_load_does_not_poison_discovery(monkeypatch):
    state = {"calls": 0}

    class WorkingBackend:
        def __init__(self, model_id):
            self.model_id = model_id

    def working_factory(model_id, api_key, *, manager, **kwargs):
        return WorkingBackend(model_id)

    class _BoomEntryPoint:
        name = "boom"

        def load(self):
            raise RuntimeError("boom")

    class _WorkingEntryPoint:
        name = "fixed"

        def load(self):
            return working_factory

    def fake_entry_points(*, group):
        state["calls"] += 1
        if state["calls"] == 1:
            return [_BoomEntryPoint()]
        return [_WorkingEntryPoint()]

    _reset_entry_point_backends_for_tests()
    monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)
    try:
        mm = ModelManager()
        with pytest.raises(RuntimeError, match="boom"):
            mm._create_backend("m1", "key", "boom")
        backend = mm._create_backend("m1", "key", "fixed")
        assert isinstance(backend, WorkingBackend)
        assert state["calls"] == 2
    finally:
        BACKEND_FACTORIES.pop("fixed", None)
        _reset_entry_point_backends_for_tests()
