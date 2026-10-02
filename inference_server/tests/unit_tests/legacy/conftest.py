from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import pytest


class FakeGateway:
    """Minimal duck-surface gateway; records calls, returns canned predictions."""

    def __init__(
        self, predictions: Optional[dict] = None, model_info: Optional[dict] = None
    ):
        self.predictions = predictions or {}
        self.model_info = model_info or {}
        self.loaded: dict[str, dict] = {}
        self.calls: list[tuple] = []
        self.ensure_results: list[tuple] = []
        self.pinned: list[str] = []

    async def start(self): ...

    async def shutdown(self): ...

    async def ensure_loaded(self, model_id, instance="", api_key="", device=""):
        self.calls.append(("ensure_loaded", model_id, api_key))
        if self.ensure_results:
            return self.ensure_results.pop(0)
        self.loaded.setdefault(
            model_id, dict(self.model_info.get(model_id, {}), state="loaded")
        )
        return ("model_ready",)

    async def load(self, model_id, api_key="", timeout_s=None, pinned=True):
        self.calls.append(("load", model_id, api_key))
        if pinned:
            self.pinned.append(model_id)
        self.loaded.setdefault(
            model_id, dict(self.model_info.get(model_id, {}), state="loaded")
        )
        return ("ok",)

    async def unload(self, model_id):
        self.calls.append(("unload", model_id))
        return ("ok",) if self.loaded.pop(model_id, None) is not None else ("error", 6)

    async def infer(
        self,
        *,
        model_id,
        image=None,
        action=None,
        instance="",
        params=None,
        request=None,
    ):
        self.calls.append(("infer", model_id, action, params, image))
        value = self.predictions[(model_id, action)]
        return value(image, params) if callable(value) else value

    async def stats(self):
        return {"models": {k: dict(v, model_id=k) for k, v in self.loaded.items()}}

    async def interface(self, model_id):
        return {
            "model_id": model_id,
            "actions": self.loaded[model_id].get("actions", {}),
        }


class EvictedModelManager:
    def __init__(self, reload_error: Optional[BaseException] = None):
        self.reload_error = reload_error
        self.loaded: set[str] = set()
        self.load_calls = 0
        self.process_calls = 0
        self.executor = None

    def __contains__(self, key):
        return key in self.loaded

    def load(self, key, api_key, **kwargs):
        self.load_calls += 1
        if self.load_calls > 1 and self.reload_error is not None:
            raise self.reload_error
        self.loaded.add(key)

    def unload(self, key):
        self.loaded.discard(key)

    def stats(self):
        return {
            "models": [
                {"model_id": key, "class_names": ["cat"], "actions": {"infer": {}}}
                for key in self.loaded
            ]
        }

    def shutdown(self):
        pass

    async def process_async(self, key, **kwargs):
        self.process_calls += 1
        self.loaded.discard(key)
        raise KeyError(key)


def route_paths(app) -> set[str]:
    """Paths of every route on the app, including lazily included routers.

    fastapi>=0.140 appends an _IncludedRouter wrapper to app.routes instead of
    flattening the included routes; walk through it.
    """

    def _walk(routes):
        for route in routes:
            inner = getattr(route, "original_router", None)
            if inner is not None:
                yield from _walk(inner.routes)
            else:
                yield route.path

    return set(_walk(app.routes))


@pytest.fixture(autouse=True)
def _reset_model_stat_cache():
    from inference_server.framework import model_stat

    model_stat._reset_cache_for_tests()
    yield
    model_stat._reset_cache_for_tests()


@pytest.fixture
def fake_stat(monkeypatch):
    from inference_models.errors import ModelNotFoundError
    from inference_server.framework import model_stat

    table = {}

    def _metadata(model_id: str, api_key: Optional[str] = None, **_):
        outcome = table.get(model_id)
        if outcome is None:
            raise ModelNotFoundError(message=model_id, help_url="")
        if isinstance(outcome, Exception):
            raise outcome
        meta = SimpleNamespace(task_type=outcome[0])
        if len(outcome) > 2:
            meta.model_architecture, meta.model_variant = outcome[2], outcome[3]
        return meta

    monkeypatch.setattr(model_stat, "get_one_page_of_model_metadata", _metadata)
    return table


class KeyGatedStat:
    def __init__(self, task_type: str = "object-detection"):
        self.task_type = task_type
        self.denied_keys: set = set()
        self.calls: list[tuple] = []


@pytest.fixture
def key_gated_stat(monkeypatch):
    from inference_models.errors import UnauthorizedModelAccessError
    from inference_server.framework import model_stat

    gate = KeyGatedStat()

    def _metadata(model_id: str, api_key: Optional[str] = None, **_):
        gate.calls.append((model_id, api_key))
        if api_key in gate.denied_keys:
            raise UnauthorizedModelAccessError("denied")
        return SimpleNamespace(task_type=gate.task_type)

    monkeypatch.setattr(model_stat, "get_one_page_of_model_metadata", _metadata)
    return gate


@pytest.fixture
def legacy_client(fake_stat, monkeypatch, request):
    from fastapi.testclient import TestClient

    import inference_server.app as app_mod

    monkeypatch.setattr(
        "inference_model_manager.watchdogs.start_enabled_watchdogs", lambda: []
    )
    monkeypatch.delenv("INFERENCE_PRELOAD_MODELS", raising=False)

    def _make(gateway):
        monkeypatch.setattr(
            "inference_server.gateway_resolver.resolve_gateway", lambda: gateway
        )
        client = TestClient(app_mod.app, raise_server_exceptions=False)
        client.__enter__()
        request.addfinalizer(lambda: client.__exit__(None, None, None))
        return client

    return _make
