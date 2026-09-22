from __future__ import annotations

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

    async def load(self, model_id, api_key="", timeout_s=None):
        self.calls.append(("load", model_id, api_key))
        self.loaded.setdefault(
            model_id, dict(self.model_info.get(model_id, {}), state="loaded")
        )
        return ("ok",)

    async def unload(self, model_id):
        self.calls.append(("unload", model_id))
        return ("ok",) if self.loaded.pop(model_id, None) is not None else ("error", 6)

    async def infer(
        self, *, model_id, image=None, task=None, instance="", params=None, request=None
    ):
        self.calls.append(("infer", model_id, task, params, image))
        value = self.predictions[(model_id, task)]
        return value(image, params) if callable(value) else value

    async def stats(self):
        return {"models": {k: dict(v, model_id=k) for k, v in self.loaded.items()}}

    async def interface(self, model_id):
        return {"model_id": model_id, "tasks": self.loaded[model_id].get("tasks", {})}


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


@pytest.fixture
def fake_stat(monkeypatch):
    table = {}

    async def _stat(common):
        if common.model_id not in table:
            raise LookupError(common.model_id)
        return table[common.model_id]

    monkeypatch.setattr(
        "inference_server.legacy.bridge.stat_model_while_checking_auth", _stat
    )
    return table


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
