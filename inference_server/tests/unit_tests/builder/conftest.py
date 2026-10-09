import importlib
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI

from inference_server import configuration
from inference_server.legacy.bridge import Route
from inference_server.legacy.router import get_bridge

_ROUTES_MODULE = "inference_server.builder.routes"


def make_route(model_id: str, task_type: str) -> Route:
    return Route(
        model_id=model_id,
        registry_id=model_id,
        task_type=task_type,
        action="infer",
    )


class FakeBridge:
    def __init__(self, routes=None):
        self.routes = list(routes or [])

    async def describe(self):
        return list(self.routes)


@pytest.fixture
def builder_env(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("MODEL_CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(configuration, "MODEL_CACHE_DIR", str(cache_dir))
    sys.modules.pop(_ROUTES_MODULE, None)
    routes = importlib.import_module(_ROUTES_MODULE)
    yield routes
    sys.modules.pop(_ROUTES_MODULE, None)


@pytest.fixture
def fake_bridge():
    return FakeBridge(
        [
            make_route("ws/od/1", "object-detection"),
            make_route("ws/cls/2", "classification"),
        ]
    )


@pytest.fixture
def builder_app(builder_env, fake_bridge, monkeypatch):
    original_read_text = Path.read_text

    def fake_read_text(self, encoding="utf-8"):
        if self.name == "editor.html":
            return "Test Editor HTML: CSRF={{CSRF}}"
        return original_read_text(self, encoding=encoding)

    monkeypatch.setattr(Path, "read_text", fake_read_text)

    app = FastAPI()
    app.include_router(builder_env.router, prefix="/build")
    app.dependency_overrides[get_bridge] = lambda: fake_bridge
    return app
