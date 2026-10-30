"""Tests for the opt-in on-device WebGPU playground served at /webgpu.

http_api.py mounts the router only when ENABLE_WEBGPU_PLAYGROUND is truthy
(default: False). As in test_builder_availability.py, these tests exercise the
router on a bare FastAPI app instead of importing the whole http_api module
(which pulls in the full model-serving stack).
"""

import re

import pytest
from fastapi import FastAPI
from starlette.status import HTTP_200_OK, HTTP_404_NOT_FOUND
from starlette.testclient import TestClient

from inference.core.interfaces.http.webgpu.routes import PLAYGROUND_PAGE, router


@pytest.fixture
def webgpu_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router, prefix="/webgpu")
    return app


def test_playground_absent_when_router_not_mounted() -> None:
    """Mirrors the `if ENABLE_WEBGPU_PLAYGROUND:` branch taking the False path."""
    client = TestClient(FastAPI())

    assert client.get("/webgpu").status_code == HTTP_404_NOT_FOUND


@pytest.mark.parametrize("path", ["/webgpu", "/webgpu/"])
def test_playground_served_when_router_mounted(webgpu_app: FastAPI, path: str) -> None:
    client = TestClient(webgpu_app)

    response = client.get(path, follow_redirects=False)

    assert response.status_code == HTTP_200_OK
    assert "text/html" in response.headers["content-type"]
    assert response.headers["cache-control"] == "no-cache"
    assert "navigator.gpu" in response.text
    assert "isSecureContext" in response.text


def test_playground_adds_no_api_routes(webgpu_app: FastAPI) -> None:
    """The page runs models in the browser: the router serves HTML and nothing else."""
    paths = {
        route.path
        for route in webgpu_app.routes
        if route.path.startswith("/webgpu")
    }

    assert paths == {"/webgpu", "/webgpu/"}
    assert "/webgpu" not in webgpu_app.openapi()["paths"]


def test_playground_loads_only_the_pinned_webllm_build() -> None:
    """The only third-party script is WebLLM, pinned to an exact version."""
    page = PLAYGROUND_PAGE.read_text(encoding="utf-8")

    script_sources = re.findall(r"<script[^>]*\bsrc=", page)
    module_imports = set(re.findall(r"https://cdn\.jsdelivr\.net/npm/[^'\"]+", page))

    assert script_sources == []
    assert module_imports == {
        "https://cdn.jsdelivr.net/npm/@mlc-ai/web-llm@0.2.78/+esm"
    }
