import importlib
import sys

from fastapi import FastAPI
from starlette.testclient import TestClient

from inference_server import configuration
from inference_server.cors import PathAwareCORSMiddleware

ALLOWED_ORIGIN = "https://app.example"
FOREIGN_ORIGIN = "https://evil.example"
_BUILDER_ROUTES_MODULE = "inference_server.builder.routes"


def _build_app() -> FastAPI:
    app = FastAPI()

    @app.get("/build/api/x")
    async def _build_api():
        return {"ok": True}

    @app.get("/workflows/run")
    async def _workflows_run():
        return {"ok": True}

    @app.get("/other")
    async def _other():
        return {"ok": True}

    app.add_middleware(
        PathAwareCORSMiddleware,
        match_paths=r"^/(build/api|workflows/).*",
        allow_origins=[ALLOWED_ORIGIN],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True,
        allow_private_network=True,
    )
    return app


def test_private_network_preflight_on_matched_path():
    client = TestClient(_build_app())

    response = client.options(
        "/build/api/x",
        headers={
            "Origin": ALLOWED_ORIGIN,
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Private-Network": "true",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == ALLOWED_ORIGIN
    assert response.headers["access-control-allow-private-network"] == "true"


def test_preflight_on_unmatched_path_gets_no_cors_headers():
    client = TestClient(_build_app())

    response = client.options(
        "/other",
        headers={
            "Origin": ALLOWED_ORIGIN,
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Private-Network": "true",
        },
    )

    assert response.status_code in (404, 405)
    assert "access-control-allow-origin" not in response.headers
    assert "access-control-allow-private-network" not in response.headers


def test_simple_request_on_matched_path_carries_cors_headers():
    client = TestClient(_build_app())

    response = client.get("/workflows/run", headers={"Origin": ALLOWED_ORIGIN})

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == ALLOWED_ORIGIN


def test_disallowed_origin_private_network_preflight_is_rejected():
    client = TestClient(_build_app())

    response = client.options(
        "/build/api/x",
        headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Private-Network": "true",
        },
    )

    assert response.status_code == 400
    assert "access-control-allow-private-network" not in response.headers


def _reloaded_app(monkeypatch, tmp_path, **overrides):
    import inference_server.app as app_mod

    monkeypatch.setattr(
        "inference_model_manager.watchdogs.start_enabled_watchdogs", lambda: []
    )
    monkeypatch.delenv("INFERENCE_PRELOAD_MODELS", raising=False)
    monkeypatch.setattr(configuration, "MODEL_CACHE_DIR", str(tmp_path / "cache"))
    for name, value in overrides.items():
        monkeypatch.setattr(configuration, name, value)
    sys.modules.pop(_BUILDER_ROUTES_MODULE, None)
    return importlib.reload(app_mod)


def _restore_app(monkeypatch, module):
    monkeypatch.undo()
    sys.modules.pop(_BUILDER_ROUTES_MODULE, None)
    importlib.reload(module)


def test_general_cors_skips_builder_pages(tmp_path, monkeypatch):
    module = _reloaded_app(
        monkeypatch, tmp_path, ENABLE_BUILDER=True, ALLOW_ORIGINS=["*"]
    )
    try:
        with TestClient(module.app, raise_server_exceptions=False) as client:
            builder_page = client.get("/build", headers={"Origin": FOREIGN_ORIGIN})
            assert "access-control-allow-origin" not in builder_page.headers

            editor_page = client.get(
                "/build/edit/example", headers={"Origin": FOREIGN_ORIGIN}
            )
            assert "access-control-allow-origin" not in editor_page.headers

            root = client.get("/", headers={"Origin": FOREIGN_ORIGIN})
            assert "access-control-allow-origin" in root.headers

            builder_api = client.get(
                "/build/api/models",
                headers={"Origin": configuration.BUILDER_ORIGIN},
            )
            assert (
                builder_api.headers["access-control-allow-origin"]
                == configuration.BUILDER_ORIGIN
            )
    finally:
        _restore_app(monkeypatch, module)


def test_body_limit_applies_when_only_builder_is_enabled(tmp_path, monkeypatch):
    module = _reloaded_app(
        monkeypatch,
        tmp_path,
        ENABLE_BUILDER=True,
        LEGACY_ROUTES_ENABLED=False,
        DISABLE_WORKFLOW_ENDPOINTS=True,
        MAX_BODY_BYTES=16,
    )
    try:
        with TestClient(module.app, raise_server_exceptions=False) as client:
            response = client.post(
                "/build/api/some-id",
                content=b"{" + b" " * 64 + b"}",
                headers={"Content-Type": "application/json"},
            )
            assert response.status_code == 413
            assert response.json() == {"message": "Request payload too large."}
    finally:
        _restore_app(monkeypatch, module)
