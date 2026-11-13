import importlib
import json
import re

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from inference_sdk.config import (
    INTERNAL_REMOTE_EXEC_REQ_HEADER,
    INTERNAL_REMOTE_EXEC_REQ_VERIFIED_HEADER,
    apply_duration_minimum,
    execution_id,
    remote_processing_times,
)
from inference_server import configuration
from inference_server.hosted import serverless_auth
from inference_server.hosted.serverless_auth import AuthorizationCacheEntry
from inference_server.hosted.serverless_context import ServerlessContextMiddleware

SECRET = "internal-secret-1"
EXECUTION_HEADER = "execution_id"
GENERATED_ID = re.compile(r"^\d+_[0-9a-f]{4}$")
CONTEXT_HEADERS = (
    EXECUTION_HEADER,
    "X-Processing-Time",
    "X-Remote-Processing-Time",
    "X-Remote-Processing-Times",
    INTERNAL_REMOTE_EXEC_REQ_VERIFIED_HEADER,
)


AFTER_REQUEST = []


class ContextSnapshotMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        await self.app(scope, receive, send)
        AFTER_REQUEST.append(
            (
                execution_id.get(),
                apply_duration_minimum.get(),
                remote_processing_times.get(),
            )
        )


class BrokenError(Exception):
    pass


def _build_app(snapshot=False):
    app = FastAPI()

    @app.get("/probe")
    def probe(request: Request):
        collector = remote_processing_times.get()
        state_collector = getattr(
            request.state, "remote_processing_time_collector", None
        )
        return {
            "execution_id": execution_id.get(),
            "apply_duration_minimum": apply_duration_minimum.get(),
            "collector_is_state_collector": collector is state_collector
            and collector is not None,
        }

    @app.get("/collect")
    def collect(request: Request):
        request.state.remote_processing_time_collector.add(0.25, "m/1")
        remote_processing_times.get().add(0.5, "m/2")
        return {"ok": True}

    @app.get("/boom")
    def boom():
        raise HTTPException(status_code=500, detail="boom")

    @app.get("/broken")
    def broken():
        raise BrokenError()

    @app.exception_handler(BrokenError)
    async def _broken_handler(request, error):
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=500, content={"detail": "broken"})

    app.add_middleware(ServerlessContextMiddleware)
    if snapshot:
        app.add_middleware(ContextSnapshotMiddleware)

    return app


@pytest.fixture(autouse=True)
def _settings(monkeypatch):
    monkeypatch.setattr(configuration, "EXECUTION_ID_HEADER", EXECUTION_HEADER)
    monkeypatch.setattr(configuration, "ROBOFLOW_INTERNAL_SERVICE_SECRET", SECRET)
    monkeypatch.setattr(
        configuration, "WORKFLOWS_REMOTE_EXECUTION_TIME_FORWARDING", True
    )


@pytest.fixture
def client():
    return TestClient(_build_app())


def test_supplied_execution_id_is_echoed_and_seen_by_the_handler(client):
    response = client.get("/probe", headers={EXECUTION_HEADER: "exec-42"})

    assert response.headers[EXECUTION_HEADER] == "exec-42"
    assert response.json()["execution_id"] == "exec-42"


def test_missing_execution_id_is_generated_and_echoed(client):
    response = client.get("/probe")

    generated = response.headers[EXECUTION_HEADER]
    assert GENERATED_ID.match(generated)
    assert response.json()["execution_id"] == generated


def test_empty_execution_id_header_is_replaced_by_a_generated_one(client):
    response = client.get("/probe", headers={EXECUTION_HEADER: ""})

    assert GENERATED_ID.match(response.headers[EXECUTION_HEADER])


def test_execution_id_header_name_follows_the_setting(monkeypatch):
    monkeypatch.setattr(configuration, "EXECUTION_ID_HEADER", "X-Exec")

    response = TestClient(_build_app()).get("/probe", headers={"X-Exec": "e-1"})

    assert response.headers["X-Exec"] == "e-1"
    assert response.json()["execution_id"] == "e-1"


def test_each_request_gets_its_own_generated_execution_id(client):
    first = client.get("/probe").headers[EXECUTION_HEADER]
    second = client.get("/probe").headers[EXECUTION_HEADER]

    assert first != second


def test_verified_internal_request_disables_the_duration_minimum(client):
    response = client.get("/probe", headers={INTERNAL_REMOTE_EXEC_REQ_HEADER: SECRET})

    assert response.json()["apply_duration_minimum"] is False
    assert response.headers[INTERNAL_REMOTE_EXEC_REQ_VERIFIED_HEADER] == "true"


def test_request_without_the_internal_header_keeps_the_duration_minimum(client):
    response = client.get("/probe")

    assert response.json()["apply_duration_minimum"] is True
    assert response.headers[INTERNAL_REMOTE_EXEC_REQ_VERIFIED_HEADER] == "false"


def test_wrong_internal_header_is_not_verified(client):
    response = client.get(
        "/probe", headers={INTERNAL_REMOTE_EXEC_REQ_HEADER: "something-else"}
    )

    assert response.json()["apply_duration_minimum"] is True
    assert response.headers[INTERNAL_REMOTE_EXEC_REQ_VERIFIED_HEADER] == "false"


def test_non_ascii_internal_header_is_not_verified_and_does_not_raise(client):
    response = client.get(
        "/probe",
        headers={INTERNAL_REMOTE_EXEC_REQ_HEADER.encode(): "sécret".encode("latin-1")},
    )

    assert response.status_code == 200
    assert response.headers[INTERNAL_REMOTE_EXEC_REQ_VERIFIED_HEADER] == "false"


@pytest.mark.parametrize("secret", [None, ""])
def test_nothing_is_verified_without_a_configured_secret(monkeypatch, client, secret):
    monkeypatch.setattr(configuration, "ROBOFLOW_INTERNAL_SERVICE_SECRET", secret)

    response = client.get(
        "/probe", headers={INTERNAL_REMOTE_EXEC_REQ_HEADER: secret or ""}
    )

    assert response.json()["apply_duration_minimum"] is True
    assert response.headers[INTERNAL_REMOTE_EXEC_REQ_VERIFIED_HEADER] == "false"


def test_processing_time_header_is_numeric(client):
    response = client.get("/probe")

    assert float(response.headers["X-Processing-Time"]) >= 0.0


def test_handler_gets_the_collector_on_request_state_and_the_variable(client):
    response = client.get("/probe")

    assert response.json()["collector_is_state_collector"] is True


def test_collected_remote_times_are_forwarded(client):
    response = client.get("/collect")

    assert float(response.headers["X-Remote-Processing-Time"]) == pytest.approx(0.75)
    assert json.loads(response.headers["X-Remote-Processing-Times"]) == [
        {"m": "m/1", "t": 0.25},
        {"m": "m/2", "t": 0.5},
    ]


def test_remote_times_are_not_forwarded_when_nothing_was_collected(client):
    response = client.get("/probe")

    assert "X-Remote-Processing-Time" not in response.headers
    assert "X-Remote-Processing-Times" not in response.headers


def test_remote_times_are_not_forwarded_when_forwarding_is_off(monkeypatch, client):
    monkeypatch.setattr(
        configuration, "WORKFLOWS_REMOTE_EXECUTION_TIME_FORWARDING", False
    )

    response = client.get("/collect")

    assert "X-Remote-Processing-Time" not in response.headers
    assert "X-Remote-Processing-Times" not in response.headers


def test_oversized_remote_time_detail_is_left_out():
    app = _build_app()

    @app.get("/many")
    def many():
        collector = remote_processing_times.get()
        for index in range(400):
            collector.add(0.001, f"model/{index}")
        return {"ok": True}

    response = TestClient(app).get("/many")

    assert "X-Remote-Processing-Time" in response.headers
    assert "X-Remote-Processing-Times" not in response.headers


@pytest.mark.parametrize(
    "path,status", [("/missing", 404), ("/boom", 500), ("/broken", 500)]
)
def test_error_answers_carry_the_headers(client, path, status):
    response = client.get(path, headers={EXECUTION_HEADER: "exec-9"})

    assert response.status_code == status
    assert response.headers[EXECUTION_HEADER] == "exec-9"
    assert float(response.headers["X-Processing-Time"]) >= 0.0
    assert response.headers[INTERNAL_REMOTE_EXEC_REQ_VERIFIED_HEADER] == "false"


def test_context_variables_are_reset_after_the_response():
    AFTER_REQUEST.clear()
    client = TestClient(_build_app(snapshot=True))

    client.get(
        "/probe",
        headers={EXECUTION_HEADER: "exec-1", INTERNAL_REMOTE_EXEC_REQ_HEADER: SECRET},
    )
    client.get("/missing")

    assert AFTER_REQUEST == [(None, False, None), (None, False, None)]


def test_context_variables_are_reset_when_the_handler_fails():
    AFTER_REQUEST.clear()
    client = TestClient(_build_app(snapshot=True))

    client.get("/broken", headers={EXECUTION_HEADER: "exec-1"})

    assert AFTER_REQUEST == [(None, False, None)]


def test_second_request_does_not_inherit_the_first_ones_internal_flag(client):
    client.get("/probe", headers={INTERNAL_REMOTE_EXEC_REQ_HEADER: SECRET})

    response = client.get("/probe")

    assert response.json()["apply_duration_minimum"] is True


def test_non_http_scopes_pass_through():
    seen = []

    async def inner(scope, receive, send):
        seen.append(scope["type"])

    import asyncio

    asyncio.run(ServerlessContextMiddleware(inner)({"type": "lifespan"}, None, None))

    assert seen == ["lifespan"]


@pytest.fixture
def reloaded_app(monkeypatch):
    import inference_server.app as app_mod

    monkeypatch.setattr(
        "inference_model_manager.watchdogs.start_enabled_watchdogs", lambda: []
    )
    monkeypatch.delenv("INFERENCE_PRELOAD_MODELS", raising=False)

    def _reload(**overrides):
        for name, value in overrides.items():
            monkeypatch.setattr(configuration, name, value)

        return importlib.reload(app_mod)

    yield _reload
    monkeypatch.undo()
    importlib.reload(app_mod)


def _names(module):
    return [middleware.cls.__name__ for middleware in module.app.user_middleware]


def test_middleware_is_absent_without_gcp_serverless(reloaded_app):
    module = reloaded_app(GCP_SERVERLESS=False)

    assert "ServerlessContextMiddleware" not in _names(module)


def test_middleware_sits_inside_the_serverless_auth_and_the_cors_layers(
    reloaded_app,
):
    module = reloaded_app(GCP_SERVERLESS=True)

    names = _names(module)
    context = names.index("ServerlessContextMiddleware")
    assert names.index("ServerlessAuthMiddleware") < context
    assert names.index("PathAwareCORSMiddleware") < context
    assert context < names.index("BillingIntentMiddleware")
    assert context < names.index("_AuthMiddleware")


@pytest.mark.parametrize("path", ["/info", "/v2/server/ready"])
def test_app_without_gcp_serverless_adds_none_of_the_headers(reloaded_app, path):
    module = reloaded_app(GCP_SERVERLESS=False)

    with TestClient(module.app) as client:
        response = client.get(path)

    assert response.status_code == 200
    for header in (
        INTERNAL_REMOTE_EXEC_REQ_VERIFIED_HEADER,
        "X-Remote-Processing-Time",
        "X-Remote-Processing-Times",
    ):
        assert header not in response.headers
    assert EXECUTION_HEADER not in response.headers
    assert "X-Processing-Time" not in response.headers


@pytest.mark.parametrize("path", ["/info", "/v2/server/ready"])
def test_app_under_gcp_serverless_adds_the_headers_to_every_route(
    reloaded_app, monkeypatch, path
):
    module = reloaded_app(GCP_SERVERLESS=True)

    async def _authorized(request, api_key):
        return None, AuthorizationCacheEntry(expires_at=0.0, workspace_id="ws"), False

    monkeypatch.setattr(serverless_auth, "_authorize", _authorized)

    with TestClient(module.app) as client:
        response = client.get(
            path, params={"api_key": "k"}, headers={EXECUTION_HEADER: "exec-5"}
        )

    assert response.status_code == 200
    assert response.headers.get_list(EXECUTION_HEADER) == ["exec-5"]
    assert response.headers.get_list("X-Processing-Time") != []
    assert response.headers[INTERNAL_REMOTE_EXEC_REQ_VERIFIED_HEADER] == "false"
