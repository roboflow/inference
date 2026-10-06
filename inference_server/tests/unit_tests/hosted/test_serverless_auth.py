import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from inference_server import configuration, platform_http
from inference_server.errors import AuthBackendUnavailable
from inference_server.hosted import serverless_auth
from inference_server.hosted.assume_identity import (
    assume_identity_authorised_workspace_db_id,
    enforce_credits_verification,
)
from inference_server.hosted.serverless_auth import ServerlessAuthMiddleware
from inference_server.legacy.errors import LegacyHTTPError

UNAUTHORIZED = {"status": 401, "message": "Unauthorized api_key"}
SERVERLESS_UNAUTHORIZED = {
    "status": 401,
    "message": "Unauthorized api_key. This key is not authorized for serverless inference.",
}
CREDITS_MESSAGE = (
    "This workspace cannot currently spend credits for serverless inference. "
    "Verify billing or credit cap settings."
)
INCOMPLETE = {
    "status": 500,
    "message": "Serverless authorization failed because the usage check returned incomplete data.",
}


@pytest.fixture(autouse=True)
def _reset_cache():
    serverless_auth._cache.clear()
    yield
    serverless_auth._cache.clear()


@pytest.fixture
def clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(serverless_auth, "_now", lambda: now[0])
    return now


@pytest.fixture
def client():
    inner = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @inner.api_route(
        "/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"]
    )
    async def _probe(request: Request):
        body = await request.body()
        return JSONResponse(
            {
                "path": request.scope["path"],
                "body": body.decode(),
                "workspace_db_id": assume_identity_authorised_workspace_db_id.get(),
                "enforce": enforce_credits_verification.get(),
            }
        )

    inner.add_middleware(ServerlessAuthMiddleware)
    return TestClient(inner)


@pytest.fixture
def platform(monkeypatch):
    """Fake usage-check transport: ``platform.answer`` is (status, payload)."""
    state = SimpleNamespace(answer=(200, {}), calls=[], error=None)

    def _request(method, url, **kwargs):
        state.calls.append((method, url, kwargs))
        if state.error is not None:
            raise state.error
        answer = state.answer.pop(0) if isinstance(state.answer, list) else state.answer
        if isinstance(answer, Exception):
            raise answer
        status, payload = answer
        return SimpleNamespace(status_code=status, json=lambda: payload)

    monkeypatch.setattr(platform_http, "_platform_request", _request)
    return state


@pytest.fixture
def workspace_lookup(monkeypatch):
    state = SimpleNamespace(
        answer=(True, "ws-lookup"), calls=[], gateway_flags=[], error=None
    )

    async def _validate(api_key, *, through_secure_gateway):
        state.gateway_flags.append(through_secure_gateway)
        state.calls.append(api_key)
        if state.error is not None:
            raise state.error
        return state.answer

    monkeypatch.setattr(serverless_auth, "validate_api_key", _validate)
    return state


def _ok_payload(**overrides):
    payload = {"workspace": "ws-1", "workspaceId": "db-1", "underCap": True}
    payload.update(overrides)
    return payload


@pytest.mark.parametrize(
    "method,path,kwargs,skipped",
    [
        ("GET", "/", {}, True),
        ("POST", "/", {}, True),
        ("GET", "/docs", {}, True),
        ("GET", "/info", {}, True),
        ("GET", "/healthz", {}, True),
        ("GET", "/readiness", {}, True),
        ("GET", "/metrics", {}, True),
        ("GET", "/openapi.json", {}, True),
        ("GET", "/model/registry", {}, True),
        ("POST", "/model/registry", {}, True),
        ("GET", "/static/app.js", {}, True),
        ("GET", "/_next/static/chunk.js", {}, True),
        ("PUT", "/infer/object_detection", {}, True),
        ("DELETE", "/v2/models", {}, True),
        ("PATCH", "/workflows/run", {}, True),
        ("GET", "/workflows/blocks/describe", {}, True),
        ("POST", "/workflows/blocks/describe", {"json": {}}, True),
        (
            "POST",
            "/workflows/blocks/describe",
            {"json": {"dynamic_blocks_definitions": []}},
            True,
        ),
        (
            "POST",
            "/workflows/blocks/describe",
            {"json": {"dynamic_blocks_definitions": [{"type": "x"}]}},
            False,
        ),
        ("POST", "/workflows/blocks/describe", {"content": b"x"}, False),
        ("GET", "/workflows/definition/schema", {}, True),
        ("POST", "/workflows/definition/schema", {"json": {"a": 1}}, True),
        (
            "POST",
            "/workflows/definition/schema",
            {"json": {"dynamic_blocks_definitions": [{"type": "x"}]}},
            False,
        ),
        ("GET", "/redoc", {}, False),
        ("GET", "/v2/server/health", {}, False),
        ("GET", "/v2/server/ready", {}, False),
        ("POST", "/workflows/run", {"json": {}}, False),
        ("POST", "/infer/object_detection", {"json": {}}, False),
        ("GET", "/ws/1", {}, False),
        ("POST", "/v2/models/infer", {}, False),
    ],
)
def test_skip_list(client, platform, method, path, kwargs, skipped):
    response = client.request(method, path, **kwargs)

    if skipped:
        assert response.status_code == 200, response.text
        assert response.json()["path"] == path
    else:
        assert response.status_code == 401
        assert response.json() == UNAUTHORIZED
    assert platform.calls == []


def test_api_key_precedence_query_over_bearer_over_body(client, platform):
    platform.answer = (200, _ok_payload())

    client.post(
        "/infer/x?api_key=from-query",
        headers={"Authorization": "Bearer from-header"},
        json={"api_key": "from-body"},
    )
    client.post(
        "/infer/x",
        headers={"Authorization": "Bearer from-header"},
        json={"api_key": "from-body"},
    )
    client.post("/infer/x", json={"api_key": "from-body"})

    keys = [url.split("api_key=")[1].split("&")[0] for _, url, _ in platform.calls]
    assert keys == ["from-query", "from-header", "from-body"]


def test_usage_check_request_shape(client, platform, monkeypatch):
    monkeypatch.setattr(configuration, "API_BASE_URL", "https://api.example.com")
    platform.answer = (200, _ok_payload())

    client.get("/some/route?api_key=k")

    method, url, kwargs = platform.calls[0]
    assert method == "get"
    assert url == (
        "https://api.example.com/serverless/usage-check?api_key=k&nocache=true"
    )
    assert kwargs["headers"]["x-roboflow-inference-version"]
    assert kwargs["timeout"] == platform_http.API_REQUEST_TIMEOUT_S


def test_authorized_request_passes_body_and_sets_workspace_header(client, platform):
    platform.answer = (200, _ok_payload())

    response = client.post("/infer/x", json={"api_key": "k", "value": 1})

    assert response.status_code == 200
    assert response.headers["X-Workspace-Id"] == "ws-1"
    assert json.loads(response.json()["body"]) == {"api_key": "k", "value": 1}
    assert response.json()["workspace_db_id"] == "db-1"
    assert response.json()["enforce"] is True


def test_workspace_falls_back_to_workspace_id(client, platform):
    platform.answer = (200, {"workspaceId": "db-9", "underCap": True})

    response = client.get("/some/route?api_key=k")

    assert response.status_code == 200
    assert response.headers["X-Workspace-Id"] == "db-9"


def test_401_from_platform(client, platform, clock):
    platform.answer = (401, {})

    response = client.get("/some/route?api_key=k")

    assert response.status_code == 401
    assert response.json() == SERVERLESS_UNAUTHORIZED
    assert "X-Workspace-Id" not in response.headers


def test_402_from_platform_appends_error_and_workspace(client, platform):
    platform.answer = (
        402,
        {
            "workspace": "ws-2",
            "workspaceId": "db-2",
            "underCap": False,
            "error": "Credit cap reached.",
        },
    )

    response = client.get("/some/route?api_key=k")

    assert response.status_code == 402
    assert response.json() == {
        "status": 402,
        "message": f"{CREDITS_MESSAGE} Credit cap reached.",
    }
    assert response.headers["X-Workspace-Id"] == "ws-2"


def test_402_without_error_uses_base_message(client, platform):
    platform.answer = (402, {"underCap": False})

    response = client.get("/some/route?api_key=k")

    assert response.status_code == 402
    assert response.json() == {"status": 402, "message": CREDITS_MESSAGE}
    assert "X-Workspace-Id" not in response.headers


@pytest.mark.parametrize(
    "payload",
    [
        {"workspace": "ws-1", "workspaceId": "db-1"},
        {"workspace": "ws-1", "underCap": False},
        {"workspace": "ws-1", "underCap": "true"},
        {"underCap": True},
        {"workspace": "bad ws", "underCap": True},
        {"workspace": "ws-1", "workspaceId": "bad id", "underCap": True},
        {"workspace": 5, "underCap": True},
    ],
)
def test_200_with_incomplete_data_is_500_and_not_cached(client, platform, payload):
    platform.answer = (200, payload)

    first = client.get("/some/route?api_key=k")
    second = client.get("/some/route?api_key=k")

    assert first.status_code == 500 and first.json() == INCOMPLETE
    assert second.status_code == 500
    assert len(platform.calls) == 2


def test_200_requires_workspace_db_id_when_assume_identity_token_set(
    client, platform, monkeypatch
):
    monkeypatch.setattr(
        configuration, "ROBOFLOW_ASSUME_IDENTITY_SERVICE_ACCESS_TOKEN", "tok"
    )
    platform.answer = (200, {"workspace": "ws-1", "underCap": True})

    response = client.get("/some/route?api_key=k")

    assert response.status_code == 500 and response.json() == INCOMPLETE


def test_unexpected_platform_status_is_500(client, platform):
    platform.answer = (403, {})

    response = client.get("/some/route?api_key=k")

    assert response.status_code == 500
    assert response.json() == {
        "status": 500,
        "message": (
            "Serverless authorization failed because the usage check returned "
            "an unexpected status (403)."
        ),
    }
    assert len(serverless_auth._cache) == 0


def test_platform_transport_error_propagates(client, platform):
    platform.error = LegacyHTTPError(
        504, "Timeout when attempting to connect to Roboflow API."
    )

    with pytest.raises(LegacyHTTPError) as raised:
        client.get("/some/route?api_key=k")

    assert raised.value.status_code == 504
    assert len(serverless_auth._cache) == 0


def test_success_is_cached_for_3600_seconds(client, platform, clock):
    platform.answer = (200, _ok_payload())

    client.get("/some/route?api_key=k")
    clock[0] += 3599
    client.get("/some/route?api_key=k")
    assert len(platform.calls) == 1

    clock[0] += 2
    response = client.get("/some/route?api_key=k")
    assert len(platform.calls) == 2
    assert response.headers["X-Workspace-Id"] == "ws-1"


@pytest.mark.parametrize("status", [401, 402])
def test_denials_are_cached_for_60_seconds(client, platform, clock, status):
    platform.answer = (status, {"workspace": "ws-3", "underCap": False})

    first = client.get("/some/route?api_key=k")
    clock[0] += 59
    second = client.get("/some/route?api_key=k")
    assert first.status_code == second.status_code == status
    assert first.json() == second.json()
    assert len(platform.calls) == 1

    clock[0] += 2
    client.get("/some/route?api_key=k")
    assert len(platform.calls) == 2


def test_cache_is_keyed_by_api_key(client, platform):
    platform.answer = (200, _ok_payload())

    client.get("/some/route?api_key=a")
    client.get("/some/route?api_key=b")
    client.get("/some/route?api_key=a")

    assert len(platform.calls) == 2


def test_non_billable_internal_request_skips_credit_check(
    client, platform, workspace_lookup, monkeypatch
):
    monkeypatch.setattr(configuration, "ROBOFLOW_SERVICE_SECRET", "s3cret")

    response = client.get(
        "/some/route?api_key=k&countinference=false&service_secret=s3cret"
    )

    assert response.status_code == 200
    assert response.headers["X-Workspace-Id"] == "ws-lookup"
    assert response.json()["enforce"] is False
    assert response.json()["workspace_db_id"] is None
    assert platform.calls == []
    assert workspace_lookup.calls == ["k"]


def test_non_billable_flags_from_json_body(
    client, platform, workspace_lookup, monkeypatch
):
    monkeypatch.setattr(configuration, "ROBOFLOW_SERVICE_SECRET", "s3cret")

    response = client.post(
        "/some/route",
        json={"api_key": "k", "countinference": False, "service_secret": "s3cret"},
    )

    assert response.status_code == 200
    assert platform.calls == []
    assert workspace_lookup.calls == ["k"]


@pytest.mark.parametrize(
    "query",
    [
        "countinference=false&service_secret=wrong",
        "countinference=false",
        "countinference=true&service_secret=s3cret",
        "service_secret=s3cret",
    ],
)
def test_unauthenticated_non_billable_intent_still_checks_credits(
    client, platform, workspace_lookup, monkeypatch, query
):
    monkeypatch.setattr(configuration, "ROBOFLOW_SERVICE_SECRET", "s3cret")
    platform.answer = (200, _ok_payload())

    response = client.get(f"/some/route?api_key=k&{query}")

    assert response.status_code == 200
    assert len(platform.calls) == 1
    assert workspace_lookup.calls == []


def test_non_billable_without_configured_secret_checks_credits(
    client, platform, workspace_lookup, monkeypatch
):
    monkeypatch.setattr(configuration, "ROBOFLOW_SERVICE_SECRET", None)
    platform.answer = (200, _ok_payload())

    client.get("/some/route?api_key=k&countinference=false&service_secret=")

    assert len(platform.calls) == 1


def test_non_billable_invalid_key_is_401_cached_60_seconds(
    client, platform, workspace_lookup, monkeypatch, clock
):
    monkeypatch.setattr(configuration, "ROBOFLOW_SERVICE_SECRET", "s3cret")
    workspace_lookup.answer = (False, None)
    query = "api_key=k&countinference=false&service_secret=s3cret"

    first = client.get(f"/some/route?{query}")
    clock[0] += 59
    second = client.get(f"/some/route?{query}")
    clock[0] += 2
    client.get(f"/some/route?{query}")

    assert first.status_code == 401 and first.json() == UNAUTHORIZED
    assert second.json() == UNAUTHORIZED
    assert workspace_lookup.calls == ["k", "k"]


def test_non_billable_and_billable_entries_are_cached_separately(
    client, platform, workspace_lookup, monkeypatch
):
    monkeypatch.setattr(configuration, "ROBOFLOW_SERVICE_SECRET", "s3cret")
    platform.answer = (200, _ok_payload())

    client.get("/some/route?api_key=k")
    client.get("/some/route?api_key=k&countinference=false&service_secret=s3cret")
    client.get("/some/route?api_key=k")

    assert len(platform.calls) == 1
    assert workspace_lookup.calls == ["k"]


def test_workspace_lookup_backend_unavailable_propagates(
    client, platform, workspace_lookup, monkeypatch
):
    monkeypatch.setattr(configuration, "ROBOFLOW_SERVICE_SECRET", "s3cret")
    workspace_lookup.error = AuthBackendUnavailable("down")

    with pytest.raises(AuthBackendUnavailable):
        client.get("/some/route?api_key=k&countinference=false&service_secret=s3cret")

    assert len(serverless_auth._cache) == 0


def test_cached_success_without_db_id_is_refreshed_when_token_set(
    client, platform, monkeypatch
):
    platform.answer = (200, {"workspace": "ws-1", "underCap": True})
    client.get("/some/route?api_key=k")
    assert len(platform.calls) == 1

    monkeypatch.setattr(
        configuration, "ROBOFLOW_ASSUME_IDENTITY_SERVICE_ACCESS_TOKEN", "tok"
    )
    platform.answer = (200, _ok_payload())
    response = client.get("/some/route?api_key=k")

    assert len(platform.calls) == 2
    assert response.status_code == 200
    assert response.json()["workspace_db_id"] == "db-1"


def test_contextvars_are_reset_after_request(client, platform):
    platform.answer = (200, _ok_payload())

    client.get("/some/route?api_key=k")

    assert assume_identity_authorised_workspace_db_id.get() is None
    assert enforce_credits_verification.get() is True


def test_unparsable_json_body_without_key_is_401(client, platform):
    response = client.post(
        "/infer/x", content=b"{not json", headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 401
    assert response.json() == UNAUTHORIZED
    assert platform.calls == []


def test_non_string_api_key_in_body_is_401(client, platform):
    response = client.post("/infer/x", json={"api_key": 12})

    assert response.status_code == 401
    assert platform.calls == []


def test_unencodable_service_secret_still_checks_credits(
    client, platform, workspace_lookup, monkeypatch
):
    monkeypatch.setattr(configuration, "ROBOFLOW_SERVICE_SECRET", "s3cret")
    platform.answer = (200, _ok_payload())

    response = client.post(
        "/some/route",
        content=b'{"api_key": "k", "countinference": false, "service_secret": "\\ud800"}',
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 200
    assert len(platform.calls) == 1
    assert workspace_lookup.calls == []


@pytest.fixture
def retry(monkeypatch):
    sleeps = []
    monkeypatch.setattr(serverless_auth, "_sleep", sleeps.append)
    monkeypatch.setattr(configuration, "TRANSIENT_ROBOFLOW_API_ERRORS_RETRIES", 3)
    monkeypatch.setattr(
        configuration, "TRANSIENT_ROBOFLOW_API_ERRORS_RETRY_INTERVAL", 7
    )
    return sleeps


def test_transient_status_is_retried_then_succeeds(
    client, platform, retry, monkeypatch
):
    monkeypatch.setattr(configuration, "TRANSIENT_ROBOFLOW_API_ERRORS", {503})
    platform.answer = [(503, {}), (200, _ok_payload())]

    response = client.get("/some/route?api_key=k")

    assert response.status_code == 200
    assert response.headers["X-Workspace-Id"] == "ws-1"
    assert len(platform.calls) == 2
    assert retry == [7]


def test_transient_status_exhausts_retries(client, platform, retry, monkeypatch):
    monkeypatch.setattr(configuration, "TRANSIENT_ROBOFLOW_API_ERRORS", {503})
    platform.answer = [(503, {}), (503, {}), (503, {})]

    response = client.get("/some/route?api_key=k")

    assert response.status_code == 500
    assert response.json()["message"].endswith("unexpected status (503).")
    assert len(platform.calls) == 3
    assert retry == [7, 7]
    assert len(serverless_auth._cache) == 0


def test_status_outside_transient_set_is_not_retried(
    client, platform, retry, monkeypatch
):
    monkeypatch.setattr(configuration, "TRANSIENT_ROBOFLOW_API_ERRORS", set())
    platform.answer = (503, {})

    response = client.get("/some/route?api_key=k")

    assert response.status_code == 500
    assert len(platform.calls) == 1
    assert retry == []


def test_transient_set_never_retries_401_or_402(client, platform, retry, monkeypatch):
    monkeypatch.setattr(configuration, "TRANSIENT_ROBOFLOW_API_ERRORS", {401, 402})
    platform.answer = (402, {"underCap": False})

    response = client.get("/some/route?api_key=k")

    assert response.status_code == 402
    assert len(platform.calls) == 1
    assert retry == []


def test_serverless_auth_imports_without_workflows(monkeypatch):
    import importlib
    import sys

    for name in [
        module
        for module in list(sys.modules)
        if module == "roboflow_workflows" or module.startswith("roboflow_workflows.")
    ]:
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setitem(sys.modules, "inference_server.workflows.host", None)
    monkeypatch.delitem(sys.modules, "inference_server.hosted.serverless_auth")
    monkeypatch.delitem(sys.modules, "inference_server.platform_http")

    module = importlib.import_module("inference_server.hosted.serverless_auth")

    assert module.ServerlessAuthMiddleware is not None


def _connection_error():
    return LegacyHTTPError(503, "Internal error. Could not connect to Roboflow API.")


def test_connection_error_is_retried_when_enabled(client, platform, retry, monkeypatch):
    monkeypatch.setattr(configuration, "RETRY_CONNECTION_ERRORS_TO_ROBOFLOW_API", True)
    platform.answer = [_connection_error(), (200, _ok_payload())]

    response = client.get("/some/route?api_key=k")

    assert response.status_code == 200
    assert len(platform.calls) == 2
    assert retry == [7]


def test_connection_error_exhaustion_reraises_original(
    client, platform, retry, monkeypatch
):
    monkeypatch.setattr(configuration, "RETRY_CONNECTION_ERRORS_TO_ROBOFLOW_API", True)
    last = _connection_error()
    platform.answer = [_connection_error(), _connection_error(), last]

    with pytest.raises(LegacyHTTPError) as raised:
        client.get("/some/route?api_key=k")

    assert raised.value is last
    assert len(platform.calls) == 3
    assert retry == [7, 7]
    assert len(serverless_auth._cache) == 0


def test_connection_error_propagates_at_once_by_default(
    client, platform, retry, monkeypatch
):
    monkeypatch.setattr(configuration, "RETRY_CONNECTION_ERRORS_TO_ROBOFLOW_API", False)
    platform.answer = [_connection_error(), (200, _ok_payload())]

    with pytest.raises(LegacyHTTPError):
        client.get("/some/route?api_key=k")

    assert len(platform.calls) == 1
    assert retry == []


def test_timeout_is_not_retried_even_when_enabled(client, platform, retry, monkeypatch):
    monkeypatch.setattr(configuration, "RETRY_CONNECTION_ERRORS_TO_ROBOFLOW_API", True)
    platform.answer = [
        LegacyHTTPError(504, "Timeout when attempting to connect to Roboflow API."),
        (200, _ok_payload()),
    ]

    with pytest.raises(LegacyHTTPError) as raised:
        client.get("/some/route?api_key=k")

    assert raised.value.status_code == 504
    assert len(platform.calls) == 1


def test_usage_check_omits_assume_identity_headers(client, monkeypatch):
    import requests

    monkeypatch.setattr(
        configuration, "ROBOFLOW_ASSUME_IDENTITY_SERVICE_ACCESS_TOKEN", "tok"
    )
    seen = {}

    def _get(url, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(status_code=200, json=lambda: _ok_payload())

    monkeypatch.setattr(requests, "get", _get)
    reset = assume_identity_authorised_workspace_db_id.set("db-1")
    try:
        response = client.get("/some/route?api_key=k")
    finally:
        assume_identity_authorised_workspace_db_id.reset(reset)

    assert response.status_code == 200
    assert "x-assume-identity-access-token" not in seen["headers"]
    assert "x-assume-identity-authorised-workspace" not in seen["headers"]


@pytest.mark.parametrize(
    "switch,root_path,method,path,authenticated",
    [
        (False, "", "POST", "/v2/1", False),
        (False, "", "POST", "/infer/x", False),
        (False, "", "GET", "/v2/server/health", True),
        (False, "/service", "GET", "/service/v2/server/health", True),
        (False, "/service", "POST", "/service/v2/1", False),
        (True, "", "POST", "/v2/1", True),
        (True, "", "POST", "/infer/x", True),
        (True, "", "GET", "/v2/server/health", True),
        (True, "/service", "GET", "/service/v2/server/health", True),
    ],
)
def test_bearer_header_follows_switch_outside_v2_routes(
    platform, monkeypatch, switch, root_path, method, path, authenticated
):
    from inference_server.routers import v2_server

    monkeypatch.setattr(configuration, "ALLOW_API_KEY_FROM_HEADERS", switch)
    platform.answer = (200, _ok_payload())
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.include_router(v2_server.router)

    @app.api_route("/{full_path:path}", methods=["GET", "POST"])
    async def _probe(request: Request):
        return JSONResponse({})

    app.add_middleware(ServerlessAuthMiddleware)

    response = TestClient(app, root_path=root_path).request(
        method, path, headers={"Authorization": "Bearer from-header"}
    )

    keys = [url.split("api_key=")[1].split("&")[0] for _, url, _ in platform.calls]
    if authenticated:
        assert response.status_code == 200
        assert keys == ["from-header"]
    else:
        assert response.status_code == 401
        assert response.json() == UNAUTHORIZED
        assert keys == []
