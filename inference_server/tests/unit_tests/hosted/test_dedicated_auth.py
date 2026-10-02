import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from inference_server import configuration
from inference_server.errors import AuthBackendUnavailable
from inference_server.hosted import dedicated_auth
from inference_server.hosted.dedicated_auth import DedicatedAuthMiddleware

UNAUTHORIZED = {"status": 401, "message": "Unauthorized api_key"}


@pytest.fixture
def client():
    inner = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @inner.api_route(
        "/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"]
    )
    async def _probe(request: Request):
        body = await request.body()
        return JSONResponse({"path": request.scope["path"], "body": body.decode()})

    inner.add_middleware(DedicatedAuthMiddleware)
    return TestClient(inner)


@pytest.fixture
def workspace_lookup(monkeypatch):
    state = SimpleNamespace(answers={}, calls=[], error=None)

    async def _validate(api_key):
        state.calls.append(api_key)
        if state.error is not None:
            raise state.error
        return state.answers.get(api_key, (False, None))

    monkeypatch.setattr(dedicated_auth, "validate_api_key", _validate)
    return state


@pytest.fixture
def allow_list(monkeypatch):
    monkeypatch.setattr(configuration, "DEDICATED_DEPLOYMENT_WORKSPACE_URL", "ws-url")
    monkeypatch.setattr(
        configuration, "WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT", ["ws-a", "ws-b"]
    )


@pytest.mark.parametrize(
    "method,path,skipped",
    [
        ("GET", "/", True),
        ("POST", "/", True),
        ("GET", "/docs", True),
        ("GET", "/redoc", True),
        ("GET", "/info", True),
        ("GET", "/healthz", True),
        ("GET", "/readiness", True),
        ("GET", "/secure-gateway/health", True),
        ("GET", "/metrics", True),
        ("GET", "/openapi.json", True),
        ("GET", "/static/app.js", True),
        ("GET", "/_next/static/chunk.js", True),
        ("PUT", "/infer/object_detection", True),
        ("DELETE", "/v2/models", True),
        ("PATCH", "/workflows/run", True),
        ("GET", "/model/registry", False),
        ("GET", "/workflows/blocks/describe", False),
        ("POST", "/workflows/definition/schema", False),
        ("GET", "/v2/server/health", False),
        ("POST", "/workflows/run", False),
        ("POST", "/infer/object_detection", False),
        ("GET", "/ws/1", False),
    ],
)
def test_skip_list(client, workspace_lookup, allow_list, method, path, skipped):
    response = client.request(method, path)

    if skipped:
        assert response.status_code == 200, response.text
        assert response.json()["path"] == path
    else:
        assert response.status_code == 401
        assert response.json() == UNAUTHORIZED
    assert workspace_lookup.calls == []


def test_api_key_precedence_query_over_bearer_over_body(
    client, workspace_lookup, allow_list
):
    workspace_lookup.answers = {
        "from-query": (True, "ws-a"),
        "from-header": (True, "ws-a"),
        "from-body": (True, "ws-a"),
    }

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

    assert workspace_lookup.calls == ["from-query", "from-header", "from-body"]


@pytest.mark.parametrize("workspace", ["ws-url", "ws-a", "ws-b"])
def test_allowed_workspace_passes_with_header_and_body(
    client, workspace_lookup, allow_list, workspace
):
    workspace_lookup.answers = {"k": (True, workspace)}

    response = client.post("/infer/x", json={"api_key": "k", "value": 1})

    assert response.status_code == 200
    assert response.headers["X-Workspace-Id"] == workspace
    assert json.loads(response.json()["body"]) == {"api_key": "k", "value": 1}


def test_workspace_outside_allow_list_is_401(client, workspace_lookup, allow_list):
    workspace_lookup.answers = {"k": (True, "ws-other")}

    response = client.get("/infer/x?api_key=k")

    assert response.status_code == 401
    assert response.json() == UNAUTHORIZED
    assert "X-Workspace-Id" not in response.headers


def test_invalid_key_is_401(client, workspace_lookup, allow_list):
    response = client.get("/infer/x?api_key=bad")

    assert response.status_code == 401
    assert response.json() == UNAUTHORIZED


def test_only_workspace_url_configured(client, workspace_lookup, monkeypatch):
    monkeypatch.setattr(configuration, "DEDICATED_DEPLOYMENT_WORKSPACE_URL", "only")
    monkeypatch.setattr(
        configuration, "WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT", []
    )
    workspace_lookup.answers = {"k": (True, "only"), "j": (True, "other")}

    assert client.get("/infer/x?api_key=k").status_code == 200
    assert client.get("/infer/x?api_key=j").status_code == 401


def test_only_whitelist_configured(client, workspace_lookup, monkeypatch):
    monkeypatch.setattr(configuration, "DEDICATED_DEPLOYMENT_WORKSPACE_URL", None)
    monkeypatch.setattr(
        configuration, "WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT", ["ws-a"]
    )
    workspace_lookup.answers = {"k": (True, "ws-a"), "j": (True, None)}

    assert client.get("/infer/x?api_key=k").status_code == 200
    assert client.get("/infer/x?api_key=j").status_code == 401


def test_backend_unavailable_propagates(client, workspace_lookup, allow_list):
    workspace_lookup.error = AuthBackendUnavailable("down")

    with pytest.raises(AuthBackendUnavailable):
        client.get("/infer/x?api_key=k")


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
    workspace_lookup,
    allow_list,
    monkeypatch,
    switch,
    root_path,
    method,
    path,
    authenticated,
):
    from inference_server.routers import v2_server

    monkeypatch.setattr(configuration, "ALLOW_API_KEY_FROM_HEADERS", switch)
    workspace_lookup.answers = {"from-header": (True, "ws-a")}
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.include_router(v2_server.router)

    @app.api_route("/{full_path:path}", methods=["GET", "POST"])
    async def _probe(request: Request):
        return JSONResponse({})

    app.add_middleware(DedicatedAuthMiddleware)

    response = TestClient(app, root_path=root_path).request(
        method, path, headers={"Authorization": "Bearer from-header"}
    )

    if authenticated:
        assert response.status_code == 200
        assert workspace_lookup.calls == ["from-header"]
    else:
        assert response.status_code == 401
        assert response.json() == UNAUTHORIZED
        assert workspace_lookup.calls == []
