import importlib
import logging

import pytest
from fastapi.testclient import TestClient

from tests.unit_tests.legacy.conftest import route_paths

HIDDEN_ON_ANY_HOSTED_FLAG = (
    "/model/add",
    "/model/remove",
    "/model/clear",
    "/infer/object_detection",
    "/infer/instance_segmentation",
    "/infer/semantic_segmentation",
    "/infer/classification",
    "/infer/embeddings",
    "/infer/keypoints_detection",
    "/clear_cache",
    "/start/{dataset_id}/{version_id}",
    "/device/stats",
    "/notebook/start",
)
HIDDEN_ON_LAMBDA_ONLY = (
    "/model/registry",
    "/infer/lmm",
    "/infer/lmm/{model_id:path}",
)
NEVER_HIDDEN = (
    "/info",
    "/healthz",
    "/readiness",
    "/logs",
    "/clip/embed_image",
    "/sam2/segment_image",
    "/infer/depth-estimation",
    "/infer/depth-estimation/{model_id:path}",
    "/infer/action_recognition",
    "/{dataset_id}/{version_id}",
    "/workflows/run",
    "/workflows/blocks/describe",
    "/v2/models/infer",
    "/v2/server/ready",
)


@pytest.fixture
def reloaded_app(monkeypatch):
    import inference_server.app as app_mod
    from inference_server import configuration

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


def _openapi_paths(app) -> set:
    return set(TestClient(app).get("/openapi.json").json()["paths"])


def _openapi_path(path: str) -> str:
    return path.replace(":path}", "}")


def _expected(path: str, lambda_flag: bool, gcp_flag: bool) -> bool:
    if path in HIDDEN_ON_ANY_HOSTED_FLAG:
        return not (lambda_flag or gcp_flag)
    if path in HIDDEN_ON_LAMBDA_ONLY:
        return not lambda_flag
    return True


@pytest.mark.parametrize(
    "lambda_flag,gcp_flag",
    [(False, False), (True, False), (False, True), (True, True)],
)
@pytest.mark.parametrize(
    "path", HIDDEN_ON_ANY_HOSTED_FLAG + HIDDEN_ON_LAMBDA_ONLY + NEVER_HIDDEN
)
def test_hosted_route_matrix(reloaded_app, lambda_flag, gcp_flag, path):
    module = reloaded_app(LAMBDA=lambda_flag, GCP_SERVERLESS=gcp_flag)

    visible = _expected(path, lambda_flag, gcp_flag)
    assert (path in route_paths(module.app)) is visible
    assert (_openapi_path(path) in _openapi_paths(module.app)) is visible


def test_default_configuration_has_no_hosted_middleware(reloaded_app):
    module = reloaded_app()

    names = [middleware.cls.__name__ for middleware in module.app.user_middleware]
    assert "ServerlessAuthMiddleware" not in names
    assert "DedicatedAuthMiddleware" not in names
    assert names[:3] == [
        "CorrelationIdMiddleware",
        "ModelLoadHeadersMiddleware",
        "PathAwareCORSMiddleware",
    ]
    assert names.index("BillingIntentMiddleware") == names.index("_AuthMiddleware") - 1
    assert set(HIDDEN_ON_ANY_HOSTED_FLAG + HIDDEN_ON_LAMBDA_ONLY) <= route_paths(
        module.app
    )


def test_serverless_middleware_is_outermost_under_gcp_serverless(reloaded_app):
    module = reloaded_app(GCP_SERVERLESS=True)

    names = [middleware.cls.__name__ for middleware in module.app.user_middleware]
    assert names[:3] == [
        "CorrelationIdMiddleware",
        "ModelLoadHeadersMiddleware",
        "ServerlessAuthMiddleware",
    ]
    assert "DedicatedAuthMiddleware" not in names


@pytest.mark.parametrize(
    "overrides",
    [
        {"DEDICATED_DEPLOYMENT_WORKSPACE_URL": "ws"},
        {"WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT": ["ws"]},
        {"REQUIRE_API_KEY_FOR_LOCAL_DEPLOYMENT": True},
    ],
)
def test_dedicated_middleware_added_by_allow_list(reloaded_app, overrides):
    module = reloaded_app(**overrides)

    names = [middleware.cls.__name__ for middleware in module.app.user_middleware]
    assert names[2] == "DedicatedAuthMiddleware"
    assert "ServerlessAuthMiddleware" not in names


def test_dedicated_wraps_serverless_when_both_configured(reloaded_app):
    module = reloaded_app(GCP_SERVERLESS=True, DEDICATED_DEPLOYMENT_WORKSPACE_URL="ws")

    names = [middleware.cls.__name__ for middleware in module.app.user_middleware]
    assert names[2:4] == ["DedicatedAuthMiddleware", "ServerlessAuthMiddleware"]


def test_lambda_alone_adds_no_middleware(reloaded_app):
    module = reloaded_app(LAMBDA=True)

    names = [middleware.cls.__name__ for middleware in module.app.user_middleware]
    assert "ServerlessAuthMiddleware" not in names
    assert "DedicatedAuthMiddleware" not in names


def test_serverless_auth_is_wired_end_to_end(reloaded_app, monkeypatch):
    from inference_server import platform_http

    module = reloaded_app(GCP_SERVERLESS=True)
    monkeypatch.setattr(
        platform_http, "_platform_request", lambda *a, **k: pytest.fail("not expected")
    )

    with TestClient(module.app) as client:
        assert client.get("/info").status_code == 200
        response = client.post("/workflows/run", json={"specification": {}})

    assert response.status_code == 401
    assert response.json() == {"status": 401, "message": "Unauthorized api_key"}


def test_required_api_key_is_ignored_by_hosted_runtimes(reloaded_app):
    module = reloaded_app(REQUIRE_API_KEY_FOR_LOCAL_DEPLOYMENT=True, LAMBDA=True)

    names = [middleware.cls.__name__ for middleware in module.app.user_middleware]
    assert "DedicatedAuthMiddleware" not in names


@pytest.mark.parametrize(
    "host,overrides,gated",
    [
        ("127.0.0.1", {}, True),
        ("0.0.0.0", {}, False),
        ("127.0.0.1", {"REQUIRE_API_KEY_FOR_LOCAL_DEPLOYMENT": True}, False),
        ("127.0.0.1", {"GCP_SERVERLESS": True}, False),
    ],
)
def test_local_callers_gate_only_on_keyless_loopback_server(
    reloaded_app, monkeypatch, host, overrides, gated
):
    monkeypatch.setenv("HOST", host)
    monkeypatch.delenv("PUBLISHED_HOST", raising=False)
    module = reloaded_app(**overrides)

    names = [middleware.cls.__name__ for middleware in module.app.user_middleware]
    assert ("_LocalCallersOnlyMiddleware" in names) is gated


def test_required_api_key_without_a_workspace_answers_401_and_logs(reloaded_app):
    # The application logger does not propagate to the root logger.
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    app_logger = logging.getLogger("inference_server.app")
    app_logger.addHandler(handler)
    try:
        module = reloaded_app(
            REQUIRE_API_KEY_FOR_LOCAL_DEPLOYMENT=True, DEFAULT_API_KEY=None
        )
    finally:
        app_logger.removeHandler(handler)
    from inference_server.hosted.dedicated_auth import NO_WORKSPACE_MESSAGE

    client = TestClient(module.app)
    response = client.post("/workflows/run", json={"api_key": "k"})

    assert response.status_code == 401
    assert response.json()["message"] == NO_WORKSPACE_MESSAGE
    assert client.get("/healthz").status_code == 200
    assert any("SECURITY" in record.getMessage() for record in records)


def test_required_api_key_restricts_v2_to_the_server_workspace(
    reloaded_app, monkeypatch
):
    from inference_server.hosted import dedicated_auth

    module = reloaded_app(
        REQUIRE_API_KEY_FOR_LOCAL_DEPLOYMENT=True,
        DEFAULT_API_KEY="server-key",
        ENABLE_CONTROL_PLANE_ROUTES=False,
    )
    answers = {"server-key": (True, "own"), "k": (True, "own"), "j": (True, "other")}

    async def _validate(api_key, *, through_secure_gateway=False):
        return answers.get(api_key, (False, None))

    monkeypatch.setattr(dedicated_auth, "validate_api_key", _validate)
    monkeypatch.setattr(module, "validate_api_key", _validate)
    monkeypatch.setattr(dedicated_auth, "_server_workspace", None)
    client = TestClient(module.app)

    other = client.get("/v2/server/info", headers={"Authorization": "Bearer j"})
    own = client.get("/v2/server/info", headers={"Authorization": "Bearer k"})

    assert other.status_code == 401
    # Past the workspace check, the control-plane switch answers.
    assert own.status_code == 403
    assert b"control-plane routes disabled" in own.content
