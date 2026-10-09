import importlib

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
