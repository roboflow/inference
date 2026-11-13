import pytest
from fastapi.testclient import TestClient

from tests.unit_tests.legacy.conftest import route_paths
from tests.unit_tests.ops.test_device_stats import NOT_CONFIGURED_BODY

GATED_PATHS = ("/device/stats", "/notebook/start", "/secure-gateway/health")


@pytest.mark.parametrize(
    "hosted", [{"LAMBDA": True}, {"GCP_SERVERLESS": True}], ids=["lambda", "gcp"]
)
def test_gated_routes_are_absent_in_hosted_modes(reloaded_app, hosted):
    module = reloaded_app(SECURE_GATEWAY_HEALTH_ENDPOINT_ENABLED=True, **hosted)

    paths = route_paths(module.app)
    assert paths.isdisjoint(GATED_PATHS)
    assert "/logs" in paths


def test_secure_gateway_route_needs_its_flag(reloaded_app):
    disabled = route_paths(reloaded_app().app)
    enabled = route_paths(reloaded_app(SECURE_GATEWAY_HEALTH_ENDPOINT_ENABLED=True).app)

    assert "/secure-gateway/health" not in disabled
    assert {"/device/stats", "/notebook/start", "/logs"} <= disabled
    assert "/secure-gateway/health" in enabled


def test_routes_answer_themselves_instead_of_the_catch_all(reloaded_app, monkeypatch):
    from inference_server.ops import secure_gateway

    monkeypatch.setattr(secure_gateway.models_configuration, "SECURE_GATEWAY", None)
    module = reloaded_app(
        SECURE_GATEWAY_HEALTH_ENDPOINT_ENABLED=True,
        DOCKER_SOCKET_PATH=None,
        NOTEBOOK_ENABLED=False,
        ENABLE_IN_MEMORY_LOGS=False,
    )
    assert "/{dataset_id}/{version_id}" in route_paths(module.app)
    client = TestClient(module.app, follow_redirects=False)

    device_stats = client.get("/device/stats")
    gateway_health = client.get("/secure-gateway/health")
    notebook = client.get("/notebook/start?browserless=true")
    logs = client.get("/logs")

    assert device_stats.status_code == 404
    assert device_stats.content == NOT_CONFIGURED_BODY
    assert gateway_health.status_code == 404
    assert gateway_health.json()["status"] == "not_configured"
    assert notebook.status_code == 200
    assert notebook.json()["success"] is False
    assert logs.status_code == 404
    assert logs.content == b'{"detail":"Logs endpoint not available"}'


def test_ops_routes_are_absent_when_legacy_routes_are_off(reloaded_app):
    module = reloaded_app(
        LEGACY_ROUTES_ENABLED=False,
        ENABLE_IN_MEMORY_LOGS=True,
        SECURE_GATEWAY_HEALTH_ENDPOINT_ENABLED=True,
    )

    paths = route_paths(module.app)
    assert paths.isdisjoint(GATED_PATHS + ("/logs",))
