"""The `/workflows/definition/schema` route keeps its name and OpenAPI
operation id across Phase 9's DTO repoint (round-3 defect 8)."""

from unittest.mock import AsyncMock, MagicMock

from starlette.testclient import TestClient


class _DummyInstrumentator:
    def __init__(self, app, model_manager, endpoint="/metrics"):
        self.app = app
        self.model_manager = model_manager
        self.endpoint = endpoint

    def set_stream_manager_client(self, stream_manager_client) -> None:
        self.stream_manager_client = stream_manager_client


def _build_app(monkeypatch):
    import inference.core.interfaces.http.http_api as http_api

    monkeypatch.setattr(http_api, "InferenceInstrumentator", _DummyInstrumentator)
    monkeypatch.setattr(
        http_api.usage_collector, "async_push_usage_payloads", AsyncMock()
    )
    model_manager = MagicMock()
    model_manager.pingback = None
    model_manager.num_errors = 0
    return http_api.HttpInterface(model_manager=model_manager).app


def test_schema_route_keeps_its_operation_id(monkeypatch) -> None:
    app = _build_app(monkeypatch)
    route = next(
        r
        for r in app.routes
        if getattr(r, "path", None) == "/workflows/definition/schema"
    )
    assert route.name == "get_workflow_schema"
    operation = app.openapi()["paths"]["/workflows/definition/schema"]["get"]
    assert (
        operation["operationId"]
        == "get_workflow_schema_workflows_definition_schema_get"
    )


def test_schema_route_returns_the_wrapped_schema(monkeypatch) -> None:
    client = TestClient(_build_app(monkeypatch))
    response = client.get("/workflows/definition/schema")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"schema"}
    assert "$defs" in body["schema"] or "definitions" in body["schema"]
