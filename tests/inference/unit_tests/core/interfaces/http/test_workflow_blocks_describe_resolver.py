"""`/workflows/blocks/describe` compiles the request's dynamic blocks with the
authenticated api key; after Phase 9 it must also hand the SERVER workspace
resolver to that compilation, or Modal validation silently runs as `anonymous`
for an authenticated caller (round-5 defect 1)."""

from unittest.mock import AsyncMock, MagicMock

from starlette.testclient import TestClient

import inference.core.roboflow_api as roboflow_api
from inference.core.interfaces.roboflow_platform_client import SERVER_WORKSPACE_RESOLVER
from inference.core.workflows.execution_engine.v1.dynamic_blocks import (
    block_scaffolding,
    modal_executor,
)

DYNAMIC_BLOCK = {
    "type": "DynamicBlockDefinition",
    "manifest": {
        "type": "ManifestDescription",
        "block_type": "DescribeProbe",
        "inputs": {
            "a": {
                "type": "DynamicInputDefinition",
                "selector_types": ["input_parameter"],
            }
        },
        "outputs": {"out": {"type": "DynamicOutputDefinition"}},
    },
    "code": {
        "type": "PythonCode",
        "run_function_code": 'def run(self, a):\n    return {"out": a}\n',
    },
}


class _DummyInstrumentator:
    def __init__(self, app, model_manager, endpoint="/metrics"):
        self.app = app
        self.model_manager = model_manager
        self.endpoint = endpoint

    def set_stream_manager_client(self, stream_manager_client) -> None:
        self.stream_manager_client = stream_manager_client


def _client(monkeypatch) -> TestClient:
    import inference.core.interfaces.http.http_api as http_api

    monkeypatch.setattr(http_api, "InferenceInstrumentator", _DummyInstrumentator)
    monkeypatch.setattr(
        http_api.usage_collector, "async_push_usage_payloads", AsyncMock()
    )
    model_manager = MagicMock()
    model_manager.pingback = None
    model_manager.num_errors = 0
    return TestClient(http_api.HttpInterface(model_manager=model_manager).app)


def test_describe_route_hands_the_server_resolver_to_dynamic_compilation(
    monkeypatch,
) -> None:
    import inference.core.interfaces.http.handlers.workflows as handler

    captured = {}

    def recording_compile(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(handler, "compile_dynamic_blocks", recording_compile)
    response = _client(monkeypatch).post(
        "/workflows/blocks/describe",
        json={
            "dynamic_blocks_definitions": [DYNAMIC_BLOCK],
            "api_key": "authenticated-key",
        },
    )
    assert response.status_code == 200, response.text
    assert captured["api_key"] == "authenticated-key"
    assert captured["workspace_resolver"] is SERVER_WORKSPACE_RESOLVER


def test_describe_route_validates_in_the_authenticated_workspace(monkeypatch) -> None:
    """Behaviour: the workspace Modal validation receives is the one the
    authenticated key resolves to - not `anonymous`. Green before AND after
    the port: the lookup is stubbed wherever it lives (the pre-port module
    attribute, while it still exists, and the server function the adapter
    reads through)."""

    def lookup(api_key):
        return f"ws-of-{api_key}"

    monkeypatch.setattr(roboflow_api, "get_roboflow_workspace", lookup)
    if hasattr(block_scaffolding, "get_roboflow_workspace"):
        monkeypatch.setattr(block_scaffolding, "get_roboflow_workspace", lookup)
    validated = []
    monkeypatch.setattr(
        block_scaffolding, "WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE", "modal"
    )
    monkeypatch.setattr(
        modal_executor,
        "validate_code_in_modal",
        lambda python_code, workspace_id=None: validated.append(workspace_id) or True,
    )
    response = _client(monkeypatch).post(
        "/workflows/blocks/describe",
        json={
            "dynamic_blocks_definitions": [DYNAMIC_BLOCK],
            "api_key": "authenticated-key",
        },
    )
    assert response.status_code == 200, response.text
    assert validated == ["ws-of-authenticated-key"]
    assert any(
        b["manifest_type_identifier"] == "DescribeProbe"
        for b in response.json()["blocks"]
    )
