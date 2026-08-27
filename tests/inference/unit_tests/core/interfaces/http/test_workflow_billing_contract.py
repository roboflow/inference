"""Contract: the workflow-run route honours authenticated `countinference=false`.

The route declares `countinference` / `service_secret` purely so FastAPI binds
them where the request usage decorator can read them - no handler code touches
them. This is the only coverage of that seam: the usage-tracking unit tests
call decorated fakes with the parameters already bound, so the parameters could
be deleted from the route without failing any of them. Here the real route is
driven over HTTP and the usage rows the collector recorded are inspected - the
workflow row and the model row recorded during workflow execution must inherit
the request's authenticated opt-out.
"""

import json
from unittest.mock import AsyncMock, MagicMock

from starlette.testclient import TestClient

SERVICE_SECRET = "workflow-billing-contract-secret"

# The workflow runs no real model; the block records a model-category usage row
# the same way a model's decorated `infer` would, from inside a step worker.
FAKE_MODEL_BLOCK_CODE = """
def run(self, value) -> BlockResult:
    from inference.usage_tracking.collector import usage_collector

    class FakeModel:
        api_key = "__API_KEY__"
        model_id = "fake-project/1"

        @usage_collector(category="model")
        def infer(self, image, **kwargs):
            return "ok"

    FakeModel().infer(value)
    return {"result": True}
"""


class _DummyInstrumentator:
    def __init__(self, app, model_manager, endpoint="/metrics"):
        self.app = app
        self.model_manager = model_manager
        self.endpoint = endpoint

    def set_stream_manager_client(self, stream_manager_client) -> None:
        self.stream_manager_client = stream_manager_client


def _build_test_client(monkeypatch) -> TestClient:
    import inference.core.interfaces.http.http_api as http_api
    from inference.core import roboflow_api
    from inference.usage_tracking import collector as collector_module

    # The validating module and the forwarding module must agree on the secret.
    monkeypatch.setattr(roboflow_api, "ROBOFLOW_SERVICE_SECRET", SERVICE_SECRET)
    monkeypatch.setattr(collector_module, "ROBOFLOW_SERVICE_SECRET", SERVICE_SECRET)
    monkeypatch.setattr(http_api, "InferenceInstrumentator", _DummyInstrumentator)
    monkeypatch.setattr(
        http_api.usage_collector,
        "async_push_usage_payloads",
        AsyncMock(),
    )
    model_manager = MagicMock()
    model_manager.pingback = None
    model_manager.num_errors = 0
    interface = http_api.HttpInterface(model_manager=model_manager)
    return TestClient(interface.app)


def _specification(api_key: str) -> dict:
    return {
        "version": "1.0",
        "inputs": [{"type": "WorkflowParameter", "name": "value"}],
        "dynamic_blocks_definitions": [
            {
                "type": "DynamicBlockDefinition",
                "manifest": {
                    "type": "ManifestDescription",
                    "block_type": "FakeModelBlock",
                    "inputs": {
                        "value": {
                            "type": "DynamicInputDefinition",
                            "selector_types": ["input_parameter"],
                        },
                    },
                    "outputs": {
                        "result": {"type": "DynamicOutputDefinition", "kind": []}
                    },
                },
                "code": {
                    "type": "PythonCode",
                    "run_function_code": FAKE_MODEL_BLOCK_CODE.replace(
                        "__API_KEY__", api_key
                    ),
                },
            },
        ],
        "steps": [
            {"type": "FakeModelBlock", "name": "fake_model", "value": "$inputs.value"},
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "result",
                "selector": "$steps.fake_model.result",
            },
        ],
    }


def _billable_by_category(api_key: str) -> dict:
    """The `billable` flag of every usage row recorded for `api_key`, by category."""
    from inference.usage_tracking.collector import usage_collector

    return {
        key.split(":", 1)[0]: json.loads(row["resource_details"])["billable"]
        for key, row in usage_collector._usage.get(api_key, {}).items()
    }


def test_authenticated_opt_out_reaches_workflow_and_model_rows(monkeypatch):
    # given
    client = _build_test_client(monkeypatch)
    api_key = "billing-contract-opt-out-key"

    # when
    response = client.post(
        f"/workflows/run?countinference=false&service_secret={SERVICE_SECRET}",
        json={
            "api_key": api_key,
            "specification": _specification(api_key),
            "inputs": {"value": 1},
        },
    )

    # then
    assert response.status_code == 200
    assert _billable_by_category(api_key) == {
        "request": False,
        "workflows": False,
        "model": False,
    }


def test_route_without_billing_parameters_stays_billable(monkeypatch):
    # given
    client = _build_test_client(monkeypatch)
    api_key = "billing-contract-default-key"

    # when
    response = client.post(
        "/workflows/run",
        json={
            "api_key": api_key,
            "specification": _specification(api_key),
            "inputs": {"value": 1},
        },
    )

    # then
    assert response.status_code == 200
    assert _billable_by_category(api_key) == {
        "request": True,
        "workflows": True,
        "model": True,
    }
