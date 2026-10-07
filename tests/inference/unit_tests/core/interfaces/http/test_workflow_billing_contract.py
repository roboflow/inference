"""Contract: the workflow-run route honours authenticated `countinference=false`.

The route declares `countinference` / `service_secret` purely so FastAPI binds
them where the request usage decorator can read them - no handler code touches
them. This is the only coverage of that seam: the usage-tracking unit tests
call decorated fakes with the parameters already bound, so the parameters could
be deleted from the route without failing any of them. Here the real route is
driven over HTTP and the usage rows the collector recorded are inspected - the
workflow row, the model row, and the custom-Python `workflow_block` row
recorded during workflow execution must inherit the request's authenticated
opt-out.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.testclient import TestClient

SERVICE_SECRET = "workflow-billing-contract-secret"

# The real provider constructs the typed request and BaseInference records usage;
# only model computation is replaced, so no model artifacts are downloaded.
FAKE_MODEL_BLOCK_CODE = """
def run(self, value) -> BlockResult:
    from inference.core.interfaces.workflows_models_provider import ModelManagerModelsProvider
    from inference.core.models.base import BaseInference

    class FakeModel(BaseInference):
        api_key = "__API_KEY__"
        model_id = "fake-project/1"

        def preprocess(self, image, **kwargs):
            return image, {}

        def predict(self, image, **kwargs):
            return []

        def postprocess(self, predictions, metadata, **kwargs):
            return predictions

    class FakeModelManager:
        def infer_from_request_sync(self, model_id, request, **kwargs):
            assert request.source == "workflow-execution"
            return FakeModel().infer(**request.model_dump())

    ModelManagerModelsProvider(FakeModelManager()).run_object_detection(
        model_id="fake-project/1",
        images=[{"type": "numpy", "value": value}],
        api_key="__API_KEY__",
        confidence=0.5,
    )
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


def _rows_for_api_key(api_key: str) -> dict:
    """Usage rows recorded for `api_key`, keyed by the collector's usage key.

    The HTTP route decorator and the dynamic-block decorator may be bound to
    different `UsageCollector` instances when another test has reloaded the
    collector module. Read both, and accept either the raw key or its hash.
    """
    import inference.core.interfaces.http.http_api as http_api
    from inference.core.interfaces import workflows_execution_observer
    from inference.usage_tracking.collector import usage_collector

    rows = {}
    for collector in (
        usage_collector,
        http_api.usage_collector,
        workflows_execution_observer.usage_collector,
    ):
        hashed = collector._hashed_api_keys.get(api_key)
        for bucket_key in (api_key, hashed):
            if bucket_key:
                rows.update(collector._usage.get(bucket_key, {}))
    return rows


def _billable_by_category(api_key: str) -> dict:
    """The `billable` flag of every usage row recorded for `api_key`, by category."""
    return {
        key.split(":", 1)[0]: json.loads(row["resource_details"])["billable"]
        for key, row in _rows_for_api_key(api_key).items()
    }


def _preview_by_category(api_key: str) -> dict:
    """The `is_preview` flag of every usage row recorded for `api_key`, by category."""
    return {
        key.split(":", 1)[0]: json.loads(row["resource_details"]).get("is_preview")
        for key, row in _rows_for_api_key(api_key).items()
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
        "workflow_block": False,
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
        "workflow_block": True,
    }
    assert _preview_by_category(api_key)["workflow_block"] is False
    model_rows = [
        row for row in _rows_for_api_key(api_key).values() if row["category"] == "model"
    ]
    assert len(model_rows) == 1
    assert (
        json.loads(model_rows[0]["resource_details"])["source"] == "workflow-execution"
    )


def test_preview_flag_reaches_workflow_and_block_rows(monkeypatch):
    # given
    client = _build_test_client(monkeypatch)
    api_key = "preview-contract-key"

    # when
    response = client.post(
        "/workflows/run",
        json={
            "api_key": api_key,
            "specification": _specification(api_key),
            "inputs": {"value": 1},
            "is_preview": True,
        },
    )

    # then
    assert response.status_code == 200
    preview_by_category = _preview_by_category(api_key)
    assert preview_by_category["workflows"] is True
    assert preview_by_category["workflow_block"] is True
    assert _billable_by_category(api_key)["workflow_block"] is True


_FAILING_BLOCK_CODE = """
def run(self, value) -> BlockResult:
    raise RuntimeError("block exploded")
"""


def _failing_specification() -> dict:
    return {
        "version": "1.0",
        "inputs": [{"type": "WorkflowParameter", "name": "value"}],
        "dynamic_blocks_definitions": [
            {
                "type": "DynamicBlockDefinition",
                "manifest": {
                    "type": "ManifestDescription",
                    "block_type": "ExplodingBlock",
                    "inputs": {
                        "value": {
                            "type": "DynamicInputDefinition",
                            "selector_types": ["input_parameter"],
                        }
                    },
                    "outputs": {
                        "result": {"type": "DynamicOutputDefinition", "kind": []}
                    },
                },
                "code": {
                    "type": "PythonCode",
                    "run_function_code": _FAILING_BLOCK_CODE,
                },
            }
        ],
        "steps": [{"type": "ExplodingBlock", "name": "boom", "value": "$inputs.value"}],
        "outputs": [
            {"type": "JsonField", "name": "result", "selector": "$steps.boom.result"}
        ],
    }


def _row_for_category(api_key: str, category: str) -> dict:
    """The single usage row recorded for `api_key` in `category`."""
    rows = [
        row
        for key, row in _rows_for_api_key(api_key).items()
        if key.split(":", 1)[0] == category
    ]
    assert len(rows) == 1, rows
    return rows[0]


def test_workflow_row_carries_the_identity_the_engine_computed(monkeypatch):
    # given
    client = _build_test_client(monkeypatch)
    api_key = "workflow-row-identity-key"

    # when
    response = client.post(
        "/workflows/run",
        json={
            "api_key": api_key,
            "specification": _specification(api_key),
            "inputs": {"value": 1},
        },
    )

    # then - every field the removed decorator derived from its own arguments
    assert response.status_code == 200
    row = _row_for_category(api_key, "workflows")
    assert row["fps"] == 0
    assert row["api_key_hash"]
    assert row["resource_id"]
    details = json.loads(row["resource_details"])
    assert details["steps"] == ["FakeModelBlock:fake_model"]
    assert details["billable"] is True
    assert details["is_preview"] is False


def test_a_failing_run_still_bills_an_error_row(monkeypatch):
    """A run that raises inside a block is billed, and the row says why.

    Real route, real engine, real collector - nothing about the recording is
    mocked, because the error path is the one that used to live inside the
    decorator on `run_workflow`.
    """
    # given
    client = _build_test_client(monkeypatch)
    api_key = "workflow-error-row-key"

    # when
    response = client.post(
        "/workflows/run",
        json={
            "api_key": api_key,
            "specification": _failing_specification(),
            "inputs": {"value": 1},
        },
    )

    # then
    assert response.status_code != 200
    details = json.loads(_row_for_category(api_key, "workflows")["resource_details"])
    # The user's exception, not the engine's wrapper: `create_dynamic_block_code_error`
    # raises `DynamicBlockCodeError` carrying `inner_error`; it is a `WorkflowError`,
    # so `safe_execute_step` re-raises it unwrapped, and the collector prefers
    # `inner_error_type` over the exception's own class name.
    assert details["error_type"] == "RuntimeError"


def test_an_unbound_observer_records_no_workflow_row(monkeypatch):
    """The failure mode this whole phase has to make loud.

    An engine initialised without `workflows_core.execution_observer` runs the
    workflow and returns results - and bills nothing. Pinned so a root that
    loses its binding fails here as well as in the AST check.
    """
    # given
    from inference.core.workflows.execution_engine.core import ExecutionEngine

    api_key = "unbound-observer-key"

    # when
    engine = ExecutionEngine.init(
        workflow_definition=_specification(api_key),
        init_parameters={"workflows_core.api_key": api_key},
    )
    engine.run(runtime_parameters={"value": 1})

    # then
    categories = {key.split(":", 1)[0] for key in _rows_for_api_key(api_key)}
    assert "workflows" not in categories


@pytest.mark.parametrize(
    "path",
    [
        "/workflows/run",
        "/infer/workflows",
        "/test-workspace/workflows/test-workflow",
        "/infer/workflows/test-workspace/test-workflow",
    ],
)
@pytest.mark.parametrize("service_secret", [None, SERVICE_SECRET])
def test_source_tags_reach_workflow_model_and_python_usage(
    monkeypatch, path, service_secret
):
    """Keep origin rows separate without changing deployment or execution identity.

    Args:
        monkeypatch: Fixture replacing external workflow lookup and service settings.
        path (str): Inline or saved workflow route, including legacy aliases.
        service_secret (Optional[str]): Internal caller secret, or no secret.
    """
    import inference.core.interfaces.http.http_api as http_api
    from inference.core.interfaces import workflows_execution_observer
    from inference.usage_tracking import collector as collector_module

    monkeypatch.setattr(
        collector_module, "ROBOFLOW_INTERNAL_SERVICE_NAME", "async-serverless-gpu"
    )
    client = _build_test_client(monkeypatch)
    for collector in (
        http_api.usage_collector,
        collector_module.usage_collector,
        workflows_execution_observer.usage_collector,
    ):
        monkeypatch.setattr(collector, "_enqueue_usage_payload", MagicMock())
        monkeypatch.setattr(
            collector, "_usage", collector.empty_usage_dict(exec_session_id="test")
        )
    api_key = f"source-tags-{path.replace('/', '-')}-{bool(service_secret)}"
    specification = _specification(api_key)
    monkeypatch.setattr(
        http_api, "get_workflow_specification", lambda **_: specification
    )

    for source in ("app", "custom-integration"):
        params = {
            "source": source,
            "source_info": "workflow-evals",
            "countinference": "true",
        }
        if service_secret is not None:
            params["service_secret"] = service_secret

        response = client.post(
            path,
            params=params,
            json={
                "api_key": api_key,
                "specification": specification,
                "inputs": {"value": 1},
            },
        )
        assert response.status_code == 200, response.text

    rows = _rows_for_api_key(api_key)
    origins_by_category = {}
    for key, row in rows.items():
        category = key.split(":", 1)[0]
        details = json.loads(row["resource_details"])
        assert details["source_info"] == "workflow-evals", (category, details)
        assert details["billable"] is True
        assert row["roboflow_service_name"] == "async-serverless-gpu", (category, row)
        assert row["processed_frames"] == 1, (category, row)
        origins_by_category.setdefault(category, []).append(details["source"])

    assert origins_by_category.keys() == {
        "request",
        "workflows",
        "model",
        "workflow_block",
    }
    for category, origins in origins_by_category.items():
        assert sorted(origins) == ["app", "custom-integration"], (category, origins)
