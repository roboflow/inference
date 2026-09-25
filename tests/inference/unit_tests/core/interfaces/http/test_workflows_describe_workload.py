"""The two `describe_workload` HTTP routes, through the real app and the real
compiler.

Only two things are mocked: the saved-definition fetch (`get_workflow_specification`,
an external service) and the registry call INSIDE the metadata adapter. The
workflow definitions below are compiled for real by
`describe_workflow_workload`, so these tests exercise the response assembly, the
auth contract and the `USE_INFERENCE_MODELS` gate end to end.
"""

import copy
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from roboflow_workflows.errors import NotSupportedExecutionEngineError
from roboflow_workflows.execution_engine.core import ExecutionEngine
from roboflow_workflows.execution_engine.introspection.workload_entities import (
    WorkflowIntrospection,
)
from roboflow_workflows.execution_engine.v1.core import EXECUTION_ENGINE_V1_VERSION
from starlette.testclient import TestClient

import inference.core.env as inference_env
from inference.core.interfaces import workflows_workload_metadata

INLINE_ROUTE = "/workflows/describe_workload"
SAVED_ROUTE = "/my-workspace/workflows/my-workflow/describe_workload"
SAVED_ROUTE_TEMPLATE = "/{workspace_name}/workflows/{workflow_id}/describe_workload"

OBJECT_DETECTION_MODEL = "roboflow_core/roboflow_object_detection_model@v3"
CLASSIFICATION_MODEL = "roboflow_core/roboflow_classification_model@v2"
DYNAMIC_CROP = "roboflow_core/dynamic_crop@v1"

API_KEY = "my-secret-api-key"

REGISTRY_PAYLOAD = {
    "modelType": "yolov8n",
    "taskType": "object-detection",
    "modelVariant": "coco",
    "modelLatencyMs": 11.0,
}

# `dicts` in the response that are NOT entities and therefore carry no `type`.
# plain JSON maps, not entities: they carry no `type` discriminator and their
# contents (a discovery problem's open `details` included) are arbitrary data
PLAIN_DICT_FIELDS = {"steps_by_dimensionality", "configuration_equals", "details"}


class _DummyInstrumentator:
    def __init__(self, app, model_manager, endpoint="/metrics"):
        self.app = app
        self.model_manager = model_manager
        self.endpoint = endpoint

    def set_stream_manager_client(self, stream_manager_client) -> None:
        self.stream_manager_client = stream_manager_client


def _build_interface(monkeypatch, **http_api_overrides):
    """Build the real `HttpInterface` with no auth middleware, applying
    `http_api_overrides` (module-level flags) before the routes are registered."""
    import inference.core.interfaces.http.http_api as http_api

    monkeypatch.setattr(http_api, "InferenceInstrumentator", _DummyInstrumentator)
    monkeypatch.setattr(
        http_api.usage_collector, "async_push_usage_payloads", AsyncMock()
    )
    monkeypatch.setattr(http_api, "GCP_SERVERLESS", False)
    monkeypatch.setattr(http_api, "LAMBDA", False)
    monkeypatch.setattr(http_api, "DEDICATED_DEPLOYMENT_WORKSPACE_URL", None)
    monkeypatch.setattr(http_api, "WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT", None)
    for name, value in http_api_overrides.items():
        monkeypatch.setattr(http_api, name, value)
    model_manager = MagicMock()
    model_manager.pingback = None
    model_manager.num_errors = 0

    built_interface = http_api.HttpInterface(model_manager=model_manager)

    return built_interface


@pytest.fixture
def interface(monkeypatch):
    """The real `HttpInterface`, with no auth middleware - self-hosted default."""
    return _build_interface(
        monkeypatch,
        DISABLE_WORKFLOW_ENDPOINTS=False,
        DISABLE_WORKFLOW_WORKLOAD_ENDPOINTS=False,
    )


@pytest.fixture(autouse=True)
def isolated_metadata_cache():
    """The adapter's metadata cache is process wide: no test may inherit another
    test's mocked payloads or credentials."""
    workflows_workload_metadata.clear_model_metadata_cache()
    yield
    workflows_workload_metadata.clear_model_metadata_cache()


@pytest.fixture
def registry_call(monkeypatch):
    call = MagicMock(return_value=dict(REGISTRY_PAYLOAD))
    monkeypatch.setattr(
        workflows_workload_metadata.roboflow_api,
        "get_model_metadata_from_inference_models_registry",
        call,
    )
    return call


@pytest.fixture
def enrichment_enabled(monkeypatch):
    monkeypatch.setattr(inference_env, "USE_INFERENCE_MODELS", True)
    monkeypatch.setattr(inference_env, "OFFLINE_MODE", False)


@pytest.fixture
def enrichment_disabled(monkeypatch):
    monkeypatch.setattr(inference_env, "USE_INFERENCE_MODELS", False)
    monkeypatch.setattr(inference_env, "OFFLINE_MODE", False)


def _single_model_definition() -> dict:
    return {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": [
            {
                "type": OBJECT_DETECTION_MODEL,
                "name": "detection",
                "images": "$inputs.image",
                "model_id": "my-project/3",
            }
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "predictions",
                "selector": "$steps.detection.predictions",
            }
        ],
    }


def _crop_and_classify_definition() -> dict:
    """image -> detection -> crop (dimension +1) -> classification on crops."""
    return {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": [
            {
                "type": OBJECT_DETECTION_MODEL,
                "name": "detection",
                "images": "$inputs.image",
                "model_id": "my-project/3",
            },
            {
                "type": DYNAMIC_CROP,
                "name": "crop",
                "images": "$inputs.image",
                "predictions": "$steps.detection.predictions",
            },
            {
                "type": CLASSIFICATION_MODEL,
                "name": "classification",
                "images": "$steps.crop.crops",
                "model_id": "my-other-project/1",
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "predictions",
                "selector": "$steps.classification.predictions",
            }
        ],
    }


def _third_party_model_definition() -> dict:
    return {
        "version": "1.4",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": [
            {
                "type": "roboflow_core/google_gemini@v2",
                "name": "gemini",
                "images": "$inputs.image",
                "task_type": "ocr",
                "api_key": "dummy-google-key",
                "model_version": "gemini-3.1-pro-preview",
            }
        ],
        "outputs": [
            {"type": "JsonField", "name": "output", "selector": "$steps.gemini.output"}
        ],
    }


def _custom_python_definition() -> dict:
    return {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": [
            {
                "type": OBJECT_DETECTION_MODEL,
                "name": "detection",
                "images": "$inputs.image",
                "model_id": "my-project/3",
            },
            {
                "type": "CountDetections",
                "name": "counter",
                "predictions": "$steps.detection.predictions",
            },
        ],
        "outputs": [
            {"type": "JsonField", "name": "count", "selector": "$steps.counter.count"}
        ],
        "dynamic_blocks_definitions": [
            {
                "type": "DynamicBlockDefinition",
                "manifest": {
                    "type": "ManifestDescription",
                    "block_type": "CountDetections",
                    "inputs": {
                        "predictions": {
                            "type": "DynamicInputDefinition",
                            "selector_types": ["step_output"],
                            "selector_data_kind": {
                                "step_output": ["object_detection_prediction"]
                            },
                        }
                    },
                    "outputs": {
                        "count": {
                            "type": "DynamicOutputDefinition",
                            "kind": ["integer"],
                        }
                    },
                },
                "code": {
                    "type": "PythonCode",
                    "run_function_code": (
                        "def run(self, predictions):\n"
                        "    raise RuntimeError('must never run during inspection')\n"
                    ),
                },
            }
        ],
    }


def _post_inline(client, definition, *, body_key=API_KEY, headers=None):
    payload = {"specification": definition}
    if body_key is not None:
        payload["api_key"] = body_key
    return client.post(INLINE_ROUTE, json=payload, headers=headers or {})


def _models_by_id(body: dict) -> dict:
    return {model["model_id"]: model for model in body["summary"]["models"]["items"]}


def _entity_dicts(node, key=None):
    """Yield every dict in the response that must carry a `type` discriminator."""
    if isinstance(node, dict):
        if key not in PLAIN_DICT_FIELDS:
            yield node
            for child_key, value in node.items():
                yield from _entity_dicts(value, child_key)
    elif isinstance(node, list):
        for value in node:
            yield from _entity_dicts(value, key)


# --------------------------------------------------------------------------
# auth contract - mirrors describe_interface exactly
# --------------------------------------------------------------------------


def test_inline_route_describes_workload_with_the_body_api_key(
    interface, enrichment_disabled, registry_call
) -> None:
    # when
    with TestClient(interface.app) as client:
        response = _post_inline(client, _single_model_definition())

    # then
    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "workflow_introspection_v1"
    assert "schema_version" not in body
    assert [step["node_id"] for step in body["steps"]] == ["$steps.detection"]


def test_inline_route_accepts_the_bearer_header_only(
    interface, enrichment_disabled
) -> None:
    # when
    with TestClient(interface.app) as client:
        response = _post_inline(
            client,
            _single_model_definition(),
            body_key=None,
            headers={"Authorization": f"Bearer {API_KEY}"},
        )

    # then
    assert response.status_code == 200
    assert response.json()["type"] == "workflow_introspection_v1"


def test_missing_key_fails_exactly_like_describe_interface(interface) -> None:
    # when
    with TestClient(interface.app) as client:
        workload_response = client.post(
            INLINE_ROUTE, json={"specification": _single_model_definition()}
        )
        interface_response = client.post(
            "/workflows/describe_interface",
            json={"specification": _single_model_definition()},
        )

    # then - same status and same error shape
    assert workload_response.status_code == interface_response.status_code == 400
    assert "API key is missing" in workload_response.json()["message"]
    assert (
        workload_response.json().keys() == interface_response.json().keys()
    ), "the error envelope of the new route must match the existing one"


def test_saved_route_forwards_cache_and_version_options(
    monkeypatch, interface, enrichment_disabled
) -> None:
    # given
    import inference.core.interfaces.http.http_api as http_api

    fetch = MagicMock(return_value=_single_model_definition())
    monkeypatch.setattr(http_api, "get_workflow_specification", fetch)

    # when
    with TestClient(interface.app) as client:
        response = client.post(
            SAVED_ROUTE,
            json={
                "api_key": API_KEY,
                "use_cache": False,
                "workflow_version_id": "v7",
            },
        )

    # then
    assert response.status_code == 200
    assert response.json()["type"] == "workflow_introspection_v1"
    fetch.assert_called_once_with(
        api_key=API_KEY,
        workspace_id="my-workspace",
        workflow_id="my-workflow",
        use_cache=False,
        workflow_version_id="v7",
    )


def test_saved_route_defaults_match_describe_interface(
    monkeypatch, interface, enrichment_disabled
) -> None:
    # given
    import inference.core.interfaces.http.http_api as http_api

    fetch = MagicMock(return_value=_single_model_definition())
    monkeypatch.setattr(http_api, "get_workflow_specification", fetch)

    # when
    with TestClient(interface.app) as client:
        response = client.post(
            SAVED_ROUTE, headers={"Authorization": f"Bearer {API_KEY}"}, json={}
        )

    # then
    assert response.status_code == 200
    assert fetch.call_args.kwargs["use_cache"] is True
    assert fetch.call_args.kwargs["workflow_version_id"] is None


def test_saved_route_requires_an_api_key(monkeypatch, interface) -> None:
    # given
    import inference.core.interfaces.http.http_api as http_api

    fetch = MagicMock(return_value=_single_model_definition())
    monkeypatch.setattr(http_api, "get_workflow_specification", fetch)

    # when
    with TestClient(interface.app) as client:
        response = client.post(SAVED_ROUTE, json={})

    # then
    assert response.status_code == 400
    assert "API key is missing" in response.json()["message"]
    fetch.assert_not_called()


def _post_saved(monkeypatch, client, definition):
    import inference.core.interfaces.http.http_api as http_api

    fetch = MagicMock(return_value=definition)
    monkeypatch.setattr(http_api, "get_workflow_specification", fetch)
    return client.post(SAVED_ROUTE, json={"api_key": API_KEY})


def _post_to_route(route, monkeypatch, client, definition):
    if route == "inline":
        return _post_inline(client, definition)
    return _post_saved(monkeypatch, client, definition)


def _execution_engine_init_error(definition: dict) -> NotSupportedExecutionEngineError:
    # engine selection fails before any block or model is touched
    with pytest.raises(NotSupportedExecutionEngineError) as error:
        ExecutionEngine.init(workflow_definition=copy.deepcopy(definition))
    return error.value


@pytest.mark.parametrize("route", ["inline", "saved"])
@pytest.mark.parametrize("requested", ["0.9", "2.0"])
def test_unsupported_major_is_rejected_with_the_execution_error(
    monkeypatch, interface, enrichment_enabled, registry_call, route, requested
) -> None:
    # given
    definition = _single_model_definition()
    definition["version"] = requested
    init_error = _execution_engine_init_error(definition)

    # when
    with TestClient(interface.app) as client:
        response = _post_to_route(route, monkeypatch, client, definition)

    # then - same error class and message as workflow execution
    assert response.status_code == 400
    body = response.json()
    assert body["error_type"] == "NotSupportedExecutionEngineError"
    assert body["message"] == init_error.public_message
    registry_call.assert_not_called()


@pytest.mark.parametrize("route", ["inline", "saved"])
@pytest.mark.parametrize(
    "requested",
    [
        "1.0.0rc1",
        f"{EXECUTION_ENGINE_V1_VERSION.major}.{EXECUTION_ENGINE_V1_VERSION.minor}.0rc1",
    ],
)
def test_prerelease_minimum_accepted_by_execution_is_described(
    monkeypatch, interface, enrichment_disabled, route, requested
) -> None:
    # given
    definition = _single_model_definition()
    definition["version"] = requested

    # when
    with TestClient(interface.app) as client:
        response = _post_to_route(route, monkeypatch, client, definition)

    # then
    assert response.status_code == 200
    body = response.json()
    assert body["execution_engine_version"] == str(EXECUTION_ENGINE_V1_VERSION)
    assert [step["node_id"] for step in body["steps"]] == ["$steps.detection"]


def test_unmet_minimum_execution_engine_version_is_rejected(
    interface, enrichment_enabled, registry_call
) -> None:
    # given - one minor above the installed engine, so it survives releases
    definition = _single_model_definition()
    definition["version"] = (
        f"{EXECUTION_ENGINE_V1_VERSION.major}.{EXECUTION_ENGINE_V1_VERSION.minor + 1}.0"
    )

    # when
    with TestClient(interface.app) as client:
        response = _post_inline(client, definition)

    # then
    assert response.status_code == 400
    assert response.json()["error_type"] == "NotSupportedExecutionEngineError"
    registry_call.assert_not_called()


# --------------------------------------------------------------------------
# model metadata gate
# --------------------------------------------------------------------------


def test_flag_off_disables_every_model_metadata_and_makes_no_registry_call(
    interface, enrichment_disabled, registry_call
) -> None:
    # when
    with TestClient(interface.app) as client:
        response = _post_inline(client, _crop_and_classify_definition())

    # then
    body = response.json()
    models = body["summary"]["models"]["items"]
    assert len(models) == 2
    assert {model["metadata_status"] for model in models} == {"disabled"}
    assert all(model["metadata"] is None for model in models)
    # the inventory itself is NOT dropped, and the gate contributes no reason of
    # its own (the only reasons present come from step resource declarations)
    assert {model["model_id"] for model in models} == {
        "my-project/3",
        "my-other-project/1",
    }
    # the per-model histogram is compile-time data: populated with the gate off
    by_id = _models_by_id(body)
    assert by_id["my-project/3"]["steps_by_dimensionality"] == {"1": 1}
    assert by_id["my-other-project/1"]["steps_by_dimensionality"] == {"2": 1}
    assert all(
        reason["code"] == "declaration_unavailable"
        and reason["details"]["declaration"] == "resources"
        for reason in body["summary"]["models"]["unknown_reasons"]
    )
    registry_call.assert_not_called()

    # a fully-declared definition stays complete with the flag off
    with TestClient(interface.app) as client:
        simple = _post_inline(client, _single_model_definition()).json()
    assert simple["summary"]["models"]["complete"] is True
    assert simple["summary"]["models"]["items"][0]["metadata_status"] == "disabled"
    assert simple["summary"]["models"]["items"][0]["steps_by_dimensionality"] == {
        "1": 1
    }


def test_flag_on_enriches_roboflow_models_with_mapped_fields(
    interface, enrichment_enabled, registry_call
) -> None:
    # when
    with TestClient(interface.app) as client:
        response = _post_inline(client, _crop_and_classify_definition())

    # then
    body = response.json()
    models = _models_by_id(body)
    assert sorted(models) == ["my-other-project/1", "my-project/3"]
    for model in models.values():
        assert model["provider"] == "roboflow"
        assert model["metadata_status"] == "available"
        assert model["metadata"] == {
            "type": "model_metadata_v1",
            "model_type": "yolov8n",
            "model_variant": "coco",
            "task_type": "object-detection",
        }
    assert models["my-project/3"]["used_by_steps"] == ["$steps.detection"]
    assert models["my-other-project/1"]["used_by_steps"] == ["$steps.classification"]
    # enrichment does not touch the compile-time histogram
    assert models["my-project/3"]["steps_by_dimensionality"] == {"1": 1}
    assert models["my-other-project/1"]["steps_by_dimensionality"] == {"2": 1}
    # one lookup per unique model id, not per referring step
    assert registry_call.call_count == 2
    assert {call.kwargs["model_id"] for call in registry_call.call_args_list} == {
        "my-project/3",
        "my-other-project/1",
    }
    assert {call.kwargs["api_key"] for call in registry_call.call_args_list} == {
        API_KEY
    }


def test_partial_registry_payload_is_reported_available(
    interface, enrichment_enabled, registry_call
) -> None:
    # given
    registry_call.return_value = {
        "modelType": "yolov8n",
        "taskType": None,
        "modelVariant": None,
    }

    # when
    with TestClient(interface.app) as client:
        response = _post_inline(client, _single_model_definition())

    # then
    model = _models_by_id(response.json())["my-project/3"]
    assert model["metadata_status"] == "available"
    assert model["metadata"]["model_type"] == "yolov8n"
    assert model["metadata"]["task_type"] is None
    assert model["metadata"]["model_variant"] is None


def test_registry_failure_is_reported_unavailable_without_losing_the_inventory(
    interface, enrichment_enabled, registry_call
) -> None:
    # given
    registry_call.side_effect = RuntimeError("platform unreachable")

    # when
    with TestClient(interface.app) as client:
        response = _post_inline(client, _single_model_definition())

    # then
    body = response.json()
    model = _models_by_id(body)["my-project/3"]
    assert model["metadata_status"] == "unavailable"
    assert model["metadata"] is None
    # a failed lookup does not make a fully known inventory incomplete
    assert body["summary"]["models"]["complete"] is True
    assert body["summary"]["models"]["unknown_reasons"] == []


def test_third_party_model_is_unavailable_without_a_registry_call(
    interface, enrichment_enabled, registry_call
) -> None:
    # when
    with TestClient(interface.app) as client:
        response = _post_inline(client, _third_party_model_definition())

    # then
    assert response.status_code == 200
    models = response.json()["summary"]["models"]["items"]
    assert len(models) == 1
    assert models[0]["provider"] == "google"
    assert models[0]["model_id"] == "gemini-3.1-pro-preview"
    assert models[0]["metadata_status"] == "unavailable"
    registry_call.assert_not_called()


def test_offline_mode_reports_unavailable_without_a_registry_call(
    monkeypatch, interface, registry_call
) -> None:
    # given
    monkeypatch.setattr(inference_env, "USE_INFERENCE_MODELS", True)
    monkeypatch.setattr(inference_env, "OFFLINE_MODE", True)

    # when
    with TestClient(interface.app) as client:
        response = _post_inline(client, _single_model_definition())

    # then
    model = _models_by_id(response.json())["my-project/3"]
    assert model["metadata_status"] == "unavailable"
    registry_call.assert_not_called()


# --------------------------------------------------------------------------
# structural inertness and response shape
# --------------------------------------------------------------------------


def test_custom_python_block_is_described_with_local_execution_forbidden(
    monkeypatch, interface, enrichment_disabled
) -> None:
    """The introspection answer must not depend on whether THIS server is
    allowed to run custom Python: the definition is described structurally, the
    code is never compiled or executed."""
    # given
    from roboflow_workflows.execution_engine.v1.dynamic_blocks import (
        block_assembler,
        block_scaffolding,
    )

    monkeypatch.setattr(
        block_assembler, "ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS", False
    )
    monkeypatch.setattr(
        block_scaffolding, "ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS", False
    )

    # when
    with TestClient(interface.app) as client:
        response = _post_inline(client, _custom_python_definition())

    # then
    assert response.status_code == 200, response.text
    body = response.json()
    steps = {step["node_id"]: step for step in body["steps"]}
    assert sorted(steps) == ["$steps.counter", "$steps.detection"]
    custom_step = steps["$steps.counter"]
    assert "custom_python" in custom_step["operations"]["items"]
    assert custom_step["operations"]["complete"] is False
    assert custom_step["operations"]["unknown_reasons"]


@pytest.mark.parametrize("route", ["inline", "saved"])
def test_malformed_dynamic_block_definition_is_a_client_error(
    monkeypatch, interface, enrichment_enabled, registry_call, route
) -> None:
    # given - a valid custom Python block next to a malformed `{}` entry, on a
    # server that forbids local custom Python
    from roboflow_workflows.execution_engine.v1.dynamic_blocks import (
        block_assembler,
        block_scaffolding,
        modal_executor,
    )

    monkeypatch.setattr(
        block_assembler, "ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS", False
    )
    monkeypatch.setattr(
        block_scaffolding, "ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS", False
    )
    assembly = MagicMock(wraps=block_assembler.create_dynamic_block_specification)
    monkeypatch.setattr(block_assembler, "create_dynamic_block_specification", assembly)
    create_dynamic_module = MagicMock()
    monkeypatch.setattr(
        block_scaffolding, "create_dynamic_module", create_dynamic_module
    )
    validate_code_in_modal = MagicMock()
    monkeypatch.setattr(
        modal_executor, "validate_code_in_modal", validate_code_in_modal
    )
    definition = _custom_python_definition()
    definition["dynamic_blocks_definitions"].append({})

    # when
    with TestClient(interface.app) as client:
        response = _post_to_route(route, monkeypatch, client, definition)

    # then - 400 with the validation details, and nothing guarded was reached
    assert response.status_code == 400, response.text
    body = response.json()
    assert body["error_type"] == "DynamicBlockError"
    assert "index 1 is malformed" in body["message"]
    assert "manifest: Field required" in body["message"]
    assert body["inner_error_type"] == "ValidationError"
    assembly.assert_not_called()
    create_dynamic_module.assert_not_called()
    validate_code_in_modal.assert_not_called()
    registry_call.assert_not_called()


def test_branching_crop_workflow_reports_graph_dimensions_and_counts(
    interface, enrichment_disabled
) -> None:
    # when
    with TestClient(interface.app) as client:
        response = _post_inline(client, _crop_and_classify_definition())

    # then
    body = response.json()
    nodes = {node["id"]: node["kind"] for node in body["nodes"]}
    assert nodes == {
        "$inputs.image": "input",
        "$steps.detection": "step",
        "$steps.crop": "step",
        "$steps.classification": "step",
        "$outputs.predictions": "output",
    }
    edges = {(edge["source"], edge["target"], edge["kind"]) for edge in body["edges"]}
    assert ("$inputs.image", "$steps.detection", "data") in edges
    assert ("$steps.detection", "$steps.crop", "data") in edges
    assert ("$steps.crop", "$steps.classification", "data") in edges
    assert ("$steps.classification", "$outputs.predictions", "data") in edges
    # no synthetic compiler node leaks out
    assert all(not node["id"].startswith("<") for node in body["nodes"])

    dimensions = {
        step["node_id"]: (
            step["input_dimensionality"],
            step["output_dimensionality"],
        )
        for step in body["steps"]
    }
    assert dimensions["$steps.detection"] == (1, 1)
    assert dimensions["$steps.crop"] == (1, 2)
    assert dimensions["$steps.classification"] == (2, 2)

    histogram = body["summary"]["steps_by_dimensionality"]
    assert histogram == {"1": 2, "2": 1}
    assert sum(histogram.values()) == len(body["steps"])
    assert body["summary"]["max_dimensionality"] == 2

    # per-model: the detection model is referenced at depth 1 only, the
    # classification model at depth 2 only; each sums to its used_by_steps
    models = _models_by_id(body)
    assert models["my-project/3"]["steps_by_dimensionality"] == {"1": 1}
    assert models["my-other-project/1"]["steps_by_dimensionality"] == {"2": 1}
    for model in models.values():
        assert sum(model["steps_by_dimensionality"].values()) == len(
            model["used_by_steps"]
        )
        assert all(isinstance(key, str) for key in model["steps_by_dimensionality"])


def test_every_nested_object_carries_its_type_discriminator(
    interface, enrichment_enabled, registry_call
) -> None:
    # when
    with TestClient(interface.app) as client:
        response = _post_inline(client, _crop_and_classify_definition())

    # then
    body = response.json()
    missing = [entity for entity in _entity_dicts(body) if "type" not in entity]
    assert not missing, f"entities serialized without a discriminator: {missing}"
    # spot-check the nested ones the response model defaults
    assert body["type"] == "workflow_introspection_v1"
    assert "schema_version" not in body
    step = body["steps"][0]
    assert step["type"] == "step_metadata_v1"
    assert step["operations"]["type"] == "discovery_v1"
    assert step["resources"]["type"] == "discovery_v1"
    assert step["restrictions"]["type"] == "discovery_v1"
    assert body["summary"]["type"] == "workflow_summary_v1"
    assert body["summary"]["models"]["items"][0]["type"] == "model_summary_v1"
    assert (
        body["summary"]["models"]["items"][0]["metadata"]["type"] == "model_metadata_v1"
    )
    detection = next(
        step for step in body["steps"] if step["node_id"] == "$steps.detection"
    )
    resource = detection["resources"]["items"][0]
    assert resource["type"] == "dependent_resource_v1"
    assert resource["resource_type"] == "roboflow_platform_model"
    assert resource["metadata"]["type"] == "roboflow_platform_model_v1"


def test_response_round_trips_through_the_response_model(
    interface, enrichment_enabled, registry_call
) -> None:
    # when
    with TestClient(interface.app) as client:
        response = _post_inline(client, _crop_and_classify_definition())

    # then
    parsed = WorkflowIntrospection.model_validate_json(response.text)
    assert parsed.model_dump(mode="json") == response.json()


def test_response_never_leaks_the_api_key(
    interface, enrichment_enabled, registry_call
) -> None:
    # when
    with TestClient(interface.app) as client:
        response = _post_inline(client, _crop_and_classify_definition())

    # then
    assert API_KEY not in response.text
    # ... but the lookups really did run under that credential, and the helper
    # was called with its default cache prefix: nothing credential-derived is
    # passed to it, so nothing credential-derived can reach the shared cache
    assert registry_call.call_count == 2
    assert {call.kwargs["api_key"] for call in registry_call.call_args_list} == {
        API_KEY
    }
    assert {tuple(sorted(call.kwargs)) for call in registry_call.call_args_list} == {
        ("api_key", "model_id")
    }


def test_existing_describe_interface_route_is_unchanged(
    interface, enrichment_disabled
) -> None:
    # when
    with TestClient(interface.app) as client:
        response = client.post(
            "/workflows/describe_interface",
            json={"api_key": API_KEY, "specification": _single_model_definition()},
        )

    # then
    assert response.status_code == 200
    assert sorted(response.json()) == [
        "inputs",
        "kinds_schemas",
        "outputs",
        "typing_hints",
    ]


def test_routes_are_registered_and_documented(interface) -> None:
    # given
    paths = {route.path for route in interface.app.routes}

    # then
    assert INLINE_ROUTE in paths
    assert SAVED_ROUTE_TEMPLATE in paths
    # the response model really is exported into the OpenAPI document, generics
    # included - a schema that cannot be generated breaks /openapi.json for
    # every route, not just these two
    with TestClient(interface.app) as client:
        spec = client.get("/openapi.json").json()
    assert INLINE_ROUTE in spec["paths"]
    assert "WorkflowIntrospection" in spec["components"]["schemas"]
    # the public schema advertises the versioned top-level tag and no
    # standalone schema version
    introspection_schema = spec["components"]["schemas"]["WorkflowIntrospection"]
    introspection_properties = introspection_schema["properties"]
    assert introspection_properties["type"]["default"] == "workflow_introspection_v1"
    assert "schema_version" not in introspection_properties
    assert "execution_engine_version" in introspection_properties
    assert {
        "Discovery_DependentResource_",
        "Discovery_ModelSummary_",
        "Discovery_RestrictionMetadata_",
        "Discovery_WorkOperation_",
    } <= set(spec["components"]["schemas"])


def test_both_routes_are_marked_experimental_in_openapi(interface) -> None:
    # when
    with TestClient(interface.app) as client:
        spec = client.get("/openapi.json").json()

    # then
    for path in (INLINE_ROUTE, SAVED_ROUTE_TEMPLATE):
        operation = spec["paths"][path]["post"]
        assert operation["summary"].startswith("[EXPERIMENTAL] ")
        assert operation["description"].startswith("[EXPERIMENTAL] ")
        # the original description text is kept after the marker
        assert "Nothing is executed" in operation["description"]


def test_routes_follow_the_workflow_endpoints_kill_switch(monkeypatch) -> None:
    # when - the global switch wins even with the dedicated switch left off
    disabled_interface = _build_interface(
        monkeypatch,
        DISABLE_WORKFLOW_ENDPOINTS=True,
        DISABLE_WORKFLOW_WORKLOAD_ENDPOINTS=False,
    )

    # then - the new routes are gated exactly like describe_interface
    paths = {route.path for route in disabled_interface.app.routes}
    assert INLINE_ROUTE not in paths
    assert SAVED_ROUTE_TEMPLATE not in paths
    assert "/workflows/describe_interface" not in paths


def test_dedicated_switch_removes_only_the_workload_routes(
    monkeypatch, enrichment_disabled
) -> None:
    # given - the legacy POST `/{dataset_id}/{version_id}` route would otherwise
    # capture the two-segment inline path and run its handler once the
    # workload route is gone
    disabled_interface = _build_interface(
        monkeypatch,
        DISABLE_WORKFLOW_ENDPOINTS=False,
        DISABLE_WORKFLOW_WORKLOAD_ENDPOINTS=True,
        LEGACY_ROUTE_ENABLED=False,
    )

    # when
    with TestClient(disabled_interface.app) as client:
        spec = client.get("/openapi.json").json()
        inline_response = _post_inline(client, _single_model_definition())
        saved_response = client.post(SAVED_ROUTE, json={"api_key": API_KEY})
        interface_response = client.post(
            "/workflows/describe_interface",
            json={"api_key": API_KEY, "specification": _single_model_definition()},
        )

    # then - both workload routes are gone from routing and from OpenAPI
    paths = {route.path for route in disabled_interface.app.routes}
    assert INLINE_ROUTE not in paths
    assert SAVED_ROUTE_TEMPLATE not in paths
    assert INLINE_ROUTE not in spec["paths"]
    assert SAVED_ROUTE_TEMPLATE not in spec["paths"]
    # no POST route serves either path any more. The unmatched request falls
    # through to existing routes, including the StaticFiles app mounted at "/",
    # which answers 405 to a POST; 404 and 405 both mean "not served here".
    assert inline_response.status_code in {404, 405}
    assert saved_response.status_code in {404, 405}
    # every other Workflow route stays registered and working
    assert {
        "/workflows/describe_interface",
        "/{workspace_name}/workflows/{workflow_id}/describe_interface",
        "/workflows/run",
        "/{workspace_name}/workflows/{workflow_id}",
    } <= paths
    assert interface_response.status_code == 200


@pytest.mark.parametrize(
    "raw_value, expected_value",
    [(None, False), ("False", False), ("True", True)],
)
def test_dedicated_switch_is_parsed_from_the_environment(
    tmp_path, raw_value, expected_value
) -> None:
    # given - a fresh interpreter, so the module-level value is read anew; the
    # temporary working directory keeps any checkout `.env` file out of play
    repository_root = Path(__file__).resolve().parents[6]
    environment = os.environ.copy()
    environment.pop("DISABLE_WORKFLOW_WORKLOAD_ENDPOINTS", None)
    if raw_value is not None:
        environment["DISABLE_WORKFLOW_WORKLOAD_ENDPOINTS"] = raw_value
    existing_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        [
            str(repository_root),
            *([existing_python_path] if existing_python_path else []),
        ]
    )

    # when
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from inference.core.env import DISABLE_WORKFLOW_WORKLOAD_ENDPOINTS;"
            "print(repr(DISABLE_WORKFLOW_WORKLOAD_ENDPOINTS))",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    # then
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == repr(expected_value)


# ---------------------------------------------------------------------------
# the identity a lookup is served under, through the REAL middleware.
#
# The auth middleware resolves the caller from query > header > body; this route
# only ever materialises header/body. So two requests can carry the SAME body
# key and still be authorised as different workspaces, and the registry call
# carries that workspace in the `x-assume-identity-authorised-workspace` header
# `_add_assume_identity_headers` fills from a per-request ContextVar. The
# adapter's in-memory cache therefore keys on that workspace as well as on the
# api key, the model id and `MODELS_CACHE_AUTH_ENABLED` - every request identity
# input, whatever precedence the platform gives the header over the body key.
#
# Both authorization policies are exercised below - real middleware, real
# routing, real compiler, real adapter, real registry helper and its real shared
# cache; only the external authentication and registry HTTP calls are stubbed.
# ---------------------------------------------------------------------------

ASSUME_IDENTITY_TOKEN = "dummy-assume-token"
WORKSPACE_SCOPED_MODEL_ID = "project/1"
DEFAULT_REGISTRY_CACHE_PREFIX = "roboflow_api_data:inference_models_registry"
SHARED_CACHE_KEY = f"{DEFAULT_REGISTRY_CACHE_PREFIX}:{WORKSPACE_SCOPED_MODEL_ID}"


def _workspace_scoped_definition() -> dict:
    return {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": [
            {
                "type": OBJECT_DETECTION_MODEL,
                "name": "det",
                "images": "$inputs.image",
                "model_id": WORKSPACE_SCOPED_MODEL_ID,
            }
        ],
        "outputs": [],
    }


def _serverless_assume_identity(monkeypatch, models_cache_auth_enabled: bool):
    """Serverless interface whose middleware fills the assume-identity context.

    Returns `(interface, fetched, store)`: the list of authorised-workspace
    scopes that actually reached the registry, and the shared cache dict the
    registry helper reads and writes.
    """
    import inference.core.interfaces.http.http_api as http_api
    from inference.core import roboflow_api

    monkeypatch.setattr(http_api, "InferenceInstrumentator", _DummyInstrumentator)
    monkeypatch.setattr(
        http_api.usage_collector, "async_push_usage_payloads", AsyncMock()
    )
    for name, value in {
        "GCP_SERVERLESS": True,
        "LAMBDA": False,
        "OFFLINE_MODE": False,
        "DEDICATED_DEPLOYMENT_WORKSPACE_URL": None,
        "WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT": None,
        "ROBOFLOW_ASSUME_IDENTITY_SERVICE_ACCESS_TOKEN": ASSUME_IDENTITY_TOKEN,
    }.items():
        monkeypatch.setattr(http_api, name, value)
    monkeypatch.setattr(inference_env, "USE_INFERENCE_MODELS", True)
    monkeypatch.setattr(inference_env, "OFFLINE_MODE", False)

    async def authorize(api_key):
        suffix = "a" if api_key == "query-a" else "b"
        return roboflow_api.ServerlessUsageCheckResponse(
            status_code=200,
            workspace_id=f"workspace-{suffix}",
            workspace_db_id=f"workspace-db-{suffix}",
            under_cap=True,
        )

    monkeypatch.setattr(http_api, "get_serverless_usage_check_async", authorize)
    monkeypatch.setattr(
        roboflow_api, "MODELS_CACHE_AUTH_ENABLED", models_cache_auth_enabled
    )
    monkeypatch.setattr(roboflow_api, "GCP_SERVERLESS", True)
    monkeypatch.setattr(roboflow_api, "ENFORCE_CREDITS_VERIFICATION", False)
    monkeypatch.setattr(roboflow_api, "ROBOFLOW_INTERNAL_SERVICE_SECRET", None)
    monkeypatch.setattr(
        roboflow_api,
        "ROBOFLOW_ASSUME_IDENTITY_SERVICE_ACCESS_TOKEN",
        ASSUME_IDENTITY_TOKEN,
    )

    store = {}

    class _Cache:
        def get(self, key):
            return store.get(key)

        def set(self, key, value, expire=None):
            store[key] = value

    fetched = []

    def _fetch(url, headers=None, json_response=True):
        scope = headers[roboflow_api.ASSUME_IDENTITY_AUTHORISED_WORKSPACE_HEADER]
        fetched.append(scope)
        return {
            "modelMetadata": {
                "modelArchitecture": scope,
                "taskType": "object-detection",
            }
        }

    monkeypatch.setattr(roboflow_api, "cache", _Cache())
    monkeypatch.setattr(roboflow_api, "_get_from_url", _fetch)

    model_manager = MagicMock()
    model_manager.pingback = None
    model_manager.num_errors = 0
    return http_api.HttpInterface(model_manager=model_manager), fetched, store


@pytest.fixture
def serverless_auth_enforced(monkeypatch):
    """`MODELS_CACHE_AUTH_ENABLED=True` - authorization enforced per lookup."""
    return _serverless_assume_identity(monkeypatch, models_cache_auth_enabled=True)


@pytest.fixture
def serverless_auth_not_enforced(monkeypatch):
    """`MODELS_CACHE_AUTH_ENABLED=False` - the shared-cache default policy."""
    return _serverless_assume_identity(monkeypatch, models_cache_auth_enabled=False)


def _model_type_of(response) -> str:
    return response.json()["summary"]["models"]["items"][0]["metadata"]["model_type"]


def _post_as(client, query_key: str, body_key: str = "query-b"):
    return client.post(
        INLINE_ROUTE,
        params={"api_key": query_key},
        json={"api_key": body_key, "specification": _workspace_scoped_definition()},
    )


def test_enforced_auth_isolates_lookups_per_authorized_workspace(
    serverless_auth_enforced,
) -> None:
    """The same body credential, two different authorised workspaces.

    With enforcement on, the registry helper does not read its shared cache, and
    the adapter's memory key carries the authorised workspace - so each request
    reaches the platform with the identity headers its own middleware filled.
    """
    interface, fetched, _ = serverless_auth_enforced

    # when - same body key, two query keys, so two authorised workspaces
    with TestClient(interface.app) as client:
        responses = [
            _post_as(client, query_key) for query_key in ("query-a", "query-b")
        ]

    # then
    for response in responses:
        assert response.status_code == 200, response.text
    assert fetched == ["workspace-db-a", "workspace-db-b"]
    assert [_model_type_of(response) for response in responses] == [
        "workspace-db-a",
        "workspace-db-b",
    ]


def test_enforced_auth_reuses_the_memory_entry_within_one_workspace(
    serverless_auth_enforced,
) -> None:
    """The key must not degenerate into "never cache": two requests in the same
    identity share one registry lookup for the TTL."""
    interface, fetched, store = serverless_auth_enforced

    # when - the same query credential twice, so the same authorised workspace
    with TestClient(interface.app) as client:
        responses = [_post_as(client, "query-a") for _ in range(2)]

    # then
    for response in responses:
        assert response.status_code == 200, response.text
    assert fetched == ["workspace-db-a"], "the second request must be a memory hit"
    assert [_model_type_of(response) for response in responses] == [
        "workspace-db-a",
        "workspace-db-a",
    ]
    # the helper still writes its own shared entry, keyed by model id only
    assert sorted(store) == [SHARED_CACHE_KEY]


def test_unenforced_auth_keeps_the_helpers_shared_model_id_cache_policy(
    serverless_auth_not_enforced,
) -> None:
    """The accepted behaviour of `MODELS_CACHE_AUTH_ENABLED=False`.

    The registry helper READS its shared, model-id-keyed cache for every caller.
    A memory miss in another workspace is therefore answered from the entry the
    first workspace populated, with no second lookup. That is the helper's
    pre-existing policy for all of its callers and is unchanged by workload
    introspection; hosted per-workspace isolation relies on enabling
    `MODELS_CACHE_AUTH_ENABLED`.
    """
    interface, fetched, store = serverless_auth_not_enforced

    # when - two different authorised workspaces, same model id
    with TestClient(interface.app) as client:
        responses = [
            _post_as(client, query_key) for query_key in ("query-a", "query-b")
        ]

    # then - one platform call; the shared entry answered the second workspace
    for response in responses:
        assert response.status_code == 200, response.text
    assert fetched == ["workspace-db-a"]
    assert [_model_type_of(response) for response in responses] == [
        "workspace-db-a",
        "workspace-db-a",
    ]
    # and the shared key carries no credential material - just the model id
    assert sorted(store) == [SHARED_CACHE_KEY]


def test_credentials_never_reach_the_response(serverless_auth_enforced) -> None:
    """Neither api key that took part in the request is public."""
    # given
    interface, _, _ = serverless_auth_enforced

    # when
    with TestClient(interface.app) as client:
        response = _post_as(client, "query-a")

    # then
    assert response.status_code == 200
    assert "query-a" not in response.text
    assert "query-b" not in response.text
