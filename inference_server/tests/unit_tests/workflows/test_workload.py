"""The two `describe_workload` routes through the real app and the real
compiler. Only the saved-definition fetch and the registry metadata call are
mocked; the definitions are compiled for real by `describe_workflow_workload`.
"""

import copy
import importlib
import urllib.parse
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import requests_mock as rm
from roboflow_workflows.errors import NotSupportedExecutionEngineError
from roboflow_workflows.execution_engine.core import ExecutionEngine
from roboflow_workflows.execution_engine.introspection.workload_entities import (
    WorkflowIntrospection,
)

from inference_models.weights_providers import roboflow as roboflow_provider
from inference_server.workflows import workload
from tests.unit_tests.legacy.conftest import FakeGateway, route_paths

INLINE_ROUTE = "/workflows/describe_workload"
SAVED_ROUTE = "/my-workspace/workflows/my-workflow/describe_workload"
SAVED_ROUTE_TEMPLATE = "/{workspace_name}/workflows/{workflow_id}/describe_workload"

OBJECT_DETECTION_MODEL = "roboflow_core/roboflow_object_detection_model@v3"
CLASSIFICATION_MODEL = "roboflow_core/roboflow_classification_model@v2"
DYNAMIC_CROP = "roboflow_core/dynamic_crop@v1"

API_KEY = "my-secret-api-key"


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


@pytest.fixture(autouse=True)
def _clear_metadata_cache():
    workload.clear_model_metadata_cache()
    yield
    workload.clear_model_metadata_cache()


@pytest.fixture
def registry_call(monkeypatch) -> MagicMock:
    call = MagicMock(
        return_value=SimpleNamespace(
            model_architecture="yolov8n",
            model_variant="coco",
            task_type="object-detection",
        )
    )
    monkeypatch.setattr(workload, "get_one_page_of_model_metadata", call)
    return call


@pytest.fixture
def saved_fetch(monkeypatch) -> MagicMock:
    fetch = MagicMock(return_value=_single_model_definition())
    monkeypatch.setattr(
        "inference_server.workflows.host.get_workflow_specification", fetch
    )
    return fetch


@pytest.fixture
def client(legacy_client):
    return legacy_client(FakeGateway())


def _post_inline(client, definition, *, body_key=API_KEY, headers=None):
    payload = {"specification": definition}
    if body_key is not None:
        payload["api_key"] = body_key
    return client.post(INLINE_ROUTE, json=payload, headers=headers or {})


def _models_by_id(body: dict) -> dict:
    return {model["model_id"]: model for model in body["summary"]["models"]["items"]}


# --------------------------------------------------------------------------
# routing and auth contract
# --------------------------------------------------------------------------


def test_routes_are_registered_and_marked_experimental(client) -> None:
    import inference_server.app as app_mod

    paths = route_paths(app_mod.app)
    assert INLINE_ROUTE in paths
    assert SAVED_ROUTE_TEMPLATE in paths
    spec = client.get("/openapi.json").json()
    assert "WorkflowIntrospection" in spec["components"]["schemas"]
    for path in (INLINE_ROUTE, SAVED_ROUTE_TEMPLATE):
        operation = spec["paths"][path]["post"]
        assert operation["summary"].startswith("[EXPERIMENTAL] ")
        assert operation["description"].startswith("[EXPERIMENTAL] ")


@pytest.mark.parametrize("path", [INLINE_ROUTE, SAVED_ROUTE])
def test_routes_are_not_shadowed_by_the_legacy_catch_all(client, path) -> None:
    # without the routes, the two-segment inline path falls into
    # `/{dataset_id}/{version_id}` and answers 404 "Requested Roboflow resource
    # not found"; the dedicated route answers the missing-key 400 instead
    response = client.post(path, json={"specification": _single_model_definition()})

    assert response.status_code == 400, response.text
    assert "API key is missing" in response.json()["message"]


def test_inline_route_describes_workload_with_the_body_api_key(
    client, registry_call
) -> None:
    response = _post_inline(client, _single_model_definition())

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["type"] == "workflow_introspection_v1"
    assert "schema_version" not in body
    assert [step["node_id"] for step in body["steps"]] == ["$steps.detection"]
    WorkflowIntrospection.model_validate_json(response.text)


def test_inline_route_accepts_the_bearer_header_only(client, registry_call) -> None:
    response = _post_inline(
        client,
        _single_model_definition(),
        body_key=None,
        headers={"Authorization": f"Bearer {API_KEY}"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["type"] == "workflow_introspection_v1"
    assert registry_call.call_args.kwargs["api_key"] == API_KEY


def test_missing_key_fails_exactly_like_describe_interface(client) -> None:
    workload_response = client.post(
        INLINE_ROUTE, json={"specification": _single_model_definition()}
    )
    interface_response = client.post(
        "/workflows/describe_interface",
        json={"specification": _single_model_definition()},
    )

    assert workload_response.status_code == interface_response.status_code == 400
    assert workload_response.json() == interface_response.json()
    assert "Required Roboflow API key is missing" in workload_response.json()["message"]


def test_saved_route_forwards_cache_and_version_options(
    client, saved_fetch, registry_call
) -> None:
    response = client.post(
        SAVED_ROUTE,
        json={"api_key": API_KEY, "use_cache": False, "workflow_version_id": "v7"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["type"] == "workflow_introspection_v1"
    saved_fetch.assert_called_once_with(
        api_key=API_KEY,
        workspace_id="my-workspace",
        workflow_id="my-workflow",
        use_cache=False,
        workflow_version_id="v7",
    )


def test_saved_route_defaults_match_describe_interface(
    client, saved_fetch, registry_call
) -> None:
    response = client.post(
        SAVED_ROUTE, headers={"Authorization": f"Bearer {API_KEY}"}, json={}
    )

    assert response.status_code == 200, response.text
    assert saved_fetch.call_args.kwargs["use_cache"] is True
    assert saved_fetch.call_args.kwargs["workflow_version_id"] is None


def test_saved_route_requires_an_api_key(client, saved_fetch) -> None:
    response = client.post(SAVED_ROUTE, json={})

    assert response.status_code == 400
    assert "API key is missing" in response.json()["message"]
    saved_fetch.assert_not_called()


@pytest.mark.parametrize("requested", ["0.9", "2.0"])
def test_unsupported_execution_engine_is_rejected_like_execution(
    client, registry_call, requested
) -> None:
    definition = _single_model_definition()
    definition["version"] = requested
    with pytest.raises(NotSupportedExecutionEngineError) as init_error:
        ExecutionEngine.init(workflow_definition=copy.deepcopy(definition))

    response = _post_inline(client, definition)

    assert response.status_code == 400, response.text
    body = response.json()
    assert body["error_type"] == "NotSupportedExecutionEngineError"
    assert body["message"] == init_error.value.public_message
    registry_call.assert_not_called()


# --------------------------------------------------------------------------
# response shape and model metadata enrichment
# --------------------------------------------------------------------------


def test_branching_crop_workflow_reports_dimensions_and_enriched_models(
    client, registry_call
) -> None:
    response = _post_inline(client, _crop_and_classify_definition())

    assert response.status_code == 200, response.text
    body = response.json()
    dimensions = {
        step["node_id"]: (step["input_dimensionality"], step["output_dimensionality"])
        for step in body["steps"]
    }
    assert dimensions["$steps.detection"] == (1, 1)
    assert dimensions["$steps.crop"] == (1, 2)
    assert dimensions["$steps.classification"] == (2, 2)
    assert body["summary"]["steps_by_dimensionality"] == {"1": 2, "2": 1}
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
    # one lookup per unique model id, under the caller's key
    assert {call.kwargs["model_id"] for call in registry_call.call_args_list} == {
        "my-project/3",
        "my-other-project/1",
    }
    assert {call.kwargs["api_key"] for call in registry_call.call_args_list} == {
        API_KEY
    }
    assert API_KEY not in response.text


def test_metadata_is_cached_per_api_key(client, registry_call) -> None:
    _post_inline(client, _single_model_definition())
    _post_inline(client, _single_model_definition())
    assert registry_call.call_count == 1

    _post_inline(client, _single_model_definition(), body_key="another-key")
    assert registry_call.call_count == 2


def test_registry_failure_is_reported_unavailable_without_losing_the_inventory(
    client, registry_call
) -> None:
    registry_call.side_effect = RuntimeError("platform unreachable")

    response = _post_inline(client, _single_model_definition())

    assert response.status_code == 200, response.text
    body = response.json()
    model = _models_by_id(body)["my-project/3"]
    assert model["metadata_status"] == "unavailable"
    assert model["metadata"] is None
    assert body["summary"]["models"]["complete"] is True


def test_third_party_model_is_unavailable_without_a_registry_call(
    client, registry_call
) -> None:
    response = _post_inline(client, _third_party_model_definition())

    assert response.status_code == 200, response.text
    models = response.json()["summary"]["models"]["items"]
    assert len(models) == 1
    assert models[0]["provider"] == "google"
    assert models[0]["metadata_status"] == "unavailable"
    registry_call.assert_not_called()


def test_offline_mode_reports_unavailable_without_a_registry_call(
    monkeypatch, client, registry_call
) -> None:
    monkeypatch.setattr("inference_server.configuration.OFFLINE_MODE", True)

    response = _post_inline(client, _single_model_definition())

    model = _models_by_id(response.json())["my-project/3"]
    assert model["metadata_status"] == "unavailable"
    registry_call.assert_not_called()


_GATEWAY = "https://gateway.example.com"


def _registry_response() -> dict:
    return {
        "modelMetadata": {
            "type": "external-model-metadata-v1",
            "modelId": "my-project/3",
            "modelArchitecture": "yolov8n",
            "taskType": "object-detection",
            "modelPackages": [],
        }
    }


def test_registry_lookup_goes_through_secure_gateway_when_configured(
    monkeypatch,
) -> None:
    monkeypatch.setattr(roboflow_provider, "SECURE_GATEWAY", _GATEWAY)
    provider = workload.RegistryModelMetadataProvider(api_key=API_KEY)
    with rm.Mocker() as m:
        m.get(rm.ANY, json=_registry_response())
        lookup = provider.resolve_model_metadata("roboflow", "my-project/3")

    assert lookup.status == "available"
    requested = m.request_history[0].url
    assert requested.startswith(f"{_GATEWAY}/proxy?url=")
    proxied = urllib.parse.unquote(requested.split("?url=", 1)[1])
    assert proxied == (
        f"{roboflow_provider.ROBOFLOW_API_HOST}/models/v1/external/weights"
        "?modelId=my-project%2F3"
    )
    assert API_KEY not in requested


def test_registry_lookup_hits_the_api_directly_without_secure_gateway(
    monkeypatch,
) -> None:
    monkeypatch.setattr(roboflow_provider, "SECURE_GATEWAY", None)
    provider = workload.RegistryModelMetadataProvider(api_key=API_KEY)
    with rm.Mocker() as m:
        m.get(rm.ANY, json=_registry_response())
        provider.resolve_model_metadata("roboflow", "my-project/3")

    assert m.request_history[0].url == (
        f"{roboflow_provider.ROBOFLOW_API_HOST}/models/v1/external/weights"
        "?modelId=my-project%2F3"
    )


# --------------------------------------------------------------------------
# kill switch
# --------------------------------------------------------------------------


def test_dedicated_switch_removes_only_the_workload_routes(monkeypatch) -> None:
    import inference_server.app as app_mod
    from inference_server import configuration
    from inference_server.workflows import router as router_mod

    monkeypatch.setattr(configuration, "DISABLE_WORKFLOW_WORKLOAD_ENDPOINTS", True)
    try:
        importlib.reload(router_mod)
        module = importlib.reload(app_mod)
        paths = route_paths(module.app)
        assert INLINE_ROUTE not in paths
        assert SAVED_ROUTE_TEMPLATE not in paths
        assert "/workflows/describe_interface" in paths
        assert "/workflows/run" in paths
    finally:
        monkeypatch.undo()
        importlib.reload(router_mod)
        importlib.reload(app_mod)
