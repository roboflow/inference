"""The two `describe_workload` HTTP routes, through the real app and the real
compiler.

Only two things are mocked: the saved-definition fetch (`get_workflow_specification`,
an external service) and the registry call INSIDE the metadata adapter. The
workflow definitions below are compiled for real by
`describe_workflow_workload`, so these tests exercise the response assembly, the
auth contract and the `USE_INFERENCE_MODELS` gate end to end.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from roboflow_workflows.execution_engine.introspection.workload_entities import (
    WorkflowIntrospection,
)
from starlette.testclient import TestClient

import inference.core.env as inference_env
from inference.core.interfaces import workflows_workload_metadata

INLINE_ROUTE = "/workflows/describe_workload"
SAVED_ROUTE = "/my-workspace/workflows/my-workflow/describe_workload"

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
PLAIN_DICT_FIELDS = {"steps_by_dimensionality", "configuration_equals"}


class _DummyInstrumentator:
    def __init__(self, app, model_manager, endpoint="/metrics"):
        self.app = app
        self.model_manager = model_manager
        self.endpoint = endpoint

    def set_stream_manager_client(self, stream_manager_client) -> None:
        self.stream_manager_client = stream_manager_client


@pytest.fixture
def interface(monkeypatch):
    """The real `HttpInterface`, with no auth middleware - self-hosted default."""
    import inference.core.interfaces.http.http_api as http_api

    monkeypatch.setattr(http_api, "InferenceInstrumentator", _DummyInstrumentator)
    monkeypatch.setattr(
        http_api.usage_collector, "async_push_usage_payloads", AsyncMock()
    )
    monkeypatch.setattr(http_api, "GCP_SERVERLESS", False)
    monkeypatch.setattr(http_api, "LAMBDA", False)
    monkeypatch.setattr(http_api, "DEDICATED_DEPLOYMENT_WORKSPACE_URL", None)
    monkeypatch.setattr(http_api, "WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT", None)
    model_manager = MagicMock()
    model_manager.pingback = None
    model_manager.num_errors = 0
    return http_api.HttpInterface(model_manager=model_manager)


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
    assert body["type"] == "workflow_introspection"
    assert body["schema_version"] == "1"
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
    assert response.json()["type"] == "workflow_introspection"


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
    assert response.json()["type"] == "workflow_introspection"
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


def test_execution_engine_v2_definition_is_rejected(
    interface, enrichment_disabled
) -> None:
    # given
    definition = _single_model_definition()
    definition["version"] = "2.0"

    # when
    with TestClient(interface.app) as client:
        response = _post_inline(client, definition)

    # then
    assert response.status_code >= 400
    assert "Execution Engine v1" in json.dumps(response.json())


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
    assert all(
        reason.startswith("step_resources_unknown:")
        for reason in body["summary"]["models"]["unknown_reasons"]
    )
    registry_call.assert_not_called()

    # a fully-declared definition stays complete with the flag off
    with TestClient(interface.app) as client:
        simple = _post_inline(client, _single_model_definition()).json()
    assert simple["summary"]["models"]["complete"] is True
    assert simple["summary"]["models"]["items"][0]["metadata_status"] == "disabled"


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
            "type": "model_metadata",
            "model_type": "yolov8n",
            "model_variant": "coco",
            "task_type": "object-detection",
        }
    assert models["my-project/3"]["used_by_steps"] == ["$steps.detection"]
    assert models["my-other-project/1"]["used_by_steps"] == ["$steps.classification"]
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
    step = body["steps"][0]
    assert step["type"] == "step_metadata"
    assert step["operations"]["type"] == "discovery"
    assert step["resources"]["type"] == "discovery"
    assert step["restrictions"]["type"] == "discovery"
    assert body["summary"]["type"] == "workflow_summary"
    assert body["summary"]["models"]["items"][0]["type"] == "model_summary"
    assert body["summary"]["models"]["items"][0]["metadata"]["type"] == "model_metadata"


def test_response_round_trips_through_the_response_model(
    interface, enrichment_enabled, registry_call
) -> None:
    # when
    with TestClient(interface.app) as client:
        response = _post_inline(client, _crop_and_classify_definition())

    # then
    parsed = WorkflowIntrospection.model_validate_json(response.text)
    assert parsed.model_dump(mode="json") == response.json()


def test_response_never_leaks_the_api_key_or_its_scope_digest(
    interface, enrichment_enabled, registry_call
) -> None:
    # when
    with TestClient(interface.app) as client:
        response = _post_inline(client, _crop_and_classify_definition())

    # then
    digest = workflows_workload_metadata.credential_scope_digest(API_KEY)
    assert API_KEY not in response.text
    assert digest not in response.text
    assert workflows_workload_metadata.WORKLOAD_CACHE_PREFIX_ROOT not in response.text
    # ... but the lookup really did run under that scope
    assert registry_call.call_args.kwargs["cache_prefix"].endswith(digest)


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
    assert "/{workspace_name}/workflows/{workflow_id}/describe_workload" in paths
    # the response model really is exported into the OpenAPI document, generics
    # included - a schema that cannot be generated breaks /openapi.json for
    # every route, not just these two
    with TestClient(interface.app) as client:
        spec = client.get("/openapi.json").json()
    assert INLINE_ROUTE in spec["paths"]
    assert "WorkflowIntrospection" in spec["components"]["schemas"]
    assert {
        "Discovery_DependentResource_",
        "Discovery_ModelSummary_",
        "Discovery_RestrictionMetadata_",
        "Discovery_WorkOperation_",
    } <= set(spec["components"]["schemas"])


def test_routes_follow_the_workflow_endpoints_kill_switch(monkeypatch) -> None:
    # given
    import inference.core.interfaces.http.http_api as http_api

    monkeypatch.setattr(http_api, "InferenceInstrumentator", _DummyInstrumentator)
    monkeypatch.setattr(
        http_api.usage_collector, "async_push_usage_payloads", AsyncMock()
    )
    monkeypatch.setattr(http_api, "GCP_SERVERLESS", False)
    monkeypatch.setattr(http_api, "LAMBDA", False)
    monkeypatch.setattr(http_api, "DEDICATED_DEPLOYMENT_WORKSPACE_URL", None)
    monkeypatch.setattr(http_api, "WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT", None)
    monkeypatch.setattr(http_api, "DISABLE_WORKFLOW_ENDPOINTS", True)
    model_manager = MagicMock()
    model_manager.pingback = None
    model_manager.num_errors = 0

    # when
    disabled_interface = http_api.HttpInterface(model_manager=model_manager)

    # then - the new routes are gated exactly like describe_interface
    paths = {route.path for route in disabled_interface.app.routes}
    assert INLINE_ROUTE not in paths
    assert "/{workspace_name}/workflows/{workflow_id}/describe_workload" not in paths
    assert "/workflows/describe_interface" not in paths


# ---------------------------------------------------------------------------
# Codex round-001 R001-F001: the registry cache must be partitioned by the
# EFFECTIVE TRUSTED SCOPE of the lookup, not by the api key alone.
#
# The auth middleware resolves the caller from query > header > body; this route
# only ever materialises header/body. So two requests can carry the SAME body
# key and still be authorised as different workspaces, and the registry call is
# authorised against the workspace, via the
# `x-assume-identity-authorised-workspace` header that
# `_add_assume_identity_headers` reads off a per-request ContextVar. With
# `MODELS_CACHE_AUTH_ENABLED=False` the helper's cache is read for every caller,
# so an api-key-only partition would hand workspace A's metadata to workspace B
# without a lookup.
#
# The reproducer below is the reviewer's, adopted verbatim in substance: real
# middleware, real routing, real compiler, real adapter, real registry cache;
# only the external authentication and registry HTTP calls are stubbed.
# ---------------------------------------------------------------------------

ASSUME_IDENTITY_TOKEN = "dummy-assume-token"


def _workspace_scoped_definition() -> dict:
    return {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": [
            {
                "type": OBJECT_DETECTION_MODEL,
                "name": "det",
                "images": "$inputs.image",
                "model_id": "project/1",
            }
        ],
        "outputs": [],
    }


@pytest.fixture
def serverless_assume_identity(monkeypatch):
    """Serverless interface whose middleware fills the assume-identity context.

    Returns `(interface, fetched, store)`: the list of authorised-workspace
    scopes that actually reached the registry, and the backing cache dict.
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
    monkeypatch.setattr(roboflow_api, "MODELS_CACHE_AUTH_ENABLED", False)
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


def _model_type_of(response) -> str:
    return response.json()["summary"]["models"]["items"][0]["metadata"]["model_type"]


def test_metadata_cache_isolated_across_actual_authorized_workspace_contexts(
    serverless_assume_identity,
) -> None:
    interface, fetched, _ = serverless_assume_identity
    definition = _workspace_scoped_definition()

    # when - the SAME body credential, two different query credentials, so the
    # middleware authorises the two requests as two different workspaces
    with TestClient(interface.app) as client:
        responses = [
            client.post(
                INLINE_ROUTE,
                params={"api_key": query_key},
                json={"api_key": "query-b", "specification": definition},
            )
            for query_key in ("query-a", "query-b")
        ]

    # then
    for response in responses:
        assert response.status_code == 200, response.text
    assert fetched == ["workspace-db-a", "workspace-db-b"]
    assert [_model_type_of(response) for response in responses] == [
        "workspace-db-a",
        "workspace-db-b",
    ]


def test_same_authorized_workspace_still_hits_the_metadata_cache(
    serverless_assume_identity,
) -> None:
    """The partition must not degenerate into "never cache": two requests in the
    same trusted scope still share one registry lookup inside the 10 s expiry."""
    interface, fetched, store = serverless_assume_identity
    definition = _workspace_scoped_definition()

    # when - same query credential twice, so the same authorised workspace
    with TestClient(interface.app) as client:
        responses = [
            client.post(
                INLINE_ROUTE,
                params={"api_key": "query-a"},
                json={"api_key": "query-b", "specification": definition},
            )
            for _ in range(2)
        ]

    # then
    for response in responses:
        assert response.status_code == 200, response.text
    assert fetched == ["workspace-db-a"], "the second request must be a cache hit"
    assert [_model_type_of(response) for response in responses] == [
        "workspace-db-a",
        "workspace-db-a",
    ]
    assert len(store) == 1


def test_workspace_scope_never_reaches_the_response(
    serverless_assume_identity,
) -> None:
    """Neither the workspace id nor any digest of the trusted scope is public."""
    interface, _, store = serverless_assume_identity
    definition = _workspace_scoped_definition()

    with TestClient(interface.app) as client:
        response = client.post(
            INLINE_ROUTE,
            params={"api_key": "query-a"},
            json={"api_key": "query-b", "specification": definition},
        )

    assert response.status_code == 200
    digest = workflows_workload_metadata.credential_scope_digest(
        "query-b", "workspace-db-a"
    )
    assert digest not in response.text
    assert workflows_workload_metadata.credential_scope_digest("query-b") not in (
        response.text
    )
    assert "query-a" not in response.text
    assert "query-b" not in response.text
    # the cache key carries the scoped digest, the response carries none of it
    assert any(digest in key for key in store)
