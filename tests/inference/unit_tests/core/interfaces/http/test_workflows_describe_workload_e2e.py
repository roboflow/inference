"""End-to-end evidence for `describe_workload`: real app, real compiler, real
block registry, real Pydantic parse - and one request over a real socket.

`test_workflows_describe_workload.py` (host-owned) pins the route contract.
This file pins the things that only become visible when the whole path runs
together:

* the branched reference example, with every count explained in
  `artifacts/validation/examples.md` and published as JSON artifacts;
* one request through a real uvicorn server over TCP, so "it works in the
  ASGI test client" is not the only transport evidence;
* the declaration census of the registry the SERVER loads (core + enterprise +
  the host plugin), in a fresh subprocess;
* the response being identical under every combination of the host flags that
  custom-Python execution depends on, while the executable engine keeps
  rejecting what it rejected before.

Only two things are ever mocked: the external registry lookup inside the
metadata adapter and (where a saved workflow is involved) the definition fetch.
No model is loaded, no block is initialised, no workflow is executed.
"""

import json
import socket
import subprocess
import sys
import threading
import time
from typing import Iterator, List
from unittest.mock import AsyncMock, MagicMock

import pytest
import requests
from roboflow_workflows.execution_engine.introspection.workload_entities import (
    WorkflowIntrospection,
)
from starlette.testclient import TestClient

import inference.core.env as inference_env
from inference.core.interfaces import workflows_workload_metadata

INLINE_ROUTE = "/workflows/describe_workload"

OBJECT_DETECTION_MODEL = "roboflow_core/roboflow_object_detection_model@v3"
CLASSIFICATION_MODEL = "roboflow_core/roboflow_classification_model@v2"
DYNAMIC_CROP = "roboflow_core/dynamic_crop@v1"

API_KEY = "my-secret-api-key"
SHARED_MODEL_ID = "my-project/3"
CLASSIFIER_MODEL_ID = "my-other-project/1"

REGISTRY_PAYLOAD = {
    "modelType": "yolov8n",
    "taskType": "object-detection",
    "modelVariant": "coco",
    "modelLatencyMs": 11.0,
}

# dicts in the response that are plain mappings, not entities
PLAIN_DICT_FIELDS = {"steps_by_dimensionality", "configuration_equals"}

ENTERPRISE_LOADER = "roboflow_workflows.enterprise_blocks.loader"
HOST_PLUGIN_LOADER = "inference.roboflow_workflows_plugin.loader"
DECLARATION_HOOKS = (
    "discover_work_operations",
    "discover_portable_restrictions",
    "discover_dependent_resources",
)
# The only registered manifest allowed to answer "unknown" for resources: a
# remote-dispatch child may pull anything, so `[]` would be untrue.
RESOURCES_INTENTIONALLY_UNKNOWN = {"roboflow_core/inner_workflow@v1"}

CENSUS_SENTINEL = "<<<CENSUS>>>"
CENSUS_CHILD = r"""
import json, sys

# server boot order: the server installs the Workflows configuration, so
# `inference.core.env` must be imported before any roboflow_workflows import
import inference.core.env  # noqa: F401
from roboflow_workflows.execution_engine.introspection.blocks_loader import (
    get_manifest_type_identifiers,
    load_workflow_blocks,
)

HOOKS = json.loads(sys.argv[1])
rows = []
for block in load_workflow_blocks():
    manifest = block.manifest_class
    identifiers = get_manifest_type_identifiers(
        block_schema=manifest.model_json_schema(),
        block_source=block.block_source,
        block_identifier=block.identifier,
    )
    rows.append(
        {
            "block_type": identifiers[0],
            "manifest_module": manifest.__module__,
            "declared": {hook: hook in vars(manifest) for hook in HOOKS},
        }
    )
print("<<<CENSUS>>>" + json.dumps(rows))
"""

INERTNESS_SENTINEL = "<<<INTROSPECTION>>>"
INERTNESS_CHILD = r"""
import json, sys

import inference.core.env as inference_env
from inference.core.interfaces import workflows_workload_metadata

# the registry lookup is the ONLY external call on this path; stub it so the
# child needs no network and every configuration gets the same metadata
workflows_workload_metadata.roboflow_api.get_model_metadata_from_inference_models_registry = (
    lambda **kwargs: json.loads(sys.argv[2])
)

from inference.core.interfaces.http.handlers.workflows import (
    handle_describe_workflow_workload,
)

definition = json.loads(sys.argv[1])
result = handle_describe_workflow_workload(definition=definition, api_key="child-key")

# the ordinary executable path must still refuse what it refused before
executable_error = None
try:
    from roboflow_workflows.execution_engine.v1.compiler.core import compile_workflow

    compile_workflow(
        workflow_definition=definition,
        init_parameters={"api_key": None, "model_manager": object()},
    )
except Exception as error:
    executable_error = type(error).__name__

print(
    "<<<INTROSPECTION>>>"
    + json.dumps(
        {
            "flags": {
                "allow_custom_python": (
                    inference_env.ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS
                ),
                "custom_python_mode": (
                    inference_env.WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE
                ),
                "tensor": inference_env.ENABLE_TENSOR_DATA_REPRESENTATION,
            },
            "introspection": result.model_dump(mode="json"),
            "executable_error": executable_error,
        }
    )
)
"""


class _DummyInstrumentator:
    def __init__(self, app, model_manager, endpoint="/metrics"):
        self.app = app
        self.model_manager = model_manager
        self.endpoint = endpoint

    def set_stream_manager_client(self, stream_manager_client) -> None:
        self.stream_manager_client = stream_manager_client


@pytest.fixture
def interface(monkeypatch):
    """The real `HttpInterface` - same construction as the host-owned suite."""
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


def branched_definition() -> dict:
    """image -> detection -> crop (+1 dimension) -> detection on the crops ->
    custom Python, plus a classification branch on the same crops.

    `detection` and `crop_detection` deliberately reference the SAME model id,
    so the inventory must hold one entry with two referring steps.
    """
    return {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": [
            {
                "type": OBJECT_DETECTION_MODEL,
                "name": "detection",
                "images": "$inputs.image",
                "model_id": SHARED_MODEL_ID,
            },
            {
                "type": DYNAMIC_CROP,
                "name": "crop",
                "images": "$inputs.image",
                "predictions": "$steps.detection.predictions",
            },
            {
                "type": OBJECT_DETECTION_MODEL,
                "name": "crop_detection",
                "images": "$steps.crop.crops",
                "model_id": SHARED_MODEL_ID,
            },
            {
                "type": "CountDetections",
                "name": "counter",
                "predictions": "$steps.crop_detection.predictions",
            },
            {
                "type": CLASSIFICATION_MODEL,
                "name": "classification",
                "images": "$steps.crop.crops",
                "model_id": CLASSIFIER_MODEL_ID,
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "crop_detections",
                "selector": "$steps.crop_detection.predictions",
            },
            {"type": "JsonField", "name": "counts", "selector": "$steps.counter.count"},
            {
                "type": "JsonField",
                "name": "classes",
                "selector": "$steps.classification.predictions",
            },
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


def _steps_by_id(body: dict) -> dict:
    return {step["node_id"]: step for step in body["steps"]}


def _models_by_id(body: dict) -> dict:
    return {model["model_id"]: model for model in body["summary"]["models"]["items"]}


def _entity_dicts(node, key=None) -> Iterator[dict]:
    if isinstance(node, dict):
        if key not in PLAIN_DICT_FIELDS:
            yield node
            for child_key, value in node.items():
                yield from _entity_dicts(value, child_key)
    elif isinstance(node, list):
        for value in node:
            yield from _entity_dicts(value, key)


def _run_child(program: str, arguments: List[str], environment: dict, sentinel: str):
    process = subprocess.run(
        [sys.executable, "-c", program, *arguments],
        env=environment,
        capture_output=True,
        text=True,
    )
    assert (
        process.returncode == 0
    ), f"child failed\n{process.stdout[-2000:]}\n{process.stderr[-4000:]}"
    lines = [line for line in process.stdout.splitlines() if line.startswith(sentinel)]
    assert len(lines) == 1, process.stdout[-2000:]
    return json.loads(lines[0][len(sentinel) :])


# --------------------------------------------------------------------------
# the branched reference example (the published JSON artifacts)
# --------------------------------------------------------------------------


def test_branched_example_without_metadata_enrichment(
    interface, enrichment_disabled, registry_call
) -> None:
    # when
    with TestClient(interface.app) as client:
        response = _post_inline(client, branched_definition())

    # then - the graph: 1 input + 5 steps + 3 outputs, 9 deduplicated edges
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["type"] == "workflow_introspection"
    assert body["schema_version"] == "1"
    assert len(body["nodes"]) == 9
    assert [node["kind"] for node in body["nodes"]].count("step") == 5
    assert {
        (edge["source"], edge["target"], edge["kind"]) for edge in body["edges"]
    } == {
        ("$inputs.image", "$steps.detection", "data"),
        ("$inputs.image", "$steps.crop", "data"),
        ("$steps.detection", "$steps.crop", "data"),
        ("$steps.crop", "$steps.crop_detection", "data"),
        ("$steps.crop", "$steps.classification", "data"),
        ("$steps.crop_detection", "$steps.counter", "data"),
        ("$steps.crop_detection", "$outputs.crop_detections", "data"),
        ("$steps.counter", "$outputs.counts", "data"),
        ("$steps.classification", "$outputs.classes", "data"),
    }
    assert len(body["edges"]) == 9
    assert all(not node["id"].startswith("<") for node in body["nodes"])

    # then - dimensionality and the histogram over compiled steps
    steps = _steps_by_id(body)
    assert {
        node_id: (step["input_dimensionality"], step["output_dimensionality"])
        for node_id, step in steps.items()
    } == {
        "$steps.detection": (1, 1),
        "$steps.crop": (1, 2),
        "$steps.crop_detection": (2, 2),
        "$steps.counter": (2, 2),
        "$steps.classification": (2, 2),
    }
    assert body["summary"]["steps_by_dimensionality"] == {"1": 2, "2": 3}
    assert sum(body["summary"]["steps_by_dimensionality"].values()) == len(
        body["steps"]
    )
    assert body["summary"]["max_dimensionality"] == 2

    # then - declarations exactly as the block owners declared them
    assert steps["$steps.detection"]["operations"]["items"] == ["model_inference"]
    assert steps["$steps.classification"]["operations"]["items"] == ["model_inference"]
    assert steps["$steps.crop"]["operations"]["items"] == ["image_crop"]
    for node_id in (
        "$steps.detection",
        "$steps.crop_detection",
        "$steps.classification",
    ):
        assert steps[node_id]["restrictions"] == {
            "type": "discovery",
            "items": [],
            "complete": True,
            "unknown_reasons": [],
        }
        assert steps[node_id]["resources"]["complete"] is True

    # then - custom Python is truthfully incomplete, and it is the ONLY unknown
    counter = steps["$steps.counter"]
    assert counter["block_type"] == "CountDetections"
    assert counter["operations"]["items"] == ["custom_python"]
    assert counter["operations"]["complete"] is False
    assert counter["operations"]["unknown_reasons"] == [
        "custom_python_internal_operations_unknown:$steps.counter"
    ]
    assert counter["restrictions"]["items"] == [
        {
            "type": "restriction",
            "code": "custom_python_execution_disabled",
            "severity": "hard",
            "when": {
                "type": "restriction_condition",
                "runtimes": None,
                "step_execution_modes": None,
                "input_modes": None,
                "configuration_equals": {
                    "ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS": False,
                    "WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE": "local",
                },
            },
        }
    ]
    assert body["summary"]["models"]["unknown_reasons"] == [
        "step_resources_unknown:$steps.counter"
    ]

    # then - one inventory entry per model id, both referring steps preserved
    models = _models_by_id(body)
    assert sorted(models) == [CLASSIFIER_MODEL_ID, SHARED_MODEL_ID]
    assert models[SHARED_MODEL_ID]["used_by_steps"] == [
        "$steps.crop_detection",
        "$steps.detection",
    ]
    assert models[CLASSIFIER_MODEL_ID]["used_by_steps"] == ["$steps.classification"]
    assert {model["provider"] for model in models.values()} == {"roboflow"}
    assert {model["metadata_status"] for model in models.values()} == {"disabled"}
    assert all(model["metadata"] is None for model in models.values())
    registry_call.assert_not_called()

    # then - discriminators everywhere and a clean client round-trip
    assert all("type" in entity for entity in _entity_dicts(body))
    parsed = WorkflowIntrospection.model_validate_json(response.text)
    assert parsed.model_dump(mode="json") == body


def test_branched_example_with_metadata_enrichment(
    interface, enrichment_enabled, registry_call
) -> None:
    # when
    with TestClient(interface.app) as client:
        response = _post_inline(client, branched_definition())

    # then - one lookup per unique model id, not per referring step
    assert response.status_code == 200, response.text
    body = response.json()
    assert registry_call.call_count == 2
    assert {call.kwargs["model_id"] for call in registry_call.call_args_list} == {
        SHARED_MODEL_ID,
        CLASSIFIER_MODEL_ID,
    }

    # then - metadata is mapped onto both entries, inventory unchanged
    models = _models_by_id(body)
    assert sorted(models) == [CLASSIFIER_MODEL_ID, SHARED_MODEL_ID]
    for model in models.values():
        assert model["metadata_status"] == "available"
        assert model["metadata"] == {
            "type": "model_metadata",
            "model_type": "yolov8n",
            "model_variant": "coco",
            "task_type": "object-detection",
        }
    assert models[SHARED_MODEL_ID]["used_by_steps"] == [
        "$steps.crop_detection",
        "$steps.detection",
    ]

    # then - enrichment changes nothing structural
    assert body["summary"]["steps_by_dimensionality"] == {"1": 2, "2": 3}
    assert body["summary"]["models"]["complete"] is False
    assert body["summary"]["models"]["unknown_reasons"] == [
        "step_resources_unknown:$steps.counter"
    ]

    # then - no credential and no cache scope reaches the client
    digest = workflows_workload_metadata.credential_scope_digest(API_KEY)
    assert API_KEY not in response.text
    assert digest not in response.text
    assert all("type" in entity for entity in _entity_dicts(body))
    assert (
        WorkflowIntrospection.model_validate_json(response.text).model_dump(mode="json")
        == body
    )


# --------------------------------------------------------------------------
# transport: one request over a real TCP socket
# --------------------------------------------------------------------------


def _wait_for_started(server, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.started:
            return
        time.sleep(0.05)
    raise AssertionError("uvicorn did not report `started` in time")


def test_describe_workload_over_a_real_socket(interface, enrichment_disabled) -> None:
    """A real uvicorn server on localhost - not the in-process ASGI client."""
    # given
    uvicorn = pytest.importorskip("uvicorn")
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    listener.close()
    server = uvicorn.Server(
        uvicorn.Config(
            interface.app, host="127.0.0.1", port=port, log_level="warning", workers=1
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)

    # when
    thread.start()
    try:
        _wait_for_started(server)
        response = requests.post(
            f"http://127.0.0.1:{port}{INLINE_ROUTE}",
            json={"api_key": API_KEY, "specification": branched_definition()},
            timeout=60,
        )
    finally:
        server.should_exit = True
        thread.join(timeout=60)

    # then
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["type"] == "workflow_introspection"
    assert len(body["steps"]) == 5
    assert body["summary"]["steps_by_dimensionality"] == {"1": 2, "2": 3}
    assert sorted(_models_by_id(body)) == [CLASSIFIER_MODEL_ID, SHARED_MODEL_ID]
    assert (
        WorkflowIntrospection.model_validate_json(response.text).model_dump(mode="json")
        == body
    )
    assert not thread.is_alive(), "the test server did not shut down"


# --------------------------------------------------------------------------
# the registry the SERVER loads: core + enterprise + host plugin
# --------------------------------------------------------------------------


def test_server_registry_declares_every_hook_on_every_block(monkeypatch) -> None:
    # given - the server's own plugin selection, in a fresh subprocess
    import os

    environment = dict(os.environ)
    environment["WORKFLOWS_PLUGINS"] = f"{ENTERPRISE_LOADER},{HOST_PLUGIN_LOADER}"
    environment["SAM3_3D_OBJECTS_ENABLED"] = "True"

    # when
    rows = _run_child(
        CENSUS_CHILD, [json.dumps(DECLARATION_HOOKS)], environment, CENSUS_SENTINEL
    )

    # then - counts derived from the registry, never hardcoded
    assert len(rows) > 200, f"registry looks truncated: {len(rows)} blocks"
    plugin_rows = [
        row
        for row in rows
        if row["manifest_module"].startswith("inference.roboflow_workflows_plugin")
    ]
    assert plugin_rows, "the host plugin contributed no block"
    for hook in DECLARATION_HOOKS:
        missing = sorted(row["block_type"] for row in rows if not row["declared"][hook])
        assert set(missing) <= RESOURCES_INTENTIONALLY_UNKNOWN, (hook, missing)
    # every host-plugin block declares all three: no plugin step can contribute
    # an unknown reason to a response
    assert all(all(row["declared"].values()) for row in plugin_rows)
    # the flag-gated block is registered when the server enables it
    assert "roboflow_core/segment_anything3_3d_objects@v1" in {
        row["block_type"] for row in rows
    }


# --------------------------------------------------------------------------
# portability: the answer must not depend on this server's configuration
# --------------------------------------------------------------------------


BASELINE_CONFIGURATION = {
    "ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS": "True",
    "WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE": "local",
    "ENABLE_TENSOR_DATA_REPRESENTATION": "False",
}


def _describe_in_a_child(overrides: dict) -> dict:
    import os

    environment = dict(os.environ)
    # USE_INFERENCE_MODELS must stay ON: the server ANDs it into
    # ENABLE_TENSOR_DATA_REPRESENTATION, so turning it off would silently
    # disable the tensor axis of this matrix. The registry call is stubbed
    # inside the child instead, so nothing reaches the network.
    environment["USE_INFERENCE_MODELS"] = "True"
    environment["OFFLINE_MODE"] = "False"
    environment.update(BASELINE_CONFIGURATION)
    environment.update(overrides)
    return _run_child(
        INERTNESS_CHILD,
        [json.dumps(branched_definition()), json.dumps(REGISTRY_PAYLOAD)],
        environment,
        INERTNESS_SENTINEL,
    )


@pytest.fixture(scope="module")
def baseline_description() -> dict:
    """The response under the default configuration - the comparison basis."""
    return _describe_in_a_child({})["introspection"]


@pytest.mark.parametrize("allow_custom_python", ["False", "True"])
@pytest.mark.parametrize("custom_python_mode", ["local", "modal"])
@pytest.mark.parametrize("tensor_mode", ["False", "True"])
def test_response_is_identical_under_every_custom_python_configuration(
    baseline_description: dict,
    allow_custom_python: str,
    custom_python_mode: str,
    tensor_mode: str,
) -> None:
    """The portable declarations describe the definition, not this deployment."""
    # when
    payload = _describe_in_a_child(
        {
            "ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS": allow_custom_python,
            "WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE": custom_python_mode,
            "ENABLE_TENSOR_DATA_REPRESENTATION": tensor_mode,
        }
    )

    # then - the child really ran under the configuration under test
    assert payload["flags"] == {
        "allow_custom_python": allow_custom_python == "True",
        "custom_python_mode": custom_python_mode,
        "tensor": tensor_mode == "True",
    }

    # then - the described workload does not depend on any of the three flags
    assert payload["introspection"] == baseline_description

    # then - the metadata really was enriched (the gate is on in every child)
    statuses = {
        model["metadata_status"]
        for model in payload["introspection"]["summary"]["models"]["items"]
    }
    assert statuses == {"available"}

    # then - the executable engine still refuses a disabled local custom-Python
    # step; describing it changed nothing about running it
    if allow_custom_python == "False" and custom_python_mode == "local":
        assert payload["executable_error"] == "WorkflowEnvironmentConfigurationError"
