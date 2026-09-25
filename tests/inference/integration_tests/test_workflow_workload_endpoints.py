"""Integration tests of `POST /workflows/describe_workload` on a running server.

Every request travels over HTTP to `server_url` - no in-process client, no
mocks. Definitions are sent inline; no image is sent, no workflow is executed
and no custom Python is evaluated. Only structural facts are asserted: model
metadata enrichment depends on the server configuration and the platform, so
it is checked for consistency, never required.
"""

import os
from typing import Dict, List, Set, Tuple

import pytest
import requests

from tests.inference.integration_tests.conftest import (
    api_key_auth_headers,
    without_api_key_in_header_mode,
)

# Structural introspection needs a key-shaped value, not a privileged one. A
# configured key is used when present (it only feeds the optional metadata
# lookup); otherwise a placeholder keeps the tests credential-free.
API_KEY = os.environ.get("API_KEY") or "workload-introspection-placeholder-key"

DESCRIBE_WORKLOAD_ROUTE = "/workflows/describe_workload"
MODEL_REGISTRY_ROUTE = "/model/registry"
REQUEST_TIMEOUT_SECONDS = 60

OBJECT_DETECTION_MODEL = "roboflow_core/roboflow_object_detection_model@v3"
DYNAMIC_CROP = "roboflow_core/dynamic_crop@v1"
IMAGE_BLUR = "roboflow_core/image_blur@v1"

# Public Roboflow detectors. Describing them needs no access to the models.
RFDETR_NANO = "rfdetr-nano"
YOLOV8N_640 = "yolov8n-640"

METADATA_STATUSES = {"available", "disabled", "unavailable"}

CUSTOM_BLOCK_TYPE = "IntrospectionInertBlock"
# Raised at module level, in `run()` and in `init()`: had the server evaluated
# any of that code, the request would fail instead of describing the step.
CUSTOM_PYTHON_SENTINEL = "workload introspection must never evaluate custom Python"

EdgeTriple = Tuple[str, str, str]


def _image_input() -> dict:
    return {"type": "WorkflowImage", "name": "image"}


def _detection_step(name: str, *, images: str, model_id: str) -> dict:
    return {
        "type": OBJECT_DETECTION_MODEL,
        "name": name,
        "images": images,
        "model_id": model_id,
    }


def _json_output(name: str, *, selector: str) -> dict:
    return {"type": "JsonField", "name": name, "selector": selector}


def _public_detectors_with_crop_definition() -> dict:
    """rfdetr-nano on the image -> crops (+1 dimension) -> rfdetr-nano and
    yolov8n-640 on the crops. rfdetr-nano is referenced twice, at two depths."""
    return {
        "version": "1.0",
        "inputs": [_image_input()],
        "steps": [
            _detection_step("detection", images="$inputs.image", model_id=RFDETR_NANO),
            {
                "type": DYNAMIC_CROP,
                "name": "crop",
                "images": "$inputs.image",
                "predictions": "$steps.detection.predictions",
            },
            _detection_step(
                "crop_rfdetr", images="$steps.crop.crops", model_id=RFDETR_NANO
            ),
            _detection_step(
                "crop_yolo", images="$steps.crop.crops", model_id=YOLOV8N_640
            ),
        ],
        "outputs": [
            _json_output("rfdetr_on_crops", selector="$steps.crop_rfdetr.predictions"),
            _json_output("yolo_on_crops", selector="$steps.crop_yolo.predictions"),
        ],
    }


def _custom_python_definition() -> dict:
    run_function_code = (
        f"raise RuntimeError({CUSTOM_PYTHON_SENTINEL!r})\n"
        "\n"
        "\n"
        "def run(self, predictions):\n"
        f"    raise RuntimeError({CUSTOM_PYTHON_SENTINEL!r})\n"
    )
    init_function_code = (
        "def init():\n" f"    raise RuntimeError({CUSTOM_PYTHON_SENTINEL!r})\n"
    )
    return {
        "version": "1.0",
        "inputs": [_image_input()],
        "dynamic_blocks_definitions": [
            {
                "type": "DynamicBlockDefinition",
                "manifest": {
                    "type": "ManifestDescription",
                    "block_type": CUSTOM_BLOCK_TYPE,
                    "inputs": {
                        "predictions": {
                            "type": "DynamicInputDefinition",
                            "selector_types": ["step_output"],
                            "selector_data_kind": {
                                "step_output": ["object_detection_prediction"]
                            },
                        },
                    },
                    "outputs": {
                        "output": {
                            "type": "DynamicOutputDefinition",
                            "kind": ["object_detection_prediction"],
                        },
                    },
                },
                "code": {
                    "type": "PythonCode",
                    "run_function_code": run_function_code,
                    "run_function_name": "run",
                    "init_function_code": init_function_code,
                    "init_function_name": "init",
                },
            },
        ],
        "steps": [
            _detection_step("detection", images="$inputs.image", model_id=YOLOV8N_640),
            {
                "type": CUSTOM_BLOCK_TYPE,
                "name": "custom",
                "predictions": "$steps.detection.predictions",
            },
        ],
        "outputs": [_json_output("custom_output", selector="$steps.custom.output")],
    }


def _describe_workload(
    server_url: str,
    definition: dict,
    *,
    auth_mode: str = "legacy",
) -> requests.Response:
    payload = {"specification": definition, "api_key": API_KEY}
    response = requests.post(
        f"{server_url}{DESCRIBE_WORKLOAD_ROUTE}",
        json=without_api_key_in_header_mode(auth_mode, payload),
        headers=api_key_auth_headers(auth_mode, API_KEY),
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    return response


def _loaded_model_ids(server_url: str) -> Set[str]:
    """Read-only snapshot of the server's model registry - nothing is cleared."""
    response = requests.get(
        f"{server_url}{MODEL_REGISTRY_ROUTE}", timeout=REQUEST_TIMEOUT_SECONDS
    )
    assert response.status_code == 200, (
        f"`GET {MODEL_REGISTRY_ROUTE}` must be available (default "
        "GET_MODEL_REGISTRY_ENABLED=True) to prove introspection loads no model, "
        f"got HTTP {response.status_code}"
    )
    loaded_model_ids = {model["model_id"] for model in response.json()["models"]}
    return loaded_model_ids


def _described_body(response: requests.Response) -> dict:
    assert response.status_code == 200, (
        f"`POST {DESCRIBE_WORKLOAD_ROUTE}` answered HTTP {response.status_code}: "
        f"{response.text[:2000]}"
    )
    key_leaked = API_KEY in response.text
    assert not key_leaked, "the response must never echo the API key"

    body = response.json()
    _assert_consistent_envelope(body=body)

    return body


def _assert_consistent_envelope(body: dict) -> None:
    """Versioned types and the cross-field facts a client relies on."""
    assert body["type"] == "workflow_introspection_v1"
    assert body["execution_engine_version"].startswith("1.")

    node_ids = [node["id"] for node in body["nodes"]]
    assert len(node_ids) == len(set(node_ids)), "graph node ids must be unique"
    assert all(node["type"] == "graph_node_v1" for node in body["nodes"])
    assert not any(node_id.startswith("<") for node_id in node_ids)
    for edge in body["edges"]:
        assert edge["type"] == "graph_edge_v1"
        assert edge["source"] in node_ids and edge["target"] in node_ids

    for step in body["steps"]:
        assert step["type"] == "step_metadata_v1"
        for declaration in ("resources", "restrictions", "operations"):
            _assert_honest_discovery(discovery=step[declaration])

    summary = body["summary"]
    assert summary["type"] == "workflow_summary_v1"
    assert sum(summary["steps_by_dimensionality"].values()) == len(body["steps"])
    _assert_honest_discovery(discovery=summary["models"])
    for model in summary["models"]["items"]:
        _assert_optional_metadata_consistent(model=model)


def _assert_honest_discovery(discovery: dict) -> None:
    assert discovery["type"] == "discovery_v1"
    assert discovery["complete"] is (len(discovery["unknown_reasons"]) == 0)
    assert all(
        reason["type"] == "discovery_problem_v1"
        for reason in discovery["unknown_reasons"]
    )


def _assert_optional_metadata_consistent(model: dict) -> None:
    # Enrichment depends on USE_INFERENCE_MODELS, OFFLINE_MODE, the key and the
    # platform: any status is valid, but it must agree with `metadata`.
    assert model["type"] == "model_summary_v1"
    assert model["metadata_status"] in METADATA_STATUSES
    if model["metadata_status"] == "available":
        assert model["metadata"]["type"] == "model_metadata_v1"
    else:
        assert model["metadata"] is None


def _node_pairs(body: dict) -> List[Tuple[str, str]]:
    return [(node["id"], node["kind"]) for node in body["nodes"]]


def _edge_triples(body: dict) -> List[EdgeTriple]:
    return [(edge["source"], edge["target"], edge["kind"]) for edge in body["edges"]]


def _steps_by_id(body: dict) -> Dict[str, dict]:
    return {step["node_id"]: step for step in body["steps"]}


def _dimensionalities(body: dict) -> Dict[str, Tuple[int, int]]:
    return {
        step["node_id"]: (step["input_dimensionality"], step["output_dimensionality"])
        for step in body["steps"]
    }


def _problems(discovery: dict) -> List[Tuple[str, dict]]:
    return [
        (reason["code"], reason["details"]) for reason in discovery["unknown_reasons"]
    ]


def _platform_model_ids(resources: dict) -> List[str]:
    return [
        resource["metadata"]["model_id"]
        for resource in resources["items"]
        if resource["resource_type"] == "roboflow_platform_model"
    ]


def test_describe_workload_of_public_detectors_with_crop_dimensionality(
    server_url: str,
    auth_mode: str,
) -> None:
    # given
    loaded_before = _loaded_model_ids(server_url)

    # when
    response = _describe_workload(
        server_url,
        _public_detectors_with_crop_definition(),
        auth_mode=auth_mode,
    )

    # then - the canonical graph, nodes in definition order, edges sorted
    body = _described_body(response)
    assert _node_pairs(body) == [
        ("$inputs.image", "input"),
        ("$steps.detection", "step"),
        ("$steps.crop", "step"),
        ("$steps.crop_rfdetr", "step"),
        ("$steps.crop_yolo", "step"),
        ("$outputs.rfdetr_on_crops", "output"),
        ("$outputs.yolo_on_crops", "output"),
    ]
    assert _edge_triples(body) == sorted(
        [
            ("$inputs.image", "$steps.detection", "data"),
            ("$inputs.image", "$steps.crop", "data"),
            ("$steps.detection", "$steps.crop", "data"),
            ("$steps.crop", "$steps.crop_rfdetr", "data"),
            ("$steps.crop", "$steps.crop_yolo", "data"),
            ("$steps.crop_rfdetr", "$outputs.rfdetr_on_crops", "data"),
            ("$steps.crop_yolo", "$outputs.yolo_on_crops", "data"),
        ]
    )

    # then - the crop adds one dimension; everything after it runs per crop
    assert _dimensionalities(body) == {
        "$steps.detection": (1, 1),
        "$steps.crop": (1, 2),
        "$steps.crop_rfdetr": (2, 2),
        "$steps.crop_yolo": (2, 2),
    }
    assert body["summary"]["steps_by_dimensionality"] == {"1": 2, "2": 2}
    assert body["summary"]["max_dimensionality"] == 2

    # then - each detector declares its literal model id and model inference
    steps = _steps_by_id(body)
    for node_id, model_id in (
        ("$steps.detection", RFDETR_NANO),
        ("$steps.crop_rfdetr", RFDETR_NANO),
        ("$steps.crop_yolo", YOLOV8N_640),
    ):
        assert steps[node_id]["block_type"] == OBJECT_DETECTION_MODEL
        assert steps[node_id]["operations"]["items"] == ["model_inference"]
        assert steps[node_id]["resources"]["complete"] is True
        assert _platform_model_ids(steps[node_id]["resources"]) == [model_id]
    assert steps["$steps.crop"]["operations"]["items"] == ["image_crop"]

    # then - one inventory entry per model, the shared one at two depths
    models = body["summary"]["models"]
    assert models["complete"] is True
    assert [
        (
            model["provider"],
            model["model_id"],
            model["used_by_steps"],
            model["steps_by_dimensionality"],
        )
        for model in models["items"]
    ] == [
        (
            "roboflow",
            RFDETR_NANO,
            ["$steps.crop_rfdetr", "$steps.detection"],
            {"1": 1, "2": 1},
        ),
        ("roboflow", YOLOV8N_640, ["$steps.crop_yolo"], {"2": 1}),
    ]

    # then - describing the models did not load any of them
    assert _loaded_model_ids(server_url) == loaded_before


def test_describe_workload_does_not_evaluate_custom_python(server_url: str) -> None:
    # given
    loaded_before = _loaded_model_ids(server_url)

    # when
    response = _describe_workload(server_url, _custom_python_definition())

    # then - a success proves no module, run() or init() code was evaluated
    body = _described_body(response)
    assert CUSTOM_PYTHON_SENTINEL not in response.text
    assert _edge_triples(body) == [
        ("$inputs.image", "$steps.detection", "data"),
        ("$steps.custom", "$outputs.custom_output", "data"),
        ("$steps.detection", "$steps.custom", "data"),
    ]
    assert _dimensionalities(body) == {
        "$steps.detection": (1, 1),
        "$steps.custom": (1, 1),
    }

    # then - custom Python is described honestly: known kind, unknown internals
    custom = _steps_by_id(body)["$steps.custom"]
    assert custom["block_type"] == CUSTOM_BLOCK_TYPE
    assert custom["operations"]["items"] == ["custom_python"]
    assert _problems(custom["operations"]) == [
        (
            "custom_python_internals_unknown",
            {"node_id": "$steps.custom", "declaration": "operations"},
        )
    ]
    resources_unavailable = (
        "declaration_unavailable",
        {
            "node_id": "$steps.custom",
            "declaration": "resources",
            "block_type": CUSTOM_BLOCK_TYPE,
        },
    )
    assert custom["resources"]["items"] == []
    assert _problems(custom["resources"]) == [resources_unavailable]
    assert "custom_python_execution_disabled" in {
        restriction["code"] for restriction in custom["restrictions"]["items"]
    }

    # then - the inventory keeps the known model and says what it cannot know
    models = body["summary"]["models"]
    assert [model["model_id"] for model in models["items"]] == [YOLOV8N_640]
    assert models["items"][0]["used_by_steps"] == ["$steps.detection"]
    assert _problems(models) == [resources_unavailable]

    # then
    assert _loaded_model_ids(server_url) == loaded_before


def test_describe_workload_keeps_model_id_selector_unresolved(
    server_url: str,
) -> None:
    # given - the parameter default must NOT be taken as the model id
    definition = {
        "version": "1.0",
        "inputs": [
            _image_input(),
            {
                "type": "WorkflowParameter",
                "name": "model_id",
                "default_value": YOLOV8N_640,
            },
        ],
        "steps": [
            _detection_step(
                "detection", images="$inputs.image", model_id="$inputs.model_id"
            ),
        ],
        "outputs": [
            _json_output("predictions", selector="$steps.detection.predictions")
        ],
    }

    # when
    response = _describe_workload(server_url, definition)

    # then
    body = _described_body(response)
    assert _edge_triples(body) == [
        ("$inputs.image", "$steps.detection", "data"),
        ("$inputs.model_id", "$steps.detection", "data"),
        ("$steps.detection", "$outputs.predictions", "data"),
    ]
    unresolved = (
        "unresolved_selector",
        {
            "node_id": "$steps.detection",
            "declaration": "resources",
            "field": "model_id",
            "selector": "$inputs.model_id",
            "resource_type": "roboflow_platform_model",
        },
    )
    resources = _steps_by_id(body)["$steps.detection"]["resources"]
    assert _platform_model_ids(resources) == ["$inputs.model_id"]
    assert _problems(resources) == [unresolved]
    models = body["summary"]["models"]
    assert models["items"] == []
    assert _problems(models) == [unresolved]


def test_describe_workload_of_workflow_without_models(server_url: str) -> None:
    # given
    definition = {
        "version": "1.0",
        "inputs": [_image_input()],
        "steps": [{"type": IMAGE_BLUR, "name": "blur", "image": "$inputs.image"}],
        "outputs": [_json_output("blurred", selector="$steps.blur.image")],
    }

    # when
    response = _describe_workload(server_url, definition)

    # then - a complete, empty inventory is a known absence, not an unknown
    body = _described_body(response)
    assert _node_pairs(body) == [
        ("$inputs.image", "input"),
        ("$steps.blur", "step"),
        ("$outputs.blurred", "output"),
    ]
    assert _edge_triples(body) == [
        ("$inputs.image", "$steps.blur", "data"),
        ("$steps.blur", "$outputs.blurred", "data"),
    ]
    blur = _steps_by_id(body)["$steps.blur"]
    assert blur["block_type"] == IMAGE_BLUR
    assert (blur["input_dimensionality"], blur["output_dimensionality"]) == (1, 1)
    assert blur["operations"]["items"] == ["image_filtering"]
    assert blur["operations"]["complete"] is True
    assert blur["resources"]["items"] == []
    assert blur["resources"]["complete"] is True
    assert body["summary"]["models"] == {
        "type": "discovery_v1",
        "items": [],
        "complete": True,
        "unknown_reasons": [],
    }
    assert body["summary"]["steps_by_dimensionality"] == {"1": 1}
    assert body["summary"]["max_dimensionality"] == 1


@pytest.mark.parametrize(
    "steps, outputs, expected_error_type",
    [
        (
            [{"type": "not_installed/block@v1", "name": "unknown"}],
            [],
            "WorkflowSyntaxError",
        ),
        (
            [
                _detection_step(
                    "detection", images="$inputs.image", model_id=YOLOV8N_640
                )
            ],
            [_json_output("result", selector="$steps.missing.predictions")],
            "InvalidReferenceTargetError",
        ),
    ],
    ids=["unknown_block_type", "missing_reference_target"],
)
def test_describe_workload_rejects_malformed_definition(
    server_url: str,
    steps: List[dict],
    outputs: List[dict],
    expected_error_type: str,
) -> None:
    # given
    definition = {
        "version": "1.0",
        "inputs": [_image_input()],
        "steps": steps,
        "outputs": outputs,
    }

    # when
    response = _describe_workload(server_url, definition)

    # then
    assert response.status_code == 400, response.text[:2000]
    response_data = response.json()
    assert response_data["error_type"] == expected_error_type
    assert response_data["message"]


def test_describe_workload_requires_api_key(server_url: str) -> None:
    # when - no key in the body and no Authorization header
    response = requests.post(
        f"{server_url}{DESCRIBE_WORKLOAD_ROUTE}",
        json={"specification": _public_detectors_with_crop_definition()},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    # then
    assert response.status_code == 400, response.text[:2000]
    assert "API key is missing" in response.json()["message"]
