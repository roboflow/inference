"""Hosted smoke tests of the experimental `describe_workload` endpoints.

The endpoints inspect a workflow structurally - nothing is executed - so these
tests check only the facts that prove the deployment answers with real
workload introspection. Environment-dependent details (restrictions, model
metadata) are deliberately not asserted.
"""

from typing import List, Tuple

import pytest
import requests

from tests.inference.hosted_platform_tests.conftest import (
    ROBOFLOW_API_KEY,
    api_key_auth_headers,
    without_api_key_in_header_mode,
)

REQUEST_TIMEOUT_SECONDS = 60

# Evaluating this code fails at import time - the endpoint must not evaluate it.
CUSTOM_PYTHON_THAT_MUST_NOT_BE_EVALUATED = """
raise RuntimeError("custom Python must not be evaluated by describe_workload")


def run(self, predictions) -> BlockResult:
    return {"max_confidence": 1.0}
"""


def _inline_definition(model_id: str) -> dict:
    return {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "dynamic_blocks_definitions": [
            {
                "type": "DynamicBlockDefinition",
                "manifest": {
                    "type": "ManifestDescription",
                    "block_type": "MaxConfidence",
                    "inputs": {
                        "predictions": {
                            "type": "DynamicInputDefinition",
                            "selector_types": ["step_output"],
                        },
                    },
                    "outputs": {
                        "max_confidence": {
                            "type": "DynamicOutputDefinition",
                            "kind": ["float_zero_to_one"],
                        }
                    },
                },
                "code": {
                    "type": "PythonCode",
                    "run_function_code": CUSTOM_PYTHON_THAT_MUST_NOT_BE_EVALUATED,
                },
            }
        ],
        "steps": [
            {
                "type": "roboflow_core/roboflow_object_detection_model@v2",
                "name": "detection",
                "images": "$inputs.image",
                "model_id": model_id,
            },
            {
                "type": "MaxConfidence",
                "name": "max_confidence",
                "predictions": "$steps.detection.predictions",
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "max_confidence",
                "selector": "$steps.max_confidence.max_confidence",
            }
        ],
    }


def _steps_by_id(response_data: dict) -> dict:
    return {step["node_id"]: step for step in response_data["steps"]}


def _reason_codes(discovery: dict) -> List[str]:
    return [reason["code"] for reason in discovery["unknown_reasons"]]


@pytest.mark.flaky(retries=4, delay=1)
def test_describing_workload_of_saved_workflow(
    object_detection_service_url: str,
    interface_discovering_workflow: Tuple[str, str],
    auth_mode: str,
) -> None:
    # given
    workspace_name, workflow_id = interface_discovering_workflow

    # when
    response = requests.post(
        f"{object_detection_service_url}/{workspace_name}/workflows/{workflow_id}/describe_workload",
        json=without_api_key_in_header_mode(
            auth_mode,
            {"api_key": ROBOFLOW_API_KEY, "use_cache": False},
        ),
        headers=api_key_auth_headers(auth_mode, ROBOFLOW_API_KEY),
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    # then
    response.raise_for_status()
    response_data = response.json()
    assert response_data["type"] == "workflow_introspection_v1"
    node_ids = {node["id"] for node in response_data["nodes"]}
    assert {
        "$inputs.image",
        "$inputs.model_id",
        "$outputs.model_predictions",
        "$outputs.bounding_box_visualization",
    }.issubset(node_ids), "Expected saved workflow interface in the graph"
    assert len(response_data["steps"]) > 0, "Expected compiled steps to be described"
    assert any(
        "model_inference" in step["operations"]["items"]
        for step in response_data["steps"]
    ), "Expected the detection step to declare model inference"
    models = response_data["summary"]["models"]
    assert (
        models["complete"] is False
    ), "Model is chosen by `$inputs.model_id` at run time - inventory is incomplete"
    assert any(
        reason["code"] == "unresolved_selector"
        and reason["details"].get("selector") == "$inputs.model_id"
        for reason in models["unknown_reasons"]
    ), "Expected the runtime model selector to be reported as the unknown"


@pytest.mark.flaky(retries=4, delay=1)
def test_describing_workload_of_inline_workflow_with_custom_python(
    object_detection_service_url: str,
    rfdetr_od_model_id: str,
    auth_mode: str,
) -> None:
    # when
    response = requests.post(
        f"{object_detection_service_url}/workflows/describe_workload",
        json=without_api_key_in_header_mode(
            auth_mode,
            {
                "specification": _inline_definition(model_id=rfdetr_od_model_id),
                "api_key": ROBOFLOW_API_KEY,
            },
        ),
        headers=api_key_auth_headers(auth_mode, ROBOFLOW_API_KEY),
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    # then
    response.raise_for_status()
    response_data = response.json()
    assert response_data["type"] == "workflow_introspection_v1"
    steps = _steps_by_id(response_data)
    assert set(steps) == {"$steps.detection", "$steps.max_confidence"}
    assert steps["$steps.detection"]["operations"]["items"] == ["model_inference"]
    models = {
        (model["provider"], model["model_id"]): model
        for model in response_data["summary"]["models"]["items"]
    }
    assert set(models) == {("roboflow", rfdetr_od_model_id)}
    assert models[("roboflow", rfdetr_od_model_id)]["used_by_steps"] == [
        "$steps.detection"
    ]

    custom_python_operations = steps["$steps.max_confidence"]["operations"]
    assert steps["$steps.max_confidence"]["block_type"] == "MaxConfidence"
    assert custom_python_operations["items"] == ["custom_python"]
    assert custom_python_operations["complete"] is False
    assert _reason_codes(custom_python_operations) == [
        "custom_python_internals_unknown"
    ]
