import os

import requests

API_KEY = os.environ.get("API_KEY")


def test_getting_blocks_descriptions(server_url) -> None:
    # when
    response = requests.get(f"{server_url}/workflows/blocks/describe")

    # then
    response.raise_for_status()
    response_data = response.json()
    assert "blocks" in response_data, "Response expected to define blocks"
    assert len(response_data["blocks"]) > 0, "Some blocs expected to be added"
    assert (
        "declared_kinds" in response_data
    ), "Declared kinds must be provided in output"
    assert len(response_data["declared_kinds"]) > 0, "Some kinds must be declared"
    assert (
        "kinds_connections" in response_data
    ), "Kinds connections expected to be declared"
    assert len(response_data["declared_kinds"]) >= len(
        response_data["kinds_connections"]
    ), "Kinds connections declared as inputs for blocks must be at most in number of all declared kinds"
    assert (
        "primitives_connections" in response_data
    ), "Primitives connections expected to be in response"
    assert (
        len(response_data["primitives_connections"]) > 0
    ), "Expected some primitive parameters for steps to be declared"


def test_getting_dynamic_outputs(server_url: str) -> None:
    # when
    response = requests.post(
        f"{server_url}/workflows/blocks/dynamic_outputs",
        json={
            "type": "LMM",
            "name": "step_1",
            "images": "$inputs.image",
            "lmm_type": "$inputs.lmm_type",
            "prompt": "This is my prompt",
            "json_output": {"field_1": "some", "field_2": "other"},
            "remote_api_key": "$inputs.open_ai_key",
        },
    )

    # then
    response.raise_for_status()
    response_data = response.json()
    assert len(response_data) == 6
    outputs_names = [output["name"] for output in response_data]
    assert outputs_names == [
        "parent_id",
        "image",
        "structured_output",
        "raw_output",
        "field_1",
        "field_2",
    ], "Expected all outputs that LMM step has - including dynamic to be provided"


def test_compilation_endpoint_when_compilation_succeeds(
    server_url: str,
) -> None:
    # given
    valid_workflow_definition = {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            {"type": "WorkflowParameter", "name": "model_id"},
            {"type": "WorkflowParameter", "name": "confidence", "default_value": 0.3},
        ],
        "steps": [
            {
                "type": "RoboflowObjectDetectionModel",
                "name": "detection",
                "image": "$inputs.image",
                "model_id": "$inputs.model_id",
                "confidence": "$inputs.confidence",
            }
        ],
        "outputs": [
            {"type": "JsonField", "name": "result", "selector": "$steps.detection.*"}
        ],
    }

    # when
    response = requests.post(
        f"{server_url}/workflows/validate",
        json=valid_workflow_definition,
    )

    # then
    response.raise_for_status()
    response_data = response.json()
    assert response_data["status"] == "ok"


def test_compilation_endpoint_when_compilation_fails(
    server_url: str,
) -> None:
    # given
    valid_workflow_definition = {
        "version": "1.0",
        "inputs": [],
        "steps": [
            {
                "type": "RoboflowObjectDetectionModel",
                "name": "detection",
                "image": "$inputs.image",
                "model_id": "$inputs.model_id",
                "confidence": "$inputs.confidence",
            }
        ],
        "outputs": [
            {"type": "JsonField", "name": "result", "selector": "$steps.detection.*"}
        ],
    }

    # when
    response = requests.post(
        f"{server_url}/workflows/validate",
        json=valid_workflow_definition,
    )

    # then
    assert response.status_code == 400, "Expected bad request to be raised"
    response_data = response.json()
    assert "message" in response_data, "Response must provide error message"
    assert (
        response_data["error_type"] == "InvalidReferenceTargetError"
    ), "Error type must be declared properly indicating invalid reference"
    assert (
        response_data["context"]
        == "workflow_compilation | execution_graph_construction"
    ), "Error context must be provided"


def test_workflow_run(
    server_url: str,
) -> None:
    # given
    valid_workflow_definition = {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            {"type": "WorkflowParameter", "name": "model_id"},
            {"type": "WorkflowParameter", "name": "confidence", "default_value": 0.3},
        ],
        "steps": [
            {
                "type": "RoboflowObjectDetectionModel",
                "name": "detection",
                "image": "$inputs.image",
                "model_id": "$inputs.model_id",
                "confidence": "$inputs.confidence",
            }
        ],
        "outputs": [
            {"type": "JsonField", "name": "result", "selector": "$steps.detection.predictions"}
        ],
    }

    # when
    response = requests.post(
        f"{server_url}/workflows/run",
        json={
            "specification": valid_workflow_definition,
            "api_key": API_KEY,
            "inputs": {
                "image": {
                    "type": "url",
                    "value": "https://media.roboflow.com/fruit.png",
                },
                "model_id": "yolov8n-640",
            },
        },
    )

    # then
    response.raise_for_status()
    response_data = response.json()
    assert (
        len(response_data["outputs"]["result"][0]["predictions"]) == 6
    ), "Expected to see 6 predictions"
