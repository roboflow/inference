from typing import Tuple

import pytest
import requests

from tests.inference.hosted_platform_tests.conftest import ROBOFLOW_API_KEY


@pytest.mark.flaky(retries=4, delay=1)
def test_getting_block_descriptions_from_legacy_get_endpoint(
    object_detection_service_url: str,
) -> None:
    # when
    response = requests.get(f"{object_detection_service_url}/workflows/blocks/describe")

    # then
    response.raise_for_status()
    response_data = response.json()
    assert set(response_data.keys()) == {
        "blocks",
        "declared_kinds",
        "kinds_connections",
        "primitives_connections",
        "universal_query_language_description",
        "dynamic_block_definition_schema",
    }
    assert len(response_data["blocks"]) > 0, "Some blocs expected to be added"
    assert len(response_data["declared_kinds"]) > 0, "Some kinds must be declared"
    assert len(response_data["declared_kinds"]) >= len(
        response_data["kinds_connections"]
    ), "Kinds connections declared as inputs for blocks must be at most in number of all declared kinds"
    assert (
        len(response_data["primitives_connections"]) > 0
    ), "Expected some primitive parameters for steps to be declared"


@pytest.mark.flaky(retries=4, delay=1)
def test_getting_block_descriptions_from_new_post_endpoint(
    object_detection_service_url: str,
) -> None:
    # when
    response = requests.post(
        f"{object_detection_service_url}/workflows/blocks/describe"
    )

    # then
    response.raise_for_status()
    response_data = response.json()
    assert set(response_data.keys()) == {
        "blocks",
        "declared_kinds",
        "kinds_connections",
        "primitives_connections",
        "universal_query_language_description",
        "dynamic_block_definition_schema",
    }
    assert len(response_data["blocks"]) > 0, "Some blocs expected to be added"
    assert len(response_data["declared_kinds"]) > 0, "Some kinds must be declared"
    assert len(response_data["declared_kinds"]) >= len(
        response_data["kinds_connections"]
    ), "Kinds connections declared as inputs for blocks must be at most in number of all declared kinds"
    assert (
        len(response_data["primitives_connections"]) > 0
    ), "Expected some primitive parameters for steps to be declared"


@pytest.mark.flaky(retries=4, delay=1)
def test_getting_schemas_from_new_post_endpoint_when_matching_execution_engine_version_is_requested(
    object_detection_service_url: str,
) -> None:
    # when
    response = requests.post(
        f"{object_detection_service_url}/workflows/blocks/describe",
        json={"execution_engine_version": "1.0.0"},
    )

    # then
    response.raise_for_status()
    response_data = response.json()
    assert set(response_data.keys()) == {
        "blocks",
        "declared_kinds",
        "kinds_connections",
        "primitives_connections",
        "universal_query_language_description",
        "dynamic_block_definition_schema",
    }
    assert len(response_data["blocks"]) > 0, "Some blocs expected to be added"
    assert len(response_data["declared_kinds"]) > 0, "Some kinds must be declared"
    assert len(response_data["declared_kinds"]) >= len(
        response_data["kinds_connections"]
    ), "Kinds connections declared as inputs for blocks must be at most in number of all declared kinds"
    assert (
        len(response_data["primitives_connections"]) > 0
    ), "Expected some primitive parameters for steps to be declared"


@pytest.mark.flaky(retries=4, delay=1)
def test_getting_schemas_from_new_post_endpoint_when_not_matching_execution_engine_version_is_requested(
    object_detection_service_url: str,
) -> None:
    # when
    response = requests.post(
        f"{object_detection_service_url}/workflows/blocks/describe",
        json={"execution_engine_version": "0.1.0"},
    )

    # then
    response.raise_for_status()
    response_data = response.json()
    assert set(response_data.keys()) == {
        "blocks",
        "declared_kinds",
        "kinds_connections",
        "primitives_connections",
        "universal_query_language_description",
        "dynamic_block_definition_schema",
    }
    assert len(response_data["blocks"]) == 0, "Expected no blocks loaded"


@pytest.mark.flaky(retries=4, delay=1)
def test_get_versions_of_execution_engine(object_detection_service_url: str) -> None:
    # when
    response = requests.get(
        f"{object_detection_service_url}/workflows/execution_engine/versions"
    )

    # then
    response.raise_for_status()
    response_data = response.json()
    assert response_data["versions"] == ["1.5.0"]


FUNCTION = """
def my_function(self, prediction: sv.Detections, crops: Batch[WorkflowImageData]) -> BlockResult:
    detection_id2bbox = {
        detection_id.item(): i for i, detection_id in enumerate(prediction.data["detection_id"])
    }
    results = []
    for crop in crops:
        parent_id = crop.parent_metadata.parent_id
        results.append({"associated_detections": prediction[detection_id2bbox[parent_id]]})
    return results
    """
DYNAMIC_BLOCKS_DEFINITION = [
    {
        "type": "DynamicBlockDefinition",
        "manifest": {
            "type": "ManifestDescription",
            "block_type": "DetectionsToCropsAssociation",
            "inputs": {
                "prediction": {
                    "type": "DynamicInputDefinition",
                    "selector_types": ["step_output"],
                    "selector_data_kind": {
                        "step_output": [
                            "Batch[object_detection_prediction]",
                            "Batch[instance_segmentation_prediction]",
                            "Batch[keypoint_detection_prediction]",
                        ]
                    },
                },
                "crops": {
                    "type": "DynamicInputDefinition",
                    "selector_types": ["step_output_image"],
                    "is_dimensionality_reference": True,
                    "dimensionality_offset": 1,
                },
            },
            "outputs": {
                "associated_detections": {
                    "type": "DynamicOutputDefinition",
                    "kind": [
                        "Batch[object_detection_prediction]",
                        "Batch[instance_segmentation_prediction]",
                        "Batch[keypoint_detection_prediction]",
                    ],
                }
            },
        },
        "code": {
            "type": "PythonCode",
            "run_function_code": FUNCTION,
            "run_function_name": "my_function",
        },
    },
]


@pytest.mark.flaky(retries=4, delay=1)
def test_getting_block_descriptions_from_new_post_endpoint_with_dynamic_blocks(
    object_detection_service_url: str,
) -> None:
    # when
    response = requests.post(
        f"{object_detection_service_url}/workflows/blocks/describe",
        json={"dynamic_blocks_definitions": DYNAMIC_BLOCKS_DEFINITION},
    )

    # then
    assert response.status_code == 500
    response_data = response.json()
    assert (
        "Cannot use dynamic blocks with custom Python code" in response_data["message"]
    ), "Expected execution to be prevented"


@pytest.mark.flaky(retries=4, delay=1)
def test_getting_block_schema_from_get_endpoint(
    object_detection_service_url: str,
) -> None:
    # when
    response = requests.get(
        f"{object_detection_service_url}/workflows/definition/schema"
    )

    # then
    response.raise_for_status()
    response_data = response.json()
    assert set(response_data.keys()) == {"schema"}
    schema = response_data["schema"]
    assert "$defs" in schema, "Response expected to define valid types"
    assert "properties" in schema, "Response expected to define schema properties"
    assert (
        "required" in schema
    ), "Response expected to define required schema properties"
    assert "title" in schema, "Response expected to define unique schema title"
    assert "type" in schema, "Response expected to define schema type"


@pytest.mark.flaky(retries=4, delay=1)
def test_getting_dynamic_outputs(object_detection_service_url: str) -> None:
    # when
    response = requests.post(
        f"{object_detection_service_url}/workflows/blocks/dynamic_outputs",
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
    assert len(response_data) == 7
    outputs_names = [output["name"] for output in response_data]
    assert outputs_names == [
        "parent_id",
        "root_parent_id",
        "image",
        "structured_output",
        "raw_output",
        "field_1",
        "field_2",
    ], "Expected all outputs that LMM step has - including dynamic to be provided"


@pytest.mark.flaky(retries=4, delay=1)
def test_compilation_endpoint_when_compilation_succeeds(
    object_detection_service_url: str,
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
        f"{object_detection_service_url}/workflows/validate",
        json=valid_workflow_definition,
    )

    # then
    response.raise_for_status()
    response_data = response.json()
    assert response_data["status"] == "ok"


@pytest.mark.flaky(retries=4, delay=1)
def test_compilation_endpoint_when_compilation_fails_due_to_invalid_requested_execution_engine_version(
    object_detection_service_url: str,
) -> None:
    # given
    valid_workflow_definition = {
        "version": "0.1.0",
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
        f"{object_detection_service_url}/workflows/validate",
        json=valid_workflow_definition,
    )

    # then
    assert (
        response.status_code == 400
    ), "Expected BadRequest response on wrong version selection"


@pytest.mark.flaky(retries=4, delay=1)
def test_compilation_endpoint_when_compilation_fails(
    object_detection_service_url: str,
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
        f"{object_detection_service_url}/workflows/validate",
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


SIMPLE_WORKFLOW_DEFINITION = {
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
        {
            "type": "JsonField",
            "name": "result",
            "selector": "$steps.detection.predictions",
        }
    ],
}


@pytest.mark.flaky(retries=4, delay=1)
def test_simple_workflow_run_when_run_expected_to_succeed(
    object_detection_service_url: str, detection_model_id: str
) -> None:
    # when
    response = requests.post(
        f"{object_detection_service_url}/workflows/run",
        json={
            "specification": SIMPLE_WORKFLOW_DEFINITION,
            "api_key": ROBOFLOW_API_KEY,
            "inputs": {
                "image": [
                    {
                        "type": "url",
                        "value": "https://media.roboflow.com/fruit.png",
                    }
                ]
                * 2,
                "model_id": detection_model_id,
            },
        },
    )

    # then
    response.raise_for_status()
    response_data = response.json()
    assert isinstance(
        response_data["outputs"], list
    ), "Expected list of elements to be returned"
    assert (
        len(response_data["outputs"]) == 2
    ), "Two images submitted - two responses expected"


@pytest.mark.flaky(retries=4, delay=1)
def test_simple_workflow_run_when_parameter_is_missing(
    object_detection_service_url: str, detection_model_id: str
) -> None:
    # when
    response = requests.post(
        f"{object_detection_service_url}/workflows/run",
        json={
            "specification": SIMPLE_WORKFLOW_DEFINITION,
            "api_key": ROBOFLOW_API_KEY,
        },
    )

    # then
    assert response.status_code == 422


@pytest.mark.flaky(retries=4, delay=1)
def test_simple_workflow_run_when_api_key_is_invalid(
    object_detection_service_url: str, detection_model_id: str
) -> None:
    # when
    response = requests.post(
        f"{object_detection_service_url}/workflows/run",
        json={
            "specification": SIMPLE_WORKFLOW_DEFINITION,
            "inputs": {
                "image": [
                    {
                        "type": "url",
                        "value": "https://media.roboflow.com/fruit.png",
                    }
                ]
                * 2,
                "model_id": detection_model_id,
            },
            "api_key": "invalid",
        },
    )

    # then
    assert (
        response.status_code == 500
    ), "Auth error is expected to be manifested as runtime error for one of the step"


CLIP_WORKFLOW_DEFINITION = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "reference"},
    ],
    "steps": [
        {
            "type": "ClipComparison",
            "name": "comparison",
            "images": "$inputs.image",
            "texts": "$inputs.reference",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "result",
            "selector": "$steps.comparison.similarity",
        }
    ],
}


@pytest.mark.flaky(retries=4, delay=1)
def test_clip_workflow_run_when_run_expected_to_succeed(
    object_detection_service_url: str, detection_model_id: str
) -> None:
    # when
    response = requests.post(
        f"{object_detection_service_url}/workflows/run",
        json={
            "specification": CLIP_WORKFLOW_DEFINITION,
            "api_key": ROBOFLOW_API_KEY,
            "inputs": {
                "image": [
                    {
                        "type": "url",
                        "value": "https://media.roboflow.com/fruit.png",
                    }
                ],
                "reference": ["cat", "dog"],
            },
        },
    )

    # then
    response.raise_for_status()
    response_data = response.json()
    assert isinstance(
        response_data["outputs"], list
    ), "Expected list of elements to be returned"
    assert (
        len(response_data["outputs"]) == 1
    ), "One image submitted - one response expected"


OCR_WORKFLOW_DEFINITION = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
    ],
    "steps": [
        {
            "type": "OCRModel",
            "name": "ocr",
            "image": "$inputs.image",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "result",
            "selector": "$steps.ocr.result",
        }
    ],
}


@pytest.mark.flaky(retries=4, delay=1)
def test_ocr_workflow_run_when_run_expected_to_succeed(
    object_detection_service_url: str, detection_model_id: str
) -> None:
    # when
    response = requests.post(
        f"{object_detection_service_url}/workflows/run",
        json={
            "specification": OCR_WORKFLOW_DEFINITION,
            "api_key": ROBOFLOW_API_KEY,
            "inputs": {
                "image": [
                    {
                        "type": "url",
                        "value": "https://media.roboflow.com/fruit.png",
                    }
                ]
                * 2,
            },
        },
    )

    # then
    response.raise_for_status()
    response_data = response.json()
    assert isinstance(
        response_data["outputs"], list
    ), "Expected list of elements to be returned"
    assert (
        len(response_data["outputs"]) == 2
    ), "Two images submitted - two response expected"


YOLO_WORLD_WORKFLOW_DEFINITION = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "class_names"},
    ],
    "steps": [
        {
            "type": "YoloWorldModel",
            "name": "detection",
            "image": "$inputs.image",
            "class_names": "$inputs.class_names",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "result",
            "selector": "$steps.detection.predictions",
        }
    ],
}


@pytest.mark.flaky(retries=4, delay=1)
def test_yolo_world_workflow_run_when_run_expected_to_succeed(
    object_detection_service_url: str,
) -> None:
    # when
    response = requests.post(
        f"{object_detection_service_url}/workflows/run",
        json={
            "specification": YOLO_WORLD_WORKFLOW_DEFINITION,
            "api_key": ROBOFLOW_API_KEY,
            "inputs": {
                "image": [
                    {
                        "type": "url",
                        "value": "https://media.roboflow.com/fruit.png",
                    }
                ]
                * 2,
                "class_names": ["banana", "apple"],
            },
        },
    )

    # then
    response.raise_for_status()
    response_data = response.json()
    assert isinstance(
        response_data["outputs"], list
    ), "Expected list of elements to be returned"
    assert (
        len(response_data["outputs"]) == 2
    ), "Two images submitted - two response expected"


FUNCTION_TO_GET_MAXIMUM_CONFIDENCE_FROM_BATCH_OF_DETECTIONS = """
def run(self, predictions: Batch[sv.Detections]) -> BlockResult:
    result = []
    for prediction in predictions:
        result.append({"max_confidence": np.max(prediction.confidence).item()})
    return result
"""

WORKFLOW_WITH_PYTHON_BLOCK_RUNNING_ON_BATCH = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "model_id"},
    ],
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
                "accepts_batch_input": True,
            },
            "code": {
                "type": "PythonCode",
                "run_function_code": FUNCTION_TO_GET_MAXIMUM_CONFIDENCE_FROM_BATCH_OF_DETECTIONS,
            },
        },
    ],
    "steps": [
        {
            "type": "RoboflowObjectDetectionModel",
            "name": "model",
            "image": "$inputs.image",
            "model_id": "$inputs.model_id",
        },
        {
            "type": "MaxConfidence",
            "name": "confidence_aggregation",
            "predictions": "$steps.model.predictions",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "max_confidence",
            "selector": "$steps.confidence_aggregation.max_confidence",
        },
    ],
}


@pytest.mark.flaky(retries=4, delay=1)
def test_workflow_run_with_dynamic_blocks(
    object_detection_service_url: str, detection_model_id: str
) -> None:
    # when
    response = requests.post(
        f"{object_detection_service_url}/workflows/run",
        json={
            "specification": WORKFLOW_WITH_PYTHON_BLOCK_RUNNING_ON_BATCH,
            "api_key": ROBOFLOW_API_KEY,
            "inputs": {
                "image": [
                    {
                        "type": "url",
                        "value": "https://media.roboflow.com/fruit.png",
                    }
                ]
                * 2,
                "model_id": detection_model_id,
            },
        },
    )

    # then
    assert response.status_code == 500
    response_data = response.json()
    assert (
        "Cannot use dynamic blocks with custom Python code" in response_data["message"]
    ), "Expected execution to be prevented"


@pytest.mark.flaky(retries=4, delay=1)
def test_workflow_validate_with_dynamic_blocks(
    object_detection_service_url: str, detection_model_id: str
) -> None:
    # when
    response = requests.post(
        f"{object_detection_service_url}/workflows/validate",
        json=WORKFLOW_WITH_PYTHON_BLOCK_RUNNING_ON_BATCH,
    )

    # then
    assert response.status_code == 500
    response_data = response.json()
    assert (
        "Cannot use dynamic blocks with custom Python code" in response_data["message"]
    ), "Expected execution to be prevented"


@pytest.mark.flaky(retries=4, delay=1)
def test_getting_block_schema_using_get_endpoint(
    object_detection_service_url: str,
) -> None:
    # when
    response = requests.get(
        f"{object_detection_service_url}/workflows/definition/schema"
    )

    # then
    response.raise_for_status()
    response_data = response.json()
    assert "schema" in response_data, "Response expected to define schema"
    schema = response_data["schema"]
    assert "$defs" in schema, "Response expected to define valid types"
    assert "properties" in schema, "Response expected to define schema properties"
    assert (
        "required" in schema
    ), "Response expected to define required schema properties"
    assert "title" in schema, "Response expected to define unique schema title"
    assert "type" in schema, "Response expected to define schema type"


@pytest.mark.flaky(retries=4, delay=1)
def test_discovering_interface_of_saved_workflow(
    object_detection_service_url: str,
    interface_discovering_workflow: Tuple[str, str],
) -> None:
    # when
    workspace_name, workflow_id = interface_discovering_workflow
    result = requests.post(
        f"{object_detection_service_url}/{workspace_name}/workflows/{workflow_id}/describe_interface",
        json={"api_key": ROBOFLOW_API_KEY},
    )

    # then
    result.raise_for_status()
    response_data = result.json()
    assert response_data["inputs"] == {
        "image": ["image"],
        "model_id": ["roboflow_model_id"],
    }
    assert response_data["outputs"] == {
        "model_predictions": {
            "predictions": ["object_detection_prediction"],
            "inference_id": ["string"],
        },
        "bounding_box_visualization": ["image"],
    }
    assert response_data["typing_hints"] == {
        "image": "dict",
        "object_detection_prediction": "dict",
        "string": "str",
        "roboflow_model_id": "str",
    }
    assert set(response_data["kinds_schemas"].keys()) == {
        "image",
        "object_detection_prediction",
    }


@pytest.mark.flaky(retries=4, delay=1)
def test_discovering_interface_of_valid_workflow_from_payload(
    object_detection_service_url: str,
) -> None:
    # given
    valid_definition = {
        "version": "1.0.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            {"type": "WorkflowParameter", "name": "model_id"},
            {"type": "WorkflowParameter", "name": "confidence"},
        ],
        "steps": [
            {
                "type": "ObjectDetectionModel",
                "name": "general_detection_1",
                "image": "$inputs.image",
                "model_id": "$inputs.model_id",
                "class_filter": ["dog"],
            },
            {
                "type": "ObjectDetectionModel",
                "name": "general_detection_2",
                "image": "$inputs.image",
                "model_id": "$inputs.model_id",
                "confidence": "$inputs.confidence",
                "class_filter": ["dog"],
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "detections",
                "selector": "$steps.general_detection_1.predictions",
            },
            {
                "type": "JsonField",
                "name": "detections",
                "selector": "$steps.general_detection_2.predictions",
            },
        ],
    }

    # when
    response = requests.post(
        f"{object_detection_service_url}/workflows/describe_interface",
        json={
            "specification": valid_definition,
            "api_key": "some",
        },
    )

    # then
    response.raise_for_status()
    response_data = response.json()
    assert response_data["outputs"] == {"detections": ["object_detection_prediction"]}
    assert response_data["inputs"] == {
        "image": ["image"],
        "model_id": ["roboflow_model_id"],
        "confidence": ["float_zero_to_one"],
    }
    assert response_data["typing_hints"] == {
        "float_zero_to_one": "float",
        "image": "dict",
        "object_detection_prediction": "dict",
        "roboflow_model_id": "str",
    }
    assert set(response_data["kinds_schemas"].keys()) == {
        "object_detection_prediction",
        "image",
    }, "Expected image and object_detection_prediction kinds to deliver schema"


@pytest.mark.flaky(retries=4, delay=1)
def test_discovering_interface_of_invalid_workflow_from_payload(
    object_detection_service_url: str,
) -> None:
    # given
    valid_definition = {
        "version": "1.0.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            {"type": "WorkflowParameter", "name": "model_id"},
        ],
        "steps": [
            {
                "type": "ObjectDetectionModel",
                "name": "general_detection_1",
                "image": "$inputs.image",
                "model_id": "$inputs.model_id",
                "class_filter": ["dog"],
            },
            {
                "type": "ObjectDetectionModel",
                "name": "general_detection_2",
                "image": "$inputs.image",
                "model_id": "$inputs.model_id",
                "confidence": "$inputs.model_id",
                "class_filter": ["dog"],
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "detections",
                "selector": "$steps.general_detection_1.predictions",
            },
            {
                "type": "JsonField",
                "name": "detections",
                "selector": "$steps.general_detection_2.predictions",
            },
        ],
    }

    # when
    response = requests.post(
        f"{object_detection_service_url}/workflows/describe_interface",
        json={
            "specification": valid_definition,
            "api_key": "some",
        },
    )

    # then
    assert response.status_code == 400


WORKFLOW_WITH_ENV_SECRET_STORE = {
    "version": "1.4.0",
    "inputs": [],
    "steps": [
        {
            "type": "roboflow_core/environment_secrets_store@v1",
            "name": "vault",
            "variables_storing_secrets": ["SOME_TOKEN"],
        }
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "status",
            "selector": "$steps.vault.some_token",
        },
    ],
}


@pytest.mark.flaky(retries=4, delay=1)
def test_extracting_secrets_from_env_based_secret_store(
    object_detection_service_url: str,
) -> None:
    # when
    response = requests.post(
        f"{object_detection_service_url}/workflows/run",
        json={
            "specification": WORKFLOW_WITH_ENV_SECRET_STORE,
            "api_key": ROBOFLOW_API_KEY,
            "inputs": {},
        },
    )

    # then
    assert response.status_code == 500
    response_data = response.json()
    assert (
        "`roboflow_core/environment_secrets_store@v1` block cannot run in this environment"
        in response_data["message"]
    ), "Expected execution to be prevented"
