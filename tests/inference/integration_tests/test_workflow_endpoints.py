import os

import requests

API_KEY = os.environ.get("API_KEY")


def test_getting_blocks_descriptions_using_legacy_get_endpoint(server_url) -> None:
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
    assert "universal_query_language_description" in response_data, "Expected universal_query_language_description key to be present in response"
    assert "dynamic_block_definition_schema" in response_data, "Expected key `dynamic_block_definition_schema` to be present in response"


def test_getting_blocks_descriptions_using_new_post_endpoint_with_dynamic_steps(server_url) -> None:
    # given
    function_code = """
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
    dynamic_blocks_definitions = [
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
                "run_function_code": function_code,
                "run_function_name": "my_function",
            },
        },
    ]

    # when
    response = requests.post(
        f"{server_url}/workflows/blocks/describe",
        json={"dynamic_blocks_definitions": dynamic_blocks_definitions}
    )

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
    assert "universal_query_language_description" in response_data, "Expected universal_query_language_description key to be present in response"
    assert "dynamic_block_definition_schema" in response_data, "Expected key `dynamic_block_definition_schema` to be present in response"


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


def test_compilation_endpoint_when_compilation_succeeds_with_custom_block(
    server_url: str,
) -> None:
    # given
    init_function = """
def init_model() -> Dict[str, Any]:
    model = YOLOv8ObjectDetection(model_id="yolov8n-640")
    return {"model": model}
"""
    infer_function = """
def infer(self, image: WorkflowImageData) -> BlockResult:
    predictions = self._init_results["model"].infer(image.numpy_image)
    return {"predictions": sv.Detections.from_inference(predictions[0].model_dump(by_alias=True, exclude_none=True))}
"""
    valid_workflow_definition = {
        "version": "1.0",
          "inputs": [
            {"type": "WorkflowImage", "name": "image"},
        ],
        "dynamic_blocks_definitions": [
            {
                "type": "DynamicBlockDefinition",
                "manifest": {
                    "type": "ManifestDescription",
                    "block_type": "CustomModel",
                    "inputs": {
                        "image": {
                            "type": "DynamicInputDefinition",
                            "selector_types": ["input_image"],
                        },
                    },
                    "outputs": {
                        "predictions": {
                            "type": "DynamicOutputDefinition",
                            "kind": [
                                "Batch[object_detection_prediction]",
                            ],
                        }
                    },
                },
                "code": {
                    "type": "PythonCode",
                    "run_function_code": infer_function,
                    "run_function_name": "infer",
                    "init_function_code": init_function,
                    "init_function_name": "init_model",
                    "imports": [
                        "from inference.models.yolov8 import YOLOv8ObjectDetection",
                    ],
                },
            },
        ],
        "steps": [
            {
                "type": "CustomModel",
                "name": "model",
                "image": "$inputs.image",
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "predictions",
                "selector": "$steps.model.predictions",
            },
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
        clean_loaded_models_fixture
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
            {
                "type": "JsonField",
                "name": "result",
                "selector": "$steps.detection.predictions",
            }
        ],
    }

    # when
    response = requests.post(
        f"{server_url}/workflows/run",
        json={
            "specification": valid_workflow_definition,
            "api_key": API_KEY,
            "inputs": {
                "image": [
                    {
                        "type": "url",
                        "value": "https://media.roboflow.com/fruit.png",
                    }
                ]
                * 2,
                "model_id": "yolov8n-640",
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
    assert (
        len(response_data["outputs"][0]["result"]["predictions"]) == 6
    ), "Expected to see 6 predictions"
    assert (
        len(response_data["outputs"][1]["result"]["predictions"]) == 6
    ), "Expected to see 6 predictions"
