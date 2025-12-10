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
    assert (
        "universal_query_language_description" in response_data
    ), "Expected universal_query_language_description key to be present in response"
    assert (
        "dynamic_block_definition_schema" in response_data
    ), "Expected key `dynamic_block_definition_schema` to be present in response"


def test_getting_blocks_descriptions_using_new_post_endpoint(server_url) -> None:
    # when
    response = requests.post(f"{server_url}/workflows/blocks/describe")

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
    assert (
        "universal_query_language_description" in response_data
    ), "Expected universal_query_language_description key to be present in response"
    assert (
        "dynamic_block_definition_schema" in response_data
    ), "Expected key `dynamic_block_definition_schema` to be present in response"
    # Assert all blocks have access_third_party property
    blocks = response_data["blocks"]
    blocks_missing_flag = [
        block["manifest_type_identifier"]
        for block in blocks
        if "access_third_party" not in block["block_schema"]
        or not isinstance(block["block_schema"]["access_third_party"], bool)
    ]
    assert (
        not blocks_missing_flag
    ), f"All blocks must declare boolean `access_third_party` flag. Missing in: {blocks_missing_flag}"


def test_getting_blocks_descriptions_using_new_post_endpoint_using_existing_execution_engine_version(server_url: str) -> None:
    # when
    response = requests.post(
        f"{server_url}/workflows/blocks/describe",
        json={"execution_engine_version": "1.0.0"}
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
    assert (
        "universal_query_language_description" in response_data
    ), "Expected universal_query_language_description key to be present in response"
    assert (
        "dynamic_block_definition_schema" in response_data
    ), "Expected key `dynamic_block_definition_schema` to be present in response"


def test_getting_blocks_descriptions_using_new_post_endpoint_requesting_non_existing_execution_engine_version(
    server_url: str
) -> None:
    # when
    response = requests.post(
        f"{server_url}/workflows/blocks/describe",
        json={"execution_engine_version": "0.1.0"}
    )

    # then
    response.raise_for_status()
    response_data = response.json()
    assert "blocks" in response_data, "Response expected to define blocks"
    assert len(response_data["blocks"]) == 0, "No blocks should be registered"


def test_getting_blocks_descriptions_using_new_post_endpoint_with_dynamic_steps(
    server_url,
) -> None:
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
                                "object_detection_prediction",
                                "instance_segmentation_prediction",
                                "keypoint_detection_prediction",
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
                            "object_detection_prediction",
                            "instance_segmentation_prediction",
                            "keypoint_detection_prediction",
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
        json={"dynamic_blocks_definitions": dynamic_blocks_definitions},
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
    assert (
        "universal_query_language_description" in response_data
    ), "Expected universal_query_language_description key to be present in response"
    assert (
        "dynamic_block_definition_schema" in response_data
    ), "Expected key `dynamic_block_definition_schema` to be present in response"
    types_compatible_with_object_detection_predictions = {
        e["manifest_type_identifier"]
        for e in response_data["kinds_connections"][
            "object_detection_prediction"
        ]
    }
    assert (
        "DetectionsToCropsAssociation"
        in types_compatible_with_object_detection_predictions
    ), "Expected dynamic block to be manifested in connections"


def test_getting_blocks_descriptions_using_new_post_endpoint_with_dynamic_steps_when_steps_are_malformed(
    server_url,
) -> None:
    # given
    function_code = """
def my_function(self, prediction: sv.Detections, crops: Batch[WorkflowImageData]) -> BlockResult:
    pass
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
                        "is_dimensionality_reference": True,
                        "selector_data_kind": {
                            "step_output": [
                                "object_detection_prediction",
                                "instance_segmentation_prediction",
                                "keypoint_detection_prediction",
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
                            "object_detection_prediction",
                            "instance_segmentation_prediction",
                            "keypoint_detection_prediction",
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
        json={"dynamic_blocks_definitions": dynamic_blocks_definitions},
    )

    # then
    assert response.status_code == 400, "Expected bad request to be manifested"
    response_data = response.json()
    assert (
        "dimensionality reference" in response_data["message"]
    ), "Expected the cause of problem being dimensionality reference declaration"


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


def test_compilation_endpoint_when_compilation_fails_due_to_invalid_requested_execution_engine_version(
    server_url: str,
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
        f"{server_url}/workflows/validate",
        json=valid_workflow_definition,
    )

    # then
    assert response.status_code == 400, "Expected BadRequest response on wrong version selection"


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
                                "object_detection_prediction",
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
    clean_loaded_models_fixture,
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
            "model_id": "yolov8n-640",
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


def test_workflow_run_when_dynamic_block_is_in_use(
    server_url: str,
    clean_loaded_models_fixture,
) -> None:
    # when
    response = requests.post(
        f"{server_url}/workflows/run",
        json={
            "specification": WORKFLOW_WITH_PYTHON_BLOCK_RUNNING_ON_BATCH,
            "api_key": API_KEY,
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
    ), "Two images submitted - two responses expected"
    assert set(response_data["outputs"][0].keys()) == {
        "max_confidence"
    }, "Expected only `max_confidence` output"
    assert set(response_data["outputs"][1].keys()) == {
        "max_confidence"
    }, "Expected only `max_confidence` output"


def test_get_versions_of_execution_engine(server_url: str) -> None:
    # when
    response = requests.get(f"{server_url}/workflows/execution_engine/versions")

    # then
    response.raise_for_status()
    response_data = response.json()
    assert response_data["versions"] == ["1.7.0"]


def test_getting_block_schema_using_get_endpoint(server_url) -> None:
    # when
    response = requests.get(f"{server_url}/workflows/definition/schema")

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


WORKFLOW_WITH_CUSTOM_DATA_TO_SERIALISE = {
    "version": "1.0",
    "inputs": [
        {"type": "InferenceImage", "name": "image"},
    ],
    "steps": [
        {
            "type": "roboflow_core/sift@v1",
            "name": "image_sift",
            "image": "$inputs.image",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "results",
            "coordinates_system": "own",
            "selector": "$steps.image_sift.*",
        }
    ],
}


def test_workflow_run_when_when_custom_serialisation_is_needed(
    server_url: str,
) -> None:
    # when
    response = requests.post(
        f"{server_url}/workflows/run",
        json={
            "specification": WORKFLOW_WITH_CUSTOM_DATA_TO_SERIALISE,
            "api_key": "some",
            "inputs": {
                "image": [
                    {
                        "type": "url",
                        "value": "https://media.roboflow.com/fruit.png",
                    }
                ],
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
    ), "One image submitted - two responses expected"
    assert set(response_data["outputs"][0].keys()) == {
        "results"
    }, "Expected only `results` output"


def test_describe_workflow_interface_when_incompatible_execution_engine_version_requested(
    server_url: str,
) -> None:
    # given
    invalid_definition = {
        "version": "2.0.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
        ],
        "steps": [
            {
                "type": "ObjectDetectionModel",
                "name": "general_detection",
                "image": "$inputs.image",
                "model_id": "yolov8n-640",
                "class_filter": ["dog"],
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "detections",
                "selector": "$steps.general_detection.predictions",
            },
        ],
    }

    # when
    response = requests.post(
        f"{server_url}/workflows/describe_interface",
        json={
            "specification": invalid_definition,
            "api_key": "some",
        }
    )

    # then
    assert response.status_code == 400, "Expected bad request to raise"


def test_describe_workflow_interface_when_definition_contains_internal_error(
    server_url: str,
) -> None:
    # given
    invalid_definition = {
        "version": "1.0.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
        ],
        "steps": [
            {
                "type": "ObjectDetectionModel",
                "name": "general_detection",
                "image": "$inputs.image",
                "model_id": "yolov8n-640",
                "class_filter": ["dog"],
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "detections",
                "selector": "$steps.invalid.predictions",
            },
        ],
    }

    # when
    response = requests.post(
        f"{server_url}/workflows/describe_interface",
        json={
            "specification": invalid_definition,
            "api_key": "some",
        }
    )

    # then
    assert response.status_code == 400, "Expected bad request to raise"


def test_describe_workflow_interface_when_valid_definition_provided(
    server_url: str,
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
        f"{server_url}/workflows/describe_interface",
        json={
            "specification": valid_definition,
            "api_key": "some",
        }
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
        "roboflow_model_id": "str"
    }
    assert set(response_data["kinds_schemas"].keys()) == {"object_detection_prediction", "image"}, \
        "Expected image and object_detection_prediction kinds to deliver schema"


def test_describe_workflow_interface_when_invalid_usage_of_inputs_detected(
    server_url: str,
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
        f"{server_url}/workflows/describe_interface",
        json={
            "specification": valid_definition,
            "api_key": "some",
        }
    )

    # then
    assert response.status_code == 400
