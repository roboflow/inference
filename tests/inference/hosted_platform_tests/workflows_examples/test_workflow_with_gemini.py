import numpy as np
import pytest

from inference_sdk import InferenceHTTPClient
from tests.inference.hosted_platform_tests.conftest import (
    GOOGLE_API_KEY,
    ROBOFLOW_API_KEY,
)

CLASSIFICATION_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "api_key"},
        {"type": "WorkflowParameter", "name": "classes"},
    ],
    "steps": [
        {
            "type": "roboflow_core/google_gemini@v1",
            "name": "gemini",
            "images": "$inputs.image",
            "task_type": "classification",
            "classes": "$inputs.classes",
            "api_key": "$inputs.api_key",
        },
        {
            "type": "roboflow_core/vlm_as_classifier@v2",
            "name": "parser",
            "image": "$inputs.image",
            "vlm_output": "$steps.gemini.output",
            "classes": "$steps.gemini.classes",
        },
        {
            "type": "roboflow_core/property_definition@v1",
            "name": "top_class",
            "operations": [
                {"type": "ClassificationPropertyExtract", "property_name": "top_class"}
            ],
            "data": "$steps.parser.predictions",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "gemini_result",
            "selector": "$steps.gemini.output",
        },
        {
            "type": "JsonField",
            "name": "top_class",
            "selector": "$steps.top_class.output",
        },
        {
            "type": "JsonField",
            "name": "parsed_prediction",
            "selector": "$steps.parser.*",
        },
    ],
}


@pytest.mark.skipif(GOOGLE_API_KEY is None, reason="No Google API key provided")
@pytest.mark.flaky(retries=4, delay=1)
def test_classification_workflow(
    object_detection_service_url: str,
    dogs_image: np.ndarray,
) -> None:
    client = InferenceHTTPClient(
        api_url=object_detection_service_url,
        api_key=ROBOFLOW_API_KEY,
    )

    # when
    result = client.run_workflow(
        specification=CLASSIFICATION_WORKFLOW,
        images={
            "image": dogs_image,
        },
        parameters={
            "api_key": GOOGLE_API_KEY,
            "classes": ["cat", "dog"],
        },
    )

    # then
    assert len(result) == 1, "Single image given, expected single output"
    assert set(result[0].keys()) == {
        "gemini_result",
        "top_class",
        "parsed_prediction",
    }, "Expected all outputs to be delivered"
    assert (
        isinstance(result[0]["gemini_result"], str)
        and len(result[0]["gemini_result"]) > 0
    ), "Expected non-empty string generated"
    assert result[0]["top_class"] == "dog"
    assert result[0]["parsed_prediction"]["error_status"] is False


STRUCTURED_PROMPTING_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "api_key"},
    ],
    "steps": [
        {
            "type": "roboflow_core/google_gemini@v1",
            "name": "gemini",
            "images": "$inputs.image",
            "task_type": "structured-answering",
            "output_structure": {
                "dogs_count": "count of dogs instances in the image",
                "cats_count": "count of cats instances in the image",
            },
            "api_key": "$inputs.api_key",
        },
        {
            "type": "roboflow_core/json_parser@v1",
            "name": "parser",
            "raw_json": "$steps.gemini.output",
            "expected_fields": ["dogs_count", "cats_count"],
        },
        {
            "type": "roboflow_core/property_definition@v1",
            "name": "property_definition",
            "operations": [{"type": "ToString"}],
            "data": "$steps.parser.dogs_count",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "result",
            "selector": "$steps.property_definition.output",
        }
    ],
}


@pytest.mark.skipif(GOOGLE_API_KEY is None, reason="No Google API key provided")
@pytest.mark.flaky(retries=4, delay=1)
def test_structured_parsing_workflow(
    object_detection_service_url: str,
    dogs_image: np.ndarray,
) -> None:
    client = InferenceHTTPClient(
        api_url=object_detection_service_url,
        api_key=ROBOFLOW_API_KEY,
    )

    # when
    result = client.run_workflow(
        specification=STRUCTURED_PROMPTING_WORKFLOW,
        images={
            "image": dogs_image,
        },
        parameters={
            "api_key": GOOGLE_API_KEY,
        },
    )

    # then
    assert len(result) == 1, "Single image given, expected single output"
    assert set(result[0].keys()) == {"result"}, "Expected all outputs to be delivered"
    assert result[0]["result"] == "2"


OBJECT_DETECTION_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "api_key"},
        {"type": "WorkflowParameter", "name": "classes"},
    ],
    "steps": [
        {
            "type": "roboflow_core/google_gemini@v1",
            "name": "gemini",
            "images": "$inputs.image",
            "task_type": "object-detection",
            "classes": "$inputs.classes",
            "api_key": "$inputs.api_key",
        },
        {
            "type": "roboflow_core/vlm_as_detector@v2",
            "name": "parser",
            "vlm_output": "$steps.gemini.output",
            "image": "$inputs.image",
            "classes": "$steps.gemini.classes",
            "model_type": "google-gemini",
            "task_type": "object-detection",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "gemini_result",
            "selector": "$steps.gemini.output",
        },
        {
            "type": "JsonField",
            "name": "parsed_prediction",
            "selector": "$steps.parser.predictions",
        },
    ],
}


@pytest.mark.skipif(GOOGLE_API_KEY is None, reason="No Google API key provided")
@pytest.mark.flaky(retries=4, delay=1)
def test_object_detection_workflow(
    object_detection_service_url: str,
    dogs_image: np.ndarray,
) -> None:
    client = InferenceHTTPClient(
        api_url=object_detection_service_url,
        api_key=ROBOFLOW_API_KEY,
    )

    # when
    result = client.run_workflow(
        specification=OBJECT_DETECTION_WORKFLOW,
        images={
            "image": dogs_image,
        },
        parameters={
            "api_key": GOOGLE_API_KEY,
            "classes": ["cat", "dog"],
        },
    )

    # then
    assert len(result) == 1, "Single image given, expected single output"
    assert set(result[0].keys()) == {
        "gemini_result",
        "parsed_prediction",
    }, "Expected all outputs to be delivered"
    assert [e["class"] for e in result[0]["parsed_prediction"]["predictions"]] == [
        "dog",
        "dog",
    ], "Expected 2 dogs to be detected"


VLM_AS_SECONDARY_CLASSIFIER_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "api_key"},
        {"type": "WorkflowParameter", "name": "model_id"},
        {
            "type": "WorkflowParameter",
            "name": "classes",
            "default_value": [
                "russell-terrier",
                "wirehaired-pointing-griffon",
                "beagle",
            ],
        },
    ],
    "steps": [
        {
            "type": "ObjectDetectionModel",
            "name": "general_detection",
            "image": "$inputs.image",
            "model_id": "$inputs.model_id",
            "class_filter": ["dog"],
        },
        {
            "type": "Crop",
            "name": "cropping",
            "image": "$inputs.image",
            "predictions": "$steps.general_detection.predictions",
        },
        {
            "type": "roboflow_core/google_gemini@v1",
            "name": "gemini",
            "images": "$steps.cropping.crops",
            "task_type": "classification",
            "classes": "$inputs.classes",
            "api_key": "$inputs.api_key",
        },
        {
            "type": "roboflow_core/vlm_as_classifier@v2",
            "name": "parser",
            "image": "$steps.cropping.crops",
            "vlm_output": "$steps.gemini.output",
            "classes": "$steps.gemini.classes",
        },
        {
            "type": "roboflow_core/detections_classes_replacement@v1",
            "name": "classes_replacement",
            "object_detection_predictions": "$steps.general_detection.predictions",
            "classification_predictions": "$steps.parser.predictions",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "predictions",
            "selector": "$steps.classes_replacement.predictions",
        },
    ],
}


@pytest.mark.skipif(GOOGLE_API_KEY is None, reason="No Google API key provided")
@pytest.mark.flaky(retries=4, delay=1)
def test_workflow_with_secondary_classifier(
    object_detection_service_url: str,
    dogs_image: np.ndarray,
    yolov8n_640_model_id: str,
) -> None:
    client = InferenceHTTPClient(
        api_url=object_detection_service_url,
        api_key=ROBOFLOW_API_KEY,
    )

    # when
    result = client.run_workflow(
        specification=VLM_AS_SECONDARY_CLASSIFIER_WORKFLOW,
        images={
            "image": dogs_image,
        },
        parameters={
            "api_key": GOOGLE_API_KEY,
            "classes": ["russell-terrier", "wirehaired-pointing-griffon", "beagle"],
            "model_id": yolov8n_640_model_id,
        },
    )

    # then
    assert len(result) == 1, "Single image given, expected single output"
    assert set(result[0].keys()) == {
        "predictions",
    }, "Expected all outputs to be delivered"
    assert "dog" not in set(
        [e["class"] for e in result[0]["predictions"]["predictions"]]
    ), "Expected classes to be substituted"
