import numpy as np
import pytest

from inference_sdk import InferenceHTTPClient
from tests.inference.hosted_platform_tests.conftest import ROBOFLOW_API_KEY

WORKFLOW_WITH_CLIP_COMPARISON_V2_AND_CLASSES_REPLACEMENT = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "detection_model_id"},
        {"type": "WorkflowParameter", "name": "clip_reference"},
        {"type": "WorkflowParameter", "name": "version", "default_value": "ViT-B-16"},
    ],
    "steps": [
        {
            "type": "ObjectDetectionModel",
            "name": "general_detection",
            "image": "$inputs.image",
            "model_id": "$inputs.detection_model_id",
            "class_filter": ["dog"],
        },
        {
            "type": "Crop",
            "name": "cropping",
            "image": "$inputs.image",
            "predictions": "$steps.general_detection.predictions",
        },
        {
            "type": "roboflow_core/clip_comparison@v2",
            "name": "comparison",
            "images": "$steps.cropping.crops",
            "classes": "$inputs.clip_reference",
            "version": "$inputs.version",
        },
        {
            "type": "DetectionsClassesReplacement",
            "name": "classes_replacement",
            "object_detection_predictions": "$steps.general_detection.predictions",
            "classification_predictions": "$steps.comparison.classification_predictions",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "original_predictions",
            "selector": "$steps.general_detection.predictions",
        },
        {
            "type": "JsonField",
            "name": "modified_predictions",
            "selector": "$steps.classes_replacement.predictions",
        },
    ],
}


@pytest.mark.flaky(retries=4, delay=1)
def test_workflow_with_clip_as_classifier_replacing_predictions(
    object_detection_service_url: str,
    yolov8n_640_model_id: str,
    dogs_image: np.ndarray,
) -> None:
    # given
    client = InferenceHTTPClient(
        api_url=object_detection_service_url,
        api_key=ROBOFLOW_API_KEY,
    )

    # when
    result = client.run_workflow(
        specification=WORKFLOW_WITH_CLIP_COMPARISON_V2_AND_CLASSES_REPLACEMENT,
        images={
            "image": dogs_image,
        },
        parameters={
            "detection_model_id": yolov8n_640_model_id,
            "clip_reference": ["small-dog", "big-shark"],
        },
    )

    # then
    assert len(result) == 1, "1 image submitted, expected one output"
    assert set(result[0].keys()) == {
        "original_predictions",
        "modified_predictions",
    }, "Expected all outputs to be registered"
    assert (
        len(result[0]["original_predictions"]["predictions"]) == 2
    ), "Expected 2 dogs detected"
    detection_confidences = [
        p["confidence"] for p in result[0]["original_predictions"]["predictions"]
    ]
    assert np.allclose(
        detection_confidences, [0.856178879737854, 0.5191817283630371], atol=5e-2
    ), "Expected predictions to match what was observed while test creation"
    assert (
        len(result[0]["modified_predictions"]["predictions"]) == 2
    ), "Expected 2 bboxes in modified prediction"
    modified_classes = [
        p["class"] for p in result[0]["modified_predictions"]["predictions"]
    ]
    assert modified_classes == [
        "small-dog",
        "small-dog",
    ], "Expected classes to be modified correctly"
