import numpy as np
import pytest

from inference_sdk import InferenceHTTPClient
from tests.inference.hosted_platform_tests.conftest import (
    ROBOFLOW_API_KEY,
    PlatformEnvironment,
)

DETECTION_PLUS_CLASSIFICATION_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "detection_model_id"},
        {"type": "WorkflowParameter", "name": "classification_model_id"},
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
            "type": "ClassificationModel",
            "name": "breds_classification",
            "image": "$steps.cropping.crops",
            "model_id": "$inputs.classification_model_id",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "detection_predictions",
            "selector": "$steps.general_detection.predictions",
        },
        {
            "type": "JsonField",
            "name": "classification_predictions",
            "selector": "$steps.breds_classification.predictions",
        },
    ],
}

CLASSIFICATION_RESULTS_FOR_ENVIRONMENT = {
    PlatformEnvironment.ROBOFLOW_STAGING_LAMBDA: [
        0.8472128510475159,
        0.9162841439247131
    ],
    PlatformEnvironment.ROBOFLOW_PLATFORM_LAMBDA: [
        0.6154301762580872,
        0.5893789529800415,
    ],
}
CLASSIFICATION_RESULTS_FOR_ENVIRONMENT[
    PlatformEnvironment.ROBOFLOW_STAGING_SERVERLESS
] = CLASSIFICATION_RESULTS_FOR_ENVIRONMENT[PlatformEnvironment.ROBOFLOW_STAGING_LAMBDA]
CLASSIFICATION_RESULTS_FOR_ENVIRONMENT[
    PlatformEnvironment.ROBOFLOW_STAGING_LOCALHOST
] = CLASSIFICATION_RESULTS_FOR_ENVIRONMENT[PlatformEnvironment.ROBOFLOW_STAGING_LAMBDA]
CLASSIFICATION_RESULTS_FOR_ENVIRONMENT[
    PlatformEnvironment.ROBOFLOW_PLATFORM_SERVERLESS
] = CLASSIFICATION_RESULTS_FOR_ENVIRONMENT[PlatformEnvironment.ROBOFLOW_PLATFORM_LAMBDA]
CLASSIFICATION_RESULTS_FOR_ENVIRONMENT[
    PlatformEnvironment.ROBOFLOW_PLATFORM_LOCALHOST
] = CLASSIFICATION_RESULTS_FOR_ENVIRONMENT[PlatformEnvironment.ROBOFLOW_PLATFORM_LAMBDA]


@pytest.mark.flaky(retries=4, delay=1)
def test_detection_plus_classification_workflow(
    platform_environment: PlatformEnvironment,
    object_detection_service_url: str,
    yolov8n_640_model_id: str,
    classification_model_id: str,
    dogs_image: np.ndarray,
    license_plate_image: np.ndarray,
) -> None:
    # given
    client = InferenceHTTPClient(
        api_url=object_detection_service_url,
        api_key=ROBOFLOW_API_KEY,
    )

    # when
    result = client.run_workflow(
        specification=DETECTION_PLUS_CLASSIFICATION_WORKFLOW,
        images={
            "image": [dogs_image, license_plate_image],
        },
        parameters={
            "detection_model_id": yolov8n_640_model_id,
            "classification_model_id": classification_model_id,
        },
    )

    # then
    assert len(result) == 2, "2 images submitted, expected two outputs"
    assert set(result[0].keys()) == {
        "detection_predictions",
        "classification_predictions",
    }, "Expected all outputs to be registered"
    assert set(result[1].keys()) == {
        "detection_predictions",
        "classification_predictions",
    }, "Expected all outputs to be registered"
    assert (
        len(result[0]["detection_predictions"]["predictions"]) == 2
    ), "Expected 2 dogs detected"
    detection_confidences = [
        p["confidence"] for p in result[0]["detection_predictions"]["predictions"]
    ]
    assert np.allclose(
        detection_confidences, [0.856178879737854, 0.5191817283630371], atol=5e-3
    ), "Expected predictions to match what was observed while test creation"
    assert (
        len(result[0]["classification_predictions"]) == 2
    ), "Expected 2 crops to be made"
    classification_confidences = [
        result[0]["classification_predictions"][0]["predictions"]["dog"]["confidence"],
        result[0]["classification_predictions"][1]["predictions"]["dog"]["confidence"],
    ]
    assert np.allclose(
        classification_confidences,
        CLASSIFICATION_RESULTS_FOR_ENVIRONMENT[platform_environment],
        atol=1e-2,
    ), "Expected classification predictions to match"
    assert (
        len(result[1]["detection_predictions"]["predictions"]) == 0
    ), "Expected 0 dogs detected"
    assert (
        len(result[1]["classification_predictions"]) == 0
    ), "Expected 0 crops to be made"
