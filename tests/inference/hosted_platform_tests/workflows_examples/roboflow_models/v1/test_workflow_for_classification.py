import numpy as np
import pytest

from inference_sdk import InferenceHTTPClient
from tests.inference.hosted_platform_tests.conftest import (
    ROBOFLOW_API_KEY,
    PlatformEnvironment,
)

MULTI_CLASS_CLASSIFICATION_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "model_id"},
    ],
    "steps": [
        {
            "type": "roboflow_core/roboflow_classification_model@v1",
            "name": "classifier",
            "image": "$inputs.image",
            "model_id": "$inputs.model_id",
            "confidence": 0.09,
        }
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "predictions",
            "selector": "$steps.classifier.predictions",
        },
        {
            "type": "JsonField",
            "name": "inference_id",
            "selector": "$steps.classifier.inference_id",
        },
    ],
}

MULTI_CLASS_CLASSIFICATION_RESULTS_FOR_ENVIRONMENT = {
    PlatformEnvironment.ROBOFLOW_STAGING: [0.3667, 0.5917],
    PlatformEnvironment.ROBOFLOW_PLATFORM: [0.8252, 0.9962],
}


@pytest.mark.flaky(retries=4, delay=1)
def test_multi_class_classification_workflow(
    platform_environment: PlatformEnvironment,
    classification_service_url: str,
    multi_class_classification_model_id: str,
    dogs_image: np.ndarray,
    license_plate_image: np.ndarray,
) -> None:
    # given
    client = InferenceHTTPClient(
        api_url=classification_service_url,
        api_key=ROBOFLOW_API_KEY,
    )

    # when
    result = client.run_workflow(
        specification=MULTI_CLASS_CLASSIFICATION_WORKFLOW,
        images={
            "image": [dogs_image, license_plate_image],
        },
        parameters={
            "model_id": multi_class_classification_model_id,
        },
    )

    # then
    assert len(result) == 2, "2 images submitted, expected two outputs"
    assert set(result[0].keys()) == {
        "predictions",
        "inference_id",
    }, "Expected all outputs to be registered"
    assert set(result[1].keys()) == {
        "predictions",
        "inference_id",
    }, "Expected all outputs to be registered"
    unique_inference_ids = {r["inference_id"] for r in result}
    assert len(unique_inference_ids) == 2, "Expected unique inference ids granted"
    predicted_confidences = [r["predictions"]["confidence"] for r in result]
    assert np.allclose(
        predicted_confidences,
        MULTI_CLASS_CLASSIFICATION_RESULTS_FOR_ENVIRONMENT[platform_environment],
        atol=1e-3,
    ), "Expected classification predictions to match expectations"


MULTI_LABEL_CLASSIFICATION_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "model_id"},
    ],
    "steps": [
        {
            "type": "roboflow_core/roboflow_multi_label_classification_model@v1",
            "name": "classifier",
            "image": "$inputs.image",
            "model_id": "$inputs.model_id",
            "confidence": 0.5,
        }
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "predictions",
            "selector": "$steps.classifier.predictions",
        },
        {
            "type": "JsonField",
            "name": "inference_id",
            "selector": "$steps.classifier.inference_id",
        },
    ],
}


MULTI_LABEL_CLASSIFICATION_RESULTS_FOR_ENVIRONMENT = {
    PlatformEnvironment.ROBOFLOW_STAGING: [
        {"dog"},
        {"cat", "dog"},
    ],
    PlatformEnvironment.ROBOFLOW_PLATFORM: [{"dog"}, set()],
}
MULTI_LABEL_CLASSIFICATION_RESULTS_FOR_ENVIRONMENT[
    PlatformEnvironment.ROBOFLOW_STAGING_LOCALHOST
] = MULTI_LABEL_CLASSIFICATION_RESULTS_FOR_ENVIRONMENT[
    PlatformEnvironment.ROBOFLOW_STAGING
]
MULTI_LABEL_CLASSIFICATION_RESULTS_FOR_ENVIRONMENT[
    PlatformEnvironment.ROBOFLOW_PLATFORM_LOCALHOST
] = MULTI_LABEL_CLASSIFICATION_RESULTS_FOR_ENVIRONMENT[
    PlatformEnvironment.ROBOFLOW_PLATFORM
]


@pytest.mark.flaky(retries=4, delay=1)
def test_multi_label_classification_workflow(
    platform_environment: PlatformEnvironment,
    classification_service_url: str,
    classification_model_id: str,
    dogs_image: np.ndarray,
    license_plate_image: np.ndarray,
) -> None:
    # given
    client = InferenceHTTPClient(
        api_url=classification_service_url,
        api_key=ROBOFLOW_API_KEY,
    )

    # when
    result = client.run_workflow(
        specification=MULTI_LABEL_CLASSIFICATION_WORKFLOW,
        images={
            "image": [dogs_image, license_plate_image],
        },
        parameters={
            "model_id": classification_model_id,
        },
    )

    # then
    assert len(result) == 2, "2 images submitted, expected two outputs"
    assert set(result[0].keys()) == {
        "predictions",
        "inference_id",
    }, "Expected all outputs to be registered"
    assert set(result[1].keys()) == {
        "predictions",
        "inference_id",
    }, "Expected all outputs to be registered"
    unique_inference_ids = {r["inference_id"] for r in result}
    assert len(unique_inference_ids) == 2, "Expected unique inference ids granted"
    predicted_classes = [set(r["predictions"]["predicted_classes"]) for r in result]
    assert (
        predicted_classes
        == MULTI_LABEL_CLASSIFICATION_RESULTS_FOR_ENVIRONMENT[platform_environment]
    )
