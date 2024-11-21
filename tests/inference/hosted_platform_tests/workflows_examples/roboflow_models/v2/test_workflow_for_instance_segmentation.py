import numpy as np
import pytest
import supervision as sv

from inference_sdk import InferenceHTTPClient
from tests.inference.hosted_platform_tests.conftest import (
    ROBOFLOW_API_KEY,
    PlatformEnvironment,
)

SEGMENTATION_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "model_id"},
    ],
    "steps": [
        {
            "type": "roboflow_core/roboflow_instance_segmentation_model@v2",
            "name": "classifier",
            "image": "$inputs.image",
            "model_id": "$inputs.model_id",
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

SEGMENTATION_RESULTS_FOR_ENVIRONMENT = {
    PlatformEnvironment.ROBOFLOW_STAGING: np.array([[23, 24, 96, 122]]),
    PlatformEnvironment.ROBOFLOW_PLATFORM: np.array([[23, 23, 96, 120]]),
}


@pytest.mark.flaky(retries=4, delay=1)
def test_segmentation_workflow(
    platform_environment: PlatformEnvironment,
    instance_segmentation_service_url: str,
    segmentation_model_id: str,
    asl_image: np.ndarray,
) -> None:
    # given
    client = InferenceHTTPClient(
        api_url=instance_segmentation_service_url,
        api_key=ROBOFLOW_API_KEY,
    )

    # when
    result = client.run_workflow(
        specification=SEGMENTATION_WORKFLOW,
        images={
            "image": [asl_image, asl_image],
        },
        parameters={
            "model_id": segmentation_model_id,
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
    first_image_predictions = sv.Detections.from_inference(result[0]["predictions"])
    assert np.allclose(
        first_image_predictions.xyxy,
        SEGMENTATION_RESULTS_FOR_ENVIRONMENT[platform_environment],
        atol=1.0,
    ), "Expected prediction to meet reference value"
    second_image_predictions = sv.Detections.from_inference(result[0]["predictions"])
    assert np.allclose(
        second_image_predictions.xyxy,
        SEGMENTATION_RESULTS_FOR_ENVIRONMENT[platform_environment],
        atol=1.0,
    ), "Expected prediction to meet reference value"
    unique_inference_ids = {r["inference_id"] for r in result}
    assert len(unique_inference_ids) == 2, "Expected unique inference ids granted"
