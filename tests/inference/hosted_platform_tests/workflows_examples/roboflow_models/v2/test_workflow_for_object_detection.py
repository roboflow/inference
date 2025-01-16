import numpy as np
import pytest
import supervision as sv

from inference_sdk import InferenceHTTPClient
from tests.inference.hosted_platform_tests.conftest import (
    ROBOFLOW_API_KEY,
    PlatformEnvironment,
)

DETECTION_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "model_id"},
    ],
    "steps": [
        {
            "type": "roboflow_core/roboflow_object_detection_model@v2",
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

DETECTION_RESULTS_FOR_ENVIRONMENT = {
    PlatformEnvironment.ROBOFLOW_STAGING: np.array(
        [
            0.84734064,
            0.83652675,
            0.81773603,
            0.80830157,
            0.76712507,
            0.75515783,
            0.72345364,
            0.71747637,
            0.71143329,
            0.56274879,
            0.5306859,
            0.42601129,
        ]
    ),
    PlatformEnvironment.ROBOFLOW_PLATFORM: np.array(
        [
            0.84734064,
            0.83652675,
            0.81773603,
            0.80830157,
            0.76712507,
            0.75515783,
            0.72345364,
            0.71747637,
            0.71143329,
            0.56274879,
            0.5306859,
            0.42601129,
        ]
    ),
}


@pytest.mark.flaky(retries=4, delay=1)
def test_detection_workflow(
    platform_environment: PlatformEnvironment,
    object_detection_service_url: str,
    yolov8n_640_model_id: str,
    crowd_image: np.ndarray,
) -> None:
    # given
    client = InferenceHTTPClient(
        api_url=object_detection_service_url,
        api_key=ROBOFLOW_API_KEY,
    )

    # when
    result = client.run_workflow(
        specification=DETECTION_WORKFLOW,
        images={
            "image": [crowd_image, crowd_image],
        },
        parameters={
            "model_id": yolov8n_640_model_id,
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
    first_detections = sv.Detections.from_inference(result[0]["predictions"])
    assert np.allclose(
        first_detections.confidence,
        DETECTION_RESULTS_FOR_ENVIRONMENT[platform_environment],
    )
    second_detections = sv.Detections.from_inference(result[1]["predictions"])
    assert np.allclose(
        second_detections.confidence,
        DETECTION_RESULTS_FOR_ENVIRONMENT[platform_environment],
    )
    unique_inference_ids = {r["inference_id"] for r in result}
    assert len(unique_inference_ids) == 2, "Expected unique inference ids granted"
