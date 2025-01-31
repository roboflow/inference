import numpy as np
import pytest

from inference_sdk import InferenceHTTPClient
from tests.inference.hosted_platform_tests.conftest import (
    ROBOFLOW_API_KEY,
    PlatformEnvironment,
)

ACTIVE_LEARNING_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "detection_model_id"},
        {"type": "WorkflowParameter", "name": "target_project"},
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
            "type": "roboflow_core/roboflow_dataset_upload@v2",
            "name": "data_collection",
            "images": "$inputs.image",
            "predictions": "$steps.general_detection.predictions",
            "target_project": "$inputs.target_project",
            "usage_quota_name": "my_quota",
            "data_percentage": 100.0,
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
            "name": "error",
            "selector": "$steps.data_collection.error_status",
        },
        {
            "type": "JsonField",
            "name": "message",
            "selector": "$steps.data_collection.message",
        },
    ],
}

CLASSIFICATION_RESULTS_FOR_ENVIRONMENT = {
    PlatformEnvironment.ROBOFLOW_STAGING: [
        0.7814103364944458,
        0.7870854139328003,
    ],
    PlatformEnvironment.ROBOFLOW_PLATFORM: [
        0.6143714189529419,
        0.6018071174621582,
    ],
}


@pytest.mark.flaky(retries=4, delay=1)
def test_detection_plus_classification_workflow(
    platform_environment: PlatformEnvironment,
    object_detection_service_url: str,
    yolov8n_640_model_id: str,
    target_project: str,
    dogs_image: np.ndarray,
    license_plate_image: np.ndarray,
) -> None:
    # given
    client = InferenceHTTPClient(
        api_url=object_detection_service_url,
        api_key=ROBOFLOW_API_KEY,
    )
    random_image_1 = (np.random.random((640, 480, 3)) * 255).astype(np.uint8)
    random_image_2 = (np.random.random((640, 480, 3)) * 255).astype(np.uint8)

    # when
    result = client.run_workflow(
        specification=ACTIVE_LEARNING_WORKFLOW,
        images={
            "image": [random_image_1, random_image_2],
        },
        parameters={
            "detection_model_id": yolov8n_640_model_id,
            "target_project": target_project,
        },
    )

    # then
    assert len(result) == 2, "2 images submitted, expected two outputs"
    assert set(result[0].keys()) == {
        "detection_predictions",
        "error",
        "message",
    }, "Expected all outputs to be registered"
    assert set(result[1].keys()) == {
        "detection_predictions",
        "error",
        "message",
    }, "Expected all outputs to be registered"
    assert result[1]["error"] is False, "Expected no error"
    assert result[1]["error"] is False, "Expected no error"
