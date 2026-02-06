import numpy as np
import pytest
import supervision as sv

from inference_sdk import InferenceHTTPClient
from tests.inference.hosted_platform_tests.conftest import (
    ROBOFLOW_API_KEY,
    PlatformEnvironment,
)

KEYPOINTS_DETECTION_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "model_id"},
    ],
    "steps": [
        {
            "type": "roboflow_core/roboflow_keypoint_detection_model@v2",
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

KEYPOINT_DETECTION_RESULTS_FOR_ENVIRONMENT = {
    PlatformEnvironment.ROBOFLOW_STAGING_LAMBDA: np.array(
        [0.84744745, 0.83828652, 0.7608133, 0.75357497, 0.71568894, 0.46073216]
    ),
    PlatformEnvironment.ROBOFLOW_PLATFORM_LAMBDA: np.array(
        [
            0.83561897,
            0.81181437,
            0.7810185,
            0.7713989,
            0.75499356,
            0.66378689,
            0.59428531,
            0.5382458,
        ]
    ),
}
KEYPOINT_DETECTION_RESULTS_FOR_ENVIRONMENT[
    PlatformEnvironment.ROBOFLOW_STAGING_SERVERLESS
] = KEYPOINT_DETECTION_RESULTS_FOR_ENVIRONMENT[
    PlatformEnvironment.ROBOFLOW_STAGING_LAMBDA
]
KEYPOINT_DETECTION_RESULTS_FOR_ENVIRONMENT[
    PlatformEnvironment.ROBOFLOW_STAGING_LOCALHOST
] = KEYPOINT_DETECTION_RESULTS_FOR_ENVIRONMENT[
    PlatformEnvironment.ROBOFLOW_STAGING_LAMBDA
]
KEYPOINT_DETECTION_RESULTS_FOR_ENVIRONMENT[
    PlatformEnvironment.ROBOFLOW_PLATFORM_SERVERLESS
] = KEYPOINT_DETECTION_RESULTS_FOR_ENVIRONMENT[
    PlatformEnvironment.ROBOFLOW_PLATFORM_LAMBDA
]
KEYPOINT_DETECTION_RESULTS_FOR_ENVIRONMENT[
    PlatformEnvironment.ROBOFLOW_PLATFORM_LOCALHOST
] = KEYPOINT_DETECTION_RESULTS_FOR_ENVIRONMENT[
    PlatformEnvironment.ROBOFLOW_PLATFORM_LAMBDA
]


@pytest.mark.flaky(retries=4, delay=1)
def test_keypoints_detection_workflow(
    platform_environment: PlatformEnvironment,
    object_detection_service_url: str,
    yolov8n_pose_640_model_id: str,
    crowd_image: np.ndarray,
) -> None:
    # given
    client = InferenceHTTPClient(
        api_url=object_detection_service_url,
        api_key=ROBOFLOW_API_KEY,
    )

    # when
    result = client.run_workflow(
        specification=KEYPOINTS_DETECTION_WORKFLOW,
        images={
            "image": [crowd_image, crowd_image],
        },
        parameters={
            "model_id": yolov8n_pose_640_model_id,
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
        KEYPOINT_DETECTION_RESULTS_FOR_ENVIRONMENT[platform_environment],
        atol=1e-2,
    )
    second_detections = sv.Detections.from_inference(result[1]["predictions"])
    assert np.allclose(
        second_detections.confidence,
        KEYPOINT_DETECTION_RESULTS_FOR_ENVIRONMENT[platform_environment],
        atol=1e-2,
    )
