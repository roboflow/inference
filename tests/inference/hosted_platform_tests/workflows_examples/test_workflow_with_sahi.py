import numpy as np
import pytest

from inference_sdk import InferenceHTTPClient
from tests.inference.hosted_platform_tests.conftest import (
    ROBOFLOW_API_KEY,
    PlatformEnvironment,
)

WORKFLOW_WITH_SAHI = {
    "version": "1.0.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "detection_model_id"},
    ],
    "steps": [
        {
            "type": "roboflow_core/image_slicer@v1",
            "name": "image_slicer",
            "image": "$inputs.image",
            "slice_width": 320,
            "slice_height": 320,
        },
        {
            "type": "roboflow_core/roboflow_object_detection_model@v1",
            "name": "detection",
            "image": "$steps.image_slicer.slices",
            "model_id": "$inputs.detection_model_id",
        },
        {
            "type": "roboflow_core/detections_stitch@v1",
            "name": "stitch",
            "reference_image": "$inputs.image",
            "predictions": "$steps.detection.predictions",
            "overlap_filtering_strategy": "nms",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "predictions",
            "selector": "$steps.stitch.predictions",
            "coordinates_system": "own",
        },
    ],
}


@pytest.mark.flaky(retries=4, delay=1)
def test_workflow_with_sahi(
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
        specification=WORKFLOW_WITH_SAHI,
        images={"image": crowd_image},
        parameters={
            "detection_model_id": yolov8n_640_model_id,
        },
    )

    # then
    assert len(result) == 1, "1 image submitted, expected one output"
    confidences = [p["confidence"] for p in result[0]["predictions"]["predictions"]]
    np.allclose(
        confidences,
        [
            0.8529196977615356,
            0.7524353265762329,
            0.6803092956542969,
            0.5056378245353699,
            0.850996732711792,
            0.7510667443275452,
            0.6945700645446777,
            0.467992901802063,
            0.9452757239341736,
            0.8821572065353394,
            0.7479097843170166,
            0.5607050657272339,
            0.5313479900360107,
            0.806734561920166,
            0.5743240118026733,
            0.9005401134490967,
            0.8576444387435913,
        ],
        atol=1e-4,
    ), "Expected predictions to be as measured while test creation"
