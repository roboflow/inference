import numpy as np
import pytest

from inference_sdk import InferenceHTTPClient
from tests.inference.hosted_platform_tests.conftest import OPENAI_KEY, ROBOFLOW_API_KEY

DESCRIPTION_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "detection_model_id"},
        {"type": "InferenceParameter", "name": "prompt"},
        {"type": "WorkflowParameter", "name": "open_ai_key"},
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
            "type": "roboflow_core/open_ai@v1",
            "name": "open_ai",
            "image": "$inputs.image",
            "prompt": "$inputs.prompt",
            "json_output_format": {
                "description": "This is the field to inject produced description",
            },
            "openai_model": "gpt-4o",
            "openai_api_key": "$inputs.open_ai_key",
            "max_tokens": 100,
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
            "name": "description",
            "selector": "$steps.open_ai.description",
        },
    ],
}


@pytest.mark.skipif(OPENAI_KEY is None, reason="No OpenAI API key provided")
def test_image_description_workflow(
    object_detection_service_url: str,
    yolov8n_640_model_id: str,
    dogs_image: np.ndarray,
) -> None:
    client = InferenceHTTPClient(
        api_url=object_detection_service_url,
        api_key=ROBOFLOW_API_KEY,
    )

    # when
    result = client.run_workflow(
        specification=DESCRIPTION_WORKFLOW,
        images={
            "image": dogs_image,
        },
        parameters={
            "detection_model_id": yolov8n_640_model_id,
            "open_ai_key": OPENAI_KEY,
            "prompt": "Provide a very short description for the image given.",
        },
    )

    # then
    assert len(result) == 1, "1 image submitted, expected one output"
    assert set(result[0].keys()) == {
        "detection_predictions",
        "description",
    }, "Expected all outputs to be registered"
    assert (
        len(result[0]["detection_predictions"]["predictions"]) == 2
    ), "Expected 2 dogs detected"
    detection_confidences = [
        p["confidence"] for p in result[0]["detection_predictions"]["predictions"]
    ]
    assert np.allclose(
        detection_confidences, [0.857235848903656, 0.5132315158843994], atol=1e-4
    ), "Expected predictions to match what was observed while test creation"
    assert len(result[0]["description"]) > 0, "Expected some description"
