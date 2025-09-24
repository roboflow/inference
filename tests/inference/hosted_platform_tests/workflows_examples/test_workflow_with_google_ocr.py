import numpy as np
import pytest

from inference_sdk import InferenceHTTPClient
from tests.inference.hosted_platform_tests.conftest import (
    GOOGLE_VISION_API_KEY,
    ROBOFLOW_API_KEY,
)

GOOGLE_VISION_OCR_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "api_key"},
    ],
    "steps": [
        {
            "type": "roboflow_core/google_vision_ocr@v1",
            "name": "google_vision_ocr",
            "image": "$inputs.image",
            "ocr_type": "text_detection",
            "api_key": "$inputs.api_key",
        },
        {
            "type": "roboflow_core/bounding_box_visualization@v1",
            "name": "bounding_box_visualization",
            "predictions": "$steps.google_vision_ocr.predictions",
            "image": "$inputs.image",
        },
        {
            "type": "roboflow_core/label_visualization@v1",
            "name": "label_visualization",
            "predictions": "$steps.google_vision_ocr.predictions",
            "image": "$steps.bounding_box_visualization.image",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "extracted_text",
            "selector": "$steps.google_vision_ocr.text",
        },
        {
            "type": "JsonField",
            "name": "text_detections",
            "selector": "$steps.google_vision_ocr.predictions",
        },
        {
            "type": "JsonField",
            "name": "text_visualised",
            "selector": "$steps.label_visualization.image",
        },
    ],
}


@pytest.mark.skipif(GOOGLE_VISION_API_KEY is None, reason="No OpenAI API key provided")
@pytest.mark.flaky(retries=4, delay=1)
def test_workflow_with_google_api_ocr(
    object_detection_service_url: str,
    license_plate_image: str,
) -> None:
    client = InferenceHTTPClient(
        api_url=object_detection_service_url,
        api_key=ROBOFLOW_API_KEY,
    )

    # when
    result = client.run_workflow(
        specification=GOOGLE_VISION_OCR_WORKFLOW,
        images={
            "image": license_plate_image,
        },
        parameters={
            "api_key": GOOGLE_VISION_API_KEY,
        },
    )

    # then
    assert len(result) == 1, "Single image given, expected single output"
    assert set(result[0].keys()) == {
        "extracted_text",
        "text_visualised",
        "text_detections",
    }, "Expected all outputs to be delivered"
    assert len(result[0]["extracted_text"]) > 0, "Expected text to be extracted"
    assert (
        len(result[0]["text_detections"]) == 2
    ), "Expected 2 text regions to be detected"
