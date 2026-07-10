import numpy as np
import pytest

from inference_sdk import InferenceHTTPClient
from tests.inference.hosted_platform_tests.conftest import ROBOFLOW_API_KEY

PP_OCR_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
    ],
    "steps": [
        {
            "type": "roboflow_core/pp_ocr@v1",
            "name": "pp_ocr",
            "images": "$inputs.image",
            "text_detection": "small",
            "text_recognition": "small",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "extracted_text",
            "selector": "$steps.pp_ocr.result",
        },
        {
            "type": "JsonField",
            "name": "text_detections",
            "selector": "$steps.pp_ocr.predictions",
        },
    ],
}


@pytest.mark.flaky(retries=4, delay=1)
def test_workflow_with_pp_ocr(
    object_detection_service_url: str,
    license_plate_image: np.ndarray,
) -> None:
    # given
    client = InferenceHTTPClient(
        api_url=object_detection_service_url,
        api_key=ROBOFLOW_API_KEY,
    )

    # when
    result = client.run_workflow(
        specification=PP_OCR_WORKFLOW,
        images={
            "image": license_plate_image,
        },
    )

    # then
    assert len(result) == 1, "Single image given, expected single output"
    assert set(result[0].keys()) == {
        "extracted_text",
        "text_detections",
    }, "Expected all outputs to be delivered"
    assert isinstance(result[0]["extracted_text"], str)
    assert len(result[0]["extracted_text"]) > 0, "Expected text to be extracted"
    assert len(result[0]["text_detections"]) > 0, "Expected text lines to be detected"
