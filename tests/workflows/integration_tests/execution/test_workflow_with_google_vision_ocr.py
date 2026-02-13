import os

import numpy as np
import pytest

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine
from tests.workflows.integration_tests.execution.workflows_gallery_collector.decorators import (
    add_to_workflows_gallery,
)

GOOGLE_VISION_API_KEY = os.getenv("WORKFLOWS_TEST_GOOGLE_VISION_API_KEY")

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


@add_to_workflows_gallery(
    category="Workflows for OCR",
    use_case_title="Google Vision OCR",
    use_case_description="""
In this example, Google Vision OCR is used to extract text from input image.
Additionally, example presents how to combine structured output of Google API
with visualisation blocks. 
    """,
    workflow_definition=GOOGLE_VISION_OCR_WORKFLOW,
    workflow_name_in_app="google-vision-ocr",
)
@pytest.mark.skipif(
    condition=GOOGLE_VISION_API_KEY is None, reason="Google API key not provided"
)
def test_workflow_with_google_ocr_when_text_should_be_detected(
    model_manager: ModelManager,
    license_plate_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=GOOGLE_VISION_OCR_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [license_plate_image],
            "api_key": GOOGLE_VISION_API_KEY,
        }
    )

    # then
    assert len(result) == 1, "Single image given, expected single output"
    assert set(result[0].keys()) == {
        "extracted_text",
        "text_visualised",
        "text_detections",
    }, "Expected all outputs to be delivered"
    # OCR results can vary slightly between runs, so check key parts are present
    extracted_text = result[0]["extracted_text"]
    assert "2398027" in extracted_text, "Expected '2398027' in extracted text"
    assert "239 8072" in extracted_text, "Expected '239 8072' in extracted text"
    assert not np.allclose(
        license_plate_image, result[0]["text_visualised"].numpy_image
    ), "Expected that visualisation will change the output image"
    assert (
        len(result[0]["text_detections"]) >= 3
    ), "Expected at least 3 text regions to be detected"


@pytest.mark.skipif(
    condition=GOOGLE_VISION_API_KEY is None, reason="Google API key not provided"
)
def test_workflow_with_google_ocr_when_no_text_should_be_detected(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=GOOGLE_VISION_OCR_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [dogs_image],
            "api_key": GOOGLE_VISION_API_KEY,
        }
    )

    # then
    assert len(result) == 1, "Single image given, expected single output"
    assert set(result[0].keys()) == {
        "extracted_text",
        "text_visualised",
        "text_detections",
    }, "Expected all outputs to be delivered"
    assert result[0]["extracted_text"] == ""
    assert np.allclose(
        dogs_image, result[0]["text_visualised"].numpy_image
    ), "Expected that visualisation will not change the output image"
    assert len(result[0]["text_detections"]) == 0, "Expected 0 text regions detected"


GOOGLE_VISION_OCR_PROXY_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
    ],
    "steps": [
        {
            "type": "roboflow_core/google_vision_ocr@v1",
            "name": "google_vision_ocr",
            "image": "$inputs.image",
            "ocr_type": "text_detection",
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
    ],
}


def test_workflow_with_google_ocr_without_api_key_via_proxy(
    model_manager: ModelManager,
    license_plate_image: np.ndarray,
    roboflow_api_key: str,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": roboflow_api_key,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=GOOGLE_VISION_OCR_PROXY_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [license_plate_image],
        }
    )

    # then
    assert len(result) == 1, "Single image given, expected single output"
    assert set(result[0].keys()) == {
        "extracted_text",
        "text_detections",
    }, "Expected all outputs to be delivered"
    # OCR results can vary slightly between runs, so check key parts are present
    extracted_text = result[0]["extracted_text"]
    assert "2398027" in extracted_text, "Expected '2398027' in extracted text"
    assert "239 8072" in extracted_text, "Expected '239 8072' in extracted text"
    assert (
        len(result[0]["text_detections"]) >= 3
    ), "Expected at least 3 text regions to be detected"
