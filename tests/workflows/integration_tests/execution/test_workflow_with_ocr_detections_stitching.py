import numpy as np

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine
from tests.workflows.integration_tests.execution.workflows_gallery_collector.decorators import (
    add_to_workflows_gallery,
)

WORKFLOW_STITCHING_OCR_DETECTIONS = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {
            "type": "WorkflowParameter",
            "name": "model_id",
            "default_value": "ocr-oy9a7/1",
        },
        {"type": "WorkflowParameter", "name": "tolerance", "default_value": 10},
        {"type": "WorkflowParameter", "name": "confidence", "default_value": 0.4},
    ],
    "steps": [
        {
            "type": "roboflow_core/roboflow_object_detection_model@v2",
            "name": "ocr_detection",
            "image": "$inputs.image",
            "model_id": "$inputs.model_id",
            "confidence": "$inputs.confidence",
        },
        {
            "type": "roboflow_core/stitch_ocr_detections@v1",
            "name": "detections_stitch",
            "predictions": "$steps.ocr_detection.predictions",
            "reading_direction": "left_to_right",
            "tolerance": "$inputs.tolerance",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "ocr_text",
            "selector": "$steps.detections_stitch.ocr_text",
        },
    ],
}


@add_to_workflows_gallery(
    category="Workflows for OCR",
    use_case_title="Workflow with model detecting individual characters and text stitching",
    use_case_description="""
This workflow extracts and organizes text from an image using OCR. It begins by analyzing the image with detection 
model to detect individual characters or words and their positions. 

Then, it groups nearby text into lines based on a specified `tolerance` for spacing and arranges them in 
reading order (`left-to-right`). 

The final output is a JSON field containing the structured text in readable, logical order, accurately reflecting 
the layout of the original image.
    """,
    workflow_definition=WORKFLOW_STITCHING_OCR_DETECTIONS,
    workflow_name_in_app="ocr-detections-stitch",
)
def test_detection_plus_classification_workflow_when_minimal_valid_input_provided(
    model_manager: ModelManager,
    multi_line_text_image: np.ndarray,
    roboflow_api_key: str,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": roboflow_api_key,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_STITCHING_OCR_DETECTIONS,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": multi_line_text_image,
            "tolerance": 20,
            "confidence": 0.6,
        }
    )

    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    assert set(result[0].keys()) == {
        "ocr_text",
    }, "Expected all declared outputs to be delivered"
    assert result[0]["ocr_text"] == "MAKE\nTHISDAY\nGREAT"
