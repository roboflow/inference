import numpy as np

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine

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
    ],
    "steps": [
        {
            "type": "roboflow_core/easy_ocr@v1",
            "name": "easy_ocr",
            "image": "$inputs.image",
            "character_set": "English",
        },
        {
            "type": "roboflow_core/stitch_ocr_detections@v2",
            "name": "detections_stitch",
            "predictions": "$steps.easy_ocr.predictions",
            "stitching_algorithm": "tolerance",
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


def test_detection_plus_classification_workflow_when_minimal_valid_input_provided(
    model_manager: ModelManager,
    multi_line_text_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
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
        }
    )

    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    assert set(result[0].keys()) == {
        "ocr_text",
    }, "Expected all declared outputs to be delivered"
    assert result[0]["ocr_text"] == "MAKe\nTHISDA Y\nG REAT!"
