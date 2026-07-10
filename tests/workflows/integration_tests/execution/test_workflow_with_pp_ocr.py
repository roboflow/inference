import numpy as np
import pytest
import supervision as sv

from inference.core.env import USE_INFERENCE_MODELS, WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine

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


@pytest.mark.skipif(
    not USE_INFERENCE_MODELS,
    reason="PP-OCR is backed by the inference-models package",
)
def test_pp_ocr_workflow_extracts_text(
    model_manager: ModelManager,
    multi_line_text_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=PP_OCR_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": multi_line_text_image,
        }
    )

    # then
    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    assert set(result[0].keys()) == {
        "extracted_text",
        "text_detections",
    }, "Expected all declared outputs to be delivered"
    normalized_text = result[0]["extracted_text"].upper().replace(" ", "")
    assert "GREAT" in normalized_text, "Expected the text to be transcribed"
    detections = result[0]["text_detections"]
    assert isinstance(detections, sv.Detections)
    assert len(detections) > 0, "Expected text lines to be detected"


@pytest.mark.skipif(
    not USE_INFERENCE_MODELS,
    reason="PP-OCR is backed by the inference-models package",
)
def test_pp_ocr_workflow_detect_only(
    model_manager: ModelManager,
    multi_line_text_image: np.ndarray,
) -> None:
    # given
    workflow_definition = {
        **PP_OCR_WORKFLOW,
        "steps": [
            {
                "type": "roboflow_core/pp_ocr@v1",
                "name": "pp_ocr",
                "images": "$inputs.image",
                "text_detection": "small",
                "text_recognition": "none",
            },
        ],
    }
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=workflow_definition,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": multi_line_text_image,
        }
    )

    # then
    assert len(result) == 1
    assert result[0]["extracted_text"] == ""
    assert len(result[0]["text_detections"]) > 0, "Expected boxes without text"
