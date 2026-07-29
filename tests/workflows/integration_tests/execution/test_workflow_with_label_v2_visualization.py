import numpy as np
import pytest

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine

LABEL_V2_VISUALIZATION_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {
            "type": "WorkflowParameter",
            "name": "model_id",
            "default_value": "yolov8n-640",
        },
        {"type": "WorkflowParameter", "name": "confidence", "default_value": 0.3},
        {
            "type": "WorkflowParameter",
            "name": "text_size_mode",
            "default_value": "Manual",
        },
    ],
    "steps": [
        {
            "type": "RoboflowObjectDetectionModel",
            "name": "detection",
            "image": "$inputs.image",
            "model_id": "$inputs.model_id",
            "confidence": "$inputs.confidence",
        },
        {
            "type": "roboflow_core/label_visualization@v2",
            "name": "label_visualization",
            "predictions": "$steps.detection.predictions",
            "image": "$inputs.image",
            "text": "Class and Confidence",
            "text_size_mode": "$inputs.text_size_mode",
        },
    ],
    "outputs": [
        {"type": "JsonField", "name": "result", "selector": "$steps.detection.*"},
        {
            "type": "JsonField",
            "name": "visualized",
            "selector": "$steps.label_visualization.image",
        },
    ],
}


@pytest.mark.parametrize("text_size_mode", ["Manual", "Automatic"])
def test_workflow_with_label_v2_visualization(
    model_manager: ModelManager,
    crowd_image: np.ndarray,
    text_size_mode: str,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=LABEL_V2_VISUALIZATION_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": crowd_image,
            "text_size_mode": text_size_mode,
        }
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1, "Single image provided - single output expected"
    assert (
        result[0]["visualized"].numpy_image.shape == crowd_image.shape
    ), "Visualization should preserve input image dimensions"
    assert not np.array_equal(
        result[0]["visualized"].numpy_image, crowd_image
    ), "Expected visualization to modify the image"


def test_workflow_with_label_v2_visualization_when_detections_are_not_present(
    model_manager: ModelManager,
    crowd_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=LABEL_V2_VISUALIZATION_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": crowd_image,
            "confidence": 0.99999,
        }
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1, "Single image provided - single output expected"
    assert (
        len(result[0]["result"]["predictions"]) == 0
    ), "Expected no predictions to be delivered"


def test_workflow_with_label_v2_visualization_when_invalid_text_size_mode_is_requested(
    model_manager: ModelManager,
    crowd_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=LABEL_V2_VISUALIZATION_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    with pytest.raises(Exception) as error:
        _ = execution_engine.run(
            runtime_parameters={
                "image": crowd_image,
                "text_size_mode": "Banana",
            }
        )

    # then - invalid modes are rejected by manifest validation (before the
    # block runs); the error must point at the offending property
    assert "text_size_mode" in str(error.value)
