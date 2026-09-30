import numpy as np
import pytest

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine

RICH_LABEL_VISUALIZATION_WORKFLOW = {
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
            "name": "font_family",
            "default_value": "geist_mono",
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
            "type": "roboflow_core/rich_label_visualization@v1",
            "name": "rich_label_visualization",
            "predictions": "$steps.detection.predictions",
            "image": "$inputs.image",
            "text": "Class and Confidence",
            "font_family": "$inputs.font_family",
        },
    ],
    "outputs": [
        {"type": "JsonField", "name": "result", "selector": "$steps.detection.*"},
        {
            "type": "JsonField",
            "name": "visualized",
            "selector": "$steps.rich_label_visualization.image",
        },
    ],
}


@pytest.mark.parametrize("font_family", ["geist_mono", "noto_sans"])
def test_workflow_with_rich_label_visualization(
    model_manager: ModelManager,
    crowd_image: np.ndarray,
    bundled_fonts,
    font_family: str,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=RICH_LABEL_VISUALIZATION_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": crowd_image,
            "font_family": font_family,
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


def test_workflow_with_rich_label_visualization_when_detections_are_not_present(
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
        workflow_definition=RICH_LABEL_VISUALIZATION_WORKFLOW,
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


def test_workflow_with_rich_label_visualization_when_unknown_font_is_requested(
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
        workflow_definition=RICH_LABEL_VISUALIZATION_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    with pytest.raises(Exception) as error:
        _ = execution_engine.run(
            runtime_parameters={
                "image": crowd_image,
                "font_family": "comic_sans",
            }
        )

    # then - unknown fonts are rejected by manifest validation (before the
    # block runs); the error must point at the offending property
    assert "font_family" in str(error.value)
