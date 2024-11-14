import numpy as np

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine

VISUALIZATION_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {
            "type": "WorkflowParameter",
            "name": "model_id",
            "default_value": "yolov8n-640",
        },
        {"type": "WorkflowParameter", "name": "confidence", "default_value": 0.3},
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
            "type": "roboflow_core/label_visualization@v1",
            "name": "label_visualization",
            "predictions": "$steps.detection.predictions",
            "image": "$inputs.image",
            "text": "Tracker Id",
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


def test_workflow_when_detections_are_not_present(
    model_manager: ModelManager,
    crowd_image: np.ndarray,
) -> None:
    """This test covers bug in label annotator block."""
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=VISUALIZATION_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={"image": crowd_image, "confidence": 0.99999}
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1, "Single image provided - single output expected"
    assert (
        len(result[0]["result"]["predictions"]) == 0
    ), "Expected no predictions to be delivered"
