import cv2
import numpy as np

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine

WORKFLOW_KEYPOINT_VISUALIZATION = {
    "version": "1.1",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {
            "type": "WorkflowParameter",
            "name": "model_id",
            "default_value": "yolov8n-pose-640",
        },
    ],
    "steps": [
        {
            "type": "KeypointsDetectionModel",
            "name": "model",
            "image": "$inputs.image",
            "model_id": "$inputs.model_id",
        },
        {
            "type": "roboflow_core/keypoint_visualization@v1",
            "name": "visualization",
            "image": "$inputs.image",
            "predictions": "$steps.model.predictions",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "visualization",
            "selector": "$steps.visualization.image",
        },
    ],
}


def test_workflow_keypoint_visualization(
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
        workflow_definition=WORKFLOW_KEYPOINT_VISUALIZATION,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": crowd_image,
        }
    )

    assert isinstance(result, list)
    assert len(result) == 1
    assert "visualization" in result[0]
    assert result[0]["visualization"].numpy_image.shape == crowd_image.shape
