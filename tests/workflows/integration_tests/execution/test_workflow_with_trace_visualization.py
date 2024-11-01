import numpy as np

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData

WORKFLOW_WITH_TRACE_VISUALIZATION = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
    ],
    "steps": [
        {
            "type": "ObjectDetectionModel",
            "name": "model",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
            "confidence": 0.2,
            "class_filter": ["person"],
        },
        {
            "type": "roboflow_core/byte_tracker@v3",
            "name": "byte_tracker",
            "image": "$inputs.image",
            "detections": "$steps.model.predictions",
        },
        {
            "type": "roboflow_core/trace_visualization@v1",
            "name": "trace_visualization",
            "image": "$inputs.image",
            "predictions": "$steps.byte_tracker.tracked_detections",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "visualization",
            "selector": "$steps.trace_visualization.image",
        }
    ],
}


def test_workflow_with_trace_visualization(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_TRACE_VISUALIZATION,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [dogs_image],
        }
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1, "Expected single result for single input image"
    assert set(result[0].keys()) == {
        "visualization"
    }, "Expected all outputs to be registered"
    assert isinstance(
        result[0]["visualization"], WorkflowImageData
    ), "Expected visualization to be image"
