import numpy as np
import pytest

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData

TRACKER_TYPES = [
    "roboflow_core/byte_tracker@v3",
    "roboflow_core/trackers_bytetrack@v1",
    "roboflow_core/trackers_sort@v1",
    "roboflow_core/trackers_ocsort@v1",
]


def _make_workflow(tracker_type: str) -> dict:
    return {
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
                "type": tracker_type,
                "name": "tracker",
                "image": "$inputs.image",
                "detections": "$steps.model.predictions",
            },
            {
                "type": "roboflow_core/trace_visualization@v1",
                "name": "trace_visualization",
                "image": "$inputs.image",
                "predictions": "$steps.tracker.tracked_detections",
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


@pytest.mark.parametrize("tracker_type", TRACKER_TYPES)
def test_workflow_with_trace_visualization(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
    tracker_type: str,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=_make_workflow(tracker_type),
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
