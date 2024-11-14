import time

import numpy as np

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData

WORKFLOW_WITH_RATE_LIMITER = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
    ],
    "steps": [
        {
            "type": "roboflow_core/roboflow_object_detection_model@v2",
            "name": "model",
            "images": "$inputs.image",
            "model_id": "yolov8n-640",
        },
        {
            "type": "roboflow_core/rate_limiter@v1",
            "name": "rate_limiter",
            "cooldown_seconds": 1,
            "depends_on": "$steps.model.predictions",
            "next_steps": ["$steps.visualization"],
        },
        {
            "type": "roboflow_core/bounding_box_visualization@v1",
            "name": "visualization",
            "image": "$inputs.image",
            "predictions": "$steps.model.predictions",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "visualization",
            "coordinates_system": "own",
            "selector": "$steps.visualization.image",
        }
    ],
}


def test_workflow_with_rate_limiter(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
    crowd_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_RATE_LIMITER,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result_1 = execution_engine.run(
        runtime_parameters={
            "image": [dogs_image, crowd_image],
        }
    )
    time.sleep(1.5)
    result_2 = execution_engine.run(
        runtime_parameters={
            "image": [dogs_image, crowd_image],
        }
    )

    assert isinstance(result_1, list), "Expected list to be delivered"
    assert len(result_1) == 2, "Expected 2 element in the output for two input images"
    assert set(result_1[0].keys()) == {
        "visualization",
    }, "Expected all declared outputs to be delivered"
    assert set(result_1[1].keys()) == {
        "visualization",
    }, "Expected all declared outputs to be delivered"
    assert isinstance(
        result_1[0]["visualization"], WorkflowImageData
    ), "Expected visualization to be created for the first image of first request"
    assert (
        result_1[1]["visualization"] is None
    ), "Expected visualization to not be created for the second image of first request due to throttling"
    assert isinstance(
        result_2[0]["visualization"], WorkflowImageData
    ), "Expected visualization to be created for the first image of second request"
    assert (
        result_2[1]["visualization"] is None
    ), "Expected visualization to not be created for the second image of second request due to throttling"
