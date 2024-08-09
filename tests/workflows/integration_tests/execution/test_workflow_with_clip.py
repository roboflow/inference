import numpy as np
import pytest

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine

CLIP_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "reference"},
    ],
    "steps": [
        {
            "type": "ClipComparison",
            "name": "comparison",
            "images": "$inputs.image",
            "texts": "$inputs.reference",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "similarity",
            "selector": "$steps.comparison.similarity",
        },
    ],
}


def test_clip_workflow_when_minimal_valid_input_provided(
    model_manager: ModelManager,
    license_plate_image: np.ndarray,
    crowd_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=CLIP_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [license_plate_image, crowd_image],
            "reference": ["car", "crowd"],
        }
    )

    # then
    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 2, "Expected 2 elements in the output for two input images"
    assert set(result[0].keys()) == {
        "similarity",
    }, "Expected all declared outputs to be delivered"
    assert set(result[1].keys()) == {
        "similarity",
    }, "Expected all declared outputs to be delivered"
    assert (
        len(result[0]["similarity"]) == 2
    ), "Expected 2 elements of similarity comparison list for first image"
    assert (
        result[0]["similarity"][0] > result[0]["similarity"][1]
    ), "Expected to predict `car` class for first image"
    assert (
        len(result[1]["similarity"]) == 2
    ), "Expected 2 elements of similarity comparison list for second image"
    assert (
        result[1]["similarity"][0] < result[1]["similarity"][1]
    ), "Expected to predict `crowd` class for second image"
