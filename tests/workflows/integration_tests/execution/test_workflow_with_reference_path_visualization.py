import numpy as np
import pytest

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.errors import RuntimeInputError
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData

WORKFLOW_WITH_REFERENCE_PATH_VISUALIZATION = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "reference_path"},
        {"type": "WorkflowParameter", "name": "thickness"},
    ],
    "steps": [
        {
            "type": "roboflow_core/reference_path_visualization@v1",
            "name": "visualization",
            "image": "$inputs.image",
            "reference_path": "$inputs.reference_path",
            "color": "rgb(255, 0, 0)",
            "thickness": "$inputs.thickness",
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


def test_workflow_with_rate_limiter_when_list_of_lists_is_given_as_reference_path(
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
        workflow_definition=WORKFLOW_WITH_REFERENCE_PATH_VISUALIZATION,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [dogs_image],
            "reference_path": [[100, 300], [110, 310], [120, 320], [130, 330]],
            "thickness": 10,
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


def test_workflow_with_rate_limiter_when_list_of_tuples_is_given_as_reference_path(
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
        workflow_definition=WORKFLOW_WITH_REFERENCE_PATH_VISUALIZATION,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [dogs_image],
            "reference_path": [(100, 300), (110, 310), (120, 320), (130, 330)],
            "thickness": 10,
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


def test_workflow_with_rate_limiter_when_invalid_thickness_is_given(
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
        workflow_definition=WORKFLOW_WITH_REFERENCE_PATH_VISUALIZATION,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    with pytest.raises(RuntimeInputError):
        _ = execution_engine.run(
            runtime_parameters={
                "image": [dogs_image],
                "reference_path": [(100, 300), (110, 310), (120, 320), (130, 330)],
                "thickness": 0,
            }
        )
