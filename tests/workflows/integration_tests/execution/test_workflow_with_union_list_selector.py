"""Integration tests for Union[List[...], Selector(...)] patterns."""

import numpy as np
import pytest

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine

WORKFLOW_WITH_SELECTOR_TO_LIST = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {
            "type": "WorkflowParameter",
            "name": "classes_to_consider",
        },
    ],
    "steps": [
        {
            "type": "RoboflowObjectDetectionModel",
            "name": "detection",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
        },
        {
            "type": "DetectionsConsensus",
            "name": "consensus",
            "predictions_batches": ["$steps.detection.predictions"],
            "required_votes": 1,
            "classes_to_consider": "$inputs.classes_to_consider",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "predictions",
            "selector": "$steps.consensus.predictions",
        }
    ],
}


WORKFLOW_WITH_LITERAL_LIST = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
    ],
    "steps": [
        {
            "type": "RoboflowObjectDetectionModel",
            "name": "detection",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
        },
        {
            "type": "DetectionsConsensus",
            "name": "consensus",
            "predictions_batches": ["$steps.detection.predictions"],
            "required_votes": 1,
            "classes_to_consider": ["person"],
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "predictions",
            "selector": "$steps.consensus.predictions",
        }
    ],
}


def test_union_list_selector_with_selector_to_list(
    model_manager: ModelManager,
    crowd_image: np.ndarray,
) -> None:
    """Test Union[List[str], Selector(...)] when using a selector to a list."""
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_SELECTOR_TO_LIST,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": crowd_image,
            "classes_to_consider": ["person"],
        }
    )

    # then
    assert isinstance(result, list), "Expected list of results"
    assert len(result) == 1, "Expected single result"
    assert "predictions" in result[0], "Expected predictions in output"
    # Verify that the selector was properly resolved and the workflow executed


def test_union_list_selector_with_literal_list(
    model_manager: ModelManager,
    crowd_image: np.ndarray,
) -> None:
    """Test Union[List[str], Selector(...)] when using a literal list."""
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_LITERAL_LIST,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": crowd_image,
        }
    )

    # then
    assert isinstance(result, list), "Expected list of results"
    assert len(result) == 1, "Expected single result"
    assert "predictions" in result[0], "Expected predictions in output"
    # Verify that the literal list was properly handled


def test_union_list_selector_validates_type_mismatch(
    model_manager: ModelManager,
    crowd_image: np.ndarray,
) -> None:
    """Test that type validation catches invalid selector resolution."""
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_SELECTOR_TO_LIST,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when/then - passing a string instead of a list should fail validation
    with pytest.raises(Exception):  # Should raise validation error
        execution_engine.run(
            runtime_parameters={
                "image": crowd_image,
                "classes_to_consider": "person",  # String instead of list
            }
        )
