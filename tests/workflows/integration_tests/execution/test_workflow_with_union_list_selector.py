"""Integration tests for Union[List[...], Selector(...)] patterns."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

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


# Tests for Union[List[Union[Selector(...), str]], Selector(...)] pattern
# This pattern allows mixed literals and selectors within a list
# These tests validate the actual selector resolution behavior

WORKFLOW_WITH_MIXED_LIST_LITERALS_AND_SELECTORS = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "dynamic_tag"},
    ],
    "steps": [
        {
            "type": "roboflow_core/roboflow_dataset_upload@v2",
            "name": "dataset_upload",
            "images": "$inputs.image",
            "target_project": "test-project",
            "usage_quota_name": "test_quota",
            "fire_and_forget": False,
            "disable_sink": True,
            "registration_tags": ["static-tag", "$inputs.dynamic_tag"],
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "message",
            "selector": "$steps.dataset_upload.message",
        }
    ],
}


WORKFLOW_WITH_SELECTOR_TO_LIST_OF_TAGS = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "tags"},
    ],
    "steps": [
        {
            "type": "roboflow_core/roboflow_dataset_upload@v2",
            "name": "dataset_upload",
            "images": "$inputs.image",
            "target_project": "test-project",
            "usage_quota_name": "test_quota",
            "fire_and_forget": False,
            "disable_sink": True,
            "registration_tags": "$inputs.tags",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "message",
            "selector": "$steps.dataset_upload.message",
        }
    ],
}


WORKFLOW_WITH_ALL_LITERAL_TAGS = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
    ],
    "steps": [
        {
            "type": "roboflow_core/roboflow_dataset_upload@v2",
            "name": "dataset_upload",
            "images": "$inputs.image",
            "target_project": "test-project",
            "usage_quota_name": "test_quota",
            "fire_and_forget": False,
            "disable_sink": True,
            "registration_tags": ["tag1", "tag2", "tag3"],
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "message",
            "selector": "$steps.dataset_upload.message",
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


@pytest.mark.xfail(
    reason="Bug: Selectors within arrays are not resolved. "
    "See schema_parser.py:368-370 which ignores selectors in list elements. "
    "This test will pass once the fix is implemented."
)
def test_registration_tags_with_mixed_literals_and_selectors(
    model_manager: ModelManager,
    crowd_image: np.ndarray,
) -> None:
    """Test Union[List[Union[Selector(...), str]], Selector(...)] with mixed literals and selectors in list.

    This test validates the core use case from the bug report:
    - registration_tags: ["static-tag", "$inputs.dynamic_tag"]
    - Should resolve to: ["static-tag", "resolved-value"]

    This tests that selectors WITHIN arrays are properly resolved by the execution engine.

    EXPECTED BEHAVIOR:
    - Input: registration_tags: ["static-tag", "$inputs.dynamic_tag"]
    - Runtime param: dynamic_tag = "resolved-value"
    - Expected result passed to block.run(): ["static-tag", "resolved-value"]

    CURRENT BEHAVIOR (BUG):
    - The selector "$inputs.dynamic_tag" is NOT resolved
    - Causes execution engine crash due to improper handling
    """
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_MIXED_LIST_LITERALS_AND_SELECTORS,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # Mock the actual upload to capture what registration_tags value is passed
    captured_tags = []

    with patch(
        "inference.core.workflows.core_steps.sinks.roboflow.dataset_upload.v2.RoboflowDatasetUploadBlockV2.run"
    ) as mock_run:
        # Configure mock to capture the registration_tags parameter and return success
        def capture_and_return(*args, **kwargs):
            captured_tags.append(kwargs.get("registration_tags"))
            return {"error_status": False, "message": "success"}

        mock_run.side_effect = capture_and_return

        # when - Execute workflow with runtime parameters
        result = execution_engine.run(
            runtime_parameters={
                "image": crowd_image,
                "dynamic_tag": "resolved-value",
            }
        )

    # then - Verify the selector was resolved correctly
    assert len(captured_tags) == 1, "Expected one call to the upload block"
    resolved_tags = captured_tags[0]

    # THIS IS THE KEY ASSERTION: verify that selectors within the array were resolved
    assert resolved_tags == ["static-tag", "resolved-value"], (
        f"Expected selectors within array to be resolved. "
        f"Got: {resolved_tags}, Expected: ['static-tag', 'resolved-value']"
    )


def test_registration_tags_with_selector_to_list(
    model_manager: ModelManager,
) -> None:
    """Test Union[List[Union[Selector(...), str]], Selector(...)] with selector to full list.

    This test validates that the workflow can be compiled with:
    - registration_tags: "$inputs.tags"
    - Where inputs.tags = ["tag1", "tag2", "tag3"]
    - Should be accepted by the schema parser
    """
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }

    # when - Initialize and compile the workflow
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_SELECTOR_TO_LIST_OF_TAGS,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # then - Verify workflow compiled successfully
    assert execution_engine is not None
    # This pattern should work: a selector that resolves to a full list


def test_registration_tags_with_all_literals(
    model_manager: ModelManager,
) -> None:
    """Test Union[List[Union[Selector(...), str]], Selector(...)] with all literal strings.

    This test validates that the workflow can be compiled with:
    - registration_tags: ["tag1", "tag2", "tag3"]
    - Should be accepted by the schema parser
    """
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }

    # when - Initialize and compile the workflow
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_ALL_LITERAL_TAGS,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # then - Verify workflow compiled successfully
    assert execution_engine is not None
    # This pattern should work: all literal strings with no selectors
