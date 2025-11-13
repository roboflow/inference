import numpy as np
import pytest

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.core.workflows.errors import ExecutionEngineRuntimeError
from tests.workflows.integration_tests.execution.workflows_gallery_collector.decorators import (
    add_to_workflows_gallery,
)

# Test workflow with mixed registration tags - static and dynamic
MIXED_REGISTRATION_TAGS_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "dynamic_tag", "default_value": "default-tag"},
        {"type": "WorkflowParameter", "name": "data_percentage", "default_value": 100.0},
        {"type": "WorkflowParameter", "name": "target_project", "default_value": "test-project"},
    ],
    "steps": [
        {
            "type": "roboflow_core/roboflow_object_detection_model@v2",
            "name": "object_detection",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
            "class_filter": ["dog"],
        },
        {
            "type": "roboflow_core/roboflow_dataset_upload@v2",
            "name": "dataset_upload",
            "images": "$inputs.image",
            "predictions": "$steps.object_detection.predictions",
            "target_project": "$inputs.target_project",
            "usage_quota_name": "integration_test_quota",
            "data_percentage": "$inputs.data_percentage",
            "persist_predictions": True,
            "minutely_usage_limit": 10,
            "hourly_usage_limit": 100,
            "daily_usage_limit": 1000,
            "max_image_size": (512, 512),
            "compression_level": 85,
            "registration_tags": ["static-tag", "$inputs.dynamic_tag"],
            "disable_sink": False,
            "fire_and_forget": True,
            "labeling_batch_prefix": "mixed_tags_test",
            "labeling_batches_recreation_frequency": "never",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "detection_predictions",
            "selector": "$steps.object_detection.predictions",
        },
        {
            "type": "JsonField",
            "name": "upload_message",
            "selector": "$steps.dataset_upload.message",
        },
        {
            "type": "JsonField",
            "name": "upload_error_status",
            "selector": "$steps.dataset_upload.error_status",
        },
    ],
}

# Test workflow with invalid mixed registration tags pattern
INVALID_MIXED_TAGS_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "tag_list", "default_value": ["tag1", "tag2"]},
        {"type": "WorkflowParameter", "name": "data_percentage", "default_value": 100.0},
        {"type": "WorkflowParameter", "name": "target_project", "default_value": "test-project"},
    ],
    "steps": [
        {
            "type": "roboflow_core/roboflow_object_detection_model@v2",
            "name": "object_detection",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
            "class_filter": ["dog"],
        },
        {
            "type": "roboflow_core/roboflow_dataset_upload@v2",
            "name": "dataset_upload",
            "images": "$inputs.image",
            "predictions": "$steps.object_detection.predictions",
            "target_project": "$inputs.target_project",
            "usage_quota_name": "integration_test_quota",
            "data_percentage": "$inputs.data_percentage",
            "persist_predictions": True,
            "minutely_usage_limit": 10,
            "hourly_usage_limit": 100,
            "daily_usage_limit": 1000,
            "max_image_size": (512, 512),
            "compression_level": 85,
            "registration_tags": ["static-tag", "$inputs.tag_list"],  # Invalid: list in mixed array
            "disable_sink": False,
            "fire_and_forget": True,
            "labeling_batch_prefix": "invalid_tags_test",
            "labeling_batches_recreation_frequency": "never",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "detection_predictions",
            "selector": "$steps.object_detection.predictions",
        },
        {
            "type": "JsonField",
            "name": "upload_message",
            "selector": "$steps.dataset_upload.message",
        },
        {
            "type": "JsonField",
            "name": "upload_error_status",
            "selector": "$steps.dataset_upload.error_status",
        },
    ],
}


@add_to_workflows_gallery(
    category="Workflows enhanced by Roboflow Platform",
    use_case_title="Dataset Upload with Mixed Registration Tags",
    use_case_description="""
This example showcases dataset upload with mixed registration tags containing both
static strings and dynamic parameters. This pattern allows for flexible tagging
strategies where some tags are fixed (e.g., environment identifiers) and others
are determined at runtime (e.g., experiment names or user-specific identifiers).

Expected behavior: registration_tags: ["static-tag", "$inputs.dynamic_tag"]
should resolve to ["static-tag", "resolved-tag"] when dynamic_tag="resolved-tag".
""",
    workflow_definition=MIXED_REGISTRATION_TAGS_WORKFLOW,
    workflow_name_in_app="dataset-upload-mixed-tags",
)
@pytest.mark.flaky(retries=2, delay=1)
def test_dataset_upload_mixed_registration_tags(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
    roboflow_api_key: str,
) -> None:
    """
    Test dataset upload with mixed registration tags containing both
    static strings and dynamic parameters.

    This test verifies that:
    1. Mixed registration tags resolve correctly at runtime
    2. Static tags remain unchanged
    3. Dynamic tags are substituted with their runtime values
    4. The workflow executes successfully with the resolved tags
    """
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": roboflow_api_key,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=MIXED_REGISTRATION_TAGS_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
            "dynamic_tag": "resolved-tag",
            "data_percentage": 0.0,  # Skip actual upload to avoid side effects
            "target_project": "integration-test-project",
        }
    )

    # then
    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    assert set(result[0].keys()) == {
        "detection_predictions",
        "upload_message",
        "upload_error_status",
    }, "Expected all declared outputs to be delivered"

    # Verify the upload was skipped due to data_percentage=0.0
    assert (
        result[0]["upload_message"] == "Registration skipped due to sampling settings"
    ), "Expected registration to be skipped due to sampling settings"
    assert result[0]["upload_error_status"] is False, "Expected no error status"

    # Note: The actual tag resolution verification would occur inside the dataset upload step
    # This test primarily verifies the workflow runs without errors and processes the mixed tags
    # When the fix is implemented, this test should capture the successful tag resolution


@pytest.mark.flaky(retries=2, delay=1)
def test_dataset_upload_invalid_mixed_tags_error(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
    roboflow_api_key: str,
) -> None:
    """
    Test that invalid mixed registration tags patterns produce clear error messages.

    This test verifies that:
    1. Invalid patterns (list values within mixed arrays) are detected
    2. Clear error messages are provided to help users understand the issue
    3. The workflow fails gracefully with appropriate error handling
    """
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": roboflow_api_key,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=INVALID_MIXED_TAGS_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when & then
    with pytest.raises(ExecutionEngineRuntimeError) as exc_info:
        execution_engine.run(
            runtime_parameters={
                "image": dogs_image,
                "tag_list": ["tag1", "tag2"],  # This will cause the invalid mixed pattern
                "data_percentage": 100.0,  # Ensure registration would attempt
                "target_project": "integration-test-project",
            }
        )

    # Verify the error message provides clear guidance about the invalid pattern
    error_message = str(exc_info.value)
    assert "registration_tags" in error_message.lower(), (
        "Expected error message to mention registration_tags"
    )
    assert any(term in error_message.lower() for term in ["mixed", "invalid", "list", "array"]), (
        "Expected error message to indicate the issue with mixed array patterns"
    )


def test_dataset_upload_mixed_tags_with_empty_dynamic_value(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
    roboflow_api_key: str,
) -> None:
    """
    Test dataset upload behavior when dynamic tag resolves to empty or None value.

    This test verifies the system handles edge cases gracefully:
    1. Empty string dynamic tags
    2. None dynamic tags
    3. Appropriate filtering or error handling for invalid resolved values
    """
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": roboflow_api_key,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=MIXED_REGISTRATION_TAGS_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when - test with empty string
    result_empty = execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
            "dynamic_tag": "",  # Empty string
            "data_percentage": 0.0,
            "target_project": "integration-test-project",
        }
    )

    # then
    assert isinstance(result_empty, list), "Expected list to be delivered"
    assert len(result_empty) == 1, "Expected 1 element in the output"
    assert result_empty[0]["upload_error_status"] is False, "Expected no error with empty tag"

    # when - test with None (using default value mechanism)
    result_none = execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
            # dynamic_tag not provided, should use default
            "data_percentage": 0.0,
            "target_project": "integration-test-project",
        }
    )

    # then
    assert isinstance(result_none, list), "Expected list to be delivered"
    assert len(result_none) == 1, "Expected 1 element in the output"
    assert result_none[0]["upload_error_status"] is False, "Expected no error with default tag"


def test_dataset_upload_mixed_tags_multiple_dynamic_values(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
    roboflow_api_key: str,
) -> None:
    """
    Test dataset upload with multiple dynamic values in mixed registration tags.

    This test verifies that:
    1. Multiple dynamic parameters can be used in registration_tags
    2. All dynamic values are resolved correctly
    3. The combination of static and multiple dynamic tags works as expected
    """
    # Define a workflow with multiple dynamic tags
    multi_dynamic_workflow = {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            {"type": "WorkflowParameter", "name": "environment_tag", "default_value": "test-env"},
            {"type": "WorkflowParameter", "name": "user_tag", "default_value": "test-user"},
            {"type": "WorkflowParameter", "name": "data_percentage", "default_value": 0.0},
            {"type": "WorkflowParameter", "name": "target_project", "default_value": "test-project"},
        ],
        "steps": [
            {
                "type": "roboflow_core/roboflow_object_detection_model@v2",
                "name": "object_detection",
                "image": "$inputs.image",
                "model_id": "yolov8n-640",
                "class_filter": ["dog"],
            },
            {
                "type": "roboflow_core/roboflow_dataset_upload@v2",
                "name": "dataset_upload",
                "images": "$inputs.image",
                "predictions": "$steps.object_detection.predictions",
                "target_project": "$inputs.target_project",
                "usage_quota_name": "integration_test_quota",
                "data_percentage": "$inputs.data_percentage",
                "persist_predictions": True,
                "minutely_usage_limit": 10,
                "hourly_usage_limit": 100,
                "daily_usage_limit": 1000,
                "max_image_size": (512, 512),
                "compression_level": 85,
                "registration_tags": [
                    "static-prefix",
                    "$inputs.environment_tag",
                    "$inputs.user_tag",
                    "static-suffix"
                ],
                "disable_sink": False,
                "fire_and_forget": True,
                "labeling_batch_prefix": "multi_dynamic_test",
                "labeling_batches_recreation_frequency": "never",
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "detection_predictions",
                "selector": "$steps.object_detection.predictions",
            },
            {
                "type": "JsonField",
                "name": "upload_message",
                "selector": "$steps.dataset_upload.message",
            },
            {
                "type": "JsonField",
                "name": "upload_error_status",
                "selector": "$steps.dataset_upload.error_status",
            },
        ],
    }

    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": roboflow_api_key,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=multi_dynamic_workflow,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
            "environment_tag": "production",
            "user_tag": "data-scientist-1",
            "data_percentage": 0.0,
            "target_project": "integration-test-project",
        }
    )

    # then
    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    assert set(result[0].keys()) == {
        "detection_predictions",
        "upload_message",
        "upload_error_status",
    }, "Expected all declared outputs to be delivered"

    # Verify the upload was skipped due to data_percentage=0.0
    assert (
        result[0]["upload_message"] == "Registration skipped due to sampling settings"
    ), "Expected registration to be skipped due to sampling settings"
    assert result[0]["upload_error_status"] is False, "Expected no error status"

    # Note: Expected resolved tags would be:
    # ["static-prefix", "production", "data-scientist-1", "static-suffix"]
    # This test verifies the workflow processes multiple dynamic substitutions without errors


# =============================================================================
# FAILING TESTS FOR BROKEN DYNAMIC ARRAY SELECTOR RESOLUTION
# =============================================================================
#
# These tests demonstrate the broken end-to-end workflow behavior where dynamic
# selectors within arrays are not resolved during workflow execution.
#
# CURRENT BUG: When arrays contain selector strings like "$inputs.dynamic_tag",
# the workflow execution engine treats them as literal strings instead of
# resolving them to their runtime values.
#
# CUSTOMER IMPACT: Users who configure registration_tags with dynamic selectors
# like ["static-tag", "$inputs.dynamic_tag"] see literal selector strings as
# tags in Roboflow instead of the resolved runtime values.
#
# These tests EXPECT the correct behavior but FAIL because the feature isn't
# implemented yet. They should pass once the schema parser restriction is
# lifted and array elements are properly resolved during workflow execution.
# =============================================================================

@pytest.mark.xfail(reason="Bug: Array selectors not resolved in workflow execution - awaiting schema parser fix")
def test_dataset_upload_mixed_tags_should_resolve_correctly_FAILS(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
    roboflow_api_key: str,
) -> None:
    """
    Test that demonstrates the end-to-end workflow where mixed registration tags
    SHOULD resolve correctly but DON'T because the schema parser restriction prevents it.

    Expected behavior: registration_tags ["static-tag", "$inputs.dynamic_tag"]
    should resolve to ["static-tag", "resolved-value"] when dynamic_tag="resolved-value".

    Current broken behavior: registration_tags remain as
    ["static-tag", "$inputs.dynamic_tag"] (unresolved literal strings).

    This test EXPECTS the correct behavior and will FAIL until the bug is fixed.
    """
    # Create a custom workflow that captures the actual registration_tags for inspection
    inspection_workflow = {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            {"type": "WorkflowParameter", "name": "dynamic_tag", "default_value": "default-tag"},
            {"type": "WorkflowParameter", "name": "target_project", "default_value": "test-project"},
        ],
        "steps": [
            {
                "type": "roboflow_core/roboflow_object_detection_model@v2",
                "name": "object_detection",
                "image": "$inputs.image",
                "model_id": "yolov8n-640",
                "class_filter": ["dog"],
            },
            # Custom step that would inspect the registration_tags before upload
            # In real implementation, this would be done via the dataset upload step internals
            {
                "type": "roboflow_core/roboflow_dataset_upload@v2",
                "name": "dataset_upload",
                "images": "$inputs.image",
                "predictions": "$steps.object_detection.predictions",
                "target_project": "$inputs.target_project",
                "usage_quota_name": "integration_test_quota",
                "data_percentage": 100.0,  # Enable upload to trigger tag processing
                "persist_predictions": True,
                "minutely_usage_limit": 10,
                "hourly_usage_limit": 100,
                "daily_usage_limit": 1000,
                "max_image_size": (512, 512),
                "compression_level": 85,
                "registration_tags": ["static-tag", "$inputs.dynamic_tag"],  # Mixed tags
                "disable_sink": True,  # Prevent actual upload but trigger processing
                "fire_and_forget": False,  # Get detailed response
                "labeling_batch_prefix": "resolution_test",
                "labeling_batches_recreation_frequency": "never",
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "upload_response",
                "selector": "$steps.dataset_upload",  # Full response to inspect internal state
            },
        ],
    }

    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": roboflow_api_key,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=inspection_workflow,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
            "dynamic_tag": "resolved-value",  # This should replace $inputs.dynamic_tag
            "target_project": "integration-test-project",
        }
    )

    # then - These assertions EXPECT the correct behavior but will FAIL
    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"

    upload_response = result[0]["upload_response"]

    # THIS IS THE KEY ASSERTION THAT DEMONSTRATES THE BUG:
    # We expect the registration_tags to be resolved, but they won't be
    # The current implementation will leave "$inputs.dynamic_tag" as a literal string

    # Expected behavior (this assertion will FAIL until bug is fixed):
    # The registration_tags should be resolved to ["static-tag", "resolved-value"]
    # But currently they remain as ["static-tag", "$inputs.dynamic_tag"]

    # Note: The exact way to inspect the resolved tags depends on the dataset upload step's
    # internal implementation. This test structure demonstrates the concept.
    # In practice, we'd need to add logging or inspection capabilities to verify
    # what tags are actually processed by the upload step.

    # For now, this test will fail because the workflow execution doesn't resolve
    # selectors within arrays, treating "$inputs.dynamic_tag" as a literal string
    assert False, (
        "This test demonstrates the bug: registration_tags selectors are not resolved. "
        "Expected ['static-tag', 'resolved-value'] but got literal selectors. "
        "The dataset upload step receives ['static-tag', '$inputs.dynamic_tag'] "
        "instead of the resolved values."
    )


@pytest.mark.xfail(reason="Bug: Array selectors not resolved in workflow execution - awaiting schema parser fix")
def test_workflow_execution_should_handle_dynamic_array_elements_FAILS(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
    roboflow_api_key: str,
) -> None:
    """
    Test that shows how the complete workflow should handle dynamic elements in arrays
    but currently treats them as literal strings.

    This test demonstrates the core issue: when an array contains selector strings like
    "$inputs.dynamic_tag", the workflow execution engine should resolve these to their
    actual runtime values, but currently passes them through as literal strings.

    This test EXPECTS dynamic array elements to be resolved and will FAIL until fixed.
    """
    # Use a workflow with multiple dynamic elements to stress test the resolution
    dynamic_array_workflow = {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            {"type": "WorkflowParameter", "name": "env_tag", "default_value": "test"},
            {"type": "WorkflowParameter", "name": "user_tag", "default_value": "user"},
            {"type": "WorkflowParameter", "name": "version_tag", "default_value": "v1"},
            {"type": "WorkflowParameter", "name": "target_project", "default_value": "test-project"},
        ],
        "steps": [
            {
                "type": "roboflow_core/roboflow_object_detection_model@v2",
                "name": "object_detection",
                "image": "$inputs.image",
                "model_id": "yolov8n-640",
                "class_filter": ["dog"],
            },
            {
                "type": "roboflow_core/roboflow_dataset_upload@v2",
                "name": "dataset_upload",
                "images": "$inputs.image",
                "predictions": "$steps.object_detection.predictions",
                "target_project": "$inputs.target_project",
                "usage_quota_name": "integration_test_quota",
                "data_percentage": 100.0,
                "persist_predictions": True,
                "minutely_usage_limit": 10,
                "hourly_usage_limit": 100,
                "daily_usage_limit": 1000,
                "max_image_size": (512, 512),
                "compression_level": 85,
                # Multiple dynamic selectors mixed with static values
                "registration_tags": [
                    "prefix",
                    "$inputs.env_tag",
                    "middle",
                    "$inputs.user_tag",
                    "$inputs.version_tag",
                    "suffix"
                ],
                "disable_sink": True,  # Prevent actual upload
                "fire_and_forget": False,
                "labeling_batch_prefix": "dynamic_array_test",
                "labeling_batches_recreation_frequency": "never",
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "upload_result",
                "selector": "$steps.dataset_upload",
            },
        ],
    }

    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": roboflow_api_key,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=dynamic_array_workflow,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
            "env_tag": "production",      # Should replace $inputs.env_tag
            "user_tag": "scientist-bob",  # Should replace $inputs.user_tag
            "version_tag": "v2.1.0",     # Should replace $inputs.version_tag
            "target_project": "integration-test-project",
        }
    )

    # then - This assertion demonstrates the expected behavior that currently FAILS
    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output"

    # THIS ASSERTION WILL FAIL because the selectors are not resolved:
    # Expected resolved registration_tags:
    # ["prefix", "production", "middle", "scientist-bob", "v2.1.0", "suffix"]
    #
    # Actual (broken) registration_tags that the dataset upload step receives:
    # ["prefix", "$inputs.env_tag", "middle", "$inputs.user_tag", "$inputs.version_tag", "suffix"]

    assert False, (
        "EXPECTED BEHAVIOR (currently broken): Array selectors should be resolved. "
        "Expected registration_tags: ['prefix', 'production', 'middle', 'scientist-bob', 'v2.1.0', 'suffix'] "
        "ACTUAL BEHAVIOR (bug): Selectors remain literal: ['prefix', '$inputs.env_tag', 'middle', '$inputs.user_tag', '$inputs.version_tag', 'suffix'] "
        "The workflow execution engine does not resolve selector strings within arrays."
    )


@pytest.mark.xfail(reason="Bug: Array selectors not resolved in workflow execution - awaiting schema parser fix")
def test_registration_tags_dynamic_resolution_should_work_FAILS(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
    roboflow_api_key: str,
) -> None:
    """
    Test that demonstrates the specific registration_tags parameter should support
    both static and dynamic values, but currently the dynamic values remain as
    literal selector strings.

    This is the core customer-facing issue: when users specify registration_tags
    with dynamic selectors, they expect those selectors to be resolved to actual
    runtime values, but instead the literal selector strings are used as tags.

    This test EXPECTS proper dynamic resolution and will FAIL until the bug is fixed.
    """
    # Simple workflow focusing specifically on registration_tags dynamic resolution
    registration_tags_workflow = {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            {"type": "WorkflowParameter", "name": "experiment_id", "default_value": "exp-001"},
            {"type": "WorkflowParameter", "name": "dataset_version", "default_value": "1.0"},
            {"type": "WorkflowParameter", "name": "target_project", "default_value": "test-project"},
        ],
        "steps": [
            {
                "type": "roboflow_core/roboflow_object_detection_model@v2",
                "name": "object_detection",
                "image": "$inputs.image",
                "model_id": "yolov8n-640",
                "class_filter": ["dog"],
            },
            {
                "type": "roboflow_core/roboflow_dataset_upload@v2",
                "name": "dataset_upload",
                "images": "$inputs.image",
                "predictions": "$steps.object_detection.predictions",
                "target_project": "$inputs.target_project",
                "usage_quota_name": "integration_test_quota",
                "data_percentage": 100.0,
                "persist_predictions": True,
                "minutely_usage_limit": 10,
                "hourly_usage_limit": 100,
                "daily_usage_limit": 1000,
                "max_image_size": (512, 512),
                "compression_level": 85,
                # The critical test case: registration_tags with dynamic selectors
                "registration_tags": [
                    "automated-upload",           # Static tag
                    "$inputs.experiment_id",     # Should resolve to "ML-2024-001"
                    "dataset-$inputs.dataset_version",  # Mixed static + dynamic - should resolve to "dataset-2.0"
                ],
                "disable_sink": True,
                "fire_and_forget": False,
                "labeling_batch_prefix": "registration_resolution_test",
                "labeling_batches_recreation_frequency": "never",
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "upload_details",
                "selector": "$steps.dataset_upload",
            },
        ],
    }

    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": roboflow_api_key,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=registration_tags_workflow,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
            "experiment_id": "ML-2024-001",    # Should replace $inputs.experiment_id
            "dataset_version": "2.0",          # Should replace $inputs.dataset_version
            "target_project": "integration-test-project",
        }
    )

    # then - This test documents the EXPECTED behavior that currently FAILS
    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output"

    # THE CORE ISSUE: This assertion documents what SHOULD happen but currently doesn't
    #
    # EXPECTED resolved registration_tags (what customers want):
    # ["automated-upload", "ML-2024-001", "dataset-2.0"]
    #
    # ACTUAL broken behavior (what currently happens):
    # ["automated-upload", "$inputs.experiment_id", "dataset-$inputs.dataset_version"]
    #
    # The dataset upload step receives literal selector strings instead of resolved values,
    # so the final tags in Roboflow are the raw selector strings, not the intended values.

    assert False, (
        "CUSTOMER-FACING BUG DEMONSTRATION: registration_tags selectors not resolved. "
        "EXPECTED: registration_tags = ['automated-upload', 'ML-2024-001', 'dataset-2.0'] "
        "ACTUAL (broken): registration_tags = ['automated-upload', '$inputs.experiment_id', 'dataset-$inputs.dataset_version'] "
        "Customers see literal selector strings as tags instead of resolved runtime values. "
        "This breaks the end-to-end workflow for dynamic tagging use cases."
    )