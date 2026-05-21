from unittest import mock
from unittest.mock import MagicMock

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.sinks.roboflow.edit_image_metadata import v1
from inference.core.workflows.execution_engine.core import ExecutionEngine

WORKFLOW_WITH_EDIT_IMAGE_METADATA = {
    "version": "1.3.0",
    "inputs": [
        {"type": "WorkflowBatchInput", "name": "source_id", "kind": ["string"]},
        {"type": "WorkflowParameter", "name": "location"},
        {"type": "WorkflowParameter", "name": "extra_tag"},
    ],
    "steps": [
        {
            "type": "roboflow_core/edit_image_metadata@v1",
            "name": "edit_metadata",
            "source_id": "$inputs.source_id",
            "metadata": {"location": "$inputs.location"},
            "tags": ["$inputs.extra_tag"],
            "disable_sink": False,
        }
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "message",
            "selector": "$steps.edit_metadata.message",
        }
    ],
}


@mock.patch.object(v1, "get_workspace_name")
@mock.patch.object(v1, "batch_update_image_metadata_at_roboflow")
def test_workflow_with_edit_image_metadata_auto_batches_scalar_metadata_and_tags(
    batch_update_image_metadata_at_roboflow_mock: MagicMock,
    get_workspace_name_mock: MagicMock,
    model_manager: ModelManager,
) -> None:
    # given
    get_workspace_name_mock.return_value = "my-workspace"
    batch_update_image_metadata_at_roboflow_mock.return_value = {"taskId": "task-123"}
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": "my_api_key",
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_EDIT_IMAGE_METADATA,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "source_id": ["img-1", "img-2"],
            "location": "warehouse_a",
            "extra_tag": "auto-labeled",
        }
    )

    # then
    assert len(result) == 2
    assert result[0]["message"] == "Metadata updated"
    assert result[1]["message"] == "Metadata updated"
    batch_update_image_metadata_at_roboflow_mock.assert_called_once_with(
        api_key="my_api_key",
        workspace_id="my-workspace",
        updates=[
            {
                "imageId": "img-1",
                "metadata": {"location": "warehouse_a"},
                "addTags": ["auto-labeled"],
            },
            {
                "imageId": "img-2",
                "metadata": {"location": "warehouse_a"},
                "addTags": ["auto-labeled"],
            },
        ],
    )


WORKFLOW_WITH_PER_ROW_METADATA_AND_TAGS = {
    "version": "1.3.0",
    "inputs": [
        {"type": "WorkflowBatchInput", "name": "source_id", "kind": ["string"]},
        {"type": "WorkflowBatchInput", "name": "per_row_metadata", "kind": ["dictionary"]},
        {"type": "WorkflowBatchInput", "name": "per_row_tags", "kind": ["list_of_values"]},
    ],
    "steps": [
        {
            "type": "roboflow_core/edit_image_metadata@v1",
            "name": "edit_metadata",
            "source_id": "$inputs.source_id",
            "metadata": "$inputs.per_row_metadata",
            "tags": "$inputs.per_row_tags",
            "disable_sink": False,
        }
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "message",
            "selector": "$steps.edit_metadata.message",
        }
    ],
}


@mock.patch.object(v1, "get_workspace_name")
@mock.patch.object(v1, "batch_update_image_metadata_at_roboflow")
def test_workflow_with_edit_image_metadata_accepts_per_row_metadata_and_tags(
    batch_update_image_metadata_at_roboflow_mock: MagicMock,
    get_workspace_name_mock: MagicMock,
    model_manager: ModelManager,
) -> None:
    # given
    get_workspace_name_mock.return_value = "my-workspace"
    batch_update_image_metadata_at_roboflow_mock.return_value = {"taskId": "task-123"}
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": "my_api_key",
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_PER_ROW_METADATA_AND_TAGS,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "source_id": ["img-1", "img-2"],
            "per_row_metadata": [{"color": "red"}, {"color": "blue"}],
            "per_row_tags": [["a"], ["b", "c"]],
        }
    )

    # then
    assert len(result) == 2
    assert result[0]["message"] == "Metadata updated"
    assert result[1]["message"] == "Metadata updated"
    batch_update_image_metadata_at_roboflow_mock.assert_called_once_with(
        api_key="my_api_key",
        workspace_id="my-workspace",
        updates=[
            {"imageId": "img-1", "metadata": {"color": "red"}, "addTags": ["a"]},
            {"imageId": "img-2", "metadata": {"color": "blue"}, "addTags": ["b", "c"]},
        ],
    )


WORKFLOW_WITH_INLINE_SCALAR_SELECTORS = {
    "version": "1.3.0",
    "inputs": [
        {"type": "WorkflowBatchInput", "name": "source_id", "kind": ["string"]},
        {"type": "WorkflowParameter", "name": "camera_id"},
        {"type": "WorkflowParameter", "name": "dynamic_tag"},
    ],
    "steps": [
        {
            "type": "roboflow_core/edit_image_metadata@v1",
            "name": "edit_metadata",
            "source_id": "$inputs.source_id",
            "metadata": {"literal": "abc", "camera": "$inputs.camera_id"},
            "tags": ["static-tag", "$inputs.dynamic_tag"],
            "disable_sink": False,
        }
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "message",
            "selector": "$steps.edit_metadata.message",
        }
    ],
}


@mock.patch.object(v1, "get_workspace_name")
@mock.patch.object(v1, "batch_update_image_metadata_at_roboflow")
def test_workflow_with_edit_image_metadata_resolves_inline_scalar_selectors(
    batch_update_image_metadata_at_roboflow_mock: MagicMock,
    get_workspace_name_mock: MagicMock,
    model_manager: ModelManager,
) -> None:
    # given
    get_workspace_name_mock.return_value = "my-workspace"
    batch_update_image_metadata_at_roboflow_mock.return_value = {"taskId": "task-123"}
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": "my_api_key",
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_INLINE_SCALAR_SELECTORS,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "source_id": ["img-1", "img-2"],
            "camera_id": "cam-7",
            "dynamic_tag": "auto",
        }
    )

    # then
    assert len(result) == 2
    assert result[0]["message"] == "Metadata updated"
    assert result[1]["message"] == "Metadata updated"
    batch_update_image_metadata_at_roboflow_mock.assert_called_once_with(
        api_key="my_api_key",
        workspace_id="my-workspace",
        updates=[
            {
                "imageId": "img-1",
                "metadata": {"literal": "abc", "camera": "cam-7"},
                "addTags": ["static-tag", "auto"],
            },
            {
                "imageId": "img-2",
                "metadata": {"literal": "abc", "camera": "cam-7"},
                "addTags": ["static-tag", "auto"],
            },
        ],
    )
