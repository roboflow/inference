from unittest import mock
from unittest.mock import MagicMock

import numpy as np

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.sinks.roboflow.edit_image_metadata import v1
from inference.core.workflows.execution_engine.core import ExecutionEngine

WORKFLOW_WITH_EDIT_IMAGE_METADATA = {
    "version": "1.3.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowBatchInput", "name": "source_id", "kind": ["string"]},
        {"type": "WorkflowParameter", "name": "metadata"},
        {"type": "WorkflowParameter", "name": "tags"},
    ],
    "steps": [
        {
            "type": "roboflow_core/edit_image_metadata@v1",
            "name": "edit_metadata",
            "images": "$inputs.image",
            "source_id": "$inputs.source_id",
            "metadata": "$inputs.metadata",
            "tags": "$inputs.tags",
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
    dogs_image: np.ndarray,
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
            "image": [dogs_image, dogs_image],
            "source_id": ["img-1", "img-2"],
            "metadata": {"location": "warehouse_a"},
            "tags": ["auto-labeled"],
        }
    )

    # then
    assert len(result) == 2
    assert result[0]["message"] == "Submitted as async task task-123"
    assert result[1]["message"] == "Submitted as async task task-123"
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
