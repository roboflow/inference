from unittest import mock
from unittest.mock import MagicMock

import pytest

from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.errors import (
    BlockInitParameterNotProvidedError,
    BlockInterfaceError,
)
from inference.core.workflows.execution_engine.compiler.core import compile_workflow
from inference.core.workflows.execution_engine.introspection import blocks_loader

VALID_DEFINITION = {
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
            "class_filter": ["car"],
        },
        {
            "type": "Crop",
            "name": "crops",
            "image": "$inputs.image",
            "predictions": "$steps.detection.predictions",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "crops",
            "selector": "$steps.crops.crops",
        },
    ],
}


@pytest.mark.asyncio
async def test_compilation_of_workflow_where_required_init_parameter_are_not_delivered() -> (
    None
):
    # when
    with pytest.raises(BlockInitParameterNotProvidedError):
        _ = compile_workflow(
            workflow_definition=VALID_DEFINITION,
            init_parameters={},
        )


WORKFLOW_WITH_FAULTY_INIT_BLOCK = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
    ],
    "steps": [
        {
            "type": "FaultyInit",
            "name": "faulty",
        },
    ],
    "outputs": [],
}


@pytest.mark.asyncio
@mock.patch.object(blocks_loader, "get_plugin_modules")
async def test_compilation_of_workflow_where_block_init_is_faulty(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
) -> None:
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.compilation.stub_plugins.plugin_with_faulty_init"
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }

    # when
    with pytest.raises(BlockInterfaceError):
        _ = compile_workflow(
            workflow_definition=WORKFLOW_WITH_FAULTY_INIT_BLOCK,
            init_parameters=workflow_init_parameters,
        )
