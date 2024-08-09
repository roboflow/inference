from unittest import mock
from unittest.mock import MagicMock

import pytest

from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.errors import WorkflowSyntaxError
from inference.core.workflows.execution_engine.introspection import blocks_loader
from inference.core.workflows.execution_engine.v1.compiler.core import compile_workflow
from inference.core.workflows.execution_engine.v1.core import (
    EXECUTION_ENGINE_V1_VERSION,
)

WORKFLOW_WITH_INCOMPATIBLE_STEP = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
    ],
    "steps": [
        {
            "type": "IncompatibleBlock",
            "name": "some",
            "images": "$inputs.image",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "some",
            "selector": "$steps.some.some",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_compilation_of_cyclic_workflow(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
) -> None:
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.compilation.stub_plugins.plugin_with_blocks_not_compatible_with_ee_version"
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }

    # when
    with pytest.raises(WorkflowSyntaxError) as error:
        _ = compile_workflow(
            workflow_definition=WORKFLOW_WITH_INCOMPATIBLE_STEP,
            init_parameters=workflow_init_parameters,
            execution_engine_version=EXECUTION_ENGINE_V1_VERSION,
        )

    # then
    assert (
        "Input tag 'IncompatibleBlock' found using 'type' does not match any of the expected tags"
        in str(error.value.inner_error)
    )
