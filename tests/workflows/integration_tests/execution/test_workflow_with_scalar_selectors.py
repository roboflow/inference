from unittest import mock
from unittest.mock import MagicMock

import numpy as np

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.core.workflows.execution_engine.introspection import blocks_loader

NON_BATCH_SECRET_STORE_WORKFLOW = {
    "version": "1.3.0",
    "inputs": [{"type": "WorkflowImage", "name": "image"}],
    "steps": [
        {
            "type": "secret_store",
            "name": "secret",
        },
        {
            "type": "secret_store_user",
            "name": "user",
            "image": "$inputs.image",
            "secret": "$steps.secret.secret",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "result",
            "selector": "$steps.user.output",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_workflow_with_scalar_selectors_for_batch_of_images(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.execution.stub_plugins.secret_store_plugin",
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=NON_BATCH_SECRET_STORE_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [dogs_image, dogs_image],
        }
    )

    # then
    assert len(result) == 2
    assert (
        result[0]["result"] == "my_secret"
    ), "Expected secret store value propagated into output"
    assert (
        result[1]["result"] == "my_secret"
    ), "Expected secret store value propagated into output"


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_workflow_with_scalar_selectors_for_single_image(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.execution.stub_plugins.secret_store_plugin",
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=NON_BATCH_SECRET_STORE_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [dogs_image],
        }
    )

    # then
    assert len(result) == 1
    assert (
        result[0]["result"] == "my_secret"
    ), "Expected secret store value propagated into output"


BATCH_SECRET_STORE_WORKFLOW = {
    "version": "1.3.0",
    "inputs": [{"type": "WorkflowImage", "name": "image"}],
    "steps": [
        {
            "type": "batch_secret_store",
            "name": "secret",
            "image": "$inputs.image",
        },
        {
            "type": "non_batch_secret_store_user",
            "name": "user",
            "image": "$inputs.image",
            "secret": "$steps.secret.secret",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "result",
            "selector": "$steps.user.output",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_workflow_with_batch_oriented_secret_store_for_batch_of_images(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.execution.stub_plugins.secret_store_plugin",
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=BATCH_SECRET_STORE_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [dogs_image, dogs_image],
        }
    )

    # then
    assert len(result) == 2
    assert result[0]["result"].startswith(
        "my_secret"
    ), "Expected secret store value propagated into output"
    assert result[1]["result"].startswith(
        "my_secret"
    ), "Expected secret store value propagated into output"
    assert (
        result[0]["result"] != result[1]["result"]
    ), "Expected different results for both outputs, as feature store should fire twice for two input images"
