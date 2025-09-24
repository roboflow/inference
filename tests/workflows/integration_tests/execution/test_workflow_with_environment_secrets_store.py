import os
from unittest import mock
from unittest.mock import MagicMock

import numpy as np

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.core.workflows.execution_engine.introspection import blocks_loader
from tests.workflows.integration_tests.execution.conftest import (
    DUMMY_SECRET_ENV_VARIABLE,
)

WORKFLOW_EXPOSING_EXISTING_ENV_VARIABLE_TO_NON_BATCH_ORIENTED_STEP = {
    "version": "1.4.0",
    "inputs": [],
    "steps": [
        {
            "type": "roboflow_core/environment_secrets_store@v1",
            "name": "vault",
            "variables_storing_secrets": [DUMMY_SECRET_ENV_VARIABLE],
        },
        {
            "type": "block_accepting_scalars",
            "name": "scalars_block",
            "secret": f"$steps.vault.{DUMMY_SECRET_ENV_VARIABLE.lower()}",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "output",
            "selector": "$steps.scalars_block.output",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_feeding_existing_env_variable_into_scalar_oriented_step(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
) -> None:
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.execution.stub_plugins.plugin_testing_non_simd_step_with_optional_outputs"
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_EXPOSING_EXISTING_ENV_VARIABLE_TO_NON_BATCH_ORIENTED_STEP,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(runtime_parameters={})

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1, "Single image provided, so one output element expected"
    assert set(result[0].keys()) == {
        "output",
    }, "Expected all declared outputs to be delivered"
    assert result[0]["output"] == os.environ[DUMMY_SECRET_ENV_VARIABLE]


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_feeding_existing_env_variable_into_scalar_oriented_step_when_serialization_of_secret_output_expected(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
) -> None:
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.execution.stub_plugins.plugin_testing_non_simd_step_with_optional_outputs"
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_EXPOSING_EXISTING_ENV_VARIABLE_TO_NON_BATCH_ORIENTED_STEP,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(runtime_parameters={}, serialize_results=True)

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1, "Single image provided, so one output element expected"
    assert set(result[0].keys()) == {
        "output",
    }, "Expected all declared outputs to be delivered"
    assert result[0]["output"] is not None
    assert result[0]["output"] != os.environ[DUMMY_SECRET_ENV_VARIABLE]


WORKFLOW_EXPOSING_EXISTING_ENV_VARIABLE_TO_SIMD_STEP_ACCEPTING_BATCHES = {
    "version": "1.4.0",
    "inputs": [{"type": "WorkflowImage", "name": "image"}],
    "steps": [
        {
            "type": "roboflow_core/environment_secrets_store@v1",
            "name": "vault",
            "variables_storing_secrets": [DUMMY_SECRET_ENV_VARIABLE],
        },
        {
            "type": "block_accepting_batches_of_images",
            "name": "model",
            "image": "$inputs.image",
            "secret": f"$steps.vault.{DUMMY_SECRET_ENV_VARIABLE.lower()}",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "output",
            "selector": "$steps.model.output",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_feeding_existing_env_variable_into_simd_step_accepting_batches(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
    crowd_image: np.ndarray,
) -> None:
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.execution.stub_plugins.plugin_testing_non_simd_step_with_optional_outputs"
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_EXPOSING_EXISTING_ENV_VARIABLE_TO_SIMD_STEP_ACCEPTING_BATCHES,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={"image": [crowd_image, crowd_image]}
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 2, "Single image provided, so one output element expected"
    assert set(result[0].keys()) == {
        "output",
    }, "Expected all declared outputs to be delivered"
    assert set(result[1].keys()) == {
        "output",
    }, "Expected all declared outputs to be delivered"
    assert result[0]["output"] == "ok"
    assert result[1]["output"] == "ok"


WORKFLOW_EXPOSING_EXISTING_ENV_VARIABLE_TO_SIMD_STEP_NOT_ACCEPTING_BATCHES = {
    "version": "1.4.0",
    "inputs": [{"type": "WorkflowImage", "name": "image"}],
    "steps": [
        {
            "type": "roboflow_core/environment_secrets_store@v1",
            "name": "vault",
            "variables_storing_secrets": [DUMMY_SECRET_ENV_VARIABLE],
        },
        {
            "type": "block_accepting_images",
            "name": "model",
            "image": "$inputs.image",
            "secret": f"$steps.vault.{DUMMY_SECRET_ENV_VARIABLE.lower()}",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "output",
            "selector": "$steps.model.output",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_feeding_existing_env_variable_into_simd_step_not_accepting_batches(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
    crowd_image: np.ndarray,
) -> None:
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.execution.stub_plugins.plugin_testing_non_simd_step_with_optional_outputs"
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_EXPOSING_EXISTING_ENV_VARIABLE_TO_SIMD_STEP_NOT_ACCEPTING_BATCHES,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={"image": [crowd_image, crowd_image]}
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 2, "Single image provided, so one output element expected"
    assert set(result[0].keys()) == {
        "output",
    }, "Expected all declared outputs to be delivered"
    assert set(result[1].keys()) == {
        "output",
    }, "Expected all declared outputs to be delivered"
    assert result[0]["output"] == "ok"
    assert result[1]["output"] == "ok"


WORKFLOW_EXPOSING_NON_EXISTING_ENV_VARIABLE_TO_SCALAR_STEP = {
    "version": "1.4.0",
    "inputs": [],
    "steps": [
        {
            "type": "roboflow_core/environment_secrets_store@v1",
            "name": "vault",
            "variables_storing_secrets": ["NON_EXISTING_ENV_VARIABLE"],
        },
        {
            "type": "block_accepting_scalars",
            "name": "block",
            "secret": f"$steps.vault.non_existing_env_variable",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "output",
            "selector": "$steps.block.output",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_feeding_existing_env_variable_into_scalar_step_not_accepting_empty_inputs(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
) -> None:
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.execution.stub_plugins.plugin_testing_non_simd_step_with_optional_outputs"
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_EXPOSING_NON_EXISTING_ENV_VARIABLE_TO_SCALAR_STEP,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(runtime_parameters={})

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1, "Single image provided, so one output element expected"
    assert set(result[0].keys()) == {
        "output",
    }, "Expected all declared outputs to be delivered"
    assert result[0]["output"] is None


WORKFLOW_EXPOSING_NON_EXISTING_ENV_VARIABLE_TO_SCALAR_STEP_ACCEPTING_EMPTY_INPUTS = {
    "version": "1.4.0",
    "inputs": [],
    "steps": [
        {
            "type": "roboflow_core/environment_secrets_store@v1",
            "name": "vault",
            "variables_storing_secrets": ["NON_EXISTING_ENV_VARIABLE"],
        },
        {
            "type": "block_accepting_empty_scalars",
            "name": "block",
            "secret": f"$steps.vault.non_existing_env_variable",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "output",
            "selector": "$steps.block.output",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_feeding_existing_env_variable_into_scalar_step_accepting_empty_inputs(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
) -> None:
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.execution.stub_plugins.plugin_testing_non_simd_step_with_optional_outputs"
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_EXPOSING_NON_EXISTING_ENV_VARIABLE_TO_SCALAR_STEP_ACCEPTING_EMPTY_INPUTS,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(runtime_parameters={})

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1, "Single image provided, so one output element expected"
    assert set(result[0].keys()) == {
        "output",
    }, "Expected all declared outputs to be delivered"
    assert result[0]["output"] == "modified-secret"


WORKFLOW_EXPOSING_NON_EXISTING_ENV_VARIABLE_TO_SIMD_STEP_ACCEPTING_BATCHES = {
    "version": "1.4.0",
    "inputs": [{"type": "WorkflowImage", "name": "image"}],
    "steps": [
        {
            "type": "roboflow_core/environment_secrets_store@v1",
            "name": "vault",
            "variables_storing_secrets": ["NON_EXISTING_ENV_VARIABLE"],
        },
        {
            "type": "block_accepting_batches_of_images",
            "name": "model",
            "image": "$inputs.image",
            "secret": f"$steps.vault.non_existing_env_variable",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "output",
            "selector": "$steps.model.output",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_feeding_existing_env_variable_into_simd_step_accepting_batches(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
    crowd_image: np.ndarray,
) -> None:
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.execution.stub_plugins.plugin_testing_non_simd_step_with_optional_outputs"
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_EXPOSING_NON_EXISTING_ENV_VARIABLE_TO_SIMD_STEP_ACCEPTING_BATCHES,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={"image": [crowd_image, crowd_image]}
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 2, "Single image provided, so one output element expected"
    assert set(result[0].keys()) == {
        "output",
    }, "Expected all declared outputs to be delivered"
    assert set(result[1].keys()) == {
        "output",
    }, "Expected all declared outputs to be delivered"
    assert result[0]["output"] is None
    assert result[1]["output"] is None


WORKFLOW_EXPOSING_NON_EXISTING_ENV_VARIABLE_TO_SIMD_STEP_ACCEPTING_BATCHES_NESTED_SCENARIO = {
    "version": "1.4.0",
    "inputs": [{"type": "WorkflowImage", "name": "image"}],
    "steps": [
        {
            "type": "roboflow_core/roboflow_object_detection_model@v2",
            "name": "detection",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
        },
        {
            "type": "roboflow_core/dynamic_crop@v1",
            "name": "cropping",
            "image": "$inputs.image",
            "predictions": "$steps.detection.predictions",
        },
        {
            "type": "roboflow_core/environment_secrets_store@v1",
            "name": "vault",
            "variables_storing_secrets": ["NON_EXISTING_ENV_VARIABLE"],
        },
        {
            "type": "block_accepting_batches_of_images",
            "name": "model",
            "image": "$steps.cropping.crops",
            "secret": f"$steps.vault.non_existing_env_variable",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "output",
            "selector": "$steps.model.output",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_feeding_existing_env_variable_into_simd_step_accepting_batches_in_nested_scenario(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
    crowd_image: np.ndarray,
) -> None:
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.execution.stub_plugins.plugin_testing_non_simd_step_with_optional_outputs"
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_EXPOSING_NON_EXISTING_ENV_VARIABLE_TO_SIMD_STEP_ACCEPTING_BATCHES_NESTED_SCENARIO,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={"image": [crowd_image, crowd_image]}
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 2, "Single image provided, so one output element expected"
    assert set(result[0].keys()) == {
        "output",
    }, "Expected all declared outputs to be delivered"
    assert set(result[1].keys()) == {
        "output",
    }, "Expected all declared outputs to be delivered"
    assert result[0]["output"] == [None] * 12
    assert result[1]["output"] == [None] * 12


WORKFLOW_EXPOSING_NON_EXISTING_ENV_VARIABLE_TO_SIMD_STEP_ACCEPTING_EMPTY_BATCHES = {
    "version": "1.4.0",
    "inputs": [{"type": "WorkflowImage", "name": "image"}],
    "steps": [
        {
            "type": "roboflow_core/environment_secrets_store@v1",
            "name": "vault",
            "variables_storing_secrets": ["NON_EXISTING_ENV_VARIABLE"],
        },
        {
            "type": "block_accepting_empty_batches_of_images",
            "name": "model",
            "image": "$inputs.image",
            "secret": f"$steps.vault.non_existing_env_variable",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "output",
            "selector": "$steps.model.output",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_feeding_existing_env_variable_into_simd_step_accepting_empty_batches(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
    crowd_image: np.ndarray,
) -> None:
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.execution.stub_plugins.plugin_testing_non_simd_step_with_optional_outputs"
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_EXPOSING_NON_EXISTING_ENV_VARIABLE_TO_SIMD_STEP_ACCEPTING_EMPTY_BATCHES,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={"image": [crowd_image, crowd_image]}
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 2, "Single image provided, so one output element expected"
    assert set(result[0].keys()) == {
        "output",
    }, "Expected all declared outputs to be delivered"
    assert set(result[1].keys()) == {
        "output",
    }, "Expected all declared outputs to be delivered"
    assert result[0]["output"] is "empty"
    assert result[1]["output"] is "empty"


WORKFLOW_EXPOSING_NON_EXISTING_ENV_VARIABLE_TO_SIMD_STEP_ACCEPTING_EMPTY_BATCHES_NESTED = {
    "version": "1.4.0",
    "inputs": [{"type": "WorkflowImage", "name": "image"}],
    "steps": [
        {
            "type": "roboflow_core/roboflow_object_detection_model@v2",
            "name": "detection",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
        },
        {
            "type": "roboflow_core/dynamic_crop@v1",
            "name": "cropping",
            "image": "$inputs.image",
            "predictions": "$steps.detection.predictions",
        },
        {
            "type": "roboflow_core/environment_secrets_store@v1",
            "name": "vault",
            "variables_storing_secrets": ["NON_EXISTING_ENV_VARIABLE"],
        },
        {
            "type": "block_accepting_empty_batches_of_images",
            "name": "model",
            "image": "$steps.cropping.crops",
            "secret": f"$steps.vault.non_existing_env_variable",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "output",
            "selector": "$steps.model.output",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_feeding_existing_env_variable_into_simd_step_accepting_empty_batches_nested_scenario(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
    crowd_image: np.ndarray,
) -> None:
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.execution.stub_plugins.plugin_testing_non_simd_step_with_optional_outputs"
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_EXPOSING_NON_EXISTING_ENV_VARIABLE_TO_SIMD_STEP_ACCEPTING_EMPTY_BATCHES_NESTED,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={"image": [crowd_image, crowd_image]}
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 2, "Single image provided, so one output element expected"
    assert set(result[0].keys()) == {
        "output",
    }, "Expected all declared outputs to be delivered"
    assert set(result[1].keys()) == {
        "output",
    }, "Expected all declared outputs to be delivered"
    assert result[0]["output"] == ["empty"] * 12
    assert result[1]["output"] == ["empty"] * 12


WORKFLOW_EXPOSING_NON_EXISTING_ENV_VARIABLE_TO_SIMD_STEP_NOT_ACCEPTING_BATCHES = {
    "version": "1.4.0",
    "inputs": [{"type": "WorkflowImage", "name": "image"}],
    "steps": [
        {
            "type": "roboflow_core/environment_secrets_store@v1",
            "name": "vault",
            "variables_storing_secrets": ["NON_EXISTING_ENV_VARIABLE"],
        },
        {
            "type": "block_accepting_images",
            "name": "model",
            "image": "$inputs.image",
            "secret": f"$steps.vault.non_existing_env_variable",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "output",
            "selector": "$steps.model.output",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_feeding_existing_env_variable_into_simd_step_not_accepting_batches(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
    crowd_image: np.ndarray,
) -> None:
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.execution.stub_plugins.plugin_testing_non_simd_step_with_optional_outputs"
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_EXPOSING_NON_EXISTING_ENV_VARIABLE_TO_SIMD_STEP_NOT_ACCEPTING_BATCHES,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={"image": [crowd_image, crowd_image]}
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 2, "Single image provided, so one output element expected"
    assert set(result[0].keys()) == {
        "output",
    }, "Expected all declared outputs to be delivered"
    assert set(result[1].keys()) == {
        "output",
    }, "Expected all declared outputs to be delivered"
    assert result[0]["output"] is None
    assert result[1]["output"] is None


WORKFLOW_EXPOSING_NON_EXISTING_ENV_VARIABLE_TO_SIMD_STEP_NOT_ACCEPTING_BATCHES_NESTED_SCENARIO = {
    "version": "1.4.0",
    "inputs": [{"type": "WorkflowImage", "name": "image"}],
    "steps": [
        {
            "type": "roboflow_core/roboflow_object_detection_model@v2",
            "name": "detection",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
        },
        {
            "type": "roboflow_core/dynamic_crop@v1",
            "name": "cropping",
            "image": "$inputs.image",
            "predictions": "$steps.detection.predictions",
        },
        {
            "type": "roboflow_core/environment_secrets_store@v1",
            "name": "vault",
            "variables_storing_secrets": ["NON_EXISTING_ENV_VARIABLE"],
        },
        {
            "type": "block_accepting_images",
            "name": "model",
            "image": "$steps.cropping.crops",
            "secret": f"$steps.vault.non_existing_env_variable",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "output",
            "selector": "$steps.model.output",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_feeding_existing_env_variable_into_simd_step_not_accepting_batches_in_nested_scenario(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
    crowd_image: np.ndarray,
) -> None:
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.execution.stub_plugins.plugin_testing_non_simd_step_with_optional_outputs"
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_EXPOSING_NON_EXISTING_ENV_VARIABLE_TO_SIMD_STEP_NOT_ACCEPTING_BATCHES_NESTED_SCENARIO,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={"image": [crowd_image, crowd_image]}
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 2, "Single image provided, so one output element expected"
    assert set(result[0].keys()) == {
        "output",
    }, "Expected all declared outputs to be delivered"
    assert set(result[1].keys()) == {
        "output",
    }, "Expected all declared outputs to be delivered"
    assert result[0]["output"] == [None] * 12
    assert result[1]["output"] == [None] * 12


WORKFLOW_EXPOSING_NON_EXISTING_ENV_VARIABLE_TO_SIMD_STEP_ACCEPTING_EMPTY_VALUES = {
    "version": "1.4.0",
    "inputs": [{"type": "WorkflowImage", "name": "image"}],
    "steps": [
        {
            "type": "roboflow_core/environment_secrets_store@v1",
            "name": "vault",
            "variables_storing_secrets": ["NON_EXISTING_ENV_VARIABLE"],
        },
        {
            "type": "block_accepting_empty_images",
            "name": "model",
            "image": "$inputs.image",
            "secret": f"$steps.vault.non_existing_env_variable",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "output",
            "selector": "$steps.model.output",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_feeding_existing_env_variable_into_simd_step_accepting_empty_values(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
    crowd_image: np.ndarray,
) -> None:
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.execution.stub_plugins.plugin_testing_non_simd_step_with_optional_outputs"
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_EXPOSING_NON_EXISTING_ENV_VARIABLE_TO_SIMD_STEP_ACCEPTING_EMPTY_VALUES,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={"image": [crowd_image, crowd_image]}
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 2, "Single image provided, so one output element expected"
    assert set(result[0].keys()) == {
        "output",
    }, "Expected all declared outputs to be delivered"
    assert set(result[1].keys()) == {
        "output",
    }, "Expected all declared outputs to be delivered"
    assert result[0]["output"] is "empty"
    assert result[1]["output"] is "empty"


WORKFLOW_EXPOSING_NON_EXISTING_ENV_VARIABLE_TO_SIMD_STEP_ACCEPTING_EMPTY_VALUES_NESTED = {
    "version": "1.4.0",
    "inputs": [{"type": "WorkflowImage", "name": "image"}],
    "steps": [
        {
            "type": "roboflow_core/roboflow_object_detection_model@v2",
            "name": "detection",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
        },
        {
            "type": "roboflow_core/dynamic_crop@v1",
            "name": "cropping",
            "image": "$inputs.image",
            "predictions": "$steps.detection.predictions",
        },
        {
            "type": "roboflow_core/environment_secrets_store@v1",
            "name": "vault",
            "variables_storing_secrets": ["NON_EXISTING_ENV_VARIABLE"],
        },
        {
            "type": "block_accepting_empty_images",
            "name": "model",
            "image": "$steps.cropping.crops",
            "secret": f"$steps.vault.non_existing_env_variable",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "output",
            "selector": "$steps.model.output",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_feeding_existing_env_variable_into_simd_step_accepting_empty_values_nested_scenario(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
    crowd_image: np.ndarray,
) -> None:
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.execution.stub_plugins.plugin_testing_non_simd_step_with_optional_outputs"
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_EXPOSING_NON_EXISTING_ENV_VARIABLE_TO_SIMD_STEP_ACCEPTING_EMPTY_VALUES_NESTED,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={"image": [crowd_image, crowd_image]}
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 2, "Single image provided, so one output element expected"
    assert set(result[0].keys()) == {
        "output",
    }, "Expected all declared outputs to be delivered"
    assert set(result[1].keys()) == {
        "output",
    }, "Expected all declared outputs to be delivered"
    assert result[0]["output"] == ["empty"] * 12
    assert result[1]["output"] == ["empty"] * 12
