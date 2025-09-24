from unittest import mock
from unittest.mock import MagicMock

import pytest

from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.errors import (
    StepInputDimensionalityError,
    StepOutputLineageError,
)
from inference.core.workflows.execution_engine.introspection import blocks_loader
from inference.core.workflows.execution_engine.v1.compiler.core import compile_workflow

WORKFLOW_WITH_INVALID_DIMENSIONALITY_OF_INPUT = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image_1"},
        {"type": "WorkflowImage", "name": "image_2"},
    ],
    "steps": [
        {
            "type": "BlockRequestingDifferentDims",
            "name": "problematic_dimensions",
            "images": "$inputs.image_1",
            "crops": "$inputs.image_2",
        }
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "crops",
            "selector": "$steps.problematic_dimensions.*",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_compilation_of_workflow_where_step_input_dimensionality_is_mismatched(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
) -> None:
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.compilation.stub_plugins.plugin_with_dimensionality_manipulation_blocks"
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }

    # when
    with pytest.raises(StepInputDimensionalityError):
        _ = compile_workflow(
            workflow_definition=WORKFLOW_WITH_INVALID_DIMENSIONALITY_OF_INPUT,
            init_parameters=workflow_init_parameters,
        )


WORKFLOW_WITH_INVALID_DIMENSIONALITY_OF_INPUT_BASED_ON_STEP_OUTPUT_DIMENSION = {
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
        {
            "type": "BlockRequestingDifferentDims",
            "name": "non_problematic_dimensions",
            "images": "$inputs.image",
            "crops": "$steps.crops.crops",
        },
        {
            "type": "BlockRequestingDifferentDims",
            "name": "problematic_dimensions",
            "images": "$inputs.image",
            "crops": "$steps.non_problematic_dimensions.output",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "crops",
            "selector": "$steps.problematic_dimensions.*",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_compilation_of_workflow_where_step_input_dimensionality_is_mismatched_based_on_step_output(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
) -> None:
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.compilation.stub_plugins.plugin_with_dimensionality_manipulation_blocks"
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }

    # when
    with pytest.raises(StepInputDimensionalityError):
        _ = compile_workflow(
            workflow_definition=WORKFLOW_WITH_INVALID_DIMENSIONALITY_OF_INPUT_BASED_ON_STEP_OUTPUT_DIMENSION,
            init_parameters=workflow_init_parameters,
        )


WORKFLOW_WITH_DIFFERENT_DIMENSIONALITIES_PROVIDED_FOR_STEP_REQUIRING_THE_SAME = {
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
        {
            "type": "ExpectsTheSameDimensionality",
            "name": "problematic_dimensions",
            "images": "$inputs.image",
            "crops": "$steps.crops.crops",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "crops",
            "selector": "$steps.problematic_dimensions.*",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_compilation_of_workflow_where_step_expects_the_same_dimensionality_but_different_provided(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
) -> None:
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.compilation.stub_plugins.plugin_with_dimensionality_manipulation_blocks"
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }

    # when
    with pytest.raises(StepInputDimensionalityError):
        _ = compile_workflow(
            workflow_definition=WORKFLOW_WITH_DIFFERENT_DIMENSIONALITIES_PROVIDED_FOR_STEP_REQUIRING_THE_SAME,
            init_parameters=workflow_init_parameters,
        )


WORKFLOW_ATTEMPTING_TO_REDUCE_DIM_TO_ZERO = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
    ],
    "steps": [
        {
            "type": "DecreasingDimensionality",
            "name": "problematic_dimensions",
            "images": "$inputs.image",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "crops",
            "selector": "$steps.problematic_dimensions.*",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_compilation_of_workflow_where_step_attempts_decreasing_dimensionality_to_zero(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
) -> None:
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.compilation.stub_plugins.plugin_with_dimensionality_manipulation_blocks"
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }

    # when
    with pytest.raises(StepOutputLineageError):
        _ = compile_workflow(
            workflow_definition=WORKFLOW_ATTEMPTING_TO_REDUCE_DIM_TO_ZERO,
            init_parameters=workflow_init_parameters,
        )
