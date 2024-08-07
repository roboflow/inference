from unittest import mock
from unittest.mock import MagicMock

import pytest

from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.errors import BlockInterfaceError
from inference.core.workflows.execution_engine.introspection import blocks_loader
from inference.core.workflows.execution_engine.v1.compiler.core import compile_workflow

WORKFLOW_WITH_INVALID_BLOCK_REGARDING_OFFSETS_RANGE = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image_1"},
        {"type": "WorkflowImage", "name": "image_2"},
    ],
    "steps": [
        {
            "type": "BlockOffsetsNotInProperRange",
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
def test_compilation_of_workflow_where_block_defines_out_of_range_offsets(
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
    with pytest.raises(BlockInterfaceError):
        _ = compile_workflow(
            workflow_definition=WORKFLOW_WITH_INVALID_BLOCK_REGARDING_OFFSETS_RANGE,
            init_parameters=workflow_init_parameters,
        )


WORKFLOW_WITH_INVALID_BLOCK_REGARDING_NEGATIVE_OFFSET = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image_1"},
        {"type": "WorkflowImage", "name": "image_2"},
    ],
    "steps": [
        {
            "type": "BlockWithNegativeOffset",
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
def test_compilation_of_workflow_where_block_defines_negative_offset(
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
    with pytest.raises(BlockInterfaceError):
        _ = compile_workflow(
            workflow_definition=WORKFLOW_WITH_INVALID_BLOCK_REGARDING_NEGATIVE_OFFSET,
            init_parameters=workflow_init_parameters,
        )


WORKFLOW_WITH_INVALID_BLOCK_DECLARING_OFFSET_BEING_NOT_SIMD = {
    "version": "1.0",
    "inputs": [],
    "steps": [
        {
            "type": "NonSIMDWithOutputOffset",
            "name": "problematic_dimensions",
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
def test_compilation_of_workflow_where_block_is_not_simd_but_defines_output_offset(
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
    with pytest.raises(BlockInterfaceError):
        _ = compile_workflow(
            workflow_definition=WORKFLOW_WITH_INVALID_BLOCK_DECLARING_OFFSET_BEING_NOT_SIMD,
            init_parameters=workflow_init_parameters,
        )


WORKFLOW_WITH_INVALID_BLOCK_DECLARING_DIMENSIONALITY_REFERENCE_PROPERTY_AS_NON_BATCH = {
    "version": "1.0",
    "inputs": [],
    "steps": [
        {
            "type": "DimensionalityReferencePropertyIsNotBatch",
            "name": "problematic_dimensions",
            "dim_reference": "a",
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
def test_compilation_of_workflow_where_block_declares_non_batch_dimensionality_reference_property(
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
    with pytest.raises(BlockInterfaceError):
        _ = compile_workflow(
            workflow_definition=WORKFLOW_WITH_INVALID_BLOCK_DECLARING_DIMENSIONALITY_REFERENCE_PROPERTY_AS_NON_BATCH,
            init_parameters=workflow_init_parameters,
        )


WORKFLOW_WITH_INVALID_BLOCK_DECLARING_OUT_OF_RANGE_OUTPUT_DIMENSIONALITY = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image_1"},
        {"type": "WorkflowImage", "name": "image_2"},
    ],
    "steps": [
        {
            "type": "OutputDimensionalityInInvalidRange",
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
def test_compilation_of_workflow_where_block_declares_out_of_range_output_dimensionality(
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
    with pytest.raises(BlockInterfaceError):
        _ = compile_workflow(
            workflow_definition=WORKFLOW_WITH_INVALID_BLOCK_DECLARING_OUT_OF_RANGE_OUTPUT_DIMENSIONALITY,
            init_parameters=workflow_init_parameters,
        )


WORKFLOW_WITH_INVALID_BLOCK_NOT_DECLARING_ZERO_GROUND_BATCH_PARAMETER = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image_1"},
        {"type": "WorkflowImage", "name": "image_2"},
    ],
    "steps": [
        {
            "type": "LackOfZeroGroundOffset",
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
def test_compilation_of_workflow_where_block_which_does_not_declare_zero_ground_dimensionality_offset(
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
    with pytest.raises(BlockInterfaceError):
        _ = compile_workflow(
            workflow_definition=WORKFLOW_WITH_INVALID_BLOCK_NOT_DECLARING_ZERO_GROUND_BATCH_PARAMETER,
            init_parameters=workflow_init_parameters,
        )


WORKFLOW_WITH_INVALID_BLOCK_NOT_DECLARING_REQUIRED_REFERENCE_PROPERTY_FOR_DIMENSIONALITY = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image_1"},
        {"type": "WorkflowImage", "name": "image_2"},
    ],
    "steps": [
        {
            "type": "LackOfRequiredReferenceProperty",
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
def test_compilation_of_workflow_where_block_which_does_not_declare_required_reference_property_for_dimensionality(
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
    with pytest.raises(BlockInterfaceError):
        _ = compile_workflow(
            workflow_definition=WORKFLOW_WITH_INVALID_BLOCK_NOT_DECLARING_REQUIRED_REFERENCE_PROPERTY_FOR_DIMENSIONALITY,
            init_parameters=workflow_init_parameters,
        )


WORKFLOW_WITH_INVALID_BLOCK_MANIPULATING_OUTPUT_DIMENSIONALITY_WHEN_INPUTS_HAVE_DIFFERENT_DIMENSION = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image_1"},
        {"type": "WorkflowImage", "name": "image_2"},
    ],
    "steps": [
        {
            "type": "ManipulationOutputDimensionalityWhenInvalid",
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
def test_compilation_of_workflow_where_block_manipulates_output_dimension_and_inputs_are_of_different_dimensionalties(
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
    with pytest.raises(BlockInterfaceError):
        _ = compile_workflow(
            workflow_definition=WORKFLOW_WITH_INVALID_BLOCK_MANIPULATING_OUTPUT_DIMENSIONALITY_WHEN_INPUTS_HAVE_DIFFERENT_DIMENSION,
            init_parameters=workflow_init_parameters,
        )
