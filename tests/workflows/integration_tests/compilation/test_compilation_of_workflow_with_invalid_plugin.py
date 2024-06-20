from unittest import mock
from unittest.mock import MagicMock

import pytest

from inference.core.managers.base import ModelManager
from inference.core.workflows.errors import BlockInterfaceError
from inference.core.workflows.execution_engine.compiler.core import compile_workflow
from inference.core.workflows.execution_engine.introspection import blocks_loader

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


@pytest.mark.asyncio
@mock.patch.object(blocks_loader, "get_plugin_modules")
async def test_compilation_of_workflow_where_block_defines_out_of_range_offsets(
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


@pytest.mark.asyncio
@mock.patch.object(blocks_loader, "get_plugin_modules")
async def test_compilation_of_workflow_where_block_defines_negative_offset(
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


@pytest.mark.asyncio
@mock.patch.object(blocks_loader, "get_plugin_modules")
async def test_compilation_of_workflow_where_block_is_not_simd_but_defines_output_offset(
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


@pytest.mark.asyncio
@mock.patch.object(blocks_loader, "get_plugin_modules")
async def test_compilation_of_workflow_where_block_declares_non_batch_dimensionality_reference_property(
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
    }

    # when
    with pytest.raises(BlockInterfaceError):
        _ = compile_workflow(
            workflow_definition=WORKFLOW_WITH_INVALID_BLOCK_DECLARING_DIMENSIONALITY_REFERENCE_PROPERTY_AS_NON_BATCH,
            init_parameters=workflow_init_parameters,
        )
