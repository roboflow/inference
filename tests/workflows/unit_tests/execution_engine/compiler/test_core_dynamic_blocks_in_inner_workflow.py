"""Compiler collects dynamic block definitions from nested inner workflows."""

from unittest import mock

from inference.core.workflows.execution_engine.v1.compiler import core as compiler_core
from inference.core.workflows.execution_engine.v1.inner_workflow.constants import (
    USE_INNER_WORKFLOW_BLOCK_TYPE,
)


def _dynamic_block_definition(block_type: str) -> dict:
    return {
        "type": "DynamicBlockDefinition",
        "manifest": {
            "type": "ManifestDescription",
            "block_type": block_type,
            "inputs": {},
            "outputs": {},
        },
        "code": {
            "type": "PythonCode",
            "run_function_code": "def run(self): return {}",
        },
    }


@mock.patch.object(compiler_core, "compile_dynamic_blocks")
@mock.patch.object(compiler_core, "inline_inner_workflow_steps")
@mock.patch.object(
    compiler_core,
    "validate_inner_workflow_composition_from_raw_workflow_definition",
)
@mock.patch.object(compiler_core, "parse_workflow_definition")
@mock.patch.object(compiler_core, "prepare_execution_graph")
@mock.patch.object(compiler_core, "validate_workflow_specification")
@mock.patch.object(compiler_core, "load_kinds_deserializers")
@mock.patch.object(compiler_core, "load_kinds_serializers")
@mock.patch.object(compiler_core, "load_initializers")
@mock.patch.object(compiler_core, "load_workflow_blocks", return_value=[])
def test_compile_workflow_graph_passes_inner_dynamic_blocks_to_compiler(
    _load_blocks,
    _load_initializers,
    _load_serializers,
    _load_deserializers,
    _validate_spec,
    _prepare_graph,
    parse_workflow_definition,
    _validate_composition,
    inline_inner_workflow_steps,
    compile_dynamic_blocks,
) -> None:
    child_block = _dynamic_block_definition("InnerOnlyBlock")
    child_workflow = {
        "version": "1.0",
        "dynamic_blocks_definitions": [child_block],
        "inputs": [],
        "steps": [],
        "outputs": [],
    }
    workflow_definition = {
        "version": "1.0",
        "inputs": [],
        "steps": [
            {
                "type": USE_INNER_WORKFLOW_BLOCK_TYPE,
                "name": "nested",
                "workflow_definition": child_workflow,
                "parameter_bindings": {},
            },
        ],
        "outputs": [],
    }

    compile_dynamic_blocks.return_value = []
    parse_workflow_definition.return_value = mock.Mock(
        steps=[],
        inputs=[],
        outputs=[],
    )
    inline_inner_workflow_steps.side_effect = lambda definition, **_: definition

    compiler_core.compile_workflow_graph(
        workflow_definition=workflow_definition,
        init_parameters={},
    )

    compile_dynamic_blocks.assert_called_once()
    passed_definitions = compile_dynamic_blocks.call_args.kwargs[
        "dynamic_blocks_definitions"
    ]

    assert passed_definitions == [child_block]
    assert workflow_definition["dynamic_blocks_definitions"] == [child_block]
