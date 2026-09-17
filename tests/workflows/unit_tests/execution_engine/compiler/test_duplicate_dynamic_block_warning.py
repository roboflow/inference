"""Phase 9 final fix wave, F2: `compile_workflow_graph` used to collect dynamic
block definitions twice on a cold compile - once before normalisation (to feed
the cache-bypass predicate and `ensure_dynamic_blocks_allowed` on the
cache-hit path), once after (`apply_collected_dynamic_blocks_definitions_to_workflow_root`)
- so a definition with a duplicate `block_type` logged the "skipping duplicate"
warning twice. The pre-resolution call now passes `warn_on_duplicates=False`,
so the warning is only ever logged once, by the post-normalisation collection.
"""

from unittest import mock

from inference.core.workflows.execution_engine.v1.compiler import core as compiler_core
from inference.core.workflows.execution_engine.v1.inner_workflow import (
    dynamic_blocks_collection,
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
def test_duplicate_dynamic_block_definition_warns_exactly_once_on_cold_compile(
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
    duplicated_block_type = "DuplicatedBlock"
    workflow_definition = {
        "version": "1.0",
        "inputs": [],
        "dynamic_blocks_definitions": [
            _dynamic_block_definition(duplicated_block_type),
            _dynamic_block_definition(duplicated_block_type),
        ],
        "steps": [],
        "outputs": [],
    }

    compile_dynamic_blocks.return_value = []
    parse_workflow_definition.return_value = mock.Mock(
        steps=[],
        inputs=[],
        outputs=[],
    )
    inline_inner_workflow_steps.side_effect = lambda definition, **_: definition

    with mock.patch.object(dynamic_blocks_collection, "logger") as mocked_logger:
        compiler_core.compile_workflow_graph(
            workflow_definition=workflow_definition,
            init_parameters={},
        )

    assert mocked_logger.warning.call_count == 1
