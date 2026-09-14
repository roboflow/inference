"""Authorization-sensitive inner-workflow resolution must precede cache lookup."""

from typing import Any, Dict, Optional
from unittest import mock

import pytest

from inference.core.workflows.errors import WorkflowDefinitionError
from inference.core.workflows.execution_engine.v1.compiler import core as compiler_core
from inference.core.workflows.execution_engine.v1.inner_workflow.constants import (
    USE_INNER_WORKFLOW_BLOCK_TYPE,
)
from inference.core.workflows.execution_engine.v1.inner_workflow.reference_resolution import (
    WORKFLOWS_CORE_INNER_WORKFLOW_SPEC_RESOLVER,
)


def _child_workflow() -> Dict[str, Any]:
    return {
        "version": "1.0",
        "inputs": [],
        "steps": [],
        "outputs": [],
    }


def _outer_workflow() -> Dict[str, Any]:
    return {
        "version": "1.0",
        "inputs": [],
        "steps": [
            {
                "type": USE_INNER_WORKFLOW_BLOCK_TYPE,
                "name": "private-child",
                "workflow_workspace_id": "private-workspace",
                "workflow_id": "private-workflow",
                "parameter_bindings": {},
            }
        ],
        "outputs": [],
    }


@pytest.fixture(autouse=True)
def clear_compilation_cache() -> None:
    with compiler_core.COMPILATION_CACHE._cache_lock:
        compiler_core.COMPILATION_CACHE._cache.clear()
        compiler_core.COMPILATION_CACHE._keys_buffer.clear()


@mock.patch.object(compiler_core, "compile_dynamic_blocks", return_value=[])
@mock.patch.object(compiler_core, "inline_inner_workflow_steps")
@mock.patch.object(
    compiler_core,
    "validate_inner_workflow_composition_from_raw_workflow_definition",
)
@mock.patch.object(compiler_core, "parse_workflow_definition")
@mock.patch.object(compiler_core, "prepare_execution_graph")
@mock.patch.object(compiler_core, "validate_workflow_specification")
@mock.patch.object(compiler_core, "load_kinds_deserializers", return_value=[])
@mock.patch.object(compiler_core, "load_kinds_serializers", return_value=[])
@mock.patch.object(compiler_core, "load_initializers", return_value=[])
@mock.patch.object(compiler_core, "load_workflow_blocks", return_value=[])
def test_cache_hit_cannot_skip_inner_workflow_authorization(
    _load_blocks,
    _load_initializers,
    _load_serializers,
    _load_deserializers,
    _validate_specification,
    prepare_execution_graph,
    parse_workflow_definition,
    _validate_composition,
    inline_inner_workflow_steps,
    _compile_dynamic_blocks,
) -> None:
    resolver_calls = []

    def resolver(
        workspace_id: str,
        workflow_id: str,
        workflow_version_id: Optional[str],
        init_parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        api_key = init_parameters.get("workflows_core.api_key")
        resolver_calls.append(api_key)
        if api_key != "authorized":
            raise WorkflowDefinitionError(
                public_message="Not authorized to read inner workflow",
                context="test",
            )
        return _child_workflow()

    parsed_workflow = mock.Mock(steps=[], inputs=[], outputs=[])
    parse_workflow_definition.return_value = parsed_workflow
    inline_inner_workflow_steps.side_effect = lambda definition, **_: definition
    prepare_execution_graph.return_value = mock.Mock()
    resolver_parameter = {
        WORKFLOWS_CORE_INNER_WORKFLOW_SPEC_RESOLVER: resolver,
    }

    compiler_core.compile_workflow_graph(
        workflow_definition=_outer_workflow(),
        init_parameters={
            **resolver_parameter,
            "workflows_core.api_key": "authorized",
        },
    )

    with pytest.raises(WorkflowDefinitionError, match="Not authorized"):
        compiler_core.compile_workflow_graph(
            workflow_definition=_outer_workflow(),
            init_parameters={
                **resolver_parameter,
                "workflows_core.api_key": "unauthorized",
            },
        )

    assert resolver_calls == ["authorized", "unauthorized"]
    assert parse_workflow_definition.call_count == 1
