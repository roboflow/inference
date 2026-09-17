"""Standalone Workflows cannot fetch an inner workflow from the Roboflow API,
and a resolver-dependent compilation is never served from the cache - while a
plain definition still is."""

import ast
import pathlib
from typing import Any, Dict, Optional

import pytest

from inference.core.workflows.errors import WorkflowEnvironmentConfigurationError
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.core.workflows.execution_engine.v1.compiler.core import (
    compile_workflow_graph,
)
from inference.core.workflows.execution_engine.v1.inner_workflow.reference_resolution import (
    WORKFLOWS_CORE_INNER_WORKFLOW_SPEC_RESOLVER,
    get_inner_workflow_spec_resolver,
)


def _echo_spec() -> Dict[str, Any]:
    # The same valid child `test_reference_resolution.py:18` uses;
    # `inner_workflow/inline.py:477` rejects an empty `steps` list.
    return {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowParameter", "name": "child_msg", "default_value": "x"},
        ],
        "steps": [
            {
                "type": "roboflow_core/first_non_empty_or_default@v1",
                "name": "pick",
                "data": ["$inputs.child_msg"],
                "default": "fallback-inner",
            },
        ],
        "outputs": [
            {"type": "JsonField", "name": "echo", "selector": "$steps.pick.output"},
        ],
    }


# Field names from core_steps/flow_control/inner_workflow/v1.py:56-80.
REFERENCING_DEFINITION = {
    "version": "1.0",
    "inputs": [{"type": "WorkflowParameter", "name": "p", "default_value": "hello"}],
    "steps": [
        {
            "type": "roboflow_core/inner_workflow@v1",
            "name": "nested",
            "workflow_workspace_id": "my-ws",
            "workflow_id": "wf-1",
            "parameter_bindings": {"child_msg": "$inputs.p"},
        }
    ],
    "outputs": [{"type": "JsonField", "name": "out", "selector": "$steps.nested.echo"}],
}

PLAIN_DEFINITION = {
    "version": "1.0",
    "inputs": [{"type": "WorkflowParameter", "name": "p", "default_value": "x"}],
    "steps": [
        {
            "type": "roboflow_core/first_non_empty_or_default@v1",
            "name": "pick",
            "data": ["$inputs.p"],
            "default": "d",
        }
    ],
    "outputs": [{"type": "JsonField", "name": "o", "selector": "$steps.pick.output"}],
}


def test_default_resolver_refuses_and_names_the_init_parameter() -> None:
    resolver = get_inner_workflow_spec_resolver(init_parameters={})
    with pytest.raises(WorkflowEnvironmentConfigurationError) as error:
        resolver("some-workspace", "some-workflow", None, {})
    assert WORKFLOWS_CORE_INNER_WORKFLOW_SPEC_RESOLVER in str(error.value)


def test_injected_resolver_wins() -> None:
    sentinel = {"version": "1.0"}
    resolver = get_inner_workflow_spec_resolver(
        init_parameters={
            WORKFLOWS_CORE_INNER_WORKFLOW_SPEC_RESOLVER: lambda *a, **k: sentinel
        }
    )
    assert resolver("w", "id", None, {}) is sentinel


def test_reference_resolution_does_not_import_the_server() -> None:
    # parents[5] == repo root from
    # tests/workflows/unit_tests/execution_engine/inner_workflow/
    path = (
        pathlib.Path(__file__).resolve().parents[5]
        / "inference/core/workflows/execution_engine/v1/inner_workflow/reference_resolution.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules = {
        n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module
    }
    assert "inference.core.roboflow_api" not in modules


def test_the_server_resolver_preserves_the_api_key_error_message() -> None:
    from inference.core.interfaces.roboflow_platform_client import (
        default_inner_workflow_spec_resolver,
    )
    from inference.core.workflows.errors import WorkflowDefinitionError

    with pytest.raises(WorkflowDefinitionError) as error:
        default_inner_workflow_spec_resolver("workspace", "wf", None, {})
    assert "requires a Roboflow API key" in str(error.value)


def test_a_resolver_dependent_definition_is_not_cached() -> None:
    """Round-2 defect 4: the cache returns before reference resolution, so a
    second engine's resolver would be bypassed. Resolver-dependent definitions
    therefore bypass the cache entirely."""
    from inference.core.workflows.execution_engine.v1.compiler.core import (
        _is_resolver_dependent,
    )

    assert _is_resolver_dependent(PLAIN_DEFINITION, []) is False
    assert (
        _is_resolver_dependent(PLAIN_DEFINITION, [{"type": "DynamicBlockDefinition"}])
        is True
    )
    assert _is_resolver_dependent(REFERENCING_DEFINITION, []) is True


def test_two_engines_with_different_resolvers_and_api_keys_do_not_share_a_compilation() -> (
    None
):
    """The end-to-end property the bypass exists for (round-3 defect 6): the
    SAME definition compiled twice with distinct resolvers AND distinct
    authentication contexts must consult both, and both engines must run."""
    seen = []

    def resolver_factory(tag):
        def resolver(
            workspace_id: str,
            workflow_id: str,
            workflow_version_id: Optional[str],
            init_parameters: Dict[str, Any],
        ) -> Dict[str, Any]:
            seen.append((tag, init_parameters.get("workflows_core.api_key")))
            return _echo_spec()

        return resolver

    results = []
    for tag, api_key in (("a", "key-a"), ("b", "key-b")):
        engine = ExecutionEngine.init(
            workflow_definition=REFERENCING_DEFINITION,
            init_parameters={
                "workflows_core.api_key": api_key,
                WORKFLOWS_CORE_INNER_WORKFLOW_SPEC_RESOLVER: resolver_factory(tag),
            },
        )
        results.append(engine.run(runtime_parameters={"p": "hello"}))

    assert seen == [
        ("a", "key-a"),
        ("b", "key-b"),
    ], "the second engine's resolver was bypassed by the cache"
    assert results == [[{"out": "hello"}], [{"out": "hello"}]]


def test_plain_definitions_still_reuse_the_compilation() -> None:
    """The cache exists for these; the bypass must not widen to them."""
    first = compile_workflow_graph(
        workflow_definition=PLAIN_DEFINITION, init_parameters={}
    )
    second = compile_workflow_graph(
        workflow_definition=PLAIN_DEFINITION, init_parameters={}
    )
    assert first is second
