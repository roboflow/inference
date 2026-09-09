"""Modal sandbox naming asks a resolver, not the Roboflow API client, and
compilation and execution ask the SAME resolver."""

import ast
import pathlib
from unittest import mock

from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.core.workflows.prototypes.workspace_resolver import (
    NULL_WORKSPACE_RESOLVER,
    NullWorkspaceResolver,
)


class StubResolver:
    def __init__(self, workspace=None):
        self._workspace = workspace
        self.calls = []

    def resolve_workspace(self, api_key):
        self.calls.append(api_key)
        return self._workspace


# Compiles and runs locally at HEAD (`[{'out': 7}]`, evidence E22).
# Selector value from dynamic_blocks/entities.SelectorType.INPUT_PARAMETER.
DYNAMIC_WORKFLOW = {
    "version": "1.0",
    "inputs": [{"type": "WorkflowParameter", "name": "a", "default_value": 1}],
    "dynamic_blocks_definitions": [
        {
            "type": "DynamicBlockDefinition",
            "manifest": {
                "type": "ManifestDescription",
                "block_type": "ResolverProbe",
                "inputs": {
                    "a": {
                        "type": "DynamicInputDefinition",
                        "selector_types": ["input_parameter"],
                    }
                },
                "outputs": {"out": {"type": "DynamicOutputDefinition"}},
            },
            "code": {
                "type": "PythonCode",
                "run_function_code": 'def run(self, a):\n    return {"out": a}\n',
            },
        }
    ],
    "steps": [{"type": "ResolverProbe", "name": "custom_block", "a": "$inputs.a"}],
    "outputs": [
        {"type": "JsonField", "name": "out", "selector": "$steps.custom_block.out"}
    ],
}


def test_null_resolver_returns_none() -> None:
    assert NullWorkspaceResolver().resolve_workspace("any-key") is None


def test_block_scaffolding_does_not_import_the_server() -> None:
    # parents[5] == repo root from
    # tests/workflows/unit_tests/execution_engine/dynamic_blocs/
    path = (
        pathlib.Path(__file__).resolve().parents[5]
        / "inference/core/workflows/execution_engine/v1/dynamic_blocks/block_scaffolding.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules = {
        n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module
    }
    assert "inference.core.roboflow_api" not in modules
    assert "inference.core.exceptions" not in modules


def test_server_resolver_swallows_workspace_load_error(monkeypatch) -> None:
    import inference.core.roboflow_api as roboflow_api
    from inference.core.exceptions import WorkspaceLoadError
    from inference.core.interfaces.roboflow_platform_client import (
        ServerWorkspaceResolver,
    )

    def boom(api_key):
        raise WorkspaceLoadError("no workspace")

    monkeypatch.setattr(roboflow_api, "get_roboflow_workspace", boom)
    assert ServerWorkspaceResolver().resolve_workspace("k") is None


def test_server_resolver_returns_the_workspace(monkeypatch) -> None:
    import inference.core.roboflow_api as roboflow_api
    from inference.core.interfaces.roboflow_platform_client import (
        ServerWorkspaceResolver,
    )

    monkeypatch.setattr(roboflow_api, "get_roboflow_workspace", lambda api_key: "ws-1")
    assert ServerWorkspaceResolver().resolve_workspace("k") == "ws-1"


def test_engine_mirrors_the_resolver_into_the_dynamic_namespace() -> None:
    """Generated blocks are `dynamic_workflows_blocks`, and
    `retrieve_init_parameter_values` does NOT fall back to `workflows_core.*`
    (`steps_initialiser.py:124`, `blocks_loader.load_core_blocks_initializers`).
    The engine mirrors `api_key` for exactly this reason."""
    from inference.core.workflows.execution_engine.v1.core import (
        _mirror_dynamic_block_parameters,
    )

    resolver = StubResolver()
    init_parameters = {
        "workflows_core.api_key": "k",
        "workflows_core.workspace_resolver": resolver,
    }
    _mirror_dynamic_block_parameters(init_parameters)
    assert init_parameters["dynamic_workflows_blocks.api_key"] == "k"
    assert init_parameters["dynamic_workflows_blocks.workspace_resolver"] is resolver


def test_engine_mirror_defaults_to_the_null_resolver() -> None:
    from inference.core.workflows.execution_engine.v1.core import (
        _mirror_dynamic_block_parameters,
    )

    init_parameters = {}
    _mirror_dynamic_block_parameters(init_parameters)
    assert (
        init_parameters["dynamic_workflows_blocks.workspace_resolver"]
        is NULL_WORKSPACE_RESOLVER
    )


def test_engine_mirror_respects_an_explicit_dynamic_namespace_value() -> None:
    from inference.core.workflows.execution_engine.v1.core import (
        _mirror_dynamic_block_parameters,
    )

    explicit = StubResolver()
    init_parameters = {"dynamic_workflows_blocks.workspace_resolver": explicit}
    _mirror_dynamic_block_parameters(init_parameters)
    assert init_parameters["dynamic_workflows_blocks.workspace_resolver"] is explicit


def test_compilation_and_execution_use_the_same_effective_resolver() -> None:
    """Round-2 defect 5: `create_dynamic_module` (Modal validation, compile
    time) and the generated block's `run` (execution) are two separate lookups.
    An explicit dynamic-namespace override must win in BOTH."""
    from inference.core.workflows.execution_engine.v1.compiler.core import (
        _effective_workspace_resolver,
    )

    core_resolver, dynamic_resolver = StubResolver("core"), StubResolver("dynamic")
    assert (
        _effective_workspace_resolver(
            {
                "workflows_core.workspace_resolver": core_resolver,
                "dynamic_workflows_blocks.workspace_resolver": dynamic_resolver,
            }
        )
        is dynamic_resolver
    )
    assert (
        _effective_workspace_resolver(
            {"workflows_core.workspace_resolver": core_resolver}
        )
        is core_resolver
    )
    assert _effective_workspace_resolver({}) is NULL_WORKSPACE_RESOLVER


def test_generated_block_declares_a_superset_of_the_two_parameters() -> None:
    # Superset, not equality: Phase 6 appends `execution_observer` to the same
    # list (fix-round-3-crossphase-9.md), and either order must pass.
    engine = ExecutionEngine.init(
        workflow_definition=DYNAMIC_WORKFLOW,
        init_parameters={"workflows_core.api_key": "k"},
    )
    step = engine._engine._compiled_workflow.steps["custom_block"].step
    assert {"api_key", "workspace_resolver"} <= set(type(step).get_init_parameters())
    assert engine.run(runtime_parameters={"a": 7}) == [{"out": 7}]


def test_effective_dynamic_api_key_prefers_an_explicit_dynamic_override() -> None:
    """Round-5 defect 2: compile-time validation must resolve the workspace
    with the SAME key the generated block executes with - the
    `dynamic_workflows_blocks.api_key` the engine mirrors (which a caller may
    override explicitly), not unconditionally `workflows_core.api_key`."""
    from inference.core.workflows.execution_engine.v1.compiler.core import (
        _effective_dynamic_api_key,
    )

    assert _effective_dynamic_api_key({"workflows_core.api_key": "core"}) == "core"
    assert (
        _effective_dynamic_api_key(
            {
                "workflows_core.api_key": "core",
                "dynamic_workflows_blocks.api_key": "dyn",
            }
        )
        == "dyn"
    )
    assert _effective_dynamic_api_key({}) is None


def test_dynamic_block_compilation_and_execution_share_the_effective_resolver_and_key(
    monkeypatch,
) -> None:
    """Round-3 defect 6 and round-5 defect 2, end to end: compile a REAL
    dynamic block in modal mode with different core / dynamic resolvers AND
    different core / dynamic api keys, then run it. Compile-time validation and
    run-time execution must ask the SAME resolver with the SAME key - the ones
    the generated block holds. `validate_code_in_modal` and `ModalExecutor`
    are stubbed on their module: block_scaffolding imports both lazily at call
    time, which is how the existing modal tests patch them too."""
    from inference.core.workflows.execution_engine.v1.dynamic_blocks import (
        block_scaffolding,
        modal_executor,
    )

    # Deterministic RED before Step 4 lands: the pre-port lookup (removed by
    # Step 4) must not reach the network. No-op once the attribute is gone.
    if hasattr(block_scaffolding, "get_roboflow_workspace"):
        monkeypatch.setattr(
            block_scaffolding, "get_roboflow_workspace", lambda api_key: None
        )
    with block_scaffolding._MODAL_EXECUTOR_CACHE_LOCK:
        block_scaffolding._MODAL_EXECUTOR_CACHE.clear()

    validated = []
    core, dynamic = StubResolver("core-ws"), StubResolver("dynamic-ws")
    executor = mock.MagicMock()
    executor.execute_remote.return_value = {"out": 1}
    monkeypatch.setattr(
        block_scaffolding, "WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE", "modal"
    )
    monkeypatch.setattr(
        modal_executor,
        "validate_code_in_modal",
        lambda python_code, workspace_id=None: validated.append(workspace_id) or True,
    )
    monkeypatch.setattr(
        modal_executor, "ModalExecutor", mock.MagicMock(return_value=executor)
    )
    engine = ExecutionEngine.init(
        # A distinct definition (different version string) so the plain
        # local compilation above can never be served for this one.
        workflow_definition={**DYNAMIC_WORKFLOW, "version": "1.0.1"},
        init_parameters={
            "workflows_core.api_key": "key-A",
            "dynamic_workflows_blocks.api_key": "key-B",
            "workflows_core.workspace_resolver": core,
            "dynamic_workflows_blocks.workspace_resolver": dynamic,
        },
    )
    assert validated == ["dynamic-ws"]
    step = engine._engine._compiled_workflow.steps["custom_block"].step
    assert step._workspace_resolver is dynamic
    assert step._api_key == "key-B"

    engine.run(runtime_parameters={"a": 1})

    # validation, then execution - the same resolver, the same key
    assert dynamic.calls == ["key-B", "key-B"]
    assert core.calls == []
    assert executor.execute_remote.call_args.kwargs["workspace_id"] == "dynamic-ws"
    with block_scaffolding._MODAL_EXECUTOR_CACHE_LOCK:
        block_scaffolding._MODAL_EXECUTOR_CACHE.clear()


def test_engine_works_on_a_private_copy_so_a_reused_caller_dictionary_is_not_contaminated() -> (
    None
):
    """Round-4 defect 1. `init` used to write the dynamic mirror into the
    caller's dictionary (HEAD leaks `dynamic_workflows_blocks.api_key` - E32).
    A caller that reuses one dict and swaps the core resolver must see the NEW
    resolver reach the next engine, its dict must stay clean, and a genuine
    explicit dynamic-namespace override must still win (E31)."""
    first_resolver, second_resolver = StubResolver("first"), StubResolver("second")
    caller = {
        "workflows_core.api_key": "k",
        "workflows_core.workspace_resolver": first_resolver,
    }
    first = ExecutionEngine.init(
        workflow_definition=DYNAMIC_WORKFLOW, init_parameters=caller
    )
    caller["workflows_core.workspace_resolver"] = second_resolver
    second = ExecutionEngine.init(
        workflow_definition=DYNAMIC_WORKFLOW, init_parameters=caller
    )

    assert not [
        key for key in caller if key.startswith("dynamic_workflows_blocks.")
    ], "the engine must work on a private copy of init_parameters"
    assert (
        first._engine._compiled_workflow.steps["custom_block"].step._workspace_resolver
        is first_resolver
    )
    assert (
        second._engine._compiled_workflow.steps["custom_block"].step._workspace_resolver
        is second_resolver
    )

    explicit = StubResolver("explicit")
    caller["dynamic_workflows_blocks.workspace_resolver"] = explicit
    third = ExecutionEngine.init(
        workflow_definition=DYNAMIC_WORKFLOW, init_parameters=caller
    )
    assert (
        third._engine._compiled_workflow.steps["custom_block"].step._workspace_resolver
        is explicit
    )
