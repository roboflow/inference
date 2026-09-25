"""`compile_workflow_structure` - inert structural compilation.

The structural path reuses the executable compiler's resolution, hoisting,
inlining, parsing, validation and graph construction, but:

* never initialises steps, never evaluates custom Python, never validates it in
  Modal, never resolves a workspace and never consults the host's custom-Python
  / tensor gates;
* never mutates the caller's definition;
* never reads or writes `COMPILATION_CACHE`.

`compile_workflow` / `compile_workflow_graph` keep their behaviour.
"""

import copy
import os
from typing import Any, Dict, List, Optional
from unittest import mock

import pytest
from roboflow_workflows.errors import (
    DynamicBlockError,
    InvalidReferenceTargetError,
    WorkflowEnvironmentConfigurationError,
    WorkflowSyntaxError,
)
from roboflow_workflows.execution_engine.constants import (
    NODE_COMPILATION_OUTPUT_PROPERTY,
)
from roboflow_workflows.execution_engine.v1.compiler import core as compiler_core
from roboflow_workflows.execution_engine.v1.compiler.core import (
    COMPILATION_CACHE,
    compile_workflow_graph,
    compile_workflow_structure,
)
from roboflow_workflows.execution_engine.v1.compiler.entities import (
    StepNode,
    StructuralCompilationResult,
)
from roboflow_workflows.execution_engine.v1.core import EXECUTION_ENGINE_V1_VERSION
from roboflow_workflows.execution_engine.v1.dynamic_blocks import (
    block_assembler,
    block_scaffolding,
    modal_executor,
)
from roboflow_workflows.execution_engine.v1.dynamic_blocks.entities import BLOCK_SOURCE
from roboflow_workflows.execution_engine.v1.inner_workflow.constants import (
    USE_INNER_WORKFLOW_BLOCK_TYPE,
)
from roboflow_workflows.execution_engine.v1.inner_workflow.reference_resolution import (
    WORKFLOWS_CORE_INNER_WORKFLOW_SPEC_RESOLVER,
)
from roboflow_workflows.prototypes.models_provider import ModelsProvider

from tests.unit_tests.execution_engine.dynamic_blocs._workspace_resolver_stub import (
    StubResolver,
)

OBJECT_DETECTION_MODEL = "roboflow_core/roboflow_object_detection_model@v3"


def _side_effect_code(marker_path: str) -> str:
    return f"""
import pathlib
pathlib.Path({marker_path!r}).write_text("executed")
raise RuntimeError("module-level side effect must never run during inspection")


def run(self, predictions):
    return {{"output": predictions}}
"""


def _dynamic_block_definition(
    marker_path: str,
    block_type: str = "SideEffectBlock",
    tensor_compatibility: Optional[str] = None,
) -> dict:
    manifest = {
        "type": "ManifestDescription",
        "block_type": block_type,
        "inputs": {
            "predictions": {
                "type": "DynamicInputDefinition",
                "selector_types": ["step_output"],
                "selector_data_kind": {"step_output": ["object_detection_prediction"]},
            }
        },
        "outputs": {
            "output": {
                "type": "DynamicOutputDefinition",
                "kind": ["object_detection_prediction"],
            }
        },
    }
    if tensor_compatibility is not None:
        manifest["tensor_compatibility"] = tensor_compatibility
    return {
        "type": "DynamicBlockDefinition",
        "manifest": manifest,
        "code": {
            "type": "PythonCode",
            "run_function_code": _side_effect_code(marker_path=marker_path),
        },
    }


def _plain_definition(model_id: str = "my_project/3") -> dict:
    return {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": [
            {
                "type": OBJECT_DETECTION_MODEL,
                "name": "model",
                "images": "$inputs.image",
                "model_id": model_id,
            }
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "predictions",
                "selector": "$steps.model.predictions",
            }
        ],
    }


def _definition_with_dynamic_block(
    marker_path: str, tensor_compatibility: Optional[str] = None
) -> dict:
    return {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "dynamic_blocks_definitions": [
            _dynamic_block_definition(
                marker_path=marker_path, tensor_compatibility=tensor_compatibility
            )
        ],
        "steps": [
            {
                "type": OBJECT_DETECTION_MODEL,
                "name": "model",
                "images": "$inputs.image",
                "model_id": "my_project/3",
            },
            {
                "type": "SideEffectBlock",
                "name": "custom",
                "predictions": "$steps.model.predictions",
            },
        ],
        "outputs": [
            {"type": "JsonField", "name": "result", "selector": "$steps.custom.output"}
        ],
    }


def _child_definition(marker_path: str, block_type: str = "ChildBlock") -> dict:
    return {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "dynamic_blocks_definitions": [
            _dynamic_block_definition(marker_path=marker_path, block_type=block_type)
        ],
        "steps": [
            {
                "type": OBJECT_DETECTION_MODEL,
                "name": "model",
                "images": "$inputs.image",
                "model_id": "child_project/1",
            },
            {
                "type": block_type,
                "name": "custom",
                "predictions": "$steps.model.predictions",
            },
        ],
        "outputs": [
            {"type": "JsonField", "name": "out", "selector": "$steps.custom.output"}
        ],
    }


def _definition_with_embedded_child(child: dict) -> dict:
    return {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": [
            {
                "type": USE_INNER_WORKFLOW_BLOCK_TYPE,
                "name": "inner",
                "workflow_definition": child,
                "parameter_bindings": {"image": "$inputs.image"},
            }
        ],
        "outputs": [
            {"type": "JsonField", "name": "from_child", "selector": "$steps.inner.out"}
        ],
    }


def _definition_with_remote_dispatch(child_step_extra: Dict[str, Any]) -> dict:
    child_step = {
        "type": USE_INNER_WORKFLOW_BLOCK_TYPE,
        "name": "dispatch",
        "execution_mode": "remote_dispatch",
        "parameter_bindings": {"image": "$inputs.image"},
    }
    child_step.update(child_step_extra)
    return {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": [
            {
                "type": OBJECT_DETECTION_MODEL,
                "name": "model",
                "images": "$inputs.image",
                "model_id": "my_project/3",
            },
            child_step,
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "predictions",
                "selector": "$steps.model.predictions",
            }
        ],
    }


def _step_names(result: StructuralCompilationResult) -> List[str]:
    return [step.name for step in result.parsed_workflow_definition.steps]


def _step_node(result: StructuralCompilationResult, selector: str) -> StepNode:
    node = result.execution_graph.nodes[selector][NODE_COMPILATION_OUTPUT_PROPERTY]
    assert isinstance(node, StepNode)
    return node


def _cache_snapshot() -> Dict[str, Any]:
    return {
        "entries": dict(COMPILATION_CACHE._cache),
        "keys": list(COMPILATION_CACHE._keys_buffer),
    }


@pytest.fixture
def marker_path(empty_directory: str) -> str:
    return os.path.join(empty_directory, "side_effect_marker.txt")


@pytest.fixture
def inertness_spies():
    resolver = StubResolver(workspace="some-workspace")
    models_provider = mock.MagicMock(spec=ModelsProvider)
    with mock.patch.object(
        compiler_core, "initialise_steps"
    ) as initialise_steps, mock.patch.object(
        compiler_core, "load_initializers"
    ) as load_initializers, mock.patch.object(
        block_scaffolding, "create_dynamic_module"
    ) as create_dynamic_module, mock.patch.object(
        block_scaffolding, "exec", create=True
    ) as exec_spy, mock.patch.object(
        modal_executor, "validate_code_in_modal"
    ) as validate_code_in_modal:
        yield {
            "initialise_steps": initialise_steps,
            "load_initializers": load_initializers,
            "create_dynamic_module": create_dynamic_module,
            "exec": exec_spy,
            "validate_code_in_modal": validate_code_in_modal,
            "resolver": resolver,
            "models_provider": models_provider,
            "init_parameters": {
                "workflows_core.api_key": "secret",
                "workflows_core.workspace_resolver": resolver,
                "workflows_core.model_manager": models_provider,
            },
        }


def _assert_no_code_execution(spies: dict, marker_path: str) -> None:
    """Holds after a structural compile AND after an executable compile that
    was stopped by a gate (the executable path loads initializers before the
    gate fires, so `_assert_inert` is too strict for it)."""
    spies["initialise_steps"].assert_not_called()
    spies["create_dynamic_module"].assert_not_called()
    spies["exec"].assert_not_called()
    spies["validate_code_in_modal"].assert_not_called()
    spies["models_provider"].add_model.assert_not_called()
    assert spies["resolver"].calls == []
    assert not os.path.exists(marker_path)


def _assert_inert(spies: dict, marker_path: str) -> None:
    _assert_no_code_execution(spies=spies, marker_path=marker_path)
    spies["load_initializers"].assert_not_called()


# ---------------------------------------------------------------------------
# inertness
# ---------------------------------------------------------------------------


def test_structural_compilation_builds_graph_without_executing_anything(
    inertness_spies: dict, marker_path: str
) -> None:
    # given
    definition = _definition_with_dynamic_block(marker_path=marker_path)

    # when
    result = compile_workflow_structure(
        workflow_definition=definition,
        init_parameters=inertness_spies["init_parameters"],
        execution_engine_version=EXECUTION_ENGINE_V1_VERSION,
    )

    # then
    _assert_inert(spies=inertness_spies, marker_path=marker_path)
    assert isinstance(result, StructuralCompilationResult)
    assert _step_names(result) == ["model", "custom"]
    assert set(result.execution_graph.nodes) == {
        "$inputs.image",
        "$steps.model",
        "$steps.custom",
        "$outputs.result",
    }
    dynamic_blocks = [
        block for block in result.available_blocks if block.block_source == BLOCK_SOURCE
    ]
    assert len(dynamic_blocks) == 1
    assert dynamic_blocks[0].block_class._structural_placeholder is True
    assert _step_node(result, "$steps.model").output_dimensionality == 1
    assert _step_node(result, "$steps.custom").output_dimensionality == 1


def test_structural_compilation_accepts_string_engine_version(
    inertness_spies: dict, marker_path: str
) -> None:
    result = compile_workflow_structure(
        workflow_definition=_plain_definition(),
        execution_engine_version=str(EXECUTION_ENGINE_V1_VERSION),
    )
    assert _step_names(result) == ["model"]
    _assert_inert(spies=inertness_spies, marker_path=marker_path)


@mock.patch.object(block_assembler, "ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS", False)
@mock.patch.object(block_assembler, "WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE", "local")
def test_structural_compilation_succeeds_where_executable_compilation_refuses_custom_python(
    inertness_spies: dict, marker_path: str
) -> None:
    # given
    definition = _definition_with_dynamic_block(marker_path=marker_path)

    # when
    result = compile_workflow_structure(
        workflow_definition=definition,
        init_parameters=inertness_spies["init_parameters"],
    )

    # then
    assert _step_names(result) == ["model", "custom"]
    _assert_inert(spies=inertness_spies, marker_path=marker_path)
    with pytest.raises(WorkflowEnvironmentConfigurationError):
        compile_workflow_graph(
            workflow_definition=copy.deepcopy(definition),
            init_parameters=inertness_spies["init_parameters"],
        )
    _assert_no_code_execution(spies=inertness_spies, marker_path=marker_path)


@mock.patch.object(block_assembler, "ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS", False)
@mock.patch.object(block_assembler, "WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE", "modal")
@mock.patch.object(block_scaffolding, "WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE", "modal")
def test_structural_compilation_in_modal_mode_never_reaches_modal(
    inertness_spies: dict, marker_path: str
) -> None:
    # given
    definition = _definition_with_dynamic_block(marker_path=marker_path)

    # when
    result = compile_workflow_structure(
        workflow_definition=definition,
        init_parameters=inertness_spies["init_parameters"],
    )

    # then
    assert _step_names(result) == ["model", "custom"]
    _assert_inert(spies=inertness_spies, marker_path=marker_path)


@pytest.mark.parametrize(
    "tensor_flag, execution_mode",
    [(False, "local"), (False, "modal"), (True, "modal"), (True, "local")],
)
def test_structural_compilation_of_tensor_native_block_is_configuration_invariant(
    inertness_spies: dict, marker_path: str, tensor_flag: bool, execution_mode: str
) -> None:
    # given
    definition = _definition_with_dynamic_block(
        marker_path=marker_path, tensor_compatibility="tensor_native"
    )
    with mock.patch.object(
        block_assembler, "ENABLE_TENSOR_DATA_REPRESENTATION", tensor_flag
    ), mock.patch.object(
        block_assembler, "WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE", execution_mode
    ), mock.patch.object(
        block_assembler, "ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS", True
    ):
        # when
        result = compile_workflow_structure(
            workflow_definition=definition,
            init_parameters=inertness_spies["init_parameters"],
        )

        # then
        assert _step_names(result) == ["model", "custom"]
        _assert_inert(spies=inertness_spies, marker_path=marker_path)
        if not (tensor_flag and execution_mode == "local"):
            with pytest.raises(DynamicBlockError):
                compile_workflow_graph(
                    workflow_definition=copy.deepcopy(definition),
                    init_parameters=inertness_spies["init_parameters"],
                )
            _assert_no_code_execution(spies=inertness_spies, marker_path=marker_path)


# ---------------------------------------------------------------------------
# request immutability
# ---------------------------------------------------------------------------


def test_structural_compilation_does_not_mutate_caller_definition(
    inertness_spies: dict, marker_path: str
) -> None:
    # given - nested dynamic block definitions would be hoisted to the root by
    # the executable path (which mutates); the structural path must not.
    child = _child_definition(marker_path=marker_path)
    definition = _definition_with_embedded_child(child=child)
    definition["dynamic_blocks_definitions"] = [
        _dynamic_block_definition(marker_path=marker_path, block_type="RootBlock")
    ]
    frozen = copy.deepcopy(definition)

    # when
    result = compile_workflow_structure(
        workflow_definition=definition,
        init_parameters=inertness_spies["init_parameters"],
    )

    # then
    assert definition == frozen
    assert definition["steps"][0]["workflow_definition"] is child
    assert (
        child["dynamic_blocks_definitions"]
        == frozen["steps"][0]["workflow_definition"]["dynamic_blocks_definitions"]
    )
    assert _step_names(result) == ["inner__model", "inner__custom"]
    _assert_inert(spies=inertness_spies, marker_path=marker_path)


def test_structural_compilation_does_not_mutate_init_parameters(
    inertness_spies: dict, marker_path: str
) -> None:
    init_parameters = dict(inertness_spies["init_parameters"])
    frozen = dict(init_parameters)
    compile_workflow_structure(
        workflow_definition=_definition_with_dynamic_block(marker_path=marker_path),
        init_parameters=init_parameters,
    )
    assert init_parameters == frozen


# ---------------------------------------------------------------------------
# COMPILATION_CACHE isolation
# ---------------------------------------------------------------------------


def test_structural_compilation_never_touches_compilation_cache_structural_first(
    inertness_spies: dict, marker_path: str
) -> None:
    # given
    definition = _plain_definition(model_id="cache_probe_a/1")
    before = _cache_snapshot()

    # when
    structural = compile_workflow_structure(
        workflow_definition=definition,
        execution_engine_version=EXECUTION_ENGINE_V1_VERSION,
    )

    # then - nothing read, nothing written
    assert _cache_snapshot() == before

    # and executable compilation is unaffected
    executable_1 = compile_workflow_graph(
        workflow_definition=definition,
        execution_engine_version=EXECUTION_ENGINE_V1_VERSION,
    )
    after_executable = _cache_snapshot()
    structural_2 = compile_workflow_structure(
        workflow_definition=definition,
        execution_engine_version=EXECUTION_ENGINE_V1_VERSION,
    )
    assert _cache_snapshot() == after_executable
    executable_2 = compile_workflow_graph(
        workflow_definition=definition,
        execution_engine_version=EXECUTION_ENGINE_V1_VERSION,
    )
    assert executable_2 is executable_1, "Cache hit must be preserved"
    assert structural.execution_graph is not executable_1.execution_graph
    assert structural_2.execution_graph is not structural.execution_graph
    assert structural_2.execution_graph is not executable_1.execution_graph
    for cached in COMPILATION_CACHE._cache.values():
        assert cached.execution_graph is not structural.execution_graph
        assert cached.execution_graph is not structural_2.execution_graph


def test_structural_compilation_never_touches_compilation_cache_executable_first(
    inertness_spies: dict, marker_path: str
) -> None:
    # given
    definition = _plain_definition(model_id="cache_probe_b/1")
    executable_1 = compile_workflow_graph(
        workflow_definition=definition,
        execution_engine_version=EXECUTION_ENGINE_V1_VERSION,
    )
    after_executable = _cache_snapshot()

    # when
    structural = compile_workflow_structure(
        workflow_definition=definition,
        execution_engine_version=EXECUTION_ENGINE_V1_VERSION,
    )

    # then
    assert _cache_snapshot() == after_executable
    executable_2 = compile_workflow_graph(
        workflow_definition=definition,
        execution_engine_version=EXECUTION_ENGINE_V1_VERSION,
    )
    assert executable_2 is executable_1
    assert structural.execution_graph is not executable_1.execution_graph
    assert (
        structural.parsed_workflow_definition
        is not executable_1.parsed_workflow_definition
    )


def test_structural_compilation_results_are_fresh_per_call(
    inertness_spies: dict, marker_path: str
) -> None:
    definition = _definition_with_dynamic_block(marker_path=marker_path)
    first = compile_workflow_structure(workflow_definition=definition)
    second = compile_workflow_structure(workflow_definition=definition)
    assert first.execution_graph is not second.execution_graph
    assert first.parsed_workflow_definition is not second.parsed_workflow_definition
    assert _step_names(first) == _step_names(second)


# ---------------------------------------------------------------------------
# embedded inner workflows
# ---------------------------------------------------------------------------


def test_embedded_inner_workflow_is_inlined_and_its_dynamic_blocks_compiled_structurally(
    inertness_spies: dict, marker_path: str
) -> None:
    # given
    definition = _definition_with_embedded_child(
        child=_child_definition(marker_path=marker_path)
    )

    # when
    result = compile_workflow_structure(
        workflow_definition=definition,
        init_parameters=inertness_spies["init_parameters"],
    )

    # then
    assert _step_names(result) == ["inner__model", "inner__custom"]
    assert set(result.execution_graph.nodes) == {
        "$inputs.image",
        "$steps.inner__model",
        "$steps.inner__custom",
        "$outputs.from_child",
    }
    assert not any(
        step.type == USE_INNER_WORKFLOW_BLOCK_TYPE
        for step in result.parsed_workflow_definition.steps
    )
    dynamic_blocks = [
        block for block in result.available_blocks if block.block_source == BLOCK_SOURCE
    ]
    assert len(dynamic_blocks) == 1, "Child dynamic block compiled exactly once"
    assert dynamic_blocks[0].block_class._structural_placeholder is True
    _assert_inert(spies=inertness_spies, marker_path=marker_path)


def test_saved_inner_workflow_reference_is_resolved_through_injected_resolver(
    inertness_spies: dict, marker_path: str
) -> None:
    # given
    child = _child_definition(marker_path=marker_path)
    calls = []

    def resolver(workspace_id, workflow_id, workflow_version_id, init_parameters):
        calls.append((workspace_id, workflow_id, workflow_version_id))
        return copy.deepcopy(child)

    definition = {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": [
            {
                "type": USE_INNER_WORKFLOW_BLOCK_TYPE,
                "name": "inner",
                "workflow_workspace_id": "my-workspace",
                "workflow_id": "saved-workflow",
                "workflow_version_id": "3",
                "parameter_bindings": {"image": "$inputs.image"},
            }
        ],
        "outputs": [
            {"type": "JsonField", "name": "from_child", "selector": "$steps.inner.out"}
        ],
    }
    frozen = copy.deepcopy(definition)
    init_parameters = dict(inertness_spies["init_parameters"])
    init_parameters[WORKFLOWS_CORE_INNER_WORKFLOW_SPEC_RESOLVER] = resolver

    # when
    result = compile_workflow_structure(
        workflow_definition=definition,
        init_parameters=init_parameters,
    )

    # then
    assert calls == [("my-workspace", "saved-workflow", "3")]
    assert _step_names(result) == ["inner__model", "inner__custom"]
    assert definition == frozen, "Reference fields stay in the caller's dict"
    _assert_inert(spies=inertness_spies, marker_path=marker_path)


def test_saved_inner_workflow_reference_without_resolver_raises_existing_error(
    inertness_spies: dict, marker_path: str
) -> None:
    definition = {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": [
            {
                "type": USE_INNER_WORKFLOW_BLOCK_TYPE,
                "name": "inner",
                "workflow_workspace_id": "my-workspace",
                "workflow_id": "saved-workflow",
                "parameter_bindings": {"image": "$inputs.image"},
            }
        ],
        "outputs": [],
    }
    with pytest.raises(WorkflowEnvironmentConfigurationError):
        compile_workflow_structure(workflow_definition=definition)


# ---------------------------------------------------------------------------
# remote-dispatch inner workflows stay opaque
# ---------------------------------------------------------------------------


def test_remote_dispatch_child_is_kept_as_opaque_step(
    inertness_spies: dict, marker_path: str
) -> None:
    # given - the child uses a block that is NOT installed here and declares a
    # dynamic block with side effects; neither may be touched.
    child = {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "dynamic_blocks_definitions": [
            _dynamic_block_definition(marker_path=marker_path, block_type="RemoteOnly")
        ],
        "steps": [
            {
                "type": "not_installed_plugin/some_block@v1",
                "name": "remote_step",
                "image": "$inputs.image",
            }
        ],
        "outputs": [],
    }
    definition = _definition_with_remote_dispatch(
        child_step_extra={"workflow_definition": child}
    )
    frozen = copy.deepcopy(definition)

    # when
    result = compile_workflow_structure(
        workflow_definition=definition,
        init_parameters=inertness_spies["init_parameters"],
    )

    # then
    assert _step_names(result) == ["model", "dispatch"]
    dispatch_node = _step_node(result, "$steps.dispatch")
    assert dispatch_node.step_manifest.type == USE_INNER_WORKFLOW_BLOCK_TYPE
    assert dispatch_node.step_manifest.execution_mode == "remote_dispatch"
    assert dispatch_node.step_manifest.get_actual_outputs() == []
    assert not any(
        block.block_source == BLOCK_SOURCE for block in result.available_blocks
    ), "Child-only dynamic blocks of a dispatched workflow are never compiled"
    assert definition == frozen
    _assert_inert(spies=inertness_spies, marker_path=marker_path)


def test_remote_dispatch_child_by_reference_is_never_resolved(
    inertness_spies: dict, marker_path: str
) -> None:
    # given
    resolver = mock.MagicMock(side_effect=AssertionError("must not be called"))
    definition = _definition_with_remote_dispatch(
        child_step_extra={
            "workflow_workspace_id": "my-workspace",
            "workflow_id": "saved-workflow",
        }
    )
    init_parameters = dict(inertness_spies["init_parameters"])
    init_parameters[WORKFLOWS_CORE_INNER_WORKFLOW_SPEC_RESOLVER] = resolver

    # when
    result = compile_workflow_structure(
        workflow_definition=definition,
        init_parameters=init_parameters,
    )

    # then
    resolver.assert_not_called()
    assert _step_names(result) == ["model", "dispatch"]
    _assert_inert(spies=inertness_spies, marker_path=marker_path)


# ---------------------------------------------------------------------------
# errors keep their existing types
# ---------------------------------------------------------------------------


def test_structural_compilation_raises_syntax_error_for_unknown_block(
    inertness_spies: dict, marker_path: str
) -> None:
    definition = _plain_definition()
    definition["steps"][0]["type"] = "not_installed_plugin/some_block@v1"
    with pytest.raises(WorkflowSyntaxError):
        compile_workflow_structure(workflow_definition=definition)


def test_structural_compilation_raises_reference_error_for_dangling_selector(
    inertness_spies: dict, marker_path: str
) -> None:
    definition = _plain_definition()
    definition["outputs"][0]["selector"] = "$steps.missing.predictions"
    with pytest.raises(InvalidReferenceTargetError):
        compile_workflow_structure(workflow_definition=definition)


# ---------------------------------------------------------------------------
# executable compilation keeps its behaviour
# ---------------------------------------------------------------------------


def test_executable_compilation_still_hoists_and_mutates_and_executes(
    marker_path: str,
) -> None:
    # given - pins the executable contract that structural mode deliberately
    # differs from: the caller's dict receives hoisted definitions and the
    # dynamic code is evaluated.
    child = _child_definition(marker_path=marker_path)
    definition = _definition_with_embedded_child(child=child)
    with mock.patch.object(
        block_assembler, "ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS", True
    ), mock.patch.object(
        block_assembler, "WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE", "local"
    ), mock.patch.object(
        block_scaffolding, "WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE", "local"
    ):
        # when
        with pytest.raises(Exception) as error:
            compile_workflow_graph(
                workflow_definition=definition,
                init_parameters={"workflows_core.api_key": None},
            )

    # then
    assert "module-level side effect" in str(error.value)
    assert os.path.exists(marker_path)
    assert "dynamic_blocks_definitions" in definition
