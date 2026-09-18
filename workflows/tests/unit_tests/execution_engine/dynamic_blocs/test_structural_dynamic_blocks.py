"""Structural (inert) compilation of dynamic blocks.

`compile_dynamic_blocks(..., structural=True)` must build the manifest exactly
like the executable path but never evaluate user code, never validate it in
Modal, never resolve a workspace and never consult the host's custom-Python /
tensor gates. The generated block class is a placeholder whose `__init__` and
`run` refuse to execute.
"""

import os
from typing import Optional
from unittest import mock

import pytest
from roboflow_workflows.errors import (
    DynamicBlockError,
    WorkflowEnvironmentConfigurationError,
)
from roboflow_workflows.execution_engine.entities.types import WILDCARD_KIND, Kind
from roboflow_workflows.execution_engine.v1.dynamic_blocks import (
    block_assembler,
    block_scaffolding,
    modal_executor,
)
from roboflow_workflows.execution_engine.v1.dynamic_blocks.block_assembler import (
    compile_dynamic_blocks,
    create_dynamic_block_specification,
)
from roboflow_workflows.execution_engine.v1.dynamic_blocks.block_scaffolding import (
    assembly_custom_python_block,
)
from roboflow_workflows.execution_engine.v1.dynamic_blocks.entities import (
    BLOCK_SOURCE,
    DynamicBlockDefinition,
    DynamicInputDefinition,
    DynamicOutputDefinition,
    ManifestDescription,
    PythonCode,
    SelectorType,
    TensorCompatibility,
)

from tests.unit_tests.execution_engine.dynamic_blocs._workspace_resolver_stub import (
    StubResolver,
)

KINDS_LOOKUP = {
    "*": WILDCARD_KIND,
    "string": Kind(name="string"),
    "integer": Kind(name="integer"),
}


def _side_effect_code(marker_path: str) -> str:
    # Module-level side effects: executed by `exec` in the executable path, and
    # they must NEVER run during structural compilation.
    return f"""
import pathlib
pathlib.Path({marker_path!r}).write_text("executed")
raise RuntimeError("module-level side effect must never run during inspection")


def run(self, a):
    return {{"output": a}}
"""


def _definition(
    marker_path: str,
    block_type: str = "SideEffectBlock",
    tensor_compatibility: Optional[str] = None,
) -> dict:
    manifest = {
        "type": "ManifestDescription",
        "block_type": block_type,
        "inputs": {
            "a": {
                "type": "DynamicInputDefinition",
                "selector_types": ["step_output"],
                "selector_data_kind": {"step_output": ["string", "integer"]},
            }
        },
        "outputs": {"output": {"type": "DynamicOutputDefinition", "kind": ["string"]}},
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


@pytest.fixture
def marker_path(empty_directory: str) -> str:
    return os.path.join(empty_directory, "side_effect_marker.txt")


@pytest.fixture
def inertness_spies():
    """Every path through which inspection could leak into execution."""
    resolver = StubResolver(workspace="some-workspace")
    with mock.patch.object(
        block_scaffolding, "create_dynamic_module"
    ) as create_dynamic_module, mock.patch.object(
        block_scaffolding, "exec", create=True
    ) as exec_spy, mock.patch.object(
        modal_executor, "validate_code_in_modal"
    ) as validate_code_in_modal, mock.patch.object(
        block_assembler,
        "load_all_defined_kinds",
        return_value=list(KINDS_LOOKUP.values()),
    ):
        yield {
            "create_dynamic_module": create_dynamic_module,
            "exec": exec_spy,
            "validate_code_in_modal": validate_code_in_modal,
            "resolver": resolver,
        }


def _assert_inert(spies: dict, marker_path: str) -> None:
    spies["create_dynamic_module"].assert_not_called()
    spies["exec"].assert_not_called()
    spies["validate_code_in_modal"].assert_not_called()
    assert spies["resolver"].calls == [], "Workspace must never be resolved"
    assert not os.path.exists(marker_path), "User code must never run"


# ---------------------------------------------------------------------------
# inertness
# ---------------------------------------------------------------------------


def test_structural_compilation_never_evaluates_user_code(
    inertness_spies: dict, marker_path: str
) -> None:
    # when
    result = compile_dynamic_blocks(
        dynamic_blocks_definitions=[_definition(marker_path=marker_path)],
        api_key="secret",
        workspace_resolver=inertness_spies["resolver"],
        structural=True,
    )

    # then
    _assert_inert(spies=inertness_spies, marker_path=marker_path)
    assert len(result) == 1
    assert result[0].block_source == BLOCK_SOURCE
    assert result[0].manifest_class.model_validate(
        {"type": "SideEffectBlock", "name": "custom", "a": "$steps.other.value"}
    )


def test_executable_compilation_of_the_same_code_executes_it(
    marker_path: str,
) -> None:
    # given - the sentinel is real: with the gates open, the executable path
    # execs the module and the top-level `raise` surfaces as a code error.
    with mock.patch.object(
        block_assembler, "ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS", True
    ), mock.patch.object(
        block_assembler, "WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE", "local"
    ), mock.patch.object(
        block_scaffolding, "WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE", "local"
    ), mock.patch.object(
        block_assembler,
        "load_all_defined_kinds",
        return_value=list(KINDS_LOOKUP.values()),
    ):
        # when
        with pytest.raises(Exception) as error:
            compile_dynamic_blocks(
                dynamic_blocks_definitions=[_definition(marker_path=marker_path)]
            )

    # then
    assert "module-level side effect" in str(error.value)
    assert os.path.exists(marker_path), "Executable compilation runs the code"


@mock.patch.object(block_assembler, "ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS", False)
@mock.patch.object(block_assembler, "WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE", "local")
def test_structural_compilation_ignores_custom_python_allowance_gate(
    inertness_spies: dict, marker_path: str
) -> None:
    # when
    result = compile_dynamic_blocks(
        dynamic_blocks_definitions=[_definition(marker_path=marker_path)],
        workspace_resolver=inertness_spies["resolver"],
        structural=True,
    )

    # then
    assert len(result) == 1
    _assert_inert(spies=inertness_spies, marker_path=marker_path)

    # and the executable path still enforces the gate
    with pytest.raises(WorkflowEnvironmentConfigurationError):
        compile_dynamic_blocks(
            dynamic_blocks_definitions=[_definition(marker_path=marker_path)],
            workspace_resolver=inertness_spies["resolver"],
        )
    _assert_inert(spies=inertness_spies, marker_path=marker_path)


@mock.patch.object(block_assembler, "ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS", False)
@mock.patch.object(block_assembler, "WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE", "modal")
@mock.patch.object(block_scaffolding, "WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE", "modal")
def test_structural_compilation_in_modal_mode_neither_validates_remotely_nor_resolves_workspace(
    marker_path: str,
) -> None:
    # given - only the remote validator and resolver are spied, so the
    # executable path is free to reach the Modal branch of
    # `create_dynamic_module` and prove the spies sit on the right seam.
    resolver = StubResolver(workspace="some-workspace")
    with mock.patch.object(
        modal_executor, "validate_code_in_modal", return_value=True
    ) as validate_code_in_modal, mock.patch.object(
        block_assembler,
        "load_all_defined_kinds",
        return_value=list(KINDS_LOOKUP.values()),
    ):
        # when
        result = compile_dynamic_blocks(
            dynamic_blocks_definitions=[_definition(marker_path=marker_path)],
            api_key="secret",
            workspace_resolver=resolver,
            structural=True,
        )

        # then
        assert len(result) == 1
        validate_code_in_modal.assert_not_called()
        assert resolver.calls == []
        assert not os.path.exists(marker_path)

        # and the executable path does validate remotely with the resolved workspace
        compile_dynamic_blocks(
            dynamic_blocks_definitions=[_definition(marker_path=marker_path)],
            api_key="secret",
            workspace_resolver=resolver,
        )
        validate_code_in_modal.assert_called_once()
        assert validate_code_in_modal.call_args.args[1] == "some-workspace"
        assert resolver.calls == ["secret"]
        assert not os.path.exists(marker_path), "Modal mode never execs locally"


@pytest.mark.parametrize(
    "tensor_flag, execution_mode, expected_message",
    [
        (False, "local", "numpy data representation"),
        (False, "modal", "numpy data representation"),
        (True, "modal", "not yet supported for remote"),
    ],
)
def test_structural_compilation_ignores_tensor_gates_that_executable_compilation_enforces(
    inertness_spies: dict,
    marker_path: str,
    tensor_flag: bool,
    execution_mode: str,
    expected_message: str,
) -> None:
    # given
    definition = _definition(
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
        result = compile_dynamic_blocks(
            dynamic_blocks_definitions=[definition],
            workspace_resolver=inertness_spies["resolver"],
            structural=True,
        )

        # then
        assert len(result) == 1
        assert (
            result[0].block_class._manifest_description.tensor_compatibility
            is TensorCompatibility.TENSOR_NATIVE
        )
        _assert_inert(spies=inertness_spies, marker_path=marker_path)

        # and the executable path still fails fast
        with pytest.raises(DynamicBlockError) as error:
            compile_dynamic_blocks(
                dynamic_blocks_definitions=[definition],
                workspace_resolver=inertness_spies["resolver"],
            )
        assert expected_message in str(error.value)
        _assert_inert(spies=inertness_spies, marker_path=marker_path)


@pytest.mark.parametrize("tensor_flag", [True, False])
def test_structural_compilation_of_legacy_compatibility_block_is_flag_invariant(
    inertness_spies: dict, marker_path: str, tensor_flag: bool
) -> None:
    with mock.patch.object(
        block_assembler, "ENABLE_TENSOR_DATA_REPRESENTATION", tensor_flag
    ):
        result = compile_dynamic_blocks(
            dynamic_blocks_definitions=[_definition(marker_path=marker_path)],
            structural=True,
        )
    assert (
        result[0].block_class._manifest_description.tensor_compatibility
        is TensorCompatibility.LEGACY_COMPATIBILITY
    )
    _assert_inert(spies=inertness_spies, marker_path=marker_path)


def test_structural_compilation_with_no_definitions_returns_empty_list() -> None:
    assert compile_dynamic_blocks(dynamic_blocks_definitions=[], structural=True) == []


# ---------------------------------------------------------------------------
# structural validation of the manifest still applies
# ---------------------------------------------------------------------------


def test_structural_compilation_still_validates_declared_kinds(
    inertness_spies: dict, marker_path: str
) -> None:
    # given
    definition = _definition(marker_path=marker_path)
    definition["manifest"]["inputs"]["a"]["selector_data_kind"] = {
        "step_output": ["no_such_kind"]
    }

    # when
    with pytest.raises(DynamicBlockError):
        compile_dynamic_blocks(
            dynamic_blocks_definitions=[definition],
            structural=True,
        )
    _assert_inert(spies=inertness_spies, marker_path=marker_path)


def test_structural_compilation_still_validates_dimensionality_reference(
    inertness_spies: dict, marker_path: str
) -> None:
    # given
    definition = _definition(marker_path=marker_path)
    definition["manifest"]["inputs"]["a"]["is_dimensionality_reference"] = True
    definition["manifest"]["inputs"]["b"] = {
        "type": "DynamicInputDefinition",
        "selector_types": ["step_output"],
        "is_dimensionality_reference": True,
    }

    # when
    with pytest.raises(DynamicBlockError):
        compile_dynamic_blocks(
            dynamic_blocks_definitions=[definition],
            structural=True,
        )
    _assert_inert(spies=inertness_spies, marker_path=marker_path)


def test_structural_and_executable_manifests_expose_the_same_declarations(
    marker_path: str,
) -> None:
    # given
    definition = DynamicBlockDefinition.model_validate(
        _definition(marker_path=marker_path)
    )
    definition.manifest.output_dimensionality_offset = 1
    definition.manifest.accepts_batch_input = True

    # when
    structural = create_dynamic_block_specification(
        dynamic_block_definition=definition,
        kinds_lookup=KINDS_LOOKUP,
        structural=True,
    )
    executable = create_dynamic_block_specification(
        dynamic_block_definition=definition,
        kinds_lookup=KINDS_LOOKUP,
        skip_class_eval=True,
    )

    # then
    for method in [
        "describe_outputs",
        "accepts_batch_input",
        "accepts_empty_values",
        "get_input_dimensionality_offsets",
        "get_dimensionality_reference_property",
        "get_output_dimensionality_offset",
        "get_parameters_accepting_batches",
        "get_parameters_accepting_batches_and_scalars",
        "get_parameters_enforcing_auto_batch_casting",
        "get_execution_engine_compatibility",
    ]:
        assert (
            getattr(structural.manifest_class, method)()
            == getattr(executable.manifest_class, method)()
        ), f"`{method}` must agree between structural and executable manifests"
    assert structural.block_class.get_init_parameters() == (
        executable.block_class.get_init_parameters()
    )
    assert structural.block_class.get_manifest() is structural.manifest_class


# ---------------------------------------------------------------------------
# placeholder block
# ---------------------------------------------------------------------------


def test_structural_placeholder_refuses_initialisation(
    inertness_spies: dict, marker_path: str
) -> None:
    # given
    (specification,) = compile_dynamic_blocks(
        dynamic_blocks_definitions=[_definition(marker_path=marker_path)],
        structural=True,
    )

    # when
    with pytest.raises(DynamicBlockError) as error:
        specification.block_class()

    # then
    assert "structural placeholder" in str(error.value)
    assert not os.path.exists(marker_path)


def test_structural_placeholder_refuses_to_run(
    inertness_spies: dict, marker_path: str
) -> None:
    # given
    (specification,) = compile_dynamic_blocks(
        dynamic_blocks_definitions=[_definition(marker_path=marker_path)],
        structural=True,
    )
    instance = object.__new__(specification.block_class)

    # when
    with pytest.raises(DynamicBlockError) as error:
        instance.run(a="value")

    # then
    assert "structural placeholder" in str(error.value)
    assert not os.path.exists(marker_path)


def test_structural_placeholder_is_marked_and_keeps_manifest_description(
    inertness_spies: dict, marker_path: str
) -> None:
    (specification,) = compile_dynamic_blocks(
        dynamic_blocks_definitions=[_definition(marker_path=marker_path)],
        structural=True,
    )
    assert specification.block_class._structural_placeholder is True
    assert isinstance(
        specification.block_class._manifest_description, ManifestDescription
    )
    assert specification.block_class._usage_block_type == "SideEffectBlock"


def test_assembly_custom_python_block_structural_short_circuits_module_creation(
    marker_path: str,
) -> None:
    # given
    manifest_description = ManifestDescription(
        type="ManifestDescription",
        block_type="SideEffectBlock",
        inputs={
            "a": DynamicInputDefinition(
                type="DynamicInputDefinition",
                selector_types=[SelectorType.STEP_OUTPUT],
            )
        },
        outputs={"output": DynamicOutputDefinition(type="DynamicOutputDefinition")},
    )
    manifest = block_assembler.assembly_dynamic_block_manifest(
        unique_identifier="uid",
        manifest_description=manifest_description,
        kinds_lookup=KINDS_LOOKUP,
    )
    python_code = PythonCode(
        type="PythonCode", run_function_code=_side_effect_code(marker_path)
    )
    resolver = StubResolver(workspace="ws")

    # when
    with mock.patch.object(
        block_scaffolding, "create_dynamic_module"
    ) as create_dynamic_module:
        block_class = assembly_custom_python_block(
            block_type_name="SideEffectBlock",
            unique_identifier="uid",
            manifest=manifest,
            python_code=python_code,
            api_key="secret",
            workspace_resolver=resolver,
            manifest_description=manifest_description,
            structural=True,
        )

    # then
    create_dynamic_module.assert_not_called()
    assert resolver.calls == []
    assert block_class.get_manifest() is manifest
    with pytest.raises(DynamicBlockError):
        block_class()


# ---------------------------------------------------------------------------
# workload declarations attached to dynamic manifests (both modes)
# ---------------------------------------------------------------------------


@pytest.fixture
def workload():
    return pytest.importorskip(
        "roboflow_workflows.execution_engine.entities.workload",
        reason="foundation's workload entities are not available yet",
    )


def _manifest_instance(
    specification, name: str = "custom", block_type: str = "SideEffectBlock"
):
    return specification.manifest_class.model_validate(
        {"type": block_type, "name": name, "a": "$steps.other.value"}
    )


@pytest.mark.parametrize("structural", [True, False])
def test_dynamic_manifest_declares_custom_python_operation_as_incomplete(
    workload, marker_path: str, structural: bool
) -> None:
    # given
    definition = DynamicBlockDefinition.model_validate(
        _definition(marker_path=marker_path)
    )
    specification = create_dynamic_block_specification(
        dynamic_block_definition=definition,
        kinds_lookup=KINDS_LOOKUP,
        skip_class_eval=not structural,
        structural=structural,
    )
    manifest = _manifest_instance(specification, name="custom")

    # when
    result = manifest.discover_work_operations()

    # then
    assert isinstance(result, workload.Discovery)
    assert result.items == [workload.WorkOperation.CUSTOM_PYTHON]
    assert result.complete is False
    assert result.unknown_reasons == [
        "custom_python_internal_operations_unknown:$steps.custom"
    ]


@pytest.mark.parametrize("structural", [True, False])
def test_dynamic_manifest_declares_execution_disabled_restriction(
    workload, marker_path: str, structural: bool
) -> None:
    # given
    definition = DynamicBlockDefinition.model_validate(
        _definition(marker_path=marker_path)
    )
    specification = create_dynamic_block_specification(
        dynamic_block_definition=definition,
        kinds_lookup=KINDS_LOOKUP,
        skip_class_eval=not structural,
        structural=structural,
    )
    manifest = _manifest_instance(specification, name="my_step")

    # when
    result = manifest.discover_portable_restrictions()

    # then
    assert isinstance(result, workload.Discovery)
    assert result.complete is False
    assert result.unknown_reasons == [
        "custom_python_internal_restrictions_unknown:$steps.my_step"
    ]
    assert result.items == [
        workload.RestrictionMetadata(
            code="custom_python_execution_disabled",
            severity=workload.Severity.HARD,
            when=workload.RestrictionCondition(
                configuration_equals={
                    "ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS": False,
                    "WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE": "local",
                }
            ),
        )
    ]


def test_tensor_native_dynamic_manifest_declares_tensor_restrictions(
    workload, marker_path: str
) -> None:
    # given
    definition = DynamicBlockDefinition.model_validate(
        _definition(marker_path=marker_path, tensor_compatibility="tensor_native")
    )
    specification = create_dynamic_block_specification(
        dynamic_block_definition=definition,
        kinds_lookup=KINDS_LOOKUP,
        structural=True,
    )
    manifest = _manifest_instance(specification, name="native")

    # when
    result = manifest.discover_portable_restrictions()

    # then
    assert result.complete is False
    assert {item.code for item in result.items} == {
        "custom_python_execution_disabled",
        "tensor_native_requires_tensor_representation",
        "tensor_native_unsupported_in_modal",
    }
    by_code = {item.code: item for item in result.items}
    assert all(item.severity is workload.Severity.HARD for item in result.items)
    assert by_code["tensor_native_requires_tensor_representation"].when == (
        workload.RestrictionCondition(
            configuration_equals={"ENABLE_TENSOR_DATA_REPRESENTATION": False}
        )
    )
    assert by_code["tensor_native_unsupported_in_modal"].when == (
        workload.RestrictionCondition(
            configuration_equals={"WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE": "modal"}
        )
    )


def test_dynamic_manifest_declarations_are_independent_of_host_flags(
    workload, marker_path: str
) -> None:
    # given - the declaration is conditional on the TARGET configuration, so
    # flipping this host's flags must not change it.
    definition = DynamicBlockDefinition.model_validate(
        _definition(marker_path=marker_path, tensor_compatibility="tensor_native")
    )
    results = []
    for tensor_flag, execution_mode, allow in [
        (True, "local", True),
        (False, "modal", False),
    ]:
        with mock.patch.object(
            block_assembler, "ENABLE_TENSOR_DATA_REPRESENTATION", tensor_flag
        ), mock.patch.object(
            block_assembler, "WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE", execution_mode
        ), mock.patch.object(
            block_assembler, "ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS", allow
        ):
            specification = create_dynamic_block_specification(
                dynamic_block_definition=definition,
                kinds_lookup=KINDS_LOOKUP,
                structural=True,
            )
            manifest = _manifest_instance(specification, name="native")
            results.append(
                (
                    manifest.discover_work_operations(),
                    manifest.discover_portable_restrictions(),
                )
            )

    # then
    assert results[0] == results[1]


def test_dynamic_manifest_never_reports_selected_backend(workload, marker_path):
    definition = DynamicBlockDefinition.model_validate(
        _definition(marker_path=marker_path)
    )
    specification = create_dynamic_block_specification(
        dynamic_block_definition=definition,
        kinds_lookup=KINDS_LOOKUP,
        structural=True,
    )
    manifest = _manifest_instance(specification)
    restrictions = manifest.discover_portable_restrictions()
    operations = manifest.discover_work_operations()
    dumped = restrictions.model_dump_json() + operations.model_dump_json()
    assert "selected_backend" not in dumped
    assert workload.WorkOperation.EXTERNAL_REQUEST not in operations.items
