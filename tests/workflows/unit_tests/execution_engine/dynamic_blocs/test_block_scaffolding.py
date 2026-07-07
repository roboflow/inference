from unittest import mock

import pytest

from inference.core.workflows.core_steps.formatters.expression.v1 import BlockManifest
from inference.core.workflows.errors import (
    DynamicBlockCodeError,
    DynamicBlockError,
    WorkflowEnvironmentConfigurationError,
)
from inference.core.workflows.execution_engine.v1.dynamic_blocks import (
    block_scaffolding,
)
from inference.core.workflows.execution_engine.v1.dynamic_blocks.block_scaffolding import (
    assembly_custom_python_block,
    create_dynamic_module,
)
from inference.core.workflows.execution_engine.v1.dynamic_blocks.debug_logs import (
    register_debug_session,
)
from inference.core.workflows.execution_engine.v1.dynamic_blocks.entities import (
    PythonCode,
)


def test_create_dynamic_module_when_syntax_error_happens() -> None:
    # given
    init_function = """
def init_fun() -> Dict[str, Any]:
    return {"a": 35}    
"""
    run_function = """
def run_function( -> BlockResult:
    return {"result": a + b}
"""
    python_code = PythonCode(
        type="PythonCode",
        run_function_code=run_function,
        run_function_name="run_function",
        init_function_code=init_function,
        init_function_name="init_fun",
        imports=["import math"],
    )

    # when
    with pytest.raises(DynamicBlockCodeError):
        _ = create_dynamic_module(
            block_type_name="some", python_code=python_code, module_name="my_module"
        )


def test_create_dynamic_module_when_creation_should_succeed() -> None:
    # given
    init_function = """
def init_fun() -> Dict[str, Any]:
    return {"a": 35}    
"""
    run_function = """
def run_function(a, b) -> BlockResult:
    return {"result": a + b}
"""
    python_code = PythonCode(
        type="PythonCode",
        run_function_code=run_function,
        run_function_name="run_function",
        init_function_code=init_function,
        init_function_name="init_fun",
        imports=["import math"],
    )

    # when
    module = create_dynamic_module(
        block_type_name="some", python_code=python_code, module_name="my_module"
    )

    # then
    assert module.init_fun() == {"a": 35}
    assert module.run_function(3, 5) == {"result": 8}


def test_assembly_custom_python_block() -> None:
    # given
    manifest = BlockManifest
    init_function = """
def init_fun() -> Dict[str, Any]:
    return {"a": 6}    
"""
    run_function = """
def run_function(self, a, b) -> BlockResult:
    return {"result": a + b + self._init_results["a"]}
    """
    python_code = PythonCode(
        type="PythonCode",
        run_function_code=run_function,
        run_function_name="run_function",
        init_function_code=init_function,
        init_function_name="init_fun",
        imports=["import math"],
    )

    # when
    workflow_block_class = assembly_custom_python_block(
        block_type_name="some",
        unique_identifier="unique-id",
        manifest=manifest,
        python_code=python_code,
    )
    workflow_block_instance = workflow_block_class()
    execution_result = workflow_block_instance.run(a=3, b=5)

    # then
    assert workflow_block_class.get_init_parameters() == [
        "api_key"
    ], "Expected api_key parameter defined"
    assert (
        workflow_block_class.get_manifest() == BlockManifest
    ), "Expected manifest to be returned"
    assert execution_result == {
        "result": 14
    }, "Expected result of 3 + 5 + 6 (last value from init)"


def test_assembly_custom_python_block_when_init_not_provided() -> None:
    # given
    manifest = BlockManifest
    run_function = """
def run_function(self, a, b) -> BlockResult:
    return {"result": a + b + len(self._init_results)}
    """
    python_code = PythonCode(
        type="PythonCode",
        run_function_code=run_function,
        run_function_name="run_function",
        imports=["import math"],
    )

    # when
    workflow_block_class = assembly_custom_python_block(
        block_type_name="some",
        unique_identifier="unique-id",
        manifest=manifest,
        python_code=python_code,
    )
    workflow_block_instance = workflow_block_class()
    execution_result = workflow_block_instance.run(a=3, b=5)

    # then
    assert workflow_block_class.get_init_parameters() == [
        "api_key"
    ], "Expected api_key parameters defined"
    assert (
        workflow_block_class.get_manifest() == BlockManifest
    ), "Expected manifest to be returned"
    assert execution_result == {
        "result": 8
    }, "Expected result of 3 + 5 + 0 (last value from init)"


def test_assembly_custom_python_block_when_run_function_not_found() -> None:
    # given
    manifest = BlockManifest
    init_function = """
def init_fun() -> Dict[str, Any]:
    return {"a": 6}    
"""
    run_function = """
def run_function(self, a, b) -> BlockResult:
    return {"result": a + b + self._init_results["a"]}
    """
    python_code = PythonCode(
        type="PythonCode",
        run_function_code=run_function,
        run_function_name="invalid",
        init_function_code=init_function,
        init_function_name="init_fun",
        imports=["import math"],
    )

    # when
    with pytest.raises(DynamicBlockError):
        _ = assembly_custom_python_block(
            block_type_name="some",
            unique_identifier="unique-id",
            manifest=manifest,
            python_code=python_code,
        )


def test_assembly_custom_python_block_when_init_function_not_found() -> None:
    # given
    manifest = BlockManifest
    init_function = """
def init_fun() -> Dict[str, Any]:
    return {"a": 6}    
"""
    run_function = """
def run_function(self, a, b) -> BlockResult:
    return {"result": a + b + self._init_results["a"]}
    """
    python_code = PythonCode(
        type="PythonCode",
        run_function_code=run_function,
        run_function_name="run_function",
        init_function_code=init_function,
        init_function_name="invalid",
        imports=["import math"],
    )

    # when
    with pytest.raises(DynamicBlockError):
        _ = assembly_custom_python_block(
            block_type_name="some",
            unique_identifier="unique-id",
            manifest=manifest,
            python_code=python_code,
        )


@mock.patch.object(
    block_scaffolding, "ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS", False
)
def test_run_assembled_custom_python_block_when_custom_python_forbidden() -> None:
    # given
    manifest = BlockManifest
    init_function = """
def init_fun() -> Dict[str, Any]:
    return {"a": 6}    
"""
    run_function = """
def run_function(self, a, b) -> BlockResult:
    return {"result": a + b + self._init_results["a"]}
    """
    python_code = PythonCode(
        type="PythonCode",
        run_function_code=run_function,
        run_function_name="run_function",
        init_function_code=init_function,
        init_function_name="init_fun",
        imports=["import math"],
    )

    # when
    workflow_block_class = assembly_custom_python_block(
        block_type_name="some",
        unique_identifier="unique-id",
        manifest=manifest,
        python_code=python_code,
    )
    workflow_block_instance = workflow_block_class()
    with pytest.raises(WorkflowEnvironmentConfigurationError):
        _ = workflow_block_instance.run(a=3, b=5)


def test_run_assembled_custom_python_block_captures_stdout_and_stderr_for_debug_collector() -> (
    None
):
    # given - a block that writes to both streams as part of normal execution
    manifest = BlockManifest
    run_function = """
import sys

def run_function(self, a, b) -> BlockResult:
    print("hello from block", a, b)
    print("oops", file=sys.stderr)
    return {"result": a + b}
"""
    python_code = PythonCode(
        type="PythonCode",
        run_function_code=run_function,
        run_function_name="run_function",
        imports=[],
    )
    workflow_block_class = assembly_custom_python_block(
        block_type_name="some",
        unique_identifier="debug-id",
        manifest=manifest,
        python_code=python_code,
    )
    workflow_block_instance = workflow_block_class()
    workflow_block_instance._workflow_step_name = "my_step"

    # when - run inside an active debug collector scope (ContextVar-published)
    with register_debug_session() as session:
        execution_result = workflow_block_instance.run(a=3, b=5)
        snapshot = session.output_streams.snapshot()

    # then - block result is unchanged AND stdout/stderr surfaced in collector
    assert execution_result == {"result": 8}
    assert list(snapshot.keys()) == ["my_step"]
    assert len(snapshot["my_step"]) == 1
    entry = snapshot["my_step"][0]
    assert "hello from block 3 5" in entry["stdout"]
    assert "oops" in entry["stderr"]


def test_get_workflow_context_reads_workflow_execution_id_from_contextvar() -> None:
    # given - a dynamic block that returns the workflow context it sees
    manifest = BlockManifest
    run_function = """
def run_function(self, value) -> BlockResult:
    return self.get_workflow_context()
"""
    python_code = PythonCode(
        type="PythonCode",
        run_function_code=run_function,
        run_function_name="run_function",
        imports=[],
    )
    workflow_block_class = assembly_custom_python_block(
        block_type_name="ContextProbe",
        unique_identifier="ctx-id",
        manifest=manifest,
        python_code=python_code,
    )
    workflow_block_instance = workflow_block_class()
    workflow_block_instance._workflow_step_name = "probe"
    workflow_block_instance._workflow_step_type = "ContextProbe"
    workflow_block_instance._workflow_step_selector = "$steps.probe"

    # when - the execution_id ContextVar is set, mimicking what the engine
    # does inside `safe_execute_step` for every worker thread
    from inference_sdk.config import execution_id

    token = execution_id.set("exec-from-ctxvar")
    try:
        context = workflow_block_instance.run(value=1)
    finally:
        execution_id.reset(token)

    # then - the context dict reflects the static metadata + the ContextVar
    assert context == {
        "step_name": "probe",
        "step_selector": "$steps.probe",
        "block_type": "ContextProbe",
        "workflow_execution_id": "exec-from-ctxvar",
    }


def test_run_assembled_custom_python_block_records_logs_to_collector_when_block_raises() -> (
    None
):
    # given - a block that prints and then fails
    manifest = BlockManifest
    run_function = """
import sys

def run_function(self, a, b) -> BlockResult:
    print("about to fail with", a, b)
    print("warning sign", file=sys.stderr)
    raise RuntimeError("boom")
"""
    python_code = PythonCode(
        type="PythonCode",
        run_function_code=run_function,
        run_function_name="run_function",
        imports=[],
    )
    workflow_block_class = assembly_custom_python_block(
        block_type_name="some",
        unique_identifier="failing-id",
        manifest=manifest,
        python_code=python_code,
    )
    workflow_block_instance = workflow_block_class()
    workflow_block_instance._workflow_step_name = "failing_step"

    # when - block raises inside an active debug collector scope
    with register_debug_session() as session:
        with pytest.raises(DynamicBlockCodeError):
            _ = workflow_block_instance.run(a=1, b=2)
        snapshot = session.output_streams.snapshot()

    # then - logs emitted before the failure are present in the collector
    assert list(snapshot.keys()) == ["failing_step"]
    entry = snapshot["failing_step"][0]
    assert "about to fail with 1 2" in entry["stdout"]
    assert "warning sign" in entry["stderr"]


def test_run_assembled_custom_python_block_does_not_record_when_no_collector_active() -> (
    None
):
    # given - a block writing output but no active collector in the ContextVar
    manifest = BlockManifest
    run_function = """
def run_function(self, a, b) -> BlockResult:
    print("noisy")
    return {"result": a + b}
"""
    python_code = PythonCode(
        type="PythonCode",
        run_function_code=run_function,
        run_function_name="run_function",
        imports=[],
    )
    workflow_block_class = assembly_custom_python_block(
        block_type_name="some",
        unique_identifier="quiet-id",
        manifest=manifest,
        python_code=python_code,
    )
    workflow_block_instance = workflow_block_class()

    # when - execute without an active collector
    execution_result = workflow_block_instance.run(a=1, b=2)

    # then - regular result, no exception, no recording side effect
    assert execution_result == {"result": 3}


def test_run_assembled_custom_python_block_appends_to_debug_trace() -> None:
    # given - a block that uses the injected `debug` variable
    manifest = BlockManifest
    run_function = """
def run_function(self, a, b) -> BlockResult:
    debug_traces.append({"sum": a + b, "inputs": [a, b]})
    return {"result": a + b}
"""
    python_code = PythonCode(
        type="PythonCode",
        run_function_code=run_function,
        run_function_name="run_function",
        imports=[],
    )
    workflow_block_class = assembly_custom_python_block(
        block_type_name="DebugBlock",
        unique_identifier="debug-trace-id",
        manifest=manifest,
        python_code=python_code,
    )
    workflow_block_instance = workflow_block_class()
    workflow_block_instance._workflow_step_name = "debug_step"

    from inference.core.workflows.execution_engine.v1.dynamic_blocks.workflow_debug import (
        current_debug_step_name,
    )

    # when
    with register_debug_session() as session:
        token = current_debug_step_name.set("debug_step")
        try:
            execution_result = workflow_block_instance.run(a=3, b=5)
            trace = session.debug_traces.snapshot()
        finally:
            current_debug_step_name.reset(token)

    # then
    assert execution_result == {"result": 8}
    assert trace == [
        {"step": "debug_step", "value": {"sum": 8, "inputs": [3, 5]}},
    ]


def _boundary_wiring_manifest_description():
    from inference.core.workflows.execution_engine.v1.dynamic_blocks.entities import (
        DynamicInputDefinition,
        DynamicOutputDefinition,
        ManifestDescription,
        SelectorType,
    )

    return ManifestDescription(
        type="ManifestDescription",
        block_type="BoundaryProbe",
        inputs={
            "predictions": DynamicInputDefinition(
                type="DynamicInputDefinition",
                selector_types=[SelectorType.STEP_OUTPUT],
                selector_data_kind={
                    SelectorType.STEP_OUTPUT: ["object_detection_prediction"]
                },
            ),
        },
        outputs={
            "result": DynamicOutputDefinition(
                type="DynamicOutputDefinition", kind=["object_detection_prediction"]
            ),
            "was_sv": DynamicOutputDefinition(type="DynamicOutputDefinition"),
        },
    )


def _native_od_fixture():
    import torch

    from inference_models.models.base.object_detection import Detections

    return Detections(
        xyxy=torch.tensor([[10.0, 10.0, 20.0, 20.0]]),
        class_id=torch.tensor([0]),
        confidence=torch.tensor([0.9]),
        image_metadata={"class_names": {0: "dog"}},
        bboxes_metadata=[{"detection_id": "det-1"}],
    )


def test_run_wrapper_applies_representation_boundary_both_ways() -> None:
    # given
    import supervision as sv

    from inference.core.workflows.execution_engine.v1.dynamic_blocks import (
        representation_boundary,
    )
    from inference_models.models.base.object_detection import Detections

    run_function = """
def run_function(self, predictions) -> BlockResult:
    return {"result": predictions, "was_sv": isinstance(predictions, sv.Detections)}
"""
    python_code = PythonCode(
        type="PythonCode",
        run_function_code=run_function,
        run_function_name="run_function",
    )
    block_class = assembly_custom_python_block(
        block_type_name="BoundaryProbe",
        unique_identifier="boundary-probe-id",
        manifest=BlockManifest,
        python_code=python_code,
        manifest_description=_boundary_wiring_manifest_description(),
    )

    # when
    with mock.patch.object(
        representation_boundary, "_TENSOR_REPRESENTATION_ACTIVE", True
    ):
        result = block_class().run(predictions=_native_od_fixture())

    # then - IN boundary delivered sv to user code, OUT boundary restored native
    assert result["was_sv"] is True, "User code must have received sv.Detections"
    assert isinstance(
        result["result"], Detections
    ), "Declared-kind output must be converted back to native"
    assert result["result"].bboxes_metadata[0]["detection_id"] == "det-1"


def test_run_wrapper_boundary_error_not_wrapped_as_user_code_error() -> None:
    # given - user code succeeds, but returns a value that cannot satisfy the
    # declared output kind; the boundary error must surface as itself, NOT as a
    # DynamicBlockCodeError blaming the user's code.
    from inference.core.workflows.execution_engine.v1.dynamic_blocks import (
        representation_boundary,
    )
    from inference.core.workflows.execution_engine.v1.dynamic_blocks.representation_boundary import (
        RepresentationBoundaryError,
    )

    run_function = """
def run_function(self, predictions) -> BlockResult:
    return {"result": object(), "was_sv": True}
"""
    python_code = PythonCode(
        type="PythonCode",
        run_function_code=run_function,
        run_function_name="run_function",
    )
    block_class = assembly_custom_python_block(
        block_type_name="BoundaryProbe",
        unique_identifier="boundary-probe-err-id",
        manifest=BlockManifest,
        python_code=python_code,
        manifest_description=_boundary_wiring_manifest_description(),
    )

    # when
    with mock.patch.object(
        representation_boundary, "_TENSOR_REPRESENTATION_ACTIVE", True
    ), pytest.raises(RepresentationBoundaryError) as error:
        _ = block_class().run(predictions=_native_od_fixture())

    # then
    assert not isinstance(error.value, DynamicBlockCodeError)
    assert "BoundaryProbe" in str(error.value)
    assert "result" in str(error.value)


def _legacy_sv_fixture():
    import numpy as np
    import supervision as sv

    return sv.Detections(
        xyxy=np.array([[10.0, 10.0, 20.0, 20.0]], dtype=np.float32),
        class_id=np.array([0]),
        confidence=np.array([0.9], dtype=np.float32),
        data={
            "class_name": np.array(["dog"]),
            "detection_id": np.array(["det-remote-1"]),
        },
    )


def test_run_wrapper_modal_arm_converts_kwargs_and_remote_result() -> None:
    # given - D2-REVISED Option A: the Modal arm must ship CONVERTED (sv) inputs
    # to the executor and convert the sv result coming back into native objects.
    from inference.core.workflows.execution_engine.v1.dynamic_blocks import (
        block_scaffolding,
        modal_executor,
        representation_boundary,
    )
    from inference_models.models.base.object_detection import Detections

    python_code = PythonCode(
        type="PythonCode",
        run_function_code="def run_function(self, predictions) -> BlockResult:\n    return None\n",
        run_function_name="run_function",
    )
    block_class = assembly_custom_python_block(
        block_type_name="BoundaryProbe",
        unique_identifier="boundary-probe-modal-id",
        manifest=BlockManifest,
        python_code=python_code,
        manifest_description=_boundary_wiring_manifest_description(),
    )
    executor_instance = mock.MagicMock()
    executor_instance.execute_remote.return_value = {
        "result": _legacy_sv_fixture(),
        "was_sv": True,
    }

    # when
    with mock.patch.object(
        representation_boundary, "_TENSOR_REPRESENTATION_ACTIVE", True
    ), mock.patch.object(
        block_scaffolding, "WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE", "modal"
    ), mock.patch.object(
        block_scaffolding, "get_roboflow_workspace", return_value="test-workspace"
    ), mock.patch.object(
        modal_executor, "ModalExecutor", return_value=executor_instance
    ):
        result = block_class().run(predictions=_native_od_fixture())

    # then - inputs leg: executor received sv (the `_type='sv_detections'` arm
    # territory), not native objects
    import supervision as sv

    sent_inputs = executor_instance.execute_remote.call_args.kwargs["inputs"]
    assert isinstance(
        sent_inputs["predictions"], sv.Detections
    ), "Modal arm must ship converted sv inputs"
    # then - return leg: sv result converted to native before entering the engine
    assert isinstance(
        result["result"], Detections
    ), "Modal arm must convert the sv result back to native"
    assert result["result"].bboxes_metadata[0]["detection_id"] == "det-remote-1"


def test_run_wrapper_modal_arm_is_passthrough_when_flag_off() -> None:
    # given
    from inference.core.workflows.execution_engine.v1.dynamic_blocks import (
        block_scaffolding,
        modal_executor,
        representation_boundary,
    )

    python_code = PythonCode(
        type="PythonCode",
        run_function_code="def run_function(self, predictions) -> BlockResult:\n    return None\n",
        run_function_name="run_function",
    )
    block_class = assembly_custom_python_block(
        block_type_name="BoundaryProbe",
        unique_identifier="boundary-probe-modal-off-id",
        manifest=BlockManifest,
        python_code=python_code,
        manifest_description=_boundary_wiring_manifest_description(),
    )
    legacy_input = _legacy_sv_fixture()
    remote_result = {"result": _legacy_sv_fixture(), "was_sv": True}
    executor_instance = mock.MagicMock()
    executor_instance.execute_remote.return_value = remote_result

    # when
    with mock.patch.object(
        representation_boundary, "_TENSOR_REPRESENTATION_ACTIVE", False
    ), mock.patch.object(
        block_scaffolding, "WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE", "modal"
    ), mock.patch.object(
        block_scaffolding, "get_roboflow_workspace", return_value="test-workspace"
    ), mock.patch.object(
        modal_executor, "ModalExecutor", return_value=executor_instance
    ):
        result = block_class().run(predictions=legacy_input)

    # then - flag-off byte-parity: both legs are is-identity
    sent_inputs = executor_instance.execute_remote.call_args.kwargs["inputs"]
    assert sent_inputs["predictions"] is legacy_input
    assert result is remote_result


def test_run_wrapper_local_forbidden_gate_wins_over_conversion_error() -> None:
    # given - local mode, custom python forbidden, and kwargs that WOULD raise a
    # RepresentationBoundaryError at IN conversion (wrong type for the declared
    # kind). The clearer misconfiguration error must fire first.
    import torch

    from inference.core.workflows.errors import WorkflowEnvironmentConfigurationError
    from inference.core.workflows.execution_engine.v1.dynamic_blocks import (
        block_scaffolding,
        representation_boundary,
    )

    python_code = PythonCode(
        type="PythonCode",
        run_function_code="def run_function(self, predictions) -> BlockResult:\n    return None\n",
        run_function_name="run_function",
    )
    block_class = assembly_custom_python_block(
        block_type_name="BoundaryProbe",
        unique_identifier="boundary-probe-gate-id",
        manifest=BlockManifest,
        python_code=python_code,
        manifest_description=_boundary_wiring_manifest_description(),
    )

    # when
    with mock.patch.object(
        representation_boundary, "_TENSOR_REPRESENTATION_ACTIVE", True
    ), mock.patch.object(
        block_scaffolding, "WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE", "local"
    ), mock.patch.object(
        block_scaffolding, "ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS", False
    ), pytest.raises(
        WorkflowEnvironmentConfigurationError
    ):
        _ = block_class().run(predictions=torch.tensor([1.0]))


def test_imports_lines_tensor_native_extension_tracks_the_flag() -> None:
    # given - the extension is resolved at import time (load-time swap philosophy):
    # flag-on ships the tensor-native authoring imports, flag-off must keep the
    # generated module source byte-identical to the legacy list. This test runs in
    # both CI directions, locking each side of the contract.
    from inference.core.env import ENABLE_TENSOR_DATA_REPRESENTATION
    from inference.core.workflows.execution_engine.v1.dynamic_blocks.block_scaffolding import (
        IMPORTS_LINES,
        TENSOR_NATIVE_IMPORTS_LINES,
    )

    # then
    assert len(TENSOR_NATIVE_IMPORTS_LINES) > 0
    for line in TENSOR_NATIVE_IMPORTS_LINES:
        assert (line in IMPORTS_LINES) == ENABLE_TENSOR_DATA_REPRESENTATION
    # the legacy prefix is untouched in both directions
    assert IMPORTS_LINES[0] == "from typing import Any, List, Dict, Set, Optional"
    assert "import supervision as sv" in IMPORTS_LINES
    assert "import numpy as np" in IMPORTS_LINES
