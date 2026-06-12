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
    register_debug_collector,
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
    with register_debug_collector() as collector:
        execution_result = workflow_block_instance.run(a=3, b=5)
        snapshot = collector.snapshot()

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
    with register_debug_collector() as collector:
        with pytest.raises(DynamicBlockCodeError):
            _ = workflow_block_instance.run(a=1, b=2)
        snapshot = collector.snapshot()

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
        get_active_debug_trace,
    )

    # when
    with register_debug_collector():
        token = current_debug_step_name.set("debug_step")
        try:
            execution_result = workflow_block_instance.run(a=3, b=5)
            trace = get_active_debug_trace().snapshot()
        finally:
            current_debug_step_name.reset(token)

    # then
    assert execution_result == {"result": 8}
    assert trace == [
        {"step": "debug_step", "value": {"sum": 8, "inputs": [3, 5]}},
    ]
