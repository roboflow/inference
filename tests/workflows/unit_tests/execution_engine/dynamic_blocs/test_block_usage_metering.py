"""Usage rows emitted for custom Python blocks.

Every assembled dynamic block runs through the shared
``usage_collector("workflow_block")`` entrypoint. What matters here is that the
duration on the row is the block's actual runtime: measured locally around the
user function, and taken from the sandbox's own measurement when the block ran
remotely.
"""

from unittest import mock

import pytest

from inference.core.workflows.core_steps.formatters.expression.v1 import BlockManifest
from inference.core.workflows.errors import DynamicBlockCodeError
from inference.core.workflows.execution_engine.v1.dynamic_blocks import (
    block_scaffolding,
    modal_executor,
)
from inference.core.workflows.execution_engine.v1.dynamic_blocks.block_scaffolding import (
    assembly_custom_python_block,
    compute_block_code_fingerprint,
)
from inference.core.workflows.execution_engine.v1.dynamic_blocks.entities import (
    PythonCode,
)
from inference.core.workflows.execution_engine.v1.dynamic_blocks.execution_timing import (
    record_remote_execution_duration,
)
from inference.usage_tracking.block_execution import (
    BLOCK_DURATION_SOURCE_CLIENT_WALL_CLOCK,
    BLOCK_DURATION_SOURCE_LOCAL_RUNTIME,
    BLOCK_DURATION_SOURCE_REMOTE_RUNTIME,
    BLOCK_EXECUTION_MODE_LOCAL,
    BLOCK_EXECUTION_MODE_REMOTE,
    clear_measured_block_execution,
)
from inference.usage_tracking.collector import usage_collector


@pytest.fixture(autouse=True)
def cleared_block_execution():
    clear_measured_block_execution()
    yield
    clear_measured_block_execution()


def _clear_modal_executor_cache() -> None:
    with block_scaffolding._MODAL_EXECUTOR_CACHE_LOCK:
        block_scaffolding._MODAL_EXECUTOR_CACHE.clear()


@pytest.fixture
def isolated_modal_executor_cache():
    _clear_modal_executor_cache()
    yield
    _clear_modal_executor_cache()


def _assemble_block(run_function: str, unique_identifier: str, api_key=None):
    python_code = PythonCode(
        type="PythonCode",
        run_function_code=run_function,
        run_function_name="run_function",
        imports=[],
    )
    block_class = assembly_custom_python_block(
        block_type_name="MeteredBlock",
        unique_identifier=unique_identifier,
        manifest=BlockManifest,
        python_code=python_code,
        api_key=api_key,
    )
    return block_class, python_code


_PASSTHROUGH_BLOCK = """
def run_function(self, a, b) -> BlockResult:
    return {"result": a + b}
"""

_FAILING_BLOCK = """
def run_function(self, a, b) -> BlockResult:
    raise RuntimeError("boom")
"""


def test_local_block_records_a_workflow_block_row_with_its_own_runtime():
    # given
    block_class, python_code = _assemble_block(_PASSTHROUGH_BLOCK, "metered-local")
    block = block_class(api_key="workflow-api-key")
    block._workflow_step_name = "my_step"
    block._workflow_step_type = "MeteredBlock"
    clock = iter([100.0, 100.25])

    # when
    with mock.patch.object(
        block_scaffolding.time, "monotonic", side_effect=lambda: next(clock)
    ), mock.patch.object(usage_collector, "record_usage") as record_usage:
        result = block.run(a=3, b=5)

    # then - the block still returns normally
    assert result == {"result": 8}

    usage_params = record_usage.call_args.kwargs
    assert usage_params["category"] == "workflow_block"
    assert usage_params["resource_id"] == (
        f"custom_python/{compute_block_code_fingerprint(python_code)}"
    )
    assert usage_params["api_key"] == "workflow-api-key"
    assert usage_params["resource_details"]["block_kind"] == "custom_python"
    assert usage_params["resource_details"]["block_type"] == "MeteredBlock"
    assert usage_params["resource_details"]["step_name"] == "my_step"
    assert (
        usage_params["resource_details"]["duration_source"]
        == BLOCK_DURATION_SOURCE_LOCAL_RUNTIME
    )
    assert (
        usage_params["resource_details"]["execution_mode"] == BLOCK_EXECUTION_MODE_LOCAL
    )
    assert usage_params["execution_duration"] == pytest.approx(0.25)


def test_local_block_that_raises_is_still_billed_for_the_time_it_ran():
    # given
    block_class, _ = _assemble_block(_FAILING_BLOCK, "metered-local-failing")
    block = block_class(api_key="workflow-api-key")

    # when
    with mock.patch.object(usage_collector, "record_usage") as record_usage:
        with pytest.raises(DynamicBlockCodeError):
            block.run(a=1, b=2)

    # then
    usage_params = record_usage.call_args.kwargs
    assert usage_params["category"] == "workflow_block"
    assert (
        usage_params["resource_details"]["duration_source"]
        == BLOCK_DURATION_SOURCE_LOCAL_RUNTIME
    )
    # the user's own exception, not the DynamicBlockCodeError wrapping it
    assert usage_params["resource_details"]["error_type"] == "RuntimeError"


def test_two_blocks_with_the_same_code_share_a_resource_id():
    _, first_code = _assemble_block(_PASSTHROUGH_BLOCK, "metered-identity-a")
    _, second_code = _assemble_block(_PASSTHROUGH_BLOCK, "metered-identity-b")
    _, other_code = _assemble_block(
        _PASSTHROUGH_BLOCK.replace("a + b", "a * b"), "metered-identity-c"
    )

    assert compute_block_code_fingerprint(first_code) == (
        compute_block_code_fingerprint(second_code)
    )
    assert compute_block_code_fingerprint(first_code) != (
        compute_block_code_fingerprint(other_code)
    )


def _run_modal_block(execute_remote, unique_identifier):
    block_class, _ = _assemble_block(_PASSTHROUGH_BLOCK, unique_identifier)
    executor_instance = mock.MagicMock()
    executor_instance.execute_remote.side_effect = execute_remote
    block = block_class(api_key="workflow-api-key")

    with mock.patch.object(
        block_scaffolding, "WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE", "modal"
    ), mock.patch.object(
        block_scaffolding, "get_roboflow_workspace", return_value="test-workspace"
    ), mock.patch.object(
        modal_executor, "ModalExecutor", return_value=executor_instance
    ), mock.patch.object(
        usage_collector, "record_usage"
    ) as record_usage:
        result = block.run(a=3, b=5)

    return result, record_usage.call_args.kwargs


def test_modal_block_is_billed_for_the_runtime_the_sandbox_reported(
    isolated_modal_executor_cache,
):
    # given - the sandbox measured 0.25s of user code; the client call around it
    # also covers serialization and the round trip
    def execute_remote(**kwargs):
        record_remote_execution_duration(0.25)
        return {"result": 8}

    # when
    result, usage_params = _run_modal_block(execute_remote, "metered-modal-reported")

    # then
    assert result == {"result": 8}
    assert usage_params["execution_duration"] == 0.25
    assert (
        usage_params["resource_details"]["duration_source"]
        == BLOCK_DURATION_SOURCE_REMOTE_RUNTIME
    )
    assert (
        usage_params["resource_details"]["execution_mode"]
        == BLOCK_EXECUTION_MODE_REMOTE
    )


def test_modal_block_falls_back_to_client_wall_clock_when_sandbox_reports_nothing(
    isolated_modal_executor_cache,
):
    # given - a sandbox deployment that predates the reported runtime
    def execute_remote(**kwargs):
        return {"result": 8}

    # when
    _, usage_params = _run_modal_block(execute_remote, "metered-modal-silent")

    # then
    assert (
        usage_params["resource_details"]["duration_source"]
        == BLOCK_DURATION_SOURCE_CLIENT_WALL_CLOCK
    )
    assert (
        usage_params["resource_details"]["execution_mode"]
        == BLOCK_EXECUTION_MODE_REMOTE
    )
    assert usage_params["execution_duration"] >= 0


def test_modal_runtime_is_not_reused_by_a_later_local_block(
    isolated_modal_executor_cache,
):
    # given - a remote invocation whose reported runtime nobody consumed, which
    # is what a failure inside usage recording would leave behind
    record_remote_execution_duration(9.0)

    # when - a local block runs next in the same thread
    block_class, _ = _assemble_block(_PASSTHROUGH_BLOCK, "metered-no-leak")
    block = block_class(api_key="workflow-api-key")
    with mock.patch.object(usage_collector, "record_usage") as record_usage:
        block.run(a=1, b=2)

    # then - it is billed for its own runtime
    usage_params = record_usage.call_args.kwargs
    assert usage_params["execution_duration"] < 9.0
    assert (
        usage_params["resource_details"]["duration_source"]
        == BLOCK_DURATION_SOURCE_LOCAL_RUNTIME
    )
