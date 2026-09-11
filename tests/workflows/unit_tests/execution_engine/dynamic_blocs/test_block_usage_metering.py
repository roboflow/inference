"""Usage rows emitted for custom Python blocks.

Every assembled dynamic block runs through the shared
``usage_collector("workflow_block")`` entrypoint. What matters here is that the
duration on the row is the block's actual runtime: measured locally around the
user function, and taken from the sandbox's own measurement when the block ran
remotely.
"""

from contextlib import contextmanager
from unittest import mock

import pytest

from inference.core.workflows.core_steps.formatters.expression.v1 import BlockManifest
from inference.core.workflows.errors import DynamicBlockCodeError, DynamicBlockError
from inference.core.workflows.execution_engine.entities.base import Batch
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
from inference.usage_tracking.block_execution import (
    BLOCK_DURATION_SOURCE_CLIENT_WALL_CLOCK,
    BLOCK_DURATION_SOURCE_DECORATOR_WALL_CLOCK,
    BLOCK_DURATION_SOURCE_LOCAL_RUNTIME,
    BLOCK_DURATION_SOURCE_REMOTE_RUNTIME,
    BLOCK_DURATION_SOURCE_UNAVAILABLE,
    BLOCK_EXECUTION_MODE_LOCAL,
    BLOCK_EXECUTION_MODE_REMOTE,
    clear_measured_block_execution,
    record_measured_block_execution,
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

_RESERVED_NAME_BLOCK = """
def run_function(self, usage_billable, usage_api_key) -> BlockResult:
    return {"result": [usage_billable, usage_api_key]}
"""


def test_block_input_named_after_a_usage_kwarg_reaches_the_user_function():
    """Block parameter names come from the workflow definition, unvalidated.

    One named after a usage-decorator keyword-only argument must not bind to it:
    that would suppress billing (`usage_billable=False`), redirect the row
    (`usage_api_key`), or zero the frame count (`usage_inference_test_run`) -
    and the value would never reach the user's function.
    """
    # given
    block_class, _ = _assemble_block(_RESERVED_NAME_BLOCK, "metered-reserved-names")
    block = block_class(api_key="workflow-api-key")

    # when
    with mock.patch.object(usage_collector, "record_usage") as record_usage:
        result = block.run(usage_billable=False, usage_api_key="attacker-key")

    # then - the user's function got the values, not the decorator
    assert result == {"result": [False, "attacker-key"]}

    usage_params = record_usage.call_args.kwargs
    assert usage_params["api_key"] == "workflow-api-key"
    assert usage_params["resource_details"]["billable"] is True


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
    clock = iter([100.0, 100.25])

    # when
    with mock.patch.object(
        block_scaffolding.time, "monotonic", side_effect=lambda: next(clock)
    ), mock.patch.object(usage_collector, "record_usage") as record_usage:
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
        record_measured_block_execution(
            duration=0.25, source=BLOCK_DURATION_SOURCE_REMOTE_RUNTIME
        )
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
    record_measured_block_execution(
        duration=9.0, source=BLOCK_DURATION_SOURCE_REMOTE_RUNTIME
    )

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


def _run_modal_block_expecting_error(execute_remote, unique_identifier, expected_error):
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
        with pytest.raises(expected_error):
            block.run(a=3, b=5)

    return record_usage.call_args.kwargs


def test_modal_transport_failure_is_not_billed_as_client_wall_clock(
    isolated_modal_executor_cache,
):
    def execute_remote(**kwargs):
        raise DynamicBlockError(
            public_message="Failed to connect to Modal endpoint",
            context="modal_executor | http_connection",
        )

    usage_params = _run_modal_block_expecting_error(
        execute_remote,
        "metered-modal-transport",
        DynamicBlockError,
    )

    assert usage_params["execution_duration"] == 0
    assert (
        usage_params["resource_details"]["duration_source"]
        == BLOCK_DURATION_SOURCE_UNAVAILABLE
    )
    assert (
        usage_params["resource_details"]["execution_mode"]
        == BLOCK_EXECUTION_MODE_REMOTE
    )


def test_modal_client_wall_clock_excludes_executor_acquisition(
    isolated_modal_executor_cache,
):
    class Clock:
        def __init__(self):
            self.t = 0.0

        def monotonic(self):
            return self.t

        def advance(self, dt):
            self.t += dt

    clock = Clock()

    def execute_remote(**kwargs):
        clock.advance(0.25)
        return {"result": 8}

    @contextmanager
    def slow_acquire(workspace_id):
        clock.advance(30.0)
        yield mock.MagicMock(execute_remote=execute_remote)

    block_class, _ = _assemble_block(_PASSTHROUGH_BLOCK, "metered-modal-acquire")
    block = block_class(api_key="workflow-api-key")

    with mock.patch.object(
        block_scaffolding, "WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE", "modal"
    ), mock.patch.object(
        block_scaffolding, "get_roboflow_workspace", return_value="test-workspace"
    ), mock.patch.object(
        block_scaffolding, "_acquire_modal_executor", slow_acquire
    ), mock.patch.object(
        block_scaffolding.time, "monotonic", clock.monotonic
    ), mock.patch.object(
        usage_collector, "record_usage"
    ) as record_usage:
        result = block.run(a=3, b=5)

    usage_params = record_usage.call_args.kwargs
    assert result == {"result": 8}
    assert usage_params["execution_duration"] == pytest.approx(0.25)
    assert (
        usage_params["resource_details"]["duration_source"]
        == BLOCK_DURATION_SOURCE_CLIENT_WALL_CLOCK
    )


_BATCH_BLOCK = """
def run_function(self, items) -> BlockResult:
    return [{"result": item} for item in items]
"""


def test_modal_block_ignores_a_bogus_runtime_and_falls_back_to_wall_clock(
    isolated_modal_executor_cache,
):
    """A sandbox that misreports must degrade, not corrupt the row."""

    # given
    def execute_remote(**kwargs):
        record_measured_block_execution(
            duration=float("nan"), source=BLOCK_DURATION_SOURCE_REMOTE_RUNTIME
        )
        record_measured_block_execution(
            duration=-1.0, source=BLOCK_DURATION_SOURCE_REMOTE_RUNTIME
        )
        record_measured_block_execution(
            duration="0.25", source=BLOCK_DURATION_SOURCE_REMOTE_RUNTIME
        )
        return {"result": 8}

    # when
    _, usage_params = _run_modal_block(execute_remote, "metered-modal-bogus")

    # then - none of those were usable, so the client's wall clock stands
    assert (
        usage_params["resource_details"]["duration_source"]
        == BLOCK_DURATION_SOURCE_CLIENT_WALL_CLOCK
    )
    assert usage_params["execution_duration"] >= 0


def test_modal_user_code_error_is_billed_the_runtime_the_sandbox_reported(
    isolated_modal_executor_cache,
):
    """A block that raises *inside* the sandbox still ran; bill what it spent.

    Distinct from a transport failure: `DynamicBlockCodeError` is not a
    `DynamicBlockError`, and the executor publishes the sandbox's runtime before
    raising either.
    """

    # given
    def execute_remote(**kwargs):
        record_measured_block_execution(
            duration=0.25, source=BLOCK_DURATION_SOURCE_REMOTE_RUNTIME
        )
        raise DynamicBlockCodeError(
            public_message="boom",
            context="workflow_execution | step_execution | dynamic_step",
        )

    block_class, _ = _assemble_block(_PASSTHROUGH_BLOCK, "metered-modal-user-error")
    executor_instance = mock.MagicMock()
    executor_instance.execute_remote.side_effect = execute_remote
    block = block_class(api_key="workflow-api-key")

    # when
    with mock.patch.object(
        block_scaffolding, "WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE", "modal"
    ), mock.patch.object(
        block_scaffolding, "get_roboflow_workspace", return_value="test-workspace"
    ), mock.patch.object(
        modal_executor, "ModalExecutor", return_value=executor_instance
    ), mock.patch.object(
        usage_collector, "record_usage"
    ) as record_usage, pytest.raises(
        DynamicBlockCodeError
    ):
        block.run(a=3, b=5)

    # then
    usage_params = record_usage.call_args.kwargs
    assert usage_params["execution_duration"] == 0.25
    assert (
        usage_params["resource_details"]["duration_source"]
        == BLOCK_DURATION_SOURCE_REMOTE_RUNTIME
    )


def test_block_without_an_api_key_records_no_row():
    """`record_usage` drops keyless rows; nothing should reach the payload."""
    # given
    block_class, _ = _assemble_block(_PASSTHROUGH_BLOCK, "metered-no-api-key")
    block = block_class(api_key=None)

    # when
    with mock.patch.object(usage_collector, "_update_usage_payload") as update_payload:
        result = block.run(a=3, b=5)

    # then
    assert result == {"result": 8}
    update_payload.assert_not_called()


def test_batch_block_is_billed_one_frame_per_element():
    """A batch-oriented block gets the whole batch in one `run()` call."""
    # given
    block_class, _ = _assemble_block(_BATCH_BLOCK, "metered-batch")
    block = block_class(api_key="workflow-api-key")
    batch = Batch.init(content=[1, 2, 3, 4], indices=[(i,) for i in range(4)])

    # when
    with mock.patch.object(usage_collector, "record_usage") as record_usage:
        block.run(items=batch)

    # then
    assert record_usage.call_args.kwargs["frames"] == 4


def test_two_steps_sharing_block_code_aggregate_into_one_row():
    """Identical code used by two steps is one billable resource, end to end."""
    # given - two separately assembled classes with the same body
    first_class, python_code = _assemble_block(_PASSTHROUGH_BLOCK, "metered-shared-a")
    second_class, _ = _assemble_block(_PASSTHROUGH_BLOCK, "metered-shared-b")
    first = first_class(api_key="workflow-api-key")
    first._workflow_step_name = "step_one"
    second = second_class(api_key="workflow-api-key")
    second._workflow_step_name = "step_two"

    # when - recorded against a throwaway usage dict rather than the singleton's
    recorded = usage_collector.empty_usage_dict(exec_session_id="test-session")
    with mock.patch.object(usage_collector, "_usage", recorded):
        first.run(a=1, b=2)
        second.run(a=3, b=4)

    # then - one row, both invocations counted
    rows = [
        row
        for api_key_usage in recorded.values()
        for row in api_key_usage.values()
        if row.get("category") == "workflow_block"
    ]
    assert len(rows) == 1, rows
    assert rows[0]["resource_id"] == (
        f"custom_python/{compute_block_code_fingerprint(python_code)}"
    )
    assert rows[0]["processed_frames"] == 2
