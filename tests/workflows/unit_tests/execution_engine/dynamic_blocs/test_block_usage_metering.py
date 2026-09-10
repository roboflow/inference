"""Usage rows emitted for custom Python blocks.

Every assembled dynamic block runs through the shared
``usage_collector("workflow_block")`` entrypoint. What matters here is that the
duration on the row is the block's actual runtime: measured locally around the
user function, and taken from the sandbox's own measurement when the block ran
remotely.
"""

import json
from contextlib import contextmanager
from unittest import mock

import pytest

from inference.core.interfaces.workflows_execution_observer import (
    UsageTrackingExecutionObserver,
)
from inference.core.workflows.core_steps.formatters.expression.v1 import BlockManifest
from inference.core.workflows.errors import DynamicBlockCodeError, DynamicBlockError
from inference.core.workflows.execution_engine.entities.base import Batch
from inference.core.workflows.execution_engine.v1.dynamic_blocks import (
    block_scaffolding,
    modal_executor,
)
from inference.core.workflows.execution_engine.v1.dynamic_blocks.block_duration import (
    BLOCK_DURATION_SOURCE_CLIENT_WALL_CLOCK,
    BLOCK_DURATION_SOURCE_LOCAL_RUNTIME,
    BLOCK_DURATION_SOURCE_REMOTE_RUNTIME,
    BLOCK_DURATION_SOURCE_UNAVAILABLE,
    clear_block_duration,
    record_block_duration,
)
from inference.core.workflows.execution_engine.v1.dynamic_blocks.block_scaffolding import (
    assembly_custom_python_block,
    compute_block_code_fingerprint,
)
from inference.core.workflows.execution_engine.v1.dynamic_blocks.entities import (
    PythonCode,
)
from inference.usage_tracking.block_execution import (
    BLOCK_DURATION_SOURCE_DECORATOR_WALL_CLOCK,
    BLOCK_EXECUTION_MODE_LOCAL,
    BLOCK_EXECUTION_MODE_REMOTE,
)
from inference.usage_tracking.collector import usage_collector

# These tests assert on the usage rows a *server* records, so they instantiate
# blocks the way the server's composition roots bind them.
SERVER_OBSERVER = UsageTrackingExecutionObserver()


class _StubWorkspaceResolver:
    """Phase 9's `WorkspaceResolver` shape, answered locally."""

    def __init__(self, workspace="test-workspace"):
        self._workspace = workspace
        self.calls = []

    def resolve_workspace(self, api_key):
        self.calls.append(api_key)
        return self._workspace


@pytest.fixture(autouse=True)
def cleared_block_execution():
    clear_block_duration()
    yield
    clear_block_duration()


def _clear_modal_executor_cache() -> None:
    with block_scaffolding._MODAL_EXECUTOR_CACHE_LOCK:
        block_scaffolding._MODAL_EXECUTOR_CACHE.clear()


@pytest.fixture
def isolated_modal_executor_cache():
    _clear_modal_executor_cache()
    yield
    _clear_modal_executor_cache()


def _modal_block(block_class, api_key="workflow-api-key", workspace="test-workspace"):
    """A server-observed block whose Modal arm resolves a non-anonymous workspace.

    After Phase 9 the generated class declares `workspace_resolver` and the
    block asks it; before Phase 9 the arm calls
    `block_scaffolding.get_roboflow_workspace`, which
    `_legacy_workspace_lookup_pinned` pins. Both orders run the same test body.
    """
    kwargs = {"api_key": api_key, "execution_observer": SERVER_OBSERVER}
    if "workspace_resolver" in block_class.get_init_parameters():
        kwargs["workspace_resolver"] = _StubWorkspaceResolver(workspace)
    return block_class(**kwargs)


@contextmanager
def _legacy_workspace_lookup_pinned(workspace="test-workspace"):
    if hasattr(block_scaffolding, "get_roboflow_workspace"):
        with mock.patch.object(
            block_scaffolding, "get_roboflow_workspace", return_value=workspace
        ):
            yield
    else:
        yield


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
    block = block_class(api_key="workflow-api-key", execution_observer=SERVER_OBSERVER)

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
    block = block_class(api_key="workflow-api-key", execution_observer=SERVER_OBSERVER)
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
    block = block_class(api_key="workflow-api-key", execution_observer=SERVER_OBSERVER)
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
    block = block_class(
        api_key="workflow-api-key",
        workspace_resolver=_StubWorkspaceResolver(),
        execution_observer=SERVER_OBSERVER,
    )

    with mock.patch.object(
        block_scaffolding, "WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE", "modal"
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
        record_block_duration(
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
    record_block_duration(duration=9.0, source=BLOCK_DURATION_SOURCE_REMOTE_RUNTIME)

    # when - a local block runs next in the same thread
    block_class, _ = _assemble_block(_PASSTHROUGH_BLOCK, "metered-no-leak")
    block = block_class(api_key="workflow-api-key", execution_observer=SERVER_OBSERVER)
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
    block = block_class(
        api_key="workflow-api-key",
        workspace_resolver=_StubWorkspaceResolver(),
        execution_observer=SERVER_OBSERVER,
    )

    with mock.patch.object(
        block_scaffolding, "WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE", "modal"
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
    block = block_class(
        api_key="workflow-api-key",
        workspace_resolver=_StubWorkspaceResolver(),
        execution_observer=SERVER_OBSERVER,
    )

    with mock.patch.object(
        block_scaffolding, "WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE", "modal"
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
        record_block_duration(
            duration=float("nan"), source=BLOCK_DURATION_SOURCE_REMOTE_RUNTIME
        )
        record_block_duration(
            duration=-1.0, source=BLOCK_DURATION_SOURCE_REMOTE_RUNTIME
        )
        record_block_duration(
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
        record_block_duration(
            duration=0.25, source=BLOCK_DURATION_SOURCE_REMOTE_RUNTIME
        )
        raise DynamicBlockCodeError(
            public_message="boom",
            context="workflow_execution | step_execution | dynamic_step",
        )

    block_class, _ = _assemble_block(_PASSTHROUGH_BLOCK, "metered-modal-user-error")
    executor_instance = mock.MagicMock()
    executor_instance.execute_remote.side_effect = execute_remote
    block = block_class(
        api_key="workflow-api-key",
        workspace_resolver=_StubWorkspaceResolver(),
        execution_observer=SERVER_OBSERVER,
    )

    # when
    with mock.patch.object(
        block_scaffolding, "WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE", "modal"
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
    block = block_class(api_key=None, execution_observer=SERVER_OBSERVER)

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
    block = block_class(api_key="workflow-api-key", execution_observer=SERVER_OBSERVER)
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
    first = first_class(api_key="workflow-api-key", execution_observer=SERVER_OBSERVER)
    first._workflow_step_name = "step_one"
    second = second_class(
        api_key="workflow-api-key", execution_observer=SERVER_OBSERVER
    )
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


def test_a_dynamic_workflow_compiles_and_runs_with_no_observer_bound():
    """No host, no observer keys - the block must still be constructible.

    Dynamic blocks resolve init parameters under their own plugin namespace,
    which the core initializer defaults do not cover, so the engine has to
    supply the null observer explicitly. Without that this raises
    `BlockInitParameterNotProvidedError` at compile time.
    """
    # given
    from inference.core.workflows.execution_engine.core import ExecutionEngine

    specification = {
        "version": "1.0",
        "inputs": [{"type": "WorkflowParameter", "name": "value"}],
        "dynamic_blocks_definitions": [
            {
                "type": "DynamicBlockDefinition",
                "manifest": {
                    "type": "ManifestDescription",
                    "block_type": "UnboundProbe",
                    "inputs": {
                        "value": {
                            "type": "DynamicInputDefinition",
                            "selector_types": ["input_parameter"],
                        }
                    },
                    "outputs": {
                        "result": {"type": "DynamicOutputDefinition", "kind": []}
                    },
                },
                "code": {
                    "type": "PythonCode",
                    "run_function_code": _PASSTHROUGH_BLOCK.replace(
                        "def run_function(self, a, b)", "def run(self, value)"
                    ).replace('{"result": a + b}', '{"result": value}'),
                    "run_function_name": "run",
                },
            }
        ],
        "steps": [{"type": "UnboundProbe", "name": "probe", "value": "$inputs.value"}],
        "outputs": [
            {"type": "JsonField", "name": "result", "selector": "$steps.probe.result"}
        ],
    }

    # when - no `workflows_core.execution_observer`, no dynamic override
    engine = ExecutionEngine.init(
        workflow_definition=specification,
        init_parameters={"workflows_core.api_key": "no-observer-key"},
    )
    results = engine.run(runtime_parameters={"value": 7})

    # then
    assert results[0]["result"] == 7


def _recorded_rows(run_block, frozen_clock=(100.0, 100.25)):
    """Run `run_block` and return the workflow_block rows the collector built."""
    from inference.usage_tracking import collector as collector_module

    recorded = usage_collector.empty_usage_dict(exec_session_id="test-session")
    with mock.patch.object(
        collector_module, "GCP_SERVERLESS", False
    ), mock.patch.object(
        collector_module.time, "time", side_effect=list(frozen_clock)
    ), mock.patch.object(
        usage_collector, "_usage", recorded
    ):
        run_block()
    return [
        row
        for per_key in recorded.values()
        for row in per_key.values()
        if row.get("category") == "workflow_block"
    ]


@contextmanager
def _real_modal_executor_with_faked_transport(post_execute_response):
    """The production executor, with only the HTTP round trip replaced."""
    real_executor = modal_executor.ModalExecutor("test-workspace")
    with mock.patch.object(
        block_scaffolding, "WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE", "modal"
    ), _legacy_workspace_lookup_pinned(), mock.patch.object(
        modal_executor, "MODAL_AVAILABLE", True
    ), mock.patch.object(
        modal_executor, "get_modal_executor", lambda workspace_id=None: real_executor
    ), mock.patch.object(
        modal_executor.ModalExecutor,
        "_get_endpoint_url",
        return_value="https://example.invalid",
    ), mock.patch.object(
        modal_executor.ModalExecutor,
        "_post_execute",
        return_value=post_execute_response,
    ):
        yield


def test_local_block_emits_a_row_attributed_to_local_execution():
    # given
    block_class, _ = _assemble_block(_PASSTHROUGH_BLOCK, "row-local")
    block = block_class(api_key="workflow-api-key", execution_observer=SERVER_OBSERVER)
    monotonic = iter([200.0, 200.4])

    # when
    with mock.patch.object(
        block_scaffolding.time, "monotonic", side_effect=lambda: next(monotonic)
    ):
        rows = _recorded_rows(lambda: block.run(a=3, b=5))

    # then
    assert len(rows) == 1, rows
    details = json.loads(rows[0]["resource_details"])
    assert details["duration_source"] == BLOCK_DURATION_SOURCE_LOCAL_RUNTIME
    assert details["execution_mode"] == BLOCK_EXECUTION_MODE_LOCAL
    assert rows[0]["execution_duration"] == pytest.approx(0.4)


def test_remote_block_emits_a_row_from_the_executors_own_publication(
    isolated_modal_executor_cache,
):
    """The sandbox's runtime, published by `ModalExecutor.execute_remote` itself.

    The client call around it also covers input serialization and the round
    trip; that is not what must be billed.
    """
    # given
    block_class, _ = _assemble_block(_PASSTHROUGH_BLOCK, "row-remote-success")
    block = _modal_block(block_class)
    response = {
        "success": True,
        "result": json.dumps({"result": 8}),
        "execution_time_seconds": 0.6,
    }

    # when
    def run_block():
        with _real_modal_executor_with_faked_transport(response):
            assert block.run(a=3, b=5) == {"result": 8}

    rows = _recorded_rows(run_block)

    # then
    assert len(rows) == 1, rows
    details = json.loads(rows[0]["resource_details"])
    assert details["duration_source"] == BLOCK_DURATION_SOURCE_REMOTE_RUNTIME
    assert details["execution_mode"] == BLOCK_EXECUTION_MODE_REMOTE
    assert rows[0]["execution_duration"] == pytest.approx(0.6)


def test_remote_block_failing_inside_the_sandbox_is_billed_what_it_spent(
    isolated_modal_executor_cache,
):
    """The executor publishes before it raises, so a failed run is still billed.

    Distinct from a transport failure: the block did run, in the sandbox, for
    the time the sandbox reported.
    """
    # given
    block_class, _ = _assemble_block(_PASSTHROUGH_BLOCK, "row-remote-failure")
    block = _modal_block(block_class)
    response = {
        "success": False,
        "error": "boom",
        "error_type": "ValueError",
        "execution_time_seconds": 0.4,
    }

    # when
    def run_block():
        with _real_modal_executor_with_faked_transport(response):
            with pytest.raises(DynamicBlockCodeError):
                block.run(a=3, b=5)

    rows = _recorded_rows(run_block)

    # then
    assert len(rows) == 1, rows
    details = json.loads(rows[0]["resource_details"])
    assert details["duration_source"] == BLOCK_DURATION_SOURCE_REMOTE_RUNTIME
    assert details["execution_mode"] == BLOCK_EXECUTION_MODE_REMOTE
    # The HTTP arm raises `DynamicBlockCodeError` for a sandbox-side failure
    # WITHOUT an inner exception (`modal_executor.py:684-692` folds the
    # sandbox's error type into the message), so `inner_error_type` is None
    # and the collector records the wrapper's own class - unlike the local
    # arm, where `create_dynamic_block_code_error` attaches the user's
    # exception and the row says `RuntimeError`.
    assert details["error_type"] == "DynamicBlockCodeError"
    assert rows[0]["execution_duration"] == pytest.approx(0.4)
