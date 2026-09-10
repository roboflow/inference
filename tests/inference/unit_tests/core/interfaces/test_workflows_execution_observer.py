"""The server's ExecutionObserver: usage rows and OTel context for workflows.

Workflows no longer imports the usage collector or the telemetry helpers; this
object is where both are reached. Each test names the field it protects rather
than asserting "a row was recorded".
"""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest import mock

import pytest

from inference.core.interfaces.workflows_execution_observer import (
    ServerStepContext,
    UsageTrackingExecutionObserver,
)
from inference.core.workflows.execution_engine.v1.dynamic_blocks import block_duration
from inference.core.workflows.prototypes.observer import ExecutionObserver
from inference.usage_tracking import block_execution as server_channel
from inference.usage_tracking.collector import usage_collector
from inference.usage_tracking.stream_session import stream_session_id


@pytest.fixture(autouse=True)
def _cleared_channels():
    block_duration.clear_block_duration()
    server_channel.clear_measured_block_execution()
    yield
    block_duration.clear_block_duration()
    server_channel.clear_measured_block_execution()


def _compiled_workflow(api_key: str = "observer-key") -> SimpleNamespace:
    return SimpleNamespace(
        init_parameters={"workflows_core.api_key": api_key},
        workflow_json={"steps": [{"type": "SomeBlock", "name": "a_step"}]},
    )


def test_observer_satisfies_the_workflows_protocol() -> None:
    assert isinstance(UsageTrackingExecutionObserver(), ExecutionObserver)


def test_workflow_run_records_a_workflows_row_with_every_field() -> None:
    # given
    observer = UsageTrackingExecutionObserver()

    # when
    with mock.patch.object(usage_collector, "record_usage") as record_usage:
        result = observer.observe_workflow_run(
            workflow=_compiled_workflow(),
            runtime_parameters={"image": [SimpleNamespace(_image_reference="s3://x")]},
            workflow_id="wf-internal-id",
            fps=12.5,
            is_preview=True,
            run=lambda: [{"out": 1}],
        )

    # then
    assert result == [{"out": 1}]
    params = record_usage.call_args.kwargs
    assert params["category"] == "workflows"
    assert params["resource_id"] == "wf-internal-id"
    assert params["api_key"] == "observer-key"
    assert params["fps"] == 12.5
    assert params["source"] == "s3://x"
    assert params["resource_details"]["is_preview"] is True
    assert params["resource_details"]["billable"] is True
    assert params["resource_details"]["steps"] == ["SomeBlock:a_step"]


def test_workflow_run_records_a_row_when_the_run_raises() -> None:
    observer = UsageTrackingExecutionObserver()

    def boom():
        raise RuntimeError("step blew up")

    with mock.patch.object(usage_collector, "record_usage") as record_usage:
        with pytest.raises(RuntimeError):
            observer.observe_workflow_run(
                workflow=_compiled_workflow(),
                runtime_parameters={},
                workflow_id="wf-internal-id",
                fps=0,
                is_preview=False,
                run=boom,
            )

    details = record_usage.call_args.kwargs["resource_details"]
    assert details["error_type"] == "RuntimeError"


def test_workflow_id_falls_back_to_the_definition_hash_when_absent() -> None:
    observer = UsageTrackingExecutionObserver()
    with mock.patch.object(usage_collector, "record_usage") as record_usage:
        observer.observe_workflow_run(
            workflow=_compiled_workflow(),
            runtime_parameters={},
            workflow_id=None,
            fps=0,
            is_preview=False,
            run=lambda: [],
        )
    # Not "unknown": the collector hashes the resource details it extracted.
    assert record_usage.call_args.kwargs["resource_id"] not in ("", "unknown")


def test_block_run_records_a_workflow_block_row_and_relays_the_duration() -> None:
    # given
    observer = UsageTrackingExecutionObserver()
    block = SimpleNamespace(
        _usage_resource_id="custom_python/abc123",
        _api_key="block-key",
        _usage_block_kind="custom_python",
        _workflow_step_type="MeteredBlock",
        _workflow_step_name="my_step",
    )

    def run():
        block_duration.record_block_duration(
            duration=0.25,
            source=block_duration.BLOCK_DURATION_SOURCE_LOCAL_RUNTIME,
        )
        return {"result": 8}

    # when
    with mock.patch.object(usage_collector, "record_usage") as record_usage:
        result = observer.observe_block_run(
            block=block, block_args=(), block_kwargs={"a": 1}, run=run
        )

    # then
    assert result == {"result": 8}
    params = record_usage.call_args.kwargs
    assert params["category"] == "workflow_block"
    assert params["resource_id"] == "custom_python/abc123"
    assert params["api_key"] == "block-key"
    assert params["execution_duration"] == pytest.approx(0.25)
    assert params["resource_details"]["duration_source"] == "local_runtime"
    assert params["resource_details"]["execution_mode"] == "local"
    assert params["resource_details"]["step_name"] == "my_step"


def test_block_run_relays_the_duration_even_when_the_block_raises() -> None:
    observer = UsageTrackingExecutionObserver()
    block = SimpleNamespace(_usage_resource_id="custom_python/abc123", _api_key="k")

    def run():
        block_duration.record_block_duration(
            duration=0.5,
            source=block_duration.BLOCK_DURATION_SOURCE_REMOTE_RUNTIME,
        )
        raise RuntimeError("boom")

    with mock.patch.object(usage_collector, "record_usage") as record_usage:
        with pytest.raises(RuntimeError):
            observer.observe_block_run(
                block=block, block_args=(), block_kwargs={}, run=run
            )

    params = record_usage.call_args.kwargs
    assert params["execution_duration"] == pytest.approx(0.5)
    assert params["resource_details"]["duration_source"] == "remote_runtime"


def test_a_stale_host_measurement_cannot_be_billed_to_the_next_block() -> None:
    # given - a measurement nobody consumed, as a failed usage recording leaves
    server_channel.record_measured_block_execution(
        duration=9.0, source=server_channel.BLOCK_DURATION_SOURCE_REMOTE_RUNTIME
    )
    observer = UsageTrackingExecutionObserver()
    block = SimpleNamespace(_usage_resource_id="custom_python/abc123", _api_key="k")

    # when - the next block publishes nothing of its own
    with mock.patch.object(usage_collector, "record_usage") as record_usage:
        observer.observe_block_run(
            block=block, block_args=(), block_kwargs={}, run=lambda: {"r": 1}
        )

    # then - the decorator's own wall clock, not the 9 seconds
    params = record_usage.call_args.kwargs
    assert params["execution_duration"] < 9.0
    assert params["resource_details"]["duration_source"] == "decorator_wall_clock"


def test_model_run_records_a_model_row_for_the_block_that_owns_the_model() -> None:
    observer = UsageTrackingExecutionObserver()
    block = SimpleNamespace(_api_key="sam-key")

    with mock.patch.object(usage_collector, "record_usage") as record_usage:
        result = observer.observe_model_run(
            block=block,
            model_id="sam2video",
            images=[object(), object()],
            run=lambda: [{"masks": None}],
        )

    assert result == [{"masks": None}]
    params = record_usage.call_args.kwargs
    assert params["category"] == "model"
    assert params["resource_id"] == "sam2video"
    assert params["api_key"] == "sam-key"
    assert params["frames"] == 2


def test_step_context_is_captured_in_the_caller_and_rebound_in_the_worker() -> None:
    # given - the pipeline's session id, bound in the thread that submits work
    observer = UsageTrackingExecutionObserver()
    token = stream_session_id.set("camera-7")
    try:
        context = observer.capture_step_context()
    finally:
        stream_session_id.reset(token)
    seen = {}

    def worker():
        with observer.step_scope(context=context, step_name="a_step"):
            seen["value"] = stream_session_id.get()

    # when
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(worker).result()

    # then
    assert isinstance(context, ServerStepContext)
    assert seen["value"] == "camera-7"


def test_step_scope_clears_a_previous_requests_session_id() -> None:
    # given - pool threads are reused; a previous pipeline's id must not leak
    observer = UsageTrackingExecutionObserver()
    context = observer.capture_step_context()  # no session bound here
    seen = {}

    def worker():
        stream_session_id.set("stale-stream")
        with observer.step_scope(context=context, step_name="a_step"):
            seen["value"] = stream_session_id.get()

    # when
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(worker).result()

    # then
    assert seen["value"] is None


_STREAM_PROBE_BLOCK = """
def run(self, value) -> BlockResult:
    return {"result": value}
"""

_STREAM_PROBE_WORKFLOW = {
    "version": "1.0",
    "inputs": [{"type": "WorkflowParameter", "name": "value"}],
    "dynamic_blocks_definitions": [
        {
            "type": "DynamicBlockDefinition",
            "manifest": {
                "type": "ManifestDescription",
                "block_type": "StreamProbe",
                "inputs": {
                    "value": {
                        "type": "DynamicInputDefinition",
                        "selector_types": ["input_parameter"],
                    }
                },
                "outputs": {"result": {"type": "DynamicOutputDefinition", "kind": []}},
            },
            "code": {"type": "PythonCode", "run_function_code": _STREAM_PROBE_BLOCK},
        }
    ],
    "steps": [{"type": "StreamProbe", "name": "probe", "value": "$inputs.value"}],
    "outputs": [
        {"type": "JsonField", "name": "result", "selector": "$steps.probe.result"}
    ],
}


def _rows_by_category(recorded: dict) -> dict:
    rows = [row for per_key in recorded.values() for row in per_key.values()]
    return {row["category"]: row for row in rows}


def test_stream_session_reaches_the_workflow_and_the_block_rows() -> None:
    """A pipeline's session id must be on every row its run produces.

    Real engine, real observer, real collector; only the usage dictionary is a
    throwaway. The session id is bound in the calling thread, and the block row
    is recorded inside a pool worker - so this is the end-to-end statement of
    what `capture_step_context` / `step_scope` exist for.
    """
    # given
    from inference.core.workflows.execution_engine.core import ExecutionEngine

    observer = UsageTrackingExecutionObserver()
    engine = ExecutionEngine.init(
        workflow_definition=_STREAM_PROBE_WORKFLOW,
        init_parameters={
            "workflows_core.api_key": "stream-identity-key",
            "workflows_core.execution_observer": observer,
        },
    )
    recorded = usage_collector.empty_usage_dict(exec_session_id="test-session")

    # when
    token = stream_session_id.set("camera-7")
    try:
        with mock.patch.object(usage_collector, "_usage", recorded):
            engine.run(runtime_parameters={"value": 1})
    finally:
        stream_session_id.reset(token)

    # then
    by_category = _rows_by_category(recorded)
    assert set(by_category) == {"workflows", "workflow_block"}
    assert by_category["workflows"]["stream_session_id"] == "camera-7"
    assert by_category["workflow_block"]["stream_session_id"] == "camera-7"


def test_a_run_without_a_stream_session_produces_rows_without_one() -> None:
    # given - the same workflow, no session bound anywhere
    from inference.core.workflows.execution_engine.core import ExecutionEngine

    engine = ExecutionEngine.init(
        workflow_definition=_STREAM_PROBE_WORKFLOW,
        init_parameters={
            "workflows_core.api_key": "no-stream-key",
            "workflows_core.execution_observer": UsageTrackingExecutionObserver(),
        },
    )
    recorded = usage_collector.empty_usage_dict(exec_session_id="test-session")

    # when
    with mock.patch.object(usage_collector, "_usage", recorded):
        engine.run(runtime_parameters={"value": 1})

    # then - no stale id from a previous test's pool thread
    by_category = _rows_by_category(recorded)
    assert set(by_category) == {"workflows", "workflow_block"}
    for row in by_category.values():
        assert not row.get("stream_session_id")
