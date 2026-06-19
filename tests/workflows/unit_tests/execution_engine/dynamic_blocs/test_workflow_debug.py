import pytest

from inference.core.workflows.execution_engine.v1.dynamic_blocks.debug_logs import (
    register_debug_session,
)
from inference.core.workflows.execution_engine.v1.dynamic_blocks.workflow_debug import (
    CAPACITY_EXCEEDED_MARKER,
    WorkflowDebugTrace,
    _entry_serialized_size,
    current_debug_step_name,
    current_debug_trace,
    debug_traces,
    get_active_debug_trace,
)


def test_debug_traces_proxy_is_noop_without_active_trace() -> None:
    # given / when / then
    assert get_active_debug_trace() is None
    debug_traces.append({"ignored": True})
    assert get_active_debug_trace() is None


def test_trace_records_entries_with_step_name() -> None:
    # given
    trace = WorkflowDebugTrace()

    # when
    trace.append(step_name="step_a", value={"count": 3})
    trace.append(step_name="step_b", value="done")

    # then
    assert trace.snapshot() == [
        {"step": "step_a", "value": {"count": 3}},
        {"step": "step_b", "value": "done"},
    ]


def test_debug_traces_proxy_forwards_to_active_trace_with_current_step() -> None:
    # given
    trace = WorkflowDebugTrace()
    token_trace = current_debug_trace.set(trace)
    token_step = current_debug_step_name.set("my_step")
    try:
        # when
        debug_traces.append({"value": 42})
    finally:
        current_debug_step_name.reset(token_step)
        current_debug_trace.reset(token_trace)

    # then
    assert trace.snapshot() == [{"step": "my_step", "value": {"value": 42}}]


def test_trace_serialises_non_json_values_with_repr() -> None:
    # given
    trace = WorkflowDebugTrace()

    # when
    trace.append(step_name="step", value=object())

    # then
    entry = trace.snapshot()[0]
    assert entry["step"] == "step"
    assert entry["value"].startswith("<object object at")


def test_trace_stops_recording_after_total_serialized_size_exceeded() -> None:
    # given - budget fits two small JSON entries but not three
    trace = WorkflowDebugTrace(max_total_serialized_chars=60)

    # when
    trace.append(step_name="a", value="a")
    trace.append(step_name="b", value="b")
    trace.append(step_name="c", value="c")  # exceeds cap
    trace.append(step_name="d", value="d")  # after cap

    # then - first two recorded, third replaced by marker, fourth dropped
    snapshot = trace.snapshot()
    assert snapshot[0] == {"step": "a", "value": "a"}
    assert snapshot[1] == {"step": "b", "value": "b"}
    assert snapshot[2] == {"step": "c", "value": CAPACITY_EXCEEDED_MARKER}
    assert len(snapshot) == 3


def test_trace_stops_recording_after_entry_limit() -> None:
    # given
    trace = WorkflowDebugTrace(max_entries=2)

    # when
    trace.append(step_name="a", value=1)
    trace.append(step_name="b", value=2)
    trace.append(step_name="c", value=3)
    trace.append(step_name="d", value=4)

    # then
    snapshot = trace.snapshot()
    assert snapshot[0] == {"step": "a", "value": 1}
    assert snapshot[1] == {"step": "b", "value": 2}
    assert snapshot[2] == {"step": "c", "value": CAPACITY_EXCEEDED_MARKER}
    assert len(snapshot) == 3


def test_trace_truncates_oversized_entries() -> None:
    # given
    trace = WorkflowDebugTrace(max_entry_serialized_chars=50)
    large_value = "x" * 200

    # when
    trace.append(step_name="step", value=large_value)

    # then
    entry = trace.snapshot()[0]
    assert entry["step"] == "step"
    assert entry["value"].endswith("... [entry truncated]")
    assert _entry_serialized_size(entry) <= 50


def test_trace_does_not_hang_when_metadata_exceeds_entry_cap() -> None:
    # given - a step name alone is larger than the per-entry cap, so the value
    # cannot be truncated enough to fit (previously this looped forever).
    trace = WorkflowDebugTrace(max_entry_serialized_chars=50)
    oversized_step = "s" * 500

    # when - non-string value drove the infinite-loop path
    trace.append(step_name=oversized_step, value={"payload": "x" * 200})

    # then
    snapshot = trace.snapshot()
    assert len(snapshot) == 1
    assert snapshot[0]["value"] == CAPACITY_EXCEEDED_MARKER

    # and - the trace is now closed to further entries
    trace.append(step_name="step", value="anything")
    assert len(trace.snapshot()) == 1


def test_debug_traces_proxy_append_with_timestamp() -> None:
    # given
    trace = WorkflowDebugTrace()
    token_trace = current_debug_trace.set(trace)
    token_step = current_debug_step_name.set("my_step")
    try:
        # when
        debug_traces.append({"received": "hello"}, add_timestamp=True)
    finally:
        current_debug_step_name.reset(token_step)
        current_debug_trace.reset(token_trace)

    # then
    entry = trace.snapshot()[0]
    assert entry["step"] == "my_step"
    assert entry["value"] == {"received": "hello"}
    assert "timestamp" in entry
    assert entry["timestamp"].endswith("+00:00")
    assert entry["timestamp_timezone"] == "UTC"


def test_debug_traces_proxy_append_with_custom_timezone() -> None:
    # given
    trace = WorkflowDebugTrace()
    token_trace = current_debug_trace.set(trace)
    try:
        # when
        debug_traces.append(
            {"received": "hello"},
            add_timestamp=True,
            timezone="America/New_York",
        )
    finally:
        current_debug_trace.reset(token_trace)

    # then
    entry = trace.snapshot()[0]
    assert entry["timestamp_timezone"] == "America/New_York"
    assert entry["timestamp"].endswith("-04:00") or entry["timestamp"].endswith(
        "-05:00"
    )


def test_debug_traces_proxy_append_with_invalid_timezone_raises() -> None:
    # given
    trace = WorkflowDebugTrace()
    token_trace = current_debug_trace.set(trace)
    try:
        # when / then
        with pytest.raises(ValueError, match="Invalid timezone"):
            debug_traces.append(
                {"received": "hello"}, add_timestamp=True, timezone="Not/AZone"
            )
    finally:
        current_debug_trace.reset(token_trace)


def test_debug_traces_proxy_append_without_timestamp_omits_field() -> None:
    # given
    trace = WorkflowDebugTrace()
    token_trace = current_debug_trace.set(trace)
    try:
        # when
        debug_traces.append({"received": "hello"})
    finally:
        current_debug_trace.reset(token_trace)

    # then
    entry = trace.snapshot()[0]
    assert "timestamp" not in entry


def test_register_debug_session_publishes_debug_trace() -> None:
    # given / when / then
    assert get_active_debug_trace() is None
    with register_debug_session() as session:
        assert get_active_debug_trace() is session.debug_traces
        debug_traces.append("hello")
        assert session.debug_traces.snapshot() == [{"step": None, "value": "hello"}]
    assert get_active_debug_trace() is None
