from inference.core.workflows.execution_engine.v1.dynamic_blocks.debug_logs import (
    CAPACITY_EXCEEDED_MARKER,
    STREAM_TRUNCATION_MARKER,
    DebugLogsCollector,
    get_active_collector,
    register_debug_collector,
)


def test_collector_records_entries_below_limits_unchanged() -> None:
    # given
    collector = DebugLogsCollector(max_chars_per_stream=100, max_total_chars=1000)

    # when
    collector.record(step_name="step", stdout="hello", stderr="world")

    # then
    assert collector.snapshot() == {
        "step": [{"stdout": "hello", "stderr": "world"}],
    }


def test_collector_truncates_oversized_streams() -> None:
    # given
    collector = DebugLogsCollector(max_chars_per_stream=10, max_total_chars=1000)

    # when
    collector.record(step_name="step", stdout="x" * 50, stderr="y" * 5)

    # then - stdout cut to the limit + marker, stderr untouched
    entry = collector.snapshot()["step"][0]
    assert entry["stdout"] == "x" * 10 + STREAM_TRUNCATION_MARKER
    assert entry["stderr"] == "y" * 5


def test_collector_stops_recording_after_total_capacity_exceeded() -> None:
    # given - room for two 40-char entries but not three
    collector = DebugLogsCollector(max_chars_per_stream=100, max_total_chars=100)

    # when
    collector.record(step_name="a", stdout="x" * 40, stderr=None)
    collector.record(step_name="b", stdout="y" * 40, stderr=None)
    collector.record(step_name="c", stdout="z" * 40, stderr=None)  # exceeds cap
    collector.record(step_name="d", stdout="dropped", stderr=None)  # after cap

    # then - first two recorded, third replaced by marker, fourth dropped
    snapshot = collector.snapshot()
    assert snapshot["a"] == [{"stdout": "x" * 40, "stderr": None}]
    assert snapshot["b"] == [{"stdout": "y" * 40, "stderr": None}]
    assert snapshot["c"] == [{"stdout": CAPACITY_EXCEEDED_MARKER, "stderr": None}]
    assert "d" not in snapshot


def test_collector_ignores_empty_records() -> None:
    # given
    collector = DebugLogsCollector()

    # when
    collector.record(step_name="step", stdout=None, stderr=None)

    # then
    assert collector.snapshot() == {}


def test_register_debug_collector_publishes_and_resets_contextvar() -> None:
    # given / when / then
    assert get_active_collector() is None
    with register_debug_collector() as collector:
        assert get_active_collector() is collector
    assert get_active_collector() is None
