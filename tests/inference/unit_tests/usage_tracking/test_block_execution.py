import pytest

from inference.usage_tracking.block_execution import (
    BLOCK_DURATION_SOURCE_LOCAL_RUNTIME,
    BLOCK_DURATION_SOURCE_REMOTE_RUNTIME,
    BLOCK_EXECUTION_MODE_LOCAL,
    BLOCK_EXECUTION_MODE_REMOTE,
    clear_measured_block_execution,
    consume_measured_block_execution,
    record_measured_block_execution,
)


@pytest.fixture(autouse=True)
def cleared_block_execution():
    clear_measured_block_execution()
    yield
    clear_measured_block_execution()


def test_consume_returns_none_when_nothing_was_recorded():
    assert consume_measured_block_execution() is None


def test_consume_returns_the_recorded_measurement():
    record_measured_block_execution(
        duration=0.25,
        source=BLOCK_DURATION_SOURCE_REMOTE_RUNTIME,
        execution_mode=BLOCK_EXECUTION_MODE_REMOTE,
    )

    measured = consume_measured_block_execution()

    assert measured.duration == 0.25
    assert measured.source == BLOCK_DURATION_SOURCE_REMOTE_RUNTIME
    assert measured.execution_mode == BLOCK_EXECUTION_MODE_REMOTE


def test_consume_clears_so_a_stale_duration_is_not_billed_twice():
    record_measured_block_execution(
        duration=0.5,
        source=BLOCK_DURATION_SOURCE_LOCAL_RUNTIME,
        execution_mode=BLOCK_EXECUTION_MODE_LOCAL,
    )

    assert consume_measured_block_execution() is not None
    assert consume_measured_block_execution() is None


def test_last_recorded_measurement_wins():
    record_measured_block_execution(
        duration=0.1,
        source=BLOCK_DURATION_SOURCE_LOCAL_RUNTIME,
    )
    record_measured_block_execution(
        duration=0.9,
        source=BLOCK_DURATION_SOURCE_REMOTE_RUNTIME,
    )

    measured = consume_measured_block_execution()

    assert measured.duration == 0.9
    assert measured.source == BLOCK_DURATION_SOURCE_REMOTE_RUNTIME


@pytest.mark.parametrize(
    "duration",
    [None, "0.5", -1.0, float("nan"), float("inf"), True],
)
def test_unusable_durations_are_dropped_rather_than_recorded(duration):
    record_measured_block_execution(
        duration=duration,
        source=BLOCK_DURATION_SOURCE_REMOTE_RUNTIME,
    )

    assert consume_measured_block_execution() is None


def test_zero_duration_is_recorded():
    record_measured_block_execution(
        duration=0,
        source=BLOCK_DURATION_SOURCE_LOCAL_RUNTIME,
    )

    measured = consume_measured_block_execution()

    assert measured is not None
    assert measured.duration == 0.0
