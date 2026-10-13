"""``v2/continue_if@v1``: compiled UQL gate and its per-lane ``stop_delay`` hold.

Calls run inside a hand-built execution context whose scope answers the
source and time of the call's index, as the engine's does.
"""

import threading
from fractions import Fraction
from typing import Optional

import pytest
from pydantic import ValidationError
from roboflow_workflows.v2.blocks.control.conditions import (
    CONDITION_STATEMENT_EXAMPLE,
    ContinueIfQueryBlock,
    HoldCapacityError,
    ObservationOrderError,
)
from roboflow_workflows.v2.blocks.query import QueryRuntimeError
from roboflow_workflows.v2.data.batch import (
    SampleContext,
    TemporalContext,
    TimeSpan,
    Timestamp,
)
from roboflow_workflows.v2.engine.context import (
    ExecutionContext,
    use_execution_context,
)
from roboflow_workflows.v2.engine.declaration import Select, Stop

NEXT = ["$steps.detector"]
SCORE_AT_LEAST_MINIMUM = {
    "type": "StatementGroup",
    "statements": [
        {
            "type": "BinaryStatement",
            "left_operand": {"type": "DynamicOperand", "operand_name": "score"},
            "comparator": {"type": "(Number) >="},
            "right_operand": {"type": "DynamicOperand", "operand_name": "minimum"},
        }
    ],
}


class _Scope:
    """Engine scope stand-in: one source and one time for every index."""

    def __init__(
        self, sample: Optional[SampleContext], temporal: Optional[TemporalContext]
    ) -> None:
        self._sample = sample
        self._temporal = temporal

    def sample_at(self, index):
        return self._sample

    def temporal_at(self, index):
        return self._temporal


def _at(seconds: float, clock: str = "engine.monotonic") -> TemporalContext:
    stamp = Timestamp(round(seconds * 1000), Fraction(1, 1000), clock)

    return TemporalContext(observed_coverage=stamp)


def _gate(
    block: ContinueIfQueryBlock,
    score: float,
    *,
    source: Optional[str] = "cam-a",
    seconds: Optional[float] = None,
    stop_delay: float = 0.0,
    index=(0,),
    run_id: str = "run",
    max_sources: int = 256,
    clock: str = "engine.monotonic",
) -> bool:
    sample = SampleContext(source_id=source) if source is not None else None
    temporal = _at(seconds, clock) if seconds is not None else None
    context = ExecutionContext(
        step_path=("gate",),
        block_type=ContinueIfQueryBlock.type,
        session_id="session",
        run_id=run_id,
        indices=(index,),
        call_scope=_Scope(sample, temporal),
    )
    with use_execution_context(context):
        decision = block.run(
            condition_statement=SCORE_AT_LEAST_MINIMUM,
            evaluation_parameters={"score": score, "minimum": 0.5},
            next_steps=NEXT,
            stop_delay=stop_delay,
            max_sources=max_sources,
        )
    assert isinstance(decision, Select)

    return not isinstance(decision, Stop)


def _params(**overrides):
    fields = {
        "condition_statement": SCORE_AT_LEAST_MINIMUM,
        "evaluation_parameters": {"score": "$steps.s.output", "minimum": 0.5},
        "next_steps": NEXT,
    }
    fields.update(overrides)

    return ContinueIfQueryBlock.Params(**fields)


def test_an_undeclared_operand_fails_validation_with_its_name_and_path() -> None:
    with pytest.raises(ValidationError) as error:
        _params(evaluation_parameters={"score": "$steps.s.output"})

    message = str(error.value)
    assert "operand `minimum` is not declared" in message
    assert "$.condition_statement.statements[0].right_operand" in message


def test_a_nested_filter_operand_is_checked_at_compile_time_too() -> None:
    class_in_classes = {
        "type": "StatementGroup",
        "statements": [
            {
                "type": "BinaryStatement",
                "left_operand": {
                    "type": "DynamicOperand",
                    "operations": [
                        {
                            "type": "ExtractDetectionProperty",
                            "property_name": "class_name",
                        }
                    ],
                },
                "comparator": {"type": "in (Sequence)"},
                "right_operand": {"type": "DynamicOperand", "operand_name": "classes"},
            }
        ],
    }
    any_kept = {
        "type": "StatementGroup",
        "statements": [
            {
                "type": "UnaryStatement",
                "operand": {
                    "type": "DynamicOperand",
                    "operand_name": "predictions",
                    "operations": [
                        {
                            "type": "DetectionsFilter",
                            "filter_operation": class_in_classes,
                        }
                    ],
                },
                "operator": {"type": "(Sequence) is not empty"},
            }
        ],
    }

    with pytest.raises(ValidationError, match="operand `classes` is not declared"):
        _params(
            condition_statement=any_kept,
            evaluation_parameters={"predictions": "$steps.model.predictions"},
        )


@pytest.mark.parametrize("stop_delay", [-1, -0.5, float("inf"), float("nan"), True])
def test_stop_delay_must_be_a_finite_number_of_at_least_zero(stop_delay) -> None:
    with pytest.raises(ValidationError):
        _params(stop_delay=stop_delay)


@pytest.mark.parametrize("stop_delay", [0, 0.0, 2, 2.5, "$inputs.delay"])
def test_stop_delay_accepts_zero_positive_numbers_and_a_selector(stop_delay) -> None:
    params = _params(stop_delay=stop_delay)

    assert params.stop_delay == stop_delay


def test_the_documented_example_is_a_valid_definition() -> None:
    params = _params(
        condition_statement=CONDITION_STATEMENT_EXAMPLE,
        evaluation_parameters={"prediction": "$steps.c.predictions", "minimum": 0.7},
    )

    assert params.stop_delay == 0


def test_without_delay_the_gate_follows_the_condition_and_keeps_no_state() -> None:
    block = ContinueIfQueryBlock()

    assert _gate(block, 0.9) is True
    assert _gate(block, 0.1) is False
    assert block._lanes._lanes == {}


def test_empty_next_steps_stop_even_when_the_condition_holds() -> None:
    block = ContinueIfQueryBlock()

    decision = block.run(
        condition_statement=SCORE_AT_LEAST_MINIMUM,
        evaluation_parameters={"score": 0.9, "minimum": 0.5},
        next_steps=[],
        stop_delay=5.0,
        max_sources=1,
    )

    assert isinstance(decision, Stop)


def test_a_condition_that_cannot_compare_its_values_fails_the_call() -> None:
    block = ContinueIfQueryBlock()

    with pytest.raises(QueryRuntimeError, match="cannot compare"):
        _gate(block, [0.9, 0.8])


def test_a_hold_keeps_the_gate_open_for_stop_delay_seconds_after_a_true() -> None:
    block = ContinueIfQueryBlock()

    opened = [
        _gate(block, score, seconds=seconds, stop_delay=2.0)
        for score, seconds in [
            (0.1, 0.0),
            (0.9, 1.0),
            (0.1, 2.0),
            (0.1, 3.0),
            (0.1, 3.5),
        ]
    ]

    # 3.0 - 1.0 == stop_delay is still inside; 3.5 is not.
    assert opened == [False, True, True, True, False]


def test_each_source_has_its_own_hold() -> None:
    block = ContinueIfQueryBlock()

    assert _gate(block, 0.9, source="cam-a", seconds=1.0, stop_delay=5.0)
    assert not _gate(block, 0.1, source="cam-b", seconds=2.0, stop_delay=5.0)
    assert _gate(block, 0.1, source="cam-a", seconds=2.0, stop_delay=5.0)


def test_calls_of_one_frame_are_held_only_by_earlier_trues() -> None:
    # Per-crop gates of one frame share its time; their order must not matter.
    for order in ([0.9, 0.1], [0.1, 0.9]):
        block = ContinueIfQueryBlock()
        frame_1 = {
            score: _gate(block, score, index=(0, k), seconds=1.0, stop_delay=5.0)
            for k, score in enumerate(order)
        }
        frame_2 = _gate(block, 0.1, index=(0, 0), seconds=2.0, stop_delay=5.0)

        assert frame_1 == {0.9: True, 0.1: False}
        assert frame_2 is True


def test_an_earlier_true_still_holds_a_later_frame_after_a_same_time_true() -> None:
    block = ContinueIfQueryBlock()
    _gate(block, 0.9, seconds=1.0, stop_delay=5.0)
    _gate(block, 0.9, seconds=2.0, index=(0, 1), stop_delay=5.0)

    assert _gate(block, 0.1, seconds=2.0, index=(0, 0), stop_delay=5.0)


def test_a_temporal_span_is_timed_by_its_end() -> None:
    block = ContinueIfQueryBlock()
    _gate(block, 0.9, seconds=0.0, stop_delay=1.0)
    span = TimeSpan(
        Timestamp(500, Fraction(1, 1000), "engine.monotonic"),
        Timestamp(1500, Fraction(1, 1000), "engine.monotonic"),
    )
    context = ExecutionContext(
        step_path=("gate",),
        block_type=ContinueIfQueryBlock.type,
        session_id="session",
        run_id="run",
        indices=((0,),),
        call_scope=_Scope(
            SampleContext(source_id="cam-a"), TemporalContext(observed_coverage=span)
        ),
    )
    with use_execution_context(context):
        decision = block.run(
            condition_statement=SCORE_AT_LEAST_MINIMUM,
            evaluation_parameters={"score": 0.1, "minimum": 0.5},
            next_steps=NEXT,
            stop_delay=1.0,
            max_sources=4,
        )

    assert isinstance(decision, Stop)


def test_without_source_each_batch_position_is_its_own_lane() -> None:
    block = ContinueIfQueryBlock()

    first_run = [
        _gate(block, score, source=None, index=(n,), run_id="r1", stop_delay=60.0)
        for n, score in enumerate([0.9, 0.1, 0.9])
    ]
    second_run = [
        _gate(block, 0.1, source=None, index=(n,), run_id="r2", stop_delay=60.0)
        for n in range(3)
    ]

    assert first_run == [True, False, True]
    assert second_run == [True, False, True]


def test_without_temporal_context_one_run_is_one_moment() -> None:
    block = ContinueIfQueryBlock()

    # Crops of one passive run: a true does not hold a same-run false.
    same_run = [
        _gate(block, score, source=None, index=(0, k), run_id="r1", stop_delay=60.0)
        for k, score in enumerate([0.9, 0.1])
    ]

    assert same_run == [True, False]
    assert _gate(block, 0.1, source=None, index=(0, 1), run_id="r2", stop_delay=60.0)


def _record(block: ContinueIfQueryBlock, source: str = "cam-a") -> tuple:
    record = block._lanes._lanes[("source", source)]

    return record.last_seen, record.latest_true, record.previous_true


def test_an_expired_hold_clears_its_history_but_keeps_the_lane_record() -> None:
    block = ContinueIfQueryBlock()
    _gate(block, 0.9, seconds=0.0, stop_delay=1.0)

    assert not _gate(block, 0.1, seconds=5.0, stop_delay=1.0)
    assert _record(block) == (5, None, None)


def test_the_order_guard_survives_expiry_so_a_late_true_cannot_restart() -> None:
    block = ContinueIfQueryBlock()
    _gate(block, 0.9, seconds=0.0, stop_delay=2.0)
    assert not _gate(block, 0.1, seconds=10.0, stop_delay=2.0)

    with pytest.raises(ObservationOrderError, match="time 1 s after 10 s"):
        _gate(block, 0.9, seconds=1.0, stop_delay=2.0)
    assert not _gate(block, 0.1, seconds=11.0, stop_delay=2.0)


def test_an_earlier_false_than_the_lanes_last_time_fails_without_changes() -> None:
    # Trues at 0, 2, 3, then a false observed at 1: rejected, not answered
    # from a predecessor the lane no longer keeps.
    block = ContinueIfQueryBlock()
    for seconds in (0.0, 2.0, 3.0):
        _gate(block, 0.9, seconds=seconds, stop_delay=5.0)

    with pytest.raises(ObservationOrderError, match="must not go backward"):
        _gate(block, 0.1, seconds=1.0, stop_delay=5.0)
    assert _record(block) == (3, 3, 2)


def test_a_false_only_lane_keeps_an_order_guard_and_counts_toward_capacity() -> None:
    block = ContinueIfQueryBlock()
    assert not _gate(block, 0.1, source="cam-a", seconds=5.0, stop_delay=1.0)

    with pytest.raises(ObservationOrderError):
        _gate(block, 0.9, source="cam-a", seconds=4.0, stop_delay=1.0)
    with pytest.raises(HoldCapacityError, match="max_sources=1"):
        _gate(block, 0.9, source="cam-b", seconds=6.0, stop_delay=1.0, max_sources=1)


def test_a_new_lane_beyond_capacity_fails_and_nothing_is_reclaimed_by_time() -> None:
    # cam-b's later time does not prove cam-a's hold is over: an older cam-a
    # observation may still be in flight.
    block = ContinueIfQueryBlock()
    _gate(block, 0.9, source="cam-a", seconds=0.0, stop_delay=1.0, max_sources=1)

    with pytest.raises(HoldCapacityError) as error:
        _gate(block, 0.9, source="cam-b", seconds=50.0, stop_delay=1.0, max_sources=1)
    message = str(error.value)
    assert "lane ('source', 'cam-b') would be lane 2 of max_sources=1" in message
    assert "Raise max_sources, or reset processing" in message
    assert set(block._lanes._lanes) == {("source", "cam-a")}
    assert _gate(block, 0.1, source="cam-a", seconds=1.0, stop_delay=1.0)


def test_a_known_lane_reuses_its_record_at_capacity() -> None:
    block = ContinueIfQueryBlock()
    for seconds in (0.0, 10.0, 20.0):
        _gate(block, 0.9, seconds=seconds, stop_delay=1.0, max_sources=1)

    assert not _gate(block, 0.1, seconds=30.0, stop_delay=1.0, max_sources=1)
    assert _gate(block, 0.9, seconds=31.0, stop_delay=1.0, max_sources=1)


def test_a_source_that_changes_clocks_fails_the_call() -> None:
    block = ContinueIfQueryBlock()
    _gate(block, 0.9, seconds=0.0, stop_delay=1.0)

    with pytest.raises(ObservationOrderError, match="changed clocks"):
        _gate(block, 0.1, seconds=0.5, stop_delay=1.0, clock="media")
    assert _gate(block, 0.1, seconds=0.5, stop_delay=1.0)


def test_a_clock_change_after_expiry_still_fails() -> None:
    block = ContinueIfQueryBlock()
    _gate(block, 0.9, seconds=0.0, stop_delay=1.0)
    assert not _gate(block, 0.1, seconds=5.0, stop_delay=1.0)

    with pytest.raises(ObservationOrderError, match="changed clocks"):
        _gate(block, 0.9, seconds=6.0, stop_delay=1.0, clock="media")


def test_zero_delay_bypasses_records_and_re_enabling_checks_the_last_one() -> None:
    block = ContinueIfQueryBlock()
    _gate(block, 0.9, seconds=10.0, stop_delay=5.0)

    # Delay 0: any time and clock, any new lane, no record change.
    assert _gate(block, 0.9, seconds=1.0, stop_delay=0.0)
    assert not _gate(block, 0.1, seconds=2.0, stop_delay=0.0, clock="media")
    assert not _gate(block, 0.1, source="cam-b", seconds=0.0, stop_delay=0.0)
    assert set(block._lanes._lanes) == {("source", "cam-a")}
    assert _record(block) == (10, 10, None)

    # Re-enabled: checked against 10, the last nonzero-delay observation.
    with pytest.raises(ObservationOrderError):
        _gate(block, 0.1, seconds=9.0, stop_delay=5.0)
    assert _gate(block, 0.1, seconds=15.0, stop_delay=5.0)
    assert not _gate(block, 0.1, seconds=15.5, stop_delay=5.0)


def test_a_new_block_after_processing_reset_starts_without_records() -> None:
    # A processing reset builds a new block instance.
    before_reset = ContinueIfQueryBlock()
    _gate(before_reset, 0.9, seconds=100.0, stop_delay=5.0, max_sources=1)

    after_reset = ContinueIfQueryBlock()
    assert after_reset._lanes._lanes == {}
    assert not _gate(after_reset, 0.1, seconds=1.0, stop_delay=5.0, max_sources=1)


@pytest.mark.parametrize("stop_delay", [-1.0, float("nan"), True])
def test_an_invalid_selector_bound_stop_delay_fails_the_call(stop_delay) -> None:
    block = ContinueIfQueryBlock()

    with pytest.raises(ValueError, match="stop_delay must be a finite number"):
        _gate(block, 0.1, seconds=0.0, stop_delay=stop_delay)


def test_each_call_measures_the_hold_with_its_own_stop_delay() -> None:
    block = ContinueIfQueryBlock()
    _gate(block, 0.9, seconds=0.0, stop_delay=1.0)

    assert _gate(block, 0.1, seconds=2.0, stop_delay=5.0)
    assert not _gate(block, 0.1, seconds=3.0, stop_delay=1.0)
    # The expired hold was cleared; a longer delay does not revive it.
    assert not _gate(block, 0.1, seconds=4.0, stop_delay=5.0)


def test_concurrent_calls_of_different_sources_take_turns_on_the_timers() -> None:
    block = ContinueIfQueryBlock()
    inside, release = threading.Event(), threading.Event()
    observe = block._lanes.observe

    def paused_observe(lane, **kwargs):
        if lane == ("source", "cam-a"):
            inside.set()
            assert release.wait(timeout=5)

        return observe(lane, **kwargs)

    block._lanes.observe = paused_observe
    results = {}

    def call(source: str) -> None:
        results[source] = _gate(block, 0.9, source=source, seconds=1.0, stop_delay=5.0)

    first = threading.Thread(target=call, args=("cam-a",))
    second = threading.Thread(target=call, args=("cam-b",))
    first.start()
    assert inside.wait(timeout=5)
    second.start()
    second.join(timeout=0.2)
    blocked_while_first_inside = second.is_alive()
    release.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert blocked_while_first_inside
    assert results == {"cam-a": True, "cam-b": True}
    assert set(block._lanes._lanes) == {("source", "cam-a"), ("source", "cam-b")}
