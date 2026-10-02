"""Stage gates: per-domain turns, prefix retirement, nth-call retirement, abort.

Every probe is event-driven: a thread that must wait is observed blocked
through the gate's counters, never inferred from a sleep.
"""

import threading
import time
from typing import Callable, List

import pytest
from roboflow_workflows.execution_engine.v2.pipelining.stages import (
    SERIAL,
    PipelineCounters,
    RunAborted,
    StageGate,
    StepStages,
    Ticket,
)

TIMEOUT = 5.0


def make_gate(name: str = "$steps.s#call") -> StageGate:
    gate = StageGate(name, aborted=threading.Event(), counters=PipelineCounters())

    return gate


def wait_for(condition: Callable[[], bool]) -> None:
    """Poll a state change another thread makes; fail after TIMEOUT."""
    deadline = time.monotonic() + TIMEOUT
    while not condition():
        assert time.monotonic() < deadline, "condition not reached"
        time.sleep(0.001)


def started(target: Callable[[], None]) -> threading.Thread:
    thread = threading.Thread(target=target, daemon=True)
    thread.start()

    return thread


def stats(gate: StageGate):
    stage = gate._counters.stage(gate.name)

    return stage


def test_turns_follow_ordinals_per_domain() -> None:
    gate = make_gate()
    order: List[int] = []

    def take(ordinal: int) -> None:
        ticket = Ticket("a", ordinal)
        gate.wait_turn(ticket)
        with gate.call(ticket):
            order.append(ordinal)
        gate.done(ticket)

    second = started(lambda: take(1))
    wait_for(lambda: stats(gate).waited_turn == 1)
    assert order == []

    take(0)
    second.join(TIMEOUT)

    assert order == [0, 1]


def test_out_of_order_retirement_advances_only_a_contiguous_prefix() -> None:
    gate = make_gate()

    gate.done(Ticket("a", 1))
    gate.done(Ticket("a", 2))
    assert gate._next == {"a": 0}
    assert gate._finished == {"a": {1, 2}}

    gate.done(Ticket("a", 0))

    # Consumed metadata is removed; only the next ordinal is kept.
    assert gate._next == {"a": 3}
    assert gate._finished == {}
    with pytest.raises(RuntimeError, match="already retired"):
        gate.done(Ticket("a", 1))
    with pytest.raises(RuntimeError, match="already retired"):
        gate.wait_turn(Ticket("a", 2))


def test_call_without_turn_is_rejected() -> None:
    gate = make_gate()

    with pytest.raises(RuntimeError, match="without its turn"):
        with gate.call(Ticket("a", 1)):
            pass


def test_domains_are_independent_but_share_one_call_slot() -> None:
    gate = make_gate()
    a0, b0 = Ticket("a", 0), Ticket("b", 0)
    in_call = threading.Event()
    release = threading.Event()

    def hold_a() -> None:
        with gate.call(a0):
            in_call.set()
            release.wait(TIMEOUT)

    holder = started(hold_a)
    assert in_call.wait(TIMEOUT)

    # b0 has its turn at once (another domain) but waits for the call slot.
    gate.wait_turn(b0)
    entered = threading.Event()
    waiter = started(lambda: _enter(gate, b0, entered))
    wait_for(lambda: stats(gate).waited_call == 1)
    assert not entered.is_set()

    release.set()
    holder.join(TIMEOUT)
    waiter.join(TIMEOUT)
    assert entered.is_set()
    assert gate._counters.peak("overlapping_pulses") == 1


def _enter(gate: StageGate, ticket: Ticket, entered: threading.Event) -> None:
    with gate.call(ticket):
        entered.set()


def test_abort_wakes_every_waiter() -> None:
    aborted = threading.Event()
    gate = StageGate("$steps.s#call", aborted=aborted, counters=PipelineCounters())
    raised: List[BaseException] = []

    def wait_late_turn() -> None:
        try:
            gate.wait_turn(Ticket("a", 3))
        except RunAborted as error:
            raised.append(error)

    waiter = started(wait_late_turn)
    wait_for(lambda: stats(gate).waited_turn == 1)

    aborted.set()
    gate.wake()
    waiter.join(TIMEOUT)

    assert len(raised) == 1
    # RunAborted is not an Exception: step error handlers never wrap it.
    assert not isinstance(raised[0], Exception)


def stages_of(gates: dict, ordinal: int, *, calls: int) -> StepStages:
    stages = StepStages(gates, ticket=Ticket("a", ordinal), calls=calls)

    return stages


def test_unit_retires_right_after_its_nth_call_not_at_step_end() -> None:
    gates = {"first": make_gate("s#first"), "second": make_gate("s#second")}
    pulse0 = stages_of(gates, 0, calls=2)

    with pulse0.call("first"):
        pass
    with pulse0.call("second"):
        pass
    # One call of two: pulse 0 keeps its turn at "first".
    assert gates["first"]._next == {}

    with pulse0.call("first"):
        pass
    # The second (last) call retired "first" while "second" is still in use.
    assert gates["first"]._next == {"a": 1}
    assert gates["second"]._next == {}

    with pulse0.call("second"):
        pass
    pulse0.finish()
    assert gates["second"]._next == {"a": 1}


def test_zero_calls_retire_at_once_and_never_wait() -> None:
    gates = {"call": make_gate()}

    # Ordinal 1 with zero calls: no wait, even though ordinal 0 is not done.
    stages_of(gates, 1, calls=0).finish()
    assert gates["call"]._finished == {"a": {1}}

    pulse0 = stages_of(gates, 0, calls=1)
    with pulse0.call("call"):
        pass

    assert gates["call"]._next == {"a": 2}


def test_gate_progress_with_skipped_multi_call_and_waiting_pulses() -> None:
    """P17: n = 0, n = 3 and a waiter, plus a static stage fed by two domains."""
    gate = make_gate()
    gates = {"call": gate}
    log: List[tuple] = []
    pulse1_inside = threading.Event()
    release1 = threading.Event()

    stages_of(gates, 0, calls=0).finish()

    def pulse1() -> None:
        stages = stages_of(gates, 1, calls=3)
        for number in range(3):
            with stages.call("call"):
                log.append(("a", 1, number))
                if number == 0:
                    pulse1_inside.set()
                    release1.wait(TIMEOUT)
        stages.finish()

    def pulse2() -> None:
        stages = stages_of(gates, 2, calls=1)
        with stages.call("call"):
            log.append(("a", 2, 0))

    first = started(pulse1)
    assert pulse1_inside.wait(TIMEOUT)
    second = started(pulse2)
    wait_for(lambda: stats(gate).waited_turn == 1)

    # Another domain is not blocked by domain "a"'s order, only by the slot.
    other = StepStages(gates, ticket=Ticket("b", 0), calls=1)
    entered = threading.Event()

    def other_domain() -> None:
        with other.call("call"):
            log.append(("b", 0, 0))
            entered.set()

    third = started(other_domain)
    wait_for(lambda: stats(gate).waited_call >= 1)
    release1.set()
    for thread in (first, second, third):
        thread.join(TIMEOUT)

    assert [item for item in log if item[0] == "a"] == [
        ("a", 1, 0),
        ("a", 1, 1),
        ("a", 1, 2),
        ("a", 2, 0),
    ]
    assert ("b", 0, 0) in log
    assert gate._next == {"a": 3, "b": 1}


def test_failed_call_is_never_retired_and_frees_the_call_slot() -> None:
    gates = {"call": make_gate()}
    stages = stages_of(gates, 0, calls=1)

    with pytest.raises(ValueError):
        with stages.call("call"):
            raise ValueError("block failed")

    assert gates["call"]._next == {}
    assert not gates["call"]._busy


def test_finish_refuses_a_partially_called_unit() -> None:
    gates = {"call": make_gate()}
    stages = stages_of(gates, 0, calls=2)

    with stages.call("call"):
        pass

    with pytest.raises(RuntimeError, match="fewer than 2"):
        stages.finish()
    with pytest.raises(RuntimeError, match="unknown stage unit"):
        with stages.call("missing"):
            pass


def test_calls_beyond_the_declared_number_are_rejected() -> None:
    stages = stages_of({"call": make_gate()}, 0, calls=1)
    with stages.call("call"):
        pass

    with pytest.raises(RuntimeError, match="declared 1"):
        with stages.call("call"):
            pass


def test_serial_coordination_gates_nothing() -> None:
    gate = SERIAL.gate("$groups.g#deliver")
    gate.wait_turn(Ticket("a", 7))
    with gate.call(Ticket("a", 7)), gate.exclusive(), SERIAL.callbacks():
        pass
    gate.done(Ticket("a", 7))
    SERIAL.checkpoint()
    SERIAL.abort()

    assert not SERIAL.aborted


def test_counters_are_thread_safe_and_reported_as_plain_data() -> None:
    counters = PipelineCounters()

    def churn() -> None:
        for _ in range(1000):
            with counters.track("live_states"):
                counters.add("queued", 1)
                counters.add("queued", -1)

    threads = [started(churn) for _ in range(4)]
    for thread in threads:
        thread.join(TIMEOUT)
    counters.count("end_tasks")
    counters.record_result_age("A", 30)
    counters.record_result_age("A", 10)

    snapshot = counters.snapshot()
    assert snapshot["current"]["live_states"] == 0
    assert snapshot["current"]["queued"] == 0
    assert 1 <= snapshot["peak"]["live_states"] <= 4
    assert snapshot["totals"] == {"end_tasks": 1}
    assert snapshot["result_age_ns"] == {"A": {"last": 10, "max": 30}}
    with pytest.raises(KeyError):
        counters.add("misspelled")
