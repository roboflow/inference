"""Pending operator pulses stay bounded: descendants first, then a hard limit.

The dispatcher runs the head of the deepest operator domain first, so a chain
of operators delivers one pulse's descendants before it expands the next
sibling. ``PipelineOptions.max_pending_operator_pulses`` bounds what still
waits; a hand-off that would exceed it fails the run, attributed to the
operator whose call returned the pulses::

    a -> first(Burst f) -> second(Burst f) -> sink
    pending at P1: (f - 1) of first + f of second, never f * f

Probes are ordered by events the test controls; ``WAIT`` only bounds each
wait, so a broken scheduler fails instead of hanging.
"""

import pytest
from roboflow_workflows.v2.engine.errors import ActiveRunError, ContractError
from roboflow_workflows.v2.engine.pipelining.options import (
    DEFAULT_MAX_PENDING_OPERATOR_PULSES,
    PipelineOptions,
)

from tests.v2.integration.engine.runtime.active.test_pipelined import Held, Session
from tests.v2.integration.engine.runtime.test_active_runtime import (
    WAIT,
    active,
    emit,
    group,
    source,
    step,
)
from tests.v2.integration.engine.runtime.test_threading_emissions import (
    OPERATORS,
    Burst,
    operator,
)
from tests.v2.integration.engine.runtime.test_threading_lifecycle import (
    eventually,
    settled,
)


def chain(count: int, *, tail: int = 0) -> dict:
    """a -> first(count) -> second(count) -> sink; group O reads the sink."""
    definition = active(
        [source("a")],
        [step(Held, "sink", value="$operators.second.out")],
        [group("O", "$operators.second.out", value="$steps.sink.value")],
    )
    definition["operators"] = [
        operator(Burst, "first", inputs={"v": "$sources.a.value"}, count=count),
        operator(
            Burst,
            "second",
            inputs={"v": "$operators.first.out"},
            count=count,
            tail=tail,
        ),
    ]

    return definition


def single(count: int, *, tail: int = 0) -> dict:
    """a -> first(count, tail) -> sink; group O reads the sink."""
    definition = active(
        [source("a")],
        [step(Held, "sink", value="$operators.first.out")],
        [group("O", "$operators.first.out", value="$steps.sink.value")],
    )
    definition["operators"] = [
        operator(
            Burst, "first", inputs={"v": "$sources.a.value"}, count=count, tail=tail
        ),
    ]

    return definition


def overflow_of(run) -> ActiveRunError:
    with pytest.raises(ActiveRunError) as caught:
        run.wait(WAIT)

    return caught.value


# Descendants first ------------------------------------------------------------


@pytest.mark.parametrize("parallelism", [1, 2])
def test_a_chain_delivers_descendants_before_expanding_the_next_sibling(
    parallelism,
) -> None:
    session = Session(chain(32), {"a": [emit(value=1.0)]}, catalogue=OPERATORS)
    release = session.probe.hold("sink", 100.0)  # first#0 -> second#0
    options = PipelineOptions(max_in_flight=1, max_workflow_parallelism=parallelism)

    run = session.start({}, pipeline=options)
    try:
        assert session.probe.entered("sink", 100.0).wait(WAIT)
        held = run.pipeline_counters.current("pending_operator_pulses")
        release.set()
        assert run.wait(WAIT)
    finally:
        release.set()
        if not run.done:
            run.cancel()

    # Breadth first would hold 31 * 32 + 31 = 1023 here and peak at 1024.
    assert held == 31 + 31
    assert run.pipeline_counters.peak("pending_operator_pulses") == 31 + 32
    assert run.operator_counters["second"].processed == 32 * 32
    settled(run, session)


# The pending bound ------------------------------------------------------------


def test_the_bound_is_a_keyword_only_option_with_a_documented_default() -> None:
    options = PipelineOptions(2, "latest", {"cam": "block"})

    assert options.max_pending_operator_pulses == DEFAULT_MAX_PENDING_OPERATOR_PULSES
    assert DEFAULT_MAX_PENDING_OPERATOR_PULSES == 4096
    assert options.max_workflow_parallelism == 1
    assert PipelineOptions(max_pending_operator_pulses=7).max_pending_operator_pulses
    with pytest.raises(TypeError):
        PipelineOptions(2, "block", {}, 3)
    for invalid in (0, True, 2.0):
        with pytest.raises(ContractError, match="max_pending_operator_pulses"):
            PipelineOptions(max_pending_operator_pulses=invalid)


@pytest.mark.parametrize(
    "count, options",
    [
        (17, PipelineOptions(max_in_flight=1, max_pending_operator_pulses=16)),
        (
            17,
            PipelineOptions(
                max_in_flight=2,
                max_workflow_parallelism=2,
                max_pending_operator_pulses=16,
            ),
        ),
        (DEFAULT_MAX_PENDING_OPERATOR_PULSES + 1, PipelineOptions(max_in_flight=1)),
    ],
    ids=["P1-limit16", "P2W2-limit16", "default"],
)
def test_one_oversized_call_fails_the_run_and_runs_none_of_its_pulses(
    count, options
) -> None:
    limit = options.max_pending_operator_pulses
    session = Session(single(count), {"a": [emit(value=1.0)]}, catalogue=OPERATORS)

    run = session.start({}, pipeline=options)
    error = overflow_of(run)

    assert (error.stage, error.operator) == ("operator", "first")
    assert f"returned {count} pulse(s) while 0 operator pulse(s) wait to start" in str(
        error
    )
    assert f"max_pending_operator_pulses={limit}" in str(error)
    first = run.operator_counters["first"]
    assert (first.emitted, first.processed, first.cancelled) == (count, 0, count)
    assert session.probe.order.get("sink", []) == []
    assert run.counters["a"].cancelled == 1  # its push failed
    settled(run, session)


@pytest.mark.parametrize("parallelism", [1, 2])
def test_cumulative_overflow_while_a_descendant_is_held_fails_and_settles(
    parallelism,
) -> None:
    # Park first#0 after its gate turn, before it emits descendants, until
    # first#1 holds the other team at gate. Then first#0's descendant holds
    # sink and first#1's push finds 31 pending: 31 + 32 > 40.
    definition = active(
        [source("a")],
        [
            step(Held, "gate", value="$operators.first.out"),
            step(Held, "before_second", value="$steps.gate.value"),
            step(Held, "sink", value="$operators.second.out"),
        ],
        [group("O", "$operators.second.out", value="$steps.sink.value")],
    )
    definition["operators"] = [
        operator(Burst, "first", inputs={"v": "$sources.a.value"}, count=2),
        operator(Burst, "second", inputs={"v": "$steps.before_second.value"}, count=32),
    ]
    session = Session(definition, {"a": [emit(value=1.0)]}, catalogue=OPERATORS)
    first = session.probe.hold("before_second", 10.0)
    gate = session.probe.hold("gate", 11.0)  # first#1
    sink = session.probe.hold("sink", 100.0)  # first#0 -> second#0
    options = PipelineOptions(
        max_in_flight=2,
        max_workflow_parallelism=parallelism,
        max_pending_operator_pulses=40,
    )

    run = session.start({}, pipeline=options)
    try:
        assert session.probe.entered("before_second", 10.0).wait(WAIT)
        assert session.probe.entered("gate", 11.0).wait(WAIT)
        first.set()
        assert session.probe.entered("sink", 100.0).wait(WAIT)
        assert run.pipeline_counters.current("pending_operator_pulses") == 31
        gate.set()
        eventually(lambda: run.operator_counters["second"].cancelled >= 32)
        sink.set()
        error = overflow_of(run)
    finally:
        first.set()
        gate.set()
        sink.set()
        if not run.done:
            run.cancel()

    assert (error.stage, error.operator) == ("operator", "second")
    assert "returned 32 pulse(s) while 31 operator pulse(s) wait to start" in str(error)
    assert "max_pending_operator_pulses=40" in str(error)
    second = run.operator_counters["second"]
    assert second.emitted == 64
    assert second.processed <= 1  # only the held descendant ever started
    settled(run, session)


def test_an_oversized_finish_fails_the_run_attributed_to_its_operator() -> None:
    session = Session(single(1, tail=8), {"a": [emit(value=1.0)]}, catalogue=OPERATORS)
    options = PipelineOptions(max_in_flight=1, max_pending_operator_pulses=4)

    run = session.start({}, pipeline=options)
    error = overflow_of(run)

    assert (error.stage, error.operator) == ("operator", "first")
    assert "returned 8 pulse(s) while 0 operator pulse(s)" in str(error)
    first = run.operator_counters["first"]
    assert (first.emitted, first.processed, first.cancelled) == (9, 1, 8)
    assert not first.finished
    settled(run, session)
