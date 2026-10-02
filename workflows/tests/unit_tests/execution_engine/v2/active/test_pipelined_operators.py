"""Operators in pipelined active runs: push order, EOF tails and domain ends.

A pulse whose route finishes first (reverse completion) still pushes to its
operators after the earlier pulse of its domain, so operator results equal
the serial reference. A domain ends only when it is sealed (its reader is
done, or its operator's ``finish`` returned) and none of its pulses is still
outstanding, so an operator never ends downstream while one of its pulses
runs. Probes are ordered by events, never by sleeping (see ``test_pipelined``).
"""

import dataclasses
import threading
from typing import Any, List

import pytest
from pydantic import Field
from roboflow_workflows.execution_engine.v2.active.pulses import DomainProgress
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.errors import ActiveRunError
from roboflow_workflows.execution_engine.v2.operators import (
    Operator,
    OperatorParams,
    OperatorPort,
    OperatorPulse,
)
from roboflow_workflows.execution_engine.v2.operators.alignment import Align
from roboflow_workflows.execution_engine.v2.operators.window import Window
from roboflow_workflows.execution_engine.v2.pipelining.options import PipelineOptions

from tests.unit_tests.execution_engine.v2.active.test_pipelined import (
    CATALOGUE,
    Collector,
    Held,
    Session,
    StepEvents,
    outcome,
)
from tests.unit_tests.execution_engine.v2.execution.blocks import ContinueIf
from tests.unit_tests.execution_engine.v2.test_active_runtime import (
    WAIT,
    active,
    emit,
    group,
    source,
    step,
)

LIFECYCLE: List[tuple] = []
"""Lifecycle calls of every ``Tail`` and ``Flush`` instance, in order."""

FLUSH_FINISHED = threading.Event()
"""Set by ``Flush.finish``."""


@pytest.fixture(autouse=True)
def lifecycle():
    LIFECYCLE.clear()
    FLUSH_FINISHED.clear()
    yield LIFECYCLE
    LIFECYCLE.clear()


class Tail(Operator):
    """Records every push (with its upstream pulses), end_input and finish; emits nothing."""

    type = "test/pipelined_tail@v1"
    input_roles = ("input",)

    class Params(OperatorParams):
        fail_in: str = Field(default="", description="Lifecycle method that raises.")

    @classmethod
    def plan_ports(cls, name, params, inputs):
        return {"seen": OperatorPort(*inputs[0].kinds)}

    def push(self, arrivals):
        LIFECYCLE.append(
            (self.name, "push", [(a.pulse.source, a.pulse.sequence) for a in arrivals])
        )
        if self.params.fail_in == "push":
            raise RuntimeError("push failed")
        return []

    def end_input(self, name):
        LIFECYCLE.append((self.name, "end_input", name))
        return []

    def finish(self, reason):
        LIFECYCLE.append((self.name, "finish", reason))
        return []


class Flush(Operator):
    """Keeps the last present value per input; ``end_input`` emits it."""

    type = "test/pipelined_flush@v1"
    input_roles = ("input",)

    class Params(OperatorParams):
        pass

    @classmethod
    def plan_ports(cls, name, params, inputs):
        return {"out": OperatorPort(*inputs[0].kinds)}

    def __init__(self, **arguments: Any) -> None:
        super().__init__(**arguments)
        self.kept = {}

    def push(self, arrivals):
        for arrival in arrivals:
            if not arrival.entry.is_effectively_filtered():
                self.kept[arrival.input] = arrival
        return []

    def end_input(self, name):
        LIFECYCLE.append((self.name, "end_input", name))
        kept = self.kept.pop(name, None)
        if kept is None:
            return []
        return [OperatorPulse(ports={"out": kept.entry}, causes=(kept.pulse,))]

    def finish(self, reason):
        LIFECYCLE.append((self.name, "finish", reason))
        FLUSH_FINISHED.set()
        return []


OPERATORS = Catalogue.merge(
    CATALOGUE, Catalogue([], operators=[Align, Window, Tail, Flush])
)


def operator(kind: type, name: str, **fields: Any) -> dict:
    return {"type": kind.type, "name": name, **fields}


def calls_of(name: str) -> List[tuple]:
    return [call[1:] for call in LIFECYCLE if call[0] == name]


# Reverse completion ------------------------------------------------------------------


def reverse_completion_chain() -> dict:
    """Pulse a#1 skips the held step, so its route finishes before a#0's."""
    definition = active(
        [source("a"), source("b")],
        [
            step(
                ContinueIf,
                "gate",
                value="$sources.a.value",
                threshold=0.0,
                next_steps=["$steps.slow"],
            ),
            step(Held, "slow", value="$sources.a.value"),
        ],
        [
            group(
                "P", "$operators.pair.a", a="$operators.pair.a", b="$operators.pair.b"
            ),
            group("C", "$operators.clip.a", a="$operators.clip.a"),
            group("S", "$operators.slow_rows.x", x="$operators.slow_rows.x"),
        ],
    )
    definition["operators"] = [
        operator(
            Align,
            "pair",
            inputs={"a": "$sources.a.value", "b": "$sources.b.value"},
            clock="media",
            missing="partial",
        ),
        operator(
            Window,
            "clip",
            size=2,
            step=1,
            partial="emit",
            collect={"a": "$operators.pair.a", "b": "$operators.pair.b"},
        ),
        operator(
            Window,
            "slow_rows",
            size=2,
            partial="emit",
            collect={"x": "$steps.slow.value"},
        ),
        operator(Tail, "tail", inputs={"value": "$sources.a.value"}),
    ]

    return definition


def chain_feeds() -> dict:
    feeds = {
        "a": [emit(value=v, pts=100 * i) for i, v in enumerate([1.0, -1.0, 2.0, 3.0])],
        "b": [emit(value=10.0 * (i + 1), pts=100 * i) for i in range(3)],
    }

    return feeds


def test_reverse_completion_through_align_and_windows_equals_the_serial_reference() -> (
    None
):
    serial = Session(reverse_completion_chain(), chain_feeds(), catalogue=OPERATORS)
    serial_collected = Collector()
    serial_run = serial.start(serial_collected.handlers("P", "C", "S"), pipeline=None)
    assert serial_run.wait(WAIT)
    serial_calls = list(LIFECYCLE)
    LIFECYCLE.clear()

    events = StepEvents()
    session = Session(
        reverse_completion_chain(), chain_feeds(), catalogue=OPERATORS, observer=events
    )
    release = session.probe.hold("slow", 1.0)
    collected = Collector()

    run = session.start(
        collected.handlers("P", "C", "S"),
        admission_bound=4,
        pipeline=PipelineOptions(max_in_flight=4),
    )

    assert session.probe.entered("slow", 1.0).wait(WAIT)
    assert events.finished(f"{run.run_id}:a:1", ("slow",)).wait(WAIT)
    # Both readers end while a#0 is held: no operator may learn of an end yet.
    session.log.wait_for(lambda: ("close", "a") in session.log)
    session.log.wait_for(lambda: ("close", "b") in session.log)
    assert calls_of("tail") == []
    assert not run.operator_counters["pair"].finished
    release.set()
    assert run.wait(WAIT)
    assert outcome(collected) == outcome(serial_collected)
    assert LIFECYCLE == serial_calls
    assert calls_of("tail")[:4] == [
        ("push", [("a", 0)]),
        ("push", [("a", 1)]),
        ("push", [("a", 2)]),
        ("push", [("a", 3)]),
    ]
    # Retention peaks depend on how the independent readers interleave, in
    # either mode; every other operator counter must match.
    assert {
        name: dataclasses.replace(counters, peak_retained=0)
        for name, counters in run.operator_counters.items()
    } == {
        name: dataclasses.replace(counters, peak_retained=0)
        for name, counters in serial_run.operator_counters.items()
    }
    assert {
        name: (c.read, c.admitted, c.processed, c.delivered, c.cancelled)
        for name, c in run.counters.items()
    } == {
        name: (c.read, c.admitted, c.processed, c.delivered, c.cancelled)
        for name, c in serial_run.counters.items()
    }
    counters = run.pipeline_counters
    assert counters.snapshot()["totals"]["end_tasks"] == 6  # 2 sources, 4 operators
    assert counters.current("pending_operator_pulses") == 0
    assert counters.current("live_states") == 0
    # Inline operator pulses keep their ancestors alive: workers x (1 + chain).
    assert counters.peak("live_states") <= 4 * (1 + 2)


# Domain ends ---------------------------------------------------------------------


def test_an_operator_does_not_end_downstream_while_its_end_input_pulse_runs() -> None:
    definition = active(
        [source("a"), source("b"), source("c")],
        [step(Held, "slow", value="$operators.flush.out")],
        [
            group("F", "$operators.flush.out", out="$steps.slow.value"),
            group("C", "$sources.c.value", c="$sources.c.value"),
        ],
    )
    definition["operators"] = [
        operator(
            Flush, "flush", inputs={"a": "$sources.a.value", "b": "$sources.b.value"}
        ),
        operator(Tail, "downstream", inputs={"value": "$steps.slow.value"}),
    ]
    session = Session(definition, {}, catalogue=OPERATORS)
    held = session.probe.entered("slow", 1.0)
    session.feeds.update(
        {
            # a ends at once: end_input(a) emits flush#0, which is held in slow.
            "a": [emit(value=1.0)],
            # b ends only then: end_input(b) emits nothing, so b's end finishes
            # flush while flush#0 is still running.
            "b": [held],
            # c's pulse is dispatched only after any end queued before it.
            "c": [FLUSH_FINISHED, emit(value=7.0)],
        }
    )
    release = session.probe.hold("slow", 1.0)
    delivered_c = threading.Event()
    collected = Collector()

    def on_c(result) -> None:
        collected.handler("C")(result)
        delivered_c.set()

    run = session.start(
        {"F": collected.handler("F"), "C": on_c},
        pipeline=PipelineOptions(max_in_flight=2),
    )

    assert held.wait(WAIT)
    assert FLUSH_FINISHED.wait(WAIT)
    assert delivered_c.wait(WAIT)
    # flush is sealed (finish returned) but flush#0 is outstanding: not ended.
    assert calls_of("flush") == [
        ("end_input", "a"),
        ("end_input", "b"),
        ("finish", "eof"),
    ]
    assert calls_of("downstream") == []
    release.set()
    assert run.wait(WAIT)
    assert collected.rows("F") == [{"out": 1.0}]
    assert calls_of("downstream") == [
        ("push", [("flush", 0)]),
        ("end_input", "value"),
        ("finish", "eof"),
    ]


def test_a_domain_ends_once_when_sealed_and_drained_in_either_order() -> None:
    progress = DomainProgress(["s", "o"])

    progress.add("s", 2)
    assert progress.complete("s") is None
    assert progress.seal("s", "eof") is None  # one pulse still outstanding
    assert progress.complete("s") == "eof"
    assert progress.seal("s", "stop") is None  # already ended

    assert progress.seal("o", "stop") == "stop"  # nothing outstanding


@pytest.mark.parametrize("pipeline", [None, PipelineOptions(max_in_flight=3)])
def test_a_failing_push_is_attributed_and_closes_every_operator(pipeline) -> None:
    definition = active(
        [source("a")],
        [step(Held, "slow", value="$sources.a.value")],
        [group("A", "$sources.a.value", value="$steps.slow.value")],
    )
    definition["operators"] = [
        operator(Tail, "tail", inputs={"value": "$sources.a.value"}, fail_in="push"),
        operator(Window, "clip", size=2, collect={"x": "$sources.a.value"}),
    ]
    session = Session(
        definition,
        {"a": [emit(value=float(v)) for v in range(3)]},
        catalogue=OPERATORS,
    )

    run = session.start(Collector().handlers("A"), pipeline=pipeline)

    with pytest.raises(ActiveRunError) as caught:
        run.wait(WAIT)
    assert (caught.value.stage, caught.value.operator) == ("operator", "tail")
    assert (caught.value.source, caught.value.pulse) == ("a", 0)
    assert all(counters.closed for counters in run.operator_counters.values())
    assert not any(counters.finished for counters in run.operator_counters.values())
    counters = run.counters["a"]
    assert counters.admitted == counters.processed + counters.cancelled
