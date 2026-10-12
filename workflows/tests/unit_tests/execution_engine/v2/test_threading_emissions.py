"""Operator emissions acquire pulse teams of their own, under the same P budget.

In a pipelined run an operator's returned pulses are handed to the
dispatcher while the operator's stage is held. The work that fed the
operator then finishes and frees its team; every emitted pulse runs later as
its own work. Serial runs keep running them inline and stay the reference::

    team: a#k  route, push -> burst#n.. handed off      team released
    dispatcher: lowest version; ties: emitted, ends, sources;
                only the head of each operator's emitted pulses

Probes are ordered by events the test controls; ``WAIT`` only bounds each
wait, so a broken scheduler fails instead of hanging.
"""

import threading
import time
from typing import Any, Dict, List, Tuple

import pytest
from pydantic import Field
from roboflow_workflows.execution_engine.v2.active.pipeline import PipelinedDriver
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.errors import ContractError
from roboflow_workflows.execution_engine.v2.execution.entries import scalar_entry
from roboflow_workflows.execution_engine.v2.operators import (
    Operator,
    OperatorParams,
    OperatorPort,
    OperatorPulse,
)
from roboflow_workflows.execution_engine.v2.operators.window import Window
from roboflow_workflows.execution_engine.v2.pipelining.options import PipelineOptions
from roboflow_workflows.execution_engine.v2.plan import ExecutionObserver

from tests.unit_tests.execution_engine.v2.active.test_pipelined import (
    CATALOGUE,
    Collector,
    Held,
    Session,
    outcome,
)
from tests.unit_tests.execution_engine.v2.test_active_runtime import (
    WAIT,
    active,
    emit,
    group,
    source,
    step,
)


class Burst(Operator):
    """Emits ``count`` pulses per present arrival and ``tail`` at ``finish``.

    Pulse ``i`` of value ``v`` carries ``10 * v + i``, so every emitted value
    is distinct and names its feeding value.
    """

    type = "test/threading_burst@v1"
    input_roles = ("input",)

    class Params(OperatorParams):
        count: int = Field(default=1, description="Pulses emitted per arrival.")
        tail: int = Field(default=0, description="Pulses emitted at finish.")

    @classmethod
    def plan_ports(cls, name, params, inputs):
        return {"out": OperatorPort(*inputs[0].kinds)}

    def __init__(self, **arguments: Any) -> None:
        super().__init__(**arguments)
        self.last = None

    def push(self, arrivals):
        emitted = []
        for arrival in arrivals:
            if arrival.entry.is_effectively_filtered():
                continue
            self.last = arrival
            emitted.extend(
                self._pulse(arrival, offset) for offset in range(self.params.count)
            )
        return emitted

    def end_input(self, name):
        return []

    def finish(self, reason):
        if self.last is None:
            return []
        return [
            self._pulse(self.last, 5 + offset) for offset in range(self.params.tail)
        ]

    @staticmethod
    def _pulse(arrival, offset: int) -> OperatorPulse:
        value = 10 * arrival.entry.values[()] + offset
        entry = scalar_entry(value, metadata=arrival.entry.metadata)
        return OperatorPulse(ports={"out": entry}, causes=(arrival.pulse,))


OPERATORS = Catalogue.merge(CATALOGUE, Catalogue([], operators=[Burst, Window]))

COMBINATIONS = [
    PipelineOptions(max_in_flight=1),
    PipelineOptions(max_in_flight=1, max_workflow_parallelism=2),
    PipelineOptions(max_in_flight=2),
    PipelineOptions(max_in_flight=3, max_workflow_parallelism=2),
]


def label(options: PipelineOptions) -> str:
    return f"P{options.max_in_flight}W{options.max_workflow_parallelism}"


def operator(kind: type, name: str, **fields: Any) -> dict:
    return {"type": kind.type, "name": name, **fields}


def burst_chain(count: int) -> dict:
    """a -> burst(count, tail 1) -> slow -> again(2 per arrival); groups on both."""
    definition = active(
        [source("a")],
        [
            step(Held, "fresh", value="$sources.a.value"),
            step(Held, "slow", value="$operators.burst.out"),
        ],
        [
            group("A", "$sources.a.value", fresh="$steps.fresh.value"),
            group("B", "$operators.burst.out", slow="$steps.slow.value"),
            group("C", "$operators.again.out", out="$operators.again.out"),
        ],
    )
    definition["operators"] = [
        operator(
            Burst, "burst", inputs={"value": "$sources.a.value"}, count=count, tail=1
        ),
        operator(Burst, "again", inputs={"value": "$steps.slow.value"}, count=2),
    ]

    return definition


def values_feed(*values: float) -> Dict[str, list]:
    return {"a": [emit(value=value) for value in values]}


def operator_totals(run: Any) -> Dict[str, Tuple[int, int, int, int, bool]]:
    totals = {
        name: (c.arrivals, c.emitted, c.processed, c.cancelled, c.finished)
        for name, c in run.operator_counters.items()
    }

    return totals


class PulseOrder(ExecutionObserver):
    """Records when every pulse started and finished, in callback order."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.events: List[Tuple[str, str, int]] = []

    def on_pulse_started(self, *, run_id, source, pulse):
        with self._lock:
            self.events.append(("started", source, pulse.sequence))

    def on_pulse_finished(self, *, run_id, source, pulse, error):
        with self._lock:
            self.events.append(("finished", source, pulse.sequence))

    def position(self, event: Tuple[str, str, int]) -> int:
        return self.events.index(event)


def dispatched_emissions(monkeypatch) -> List[Tuple[str, int, str]]:
    """Record each emitted pulse taken: operator, ordinal and taking thread.

    The dispatcher takes work for a free worker; a finishing worker takes
    its next unit itself. Both use
    ``_take_work`` with the driver's condition held, so the emitted queues
    cannot change meanwhile: a head that changed was taken.
    """
    taken: List[Tuple[str, int, str]] = []
    take = PipelinedDriver._take_work

    def recording(driver: PipelinedDriver):
        heads = {name: queue[0] for name, queue in driver._emitted.items() if queue}
        work = take(driver)
        for name, head in heads.items():
            queue = driver._emitted[name]
            if not queue or queue[0] is not head:
                taken.append((name, head.key.sequence, threading.current_thread().name))
        return work

    monkeypatch.setattr(PipelinedDriver, "_take_work", recording)

    return taken


# Serial reference -------------------------------------------------------------------


@pytest.mark.parametrize("options", COMBINATIONS, ids=label)
@pytest.mark.parametrize("count", [0, 1, 3], ids=lambda count: f"emits{count}")
def test_emitted_pulses_and_their_chains_equal_the_serial_reference(
    options, count
) -> None:
    serial = Session(
        burst_chain(count), values_feed(1.0, 2.0, 3.0), catalogue=OPERATORS
    )
    serial_collected = Collector()
    serial_run = serial.start(serial_collected.handlers("A", "B", "C"), pipeline=None)
    assert serial_run.wait(WAIT)

    session = Session(
        burst_chain(count), values_feed(1.0, 2.0, 3.0), catalogue=OPERATORS
    )
    collected = Collector()
    run = session.start(collected.handlers("A", "B", "C"), pipeline=options)
    assert run.wait(WAIT)

    assert outcome(collected) == outcome(serial_collected)
    assert operator_totals(run) == operator_totals(serial_run)
    burst = run.operator_counters["burst"]
    # count per source pulse, plus the finish tail of one.
    assert burst.emitted == 3 * count + 1 == burst.processed
    gauges = run.pipeline_counters
    assert gauges.current("pending_operator_pulses") == 0
    assert gauges.current("live_states") == 0
    # Pulses never nest: one live run state per busy team at most.
    assert gauges.peak("live_states") <= options.max_in_flight


# The parent frees its team ----------------------------------------------------------


def test_at_p1_the_feeding_pulse_finishes_before_its_emissions_start() -> None:
    events = PulseOrder()
    session = Session(
        burst_chain(2), values_feed(1.0), catalogue=OPERATORS, observer=events
    )

    run = session.start({}, pipeline=PipelineOptions(max_in_flight=1))

    assert run.wait(WAIT)
    finished_parent = events.position(("finished", "a", 0))
    assert finished_parent < events.position(("started", "burst", 0))
    assert finished_parent < events.position(("started", "burst", 1))
    # A chained emission runs after the emitted pulse that fed it finished.
    assert events.position(("finished", "burst", 0)) < events.position(
        ("started", "again", 0)
    )


def test_serial_runs_keep_emissions_inline_inside_the_feeding_pulse() -> None:
    events = PulseOrder()
    session = Session(
        burst_chain(2), values_feed(1.0), catalogue=OPERATORS, observer=events
    )

    run = session.start({}, pipeline=None)

    assert run.wait(WAIT)
    assert events.position(("finished", "burst", 1)) < events.position(
        ("finished", "a", 0)
    )


def test_a_later_emission_runs_on_another_team_while_an_earlier_one_is_held() -> None:
    # burst#0 (value 10) is held in slow; burst#1 (11) runs step first on
    # the second team meanwhile. Inline, it would wait for burst#0 to finish.
    definition = active(
        [source("a")],
        [
            step(Held, "first", value="$operators.burst.out"),
            step(Held, "slow", value="$steps.first.value"),
        ],
        [group("B", "$operators.burst.out", slow="$steps.slow.value")],
    )
    definition["operators"] = [
        operator(Burst, "burst", inputs={"value": "$sources.a.value"}, count=2)
    ]
    session = Session(definition, values_feed(1.0), catalogue=OPERATORS)
    release = session.probe.hold("slow", 10.0)
    collected = Collector()

    run = session.start(
        collected.handlers("B"), pipeline=PipelineOptions(max_in_flight=2)
    )

    assert session.probe.entered("slow", 10.0).wait(WAIT)
    assert session.probe.entered("first", 11.0).wait(WAIT)
    assert run.pipeline_counters.current("executing") == 2
    release.set()
    assert run.wait(WAIT)
    assert collected.rows("B") == [{"slow": 10.0}, {"slow": 11.0}]


# Dispatch order and back-pressure ---------------------------------------------------


@pytest.mark.parametrize("options", COMBINATIONS, ids=label)
def test_dispatch_takes_every_operator_pulse_in_ordinal_order(
    monkeypatch, options
) -> None:
    taken = dispatched_emissions(monkeypatch)
    session = Session(
        burst_chain(3), values_feed(1.0, 2.0, 3.0, 4.0), catalogue=OPERATORS
    )

    run = session.start({}, pipeline=options)

    assert run.wait(WAIT)
    for name, counters in run.operator_counters.items():
        ordinals = [sequence for domain, sequence, _ in taken if domain == name]
        assert ordinals == list(range(counters.emitted)), name


@pytest.mark.parametrize("parallelism", [1, 2])
def test_at_p1_the_finishing_worker_takes_every_emission_itself(
    monkeypatch, parallelism
) -> None:
    # One worker: it is busy whenever an emission is handed off, so it goes
    # on with the next unit itself and the dispatcher never takes one.
    taken = dispatched_emissions(monkeypatch)
    options = PipelineOptions(max_in_flight=1, max_workflow_parallelism=parallelism)
    session = Session(burst_chain(3), values_feed(1.0, 2.0, 3.0), catalogue=OPERATORS)

    run = session.start({}, pipeline=options)

    assert run.wait(WAIT)
    emitted = sum(counters.emitted for counters in run.operator_counters.values())
    assert len(taken) == emitted == 3 * 3 + 1 + 2 * 10
    assert all(thread.startswith("workflows-v2-worker-") for *_, thread in taken)
    for counters in run.operator_counters.values():
        assert (counters.processed, counters.cancelled) == (counters.emitted, 0)
    gauges = run.pipeline_counters
    assert gauges.peak("executing") == gauges.peak("live_states") == 1
    assert gauges.current("pending_operator_pulses") == 0


def test_with_an_idle_worker_the_dispatcher_places_the_next_emission(
    monkeypatch,
) -> None:
    # Both emissions must obtain workers while burst#0 (value 10) is held.
    # The dispatcher and finishing worker may race to take either unit.
    taken = dispatched_emissions(monkeypatch)
    definition = burst_chain(2)
    definition["operators"].pop()  # no chained operator
    definition["outputs"].pop()
    session = Session(definition, values_feed(1.0), catalogue=OPERATORS)
    release = session.probe.hold("slow", 10.0)

    run = session.start({}, pipeline=PipelineOptions(max_in_flight=2))
    try:
        assert session.probe.entered("slow", 10.0).wait(WAIT)
        deadline = time.monotonic() + WAIT
        while len(taken) < 2:
            assert time.monotonic() < deadline, "burst#1 was never taken"
            time.sleep(0.001)
        first_level = [(name, sequence) for name, sequence, _ in taken[:2]]
        assert first_level == [("burst", 0), ("burst", 1)]
    finally:
        release.set()

    assert run.wait(WAIT)


def test_pending_emissions_run_before_fresh_source_pulses() -> None:
    definition = burst_chain(3)
    definition["operators"].pop()  # no chained operator
    definition["outputs"].pop()
    session = Session(definition, values_feed(1.0, 2.0), catalogue=OPERATORS)
    release = session.probe.hold("slow", 10.0)

    run = session.start({}, pipeline=PipelineOptions(max_in_flight=1))

    assert session.probe.entered("slow", 10.0).wait(WAIT)
    # a#0 finished; burst#1 and #2 wait without a team while #0 is held.
    assert run.pipeline_counters.current("pending_operator_pulses") == 2
    assert run.pipeline_counters.current("executing") == 1
    release.set()
    assert run.wait(WAIT)
    entered = [
        (unit, value) for kind, unit, value in session.probe.events if kind == "enter"
    ]
    assert entered == [
        ("fresh", 1.0),
        ("slow", 10.0),
        ("slow", 11.0),
        ("slow", 12.0),
        ("fresh", 2.0),
        ("slow", 20.0),
        ("slow", 21.0),
        ("slow", 22.0),
        ("slow", 25.0),  # the finish tail
    ]


@pytest.mark.parametrize("parallelism", [1, 2], ids=["W1", "W2"])
def test_collection_holds_no_team_while_its_window_is_incomplete(parallelism) -> None:
    gate = threading.Event()
    definition = active(
        [source("a"), source("b")],
        [step(Held, "other", value="$sources.b.value")],
        [
            group("W", "$operators.clip.x", x="$operators.clip.x"),
            group("O", "$sources.b.value", other="$steps.other.value"),
        ],
    )
    definition["operators"] = [
        operator(Window, "clip", size=3, collect={"x": "$sources.a.value"})
    ]
    feeds = {
        "a": [emit(value=1.0), emit(value=2.0), gate, emit(value=3.0)],
        "b": [emit(value=7.0)],
    }
    session = Session(definition, feeds, catalogue=OPERATORS)
    delivered = threading.Event()
    collected = Collector()

    def on_other(result) -> None:
        collected.handler("O")(result)
        delivered.set()

    run = session.start(
        {"W": collected.handler("W"), "O": on_other},
        pipeline=PipelineOptions(max_in_flight=1, max_workflow_parallelism=parallelism),
    )

    # Two values collected, the third not yet read: b still gets the team.
    assert delivered.wait(WAIT)
    assert run.operator_counters["clip"].emitted == 0
    gate.set()
    assert run.wait(WAIT)
    assert collected.rows("W") == [{"x": [1.0, 2.0, 3.0]}]


# PipelineOptions compatibility (TC-002) ---------------------------------------------


def test_pipeline_options_keep_their_positional_order() -> None:
    latest = PipelineOptions(2, "latest")
    mixed = PipelineOptions(2, "block", {"cam": "latest"})

    assert (latest.max_in_flight, latest.overload) == (2, "latest")
    assert mixed.overload_for("cam") == "latest" and mixed.overload == "block"
    assert latest.max_workflow_parallelism == mixed.max_workflow_parallelism == 1
    assert PipelineOptions(2, max_workflow_parallelism=3).max_workflow_parallelism == 3
    with pytest.raises(TypeError):
        PipelineOptions(2, "block", {}, 3)  # W is keyword-only
    with pytest.raises(ContractError, match="max_workflow_parallelism"):
        PipelineOptions(max_workflow_parallelism=0)
