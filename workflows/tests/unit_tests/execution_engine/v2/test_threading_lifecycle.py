"""Lifecycle of independently dispatched operator emissions.

Emitted pulses are admitted work: they count in flight under the control
version they inherited, keep the run from completing, block an update
boundary and a younger reset epoch until they settle, and on an abort are
cancelled, never dropped::

    stop      emitted and finish pulses all run; nothing cancelled
    cancel    running one returns; turn-waiting and pending ones cancelled
    failure   the failing pulse is attributed; the rest cancelled
    update    boundary only after old emissions ran; results keep versions
    reset     old emissions finish on the old processing; source stays open

Every run ends with nothing pending, live or in flight and no thread left.
"""

import threading
import time
from typing import Any, Callable, List

import pytest
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.controls import InFlightWork
from roboflow_workflows.execution_engine.v2.errors import ActiveRunError
from roboflow_workflows.execution_engine.v2.pipelining.options import PipelineOptions

from tests.unit_tests.execution_engine.v2.active.test_pipelined import (
    Collector,
    Held,
    Session,
)
from tests.unit_tests.execution_engine.v2.execution.blocks import Deferred, Failing
from tests.unit_tests.execution_engine.v2.m7_controls.blocks import Tracker
from tests.unit_tests.execution_engine.v2.test_active_runtime import (
    WAIT,
    active,
    emit,
    group,
    source,
    step,
)
from tests.unit_tests.execution_engine.v2.test_threading_emissions import (
    OPERATORS,
    Burst,
    operator,
)

CATALOGUE = Catalogue.merge(OPERATORS, Catalogue([Deferred, Failing, Tracker]))


def bursting(count: int, *steps: dict, **sections: Any) -> dict:
    """a -> burst(count) -> steps; group B reads the last step's value."""
    definition = active(
        [source("a")],
        list(steps),
        [group("B", "$operators.burst.out", value=f"$steps.{steps[-1]['name']}.value")],
    )
    definition["operators"] = [
        operator(Burst, "burst", inputs={"value": "$sources.a.value"}, count=count)
    ]
    definition.update(sections)

    return definition


HELD_ROUTE = (
    step(Held, "first", value="$operators.burst.out"),
    step(Held, "slow", value="$steps.first.value"),
)


def settled(run: Any, session: Session) -> None:
    """Nothing pending, live or in flight; every engine thread of the run joined."""
    gauges = run.pipeline_counters
    assert gauges.current("pending_operator_pulses") == 0
    assert gauges.current("live_states") == 0
    assert gauges.current("executing") == 0
    assert session.session.controls.in_flight.describe() == {}
    prefix = run.run_id[:8]
    assert not [
        thread.name
        for thread in threading.enumerate()
        if thread.name.startswith(
            (f"workflows-v2-worker-{prefix}", f"workflows-v2-run-{prefix}")
        )
    ]
    for counters in run.operator_counters.values():
        assert counters.emitted == counters.processed + counters.cancelled
        assert counters.closed


def eventually(predicate: Callable[[], bool]) -> None:
    deadline = time.monotonic() + WAIT
    while not predicate():
        assert time.monotonic() < deadline, "the condition never held"
        time.sleep(0.001)


class Updater:
    """Applies one update from a host thread; the test thread keeps control."""

    def __init__(self, run: Any, candidate: Any) -> None:
        self.receipt: Any = None
        self.error: Any = None
        self.finished = threading.Event()
        self._thread = threading.Thread(target=self._apply, args=(run, candidate))
        self._thread.start()

    def _apply(self, run: Any, candidate: Any) -> None:
        try:
            self.receipt = run.apply_update(candidate, timeout=WAIT)
        except Exception as error:
            self.error = error
        finally:
            self.finished.set()

    def join(self) -> Any:
        self._thread.join(WAIT)
        assert self.error is None, self.error
        return self.receipt


OPTIONS = [
    PipelineOptions(max_in_flight=2),
    PipelineOptions(max_in_flight=2, max_workflow_parallelism=2),
]


def label(options: PipelineOptions) -> str:
    return f"P{options.max_in_flight}W{options.max_workflow_parallelism}"


# Stop, cancel, failure --------------------------------------------------------------


def test_stop_runs_every_pending_emission_and_the_finish_tail() -> None:
    gate = threading.Event()
    definition = bursting(3, *HELD_ROUTE)
    definition["operators"][0]["tail"] = 1
    session = Session(
        definition, {"a": [emit(value=1.0), gate, emit(value=2.0)]}, catalogue=CATALOGUE
    )
    release = session.probe.hold("slow", 10.0)
    collected = Collector()
    run = session.start(
        collected.handlers("B"), pipeline=PipelineOptions(max_in_flight=1)
    )

    assert session.probe.entered("slow", 10.0).wait(WAIT)
    assert run.pipeline_counters.current("pending_operator_pulses") == 2
    run.stop()
    gate.set()
    release.set()

    assert run.wait(WAIT)
    assert run.state == "finished"
    assert [row["value"] for row in collected.rows("B")] == [10.0, 11.0, 12.0, 15.0]
    burst = run.operator_counters["burst"]
    assert (burst.emitted, burst.processed, burst.cancelled) == (4, 4, 0)
    assert burst.finished
    assert run.counters["a"].unadmitted == 1
    settled(run, session)


@pytest.mark.parametrize("options", OPTIONS, ids=label)
def test_cancel_with_emissions_running_turn_waiting_and_pending(options) -> None:
    session = Session(
        bursting(3, *HELD_ROUTE), {"a": [emit(value=1.0)]}, catalogue=CATALOGUE
    )
    release = session.probe.hold("slow", 10.0)
    run = session.start(Collector().handlers("B"), pipeline=options)

    # burst#0 is held in slow; burst#1 passed first and waits for its turn
    # at slow on the second team; burst#2 has no team.
    assert session.probe.entered("slow", 10.0).wait(WAIT)
    assert session.probe.entered("first", 11.0).wait(WAIT)
    assert run.pipeline_counters.current("pending_operator_pulses") == 1
    run.cancel()
    release.set()

    assert run.wait(WAIT)
    assert run.state == "cancelled"
    burst = run.operator_counters["burst"]
    assert burst.emitted == 3 and burst.cancelled >= 2
    assert session.probe.order["slow"] == [10.0]
    settled(run, session)


def test_a_failing_emission_cancels_the_pending_ones_and_not_its_parent() -> None:
    session = Session(
        bursting(3, step(Failing, "check", value="$operators.burst.out")),
        {"a": [emit(value=-1.0)]},
        catalogue=CATALOGUE,
    )
    run = session.start(
        Collector().handlers("B"), pipeline=PipelineOptions(max_in_flight=1)
    )

    with pytest.raises(ActiveRunError, match="negative -10.0") as raised:
        run.wait(WAIT)
    assert (raised.value.stage, raised.value.operator, raised.value.pulse) == (
        "step",
        "burst",
        0,
    )
    burst = run.operator_counters["burst"]
    assert (burst.emitted, burst.processed, burst.cancelled) == (3, 0, 3)
    # Pipelined, the feeding pulse completed before its emissions ran.
    assert (run.counters["a"].processed, run.counters["a"].cancelled) == (1, 0)
    settled(run, session)


@pytest.mark.parametrize("options", OPTIONS, ids=label)
def test_a_failed_future_of_an_emitted_pulse_fails_the_run_and_settles(
    options,
) -> None:
    definition = bursting(
        2, step(Deferred, "remote", value="$operators.burst.out", fail=True)
    )
    definition["outputs"] = [
        group("B", "$operators.burst.out", doubled="$steps.remote.doubled")
    ]
    session = Session(definition, {"a": [emit(value=1.0)]}, catalogue=CATALOGUE)
    run = session.start(Collector().handlers("B"), pipeline=options)

    with pytest.raises(ActiveRunError, match="remote failure on 10.0") as raised:
        run.wait(WAIT)
    assert (raised.value.operator, raised.value.pulse) == ("burst", 0)
    settled(run, session)


# Controls ---------------------------------------------------------------------------


def test_a_younger_reset_epoch_waits_for_old_emissions_before_its_push(
    monkeypatch,
) -> None:
    draining = threading.Event()
    wait = InFlightWork.wait_drained_below

    def signalled(self, epoch, aborted=None):
        draining.set()
        return wait(self, epoch, aborted)

    monkeypatch.setattr(InFlightWork, "wait_drained_below", signalled)
    gate = threading.Event()
    definition = bursting(
        2,
        *HELD_ROUTE,
        controls={
            "tracking": {
                "type": "enable",
                "steps": ["$steps.tracker"],
                "enabled": False,
                "state": "reset_on_enable",
                "suspends_effects": True,
            }
        },
    )
    definition["steps"].append(step(Tracker, "tracker", value="$sources.a.value"))
    session = Session(
        definition, {"a": [emit(value=1.0), gate, emit(value=2.0)]}, catalogue=CATALOGUE
    )
    release = session.probe.hold("slow", 10.0)
    collected = Collector()
    run = session.start(
        collected.handlers("B"), pipeline=PipelineOptions(max_in_flight=3)
    )

    assert session.probe.entered("slow", 10.0).wait(WAIT)
    # a#0 completed; its emissions (version 0) are counted in flight.
    assert session.session.controls.in_flight.describe() == {"0": 2}
    receipt = session.session.controls.update(tracking=True)
    gate.set()
    # a#1 carries the reset epoch: it waits before pushing to burst.
    assert draining.wait(WAIT)
    assert run.operator_counters["burst"].arrivals == 1
    release.set()

    assert run.wait(WAIT)
    versions = [
        (row["value"], result.controls.version)
        for result in collected.results["B"]
        for row in result.rows()
    ]
    old, new = 0, receipt.version
    assert versions == [(10.0, old), (11.0, old), (20.0, new), (21.0, new)]
    settled(run, session)


# Graph updates ----------------------------------------------------------------------


@pytest.mark.parametrize("reset", [False, True], ids=["update", "reset"])
def test_an_update_boundary_waits_for_pending_emissions(reset) -> None:
    gate = threading.Event()
    definition = bursting(2, *HELD_ROUTE)
    session = Session(
        definition, {"a": [emit(value=1.0), gate, emit(value=2.0)]}, catalogue=CATALOGUE
    )
    release = session.probe.hold("slow", 10.0)
    collected = Collector()
    run = session.start(
        collected.handlers("B"), pipeline=PipelineOptions(max_in_flight=1)
    )
    assert session.probe.entered("slow", 10.0).wait(WAIT)

    changed = bursting(2, *HELD_ROUTE)
    changed["outputs"].append(group("A", "$sources.a.value", a="$sources.a.value"))
    plan = compile_workflow(changed, catalogue=CATALOGUE)
    updater = Updater(run, session.session.prepare_update(plan, reset=reset))
    eventually(lambda: run._driver._paused)
    # Paused with burst#0 running and burst#1 pending: no boundary yet.
    assert run.pipeline_counters.current("pending_operator_pulses") == 1
    assert not updater.finished.is_set()
    release.set()
    receipt = updater.join()
    gate.set()

    assert run.wait(WAIT)
    seen = [
        (result.pulse.sequence, result.graph_version)
        for result in collected.results["B"]
    ]
    assert seen == [(0, 0), (1, 0), (2, 1), (3, 1)]
    assert receipt.frontiers["burst"] == 2
    opened = [entry for entry in session.log if entry[0] == "open"]
    assert len(opened) == 1  # the source stayed connected
    if reset:
        assert receipt.cleanup is not None and receipt.cleanup.wait(WAIT)
    settled(run, session)
