"""External signals and run lifecycle events in real engine runs.

``ActiveRun.signal`` runs synchronous work on its caller and is admitted
atomically against ``stop()``/``cancel()``. ``$system.events.started`` comes
before any source is read; ``ended`` after the last pulse, before the drain,
and only for a run that was not cancelled or failed.
"""

import threading
from fractions import Fraction
from typing import Any, List, Tuple

import pytest
from roboflow_workflows.execution_engine.v2.active.runtime import _ACTIVE_RUNS
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.data import (
    SampleContext,
    TemporalContext,
    Timestamp,
)
from roboflow_workflows.execution_engine.v2.errors import (
    ActiveRunError,
    ContractError,
    EventEmissionError,
    ReactionError,
    WorkflowCompileError,
)
from roboflow_workflows.execution_engine.v2.events import EventPayloadError
from roboflow_workflows.execution_engine.v2.reactions.machines import TransitionCounters

from .test_machines_runtime import CATALOGUE, Journal, inspection, recorder, start
from .test_runtime import WAIT, eventually


def on_ack(name: str, mode: str = "sync") -> dict:
    """A handler of ``$signals.ack`` recording its ``note`` under ``name``."""
    handler = {
        "name": name,
        "on": "$signals.ack",
        "execution": {"mode": mode},
        "bindings": {"note": "$event.note"},
        "workflow": recorder(name, "note", "string"),
    }

    return handler


def approved(run: Any) -> bool:
    return run.machine_state("inspection_state", source_id="cam_a") == ("approved", 2)


# Signals ------------------------------------------------------------------


def test_a_signal_drives_a_source_machine_and_its_handler_sees_the_given_context():
    journal = Journal()
    gate = threading.Event()
    run = start(inspection(handlers=(on_ack("ack"),)), journal, {"cam_a": [0.9, gate]})
    eventually(lambda: approved(run))
    temporal = TemporalContext(
        observed_coverage=Timestamp(ticks=5, time_base=Fraction(1, 1), clock_id="c")
    )
    run.signal("ack", source_id="cam_a", temporal=temporal, note="ok")
    gate.set()
    assert run.wait(WAIT)

    assert run.machine_state("inspection_state", source_id="cam_a") == ("idle", 3)
    ((_, note, cause, thread),) = journal.entries("ack")
    assert note == "ok" and thread == threading.current_thread().name
    assert cause.origin_selector == "$signals.ack" and cause.kind == "signal"
    assert cause.source_id == "cam_a" and cause.temporal is temporal
    assert cause.pulse is None and cause.parent is None


def test_signal_identity_and_payload_are_checked_before_anything_applies():
    journal = Journal()
    gate = threading.Event()
    run = start(inspection(), journal, {"cam_a": [0.9, gate]})
    eventually(lambda: approved(run))

    with pytest.raises(EventEmissionError, match="'nope' is not declared"):
        run.signal("nope", source_id="cam_a")
    with pytest.raises(EventEmissionError, match="per-source state machine"):
        run.signal("ack", note="ok")
    with pytest.raises(ContractError, match="disagree|equal ones"):
        run.signal("ack", source_id="cam_a", sample=SampleContext("cam_b"), note="ok")
    with pytest.raises(EventPayloadError):
        run.signal("ack", source_id="cam_a", note=3)
    with pytest.raises(EventPayloadError):
        run.signal("ack", source_id="cam_a")
    run.signal("ack", sample=SampleContext("cam_a"), note="ok")
    gate.set()
    assert run.wait(WAIT)

    reset = run.machine_counters()["inspection_state.reset"]
    assert reset == TransitionCounters(applied=1)


def test_signals_are_rejected_once_the_run_stops_is_cancelled_or_done():
    for end in ("stop", "cancel", "done"):
        journal = Journal()
        gate = threading.Event()
        run = start(inspection(), journal, {"cam_a": [0.9, gate]})
        eventually(lambda: approved(run))
        if end == "stop":
            run.stop()
        elif end == "cancel":
            run.cancel()
        gate.set()
        assert run.wait(WAIT)

        with pytest.raises(EventEmissionError, match="rejected"):
            run.signal("ack", source_id="cam_a", note="late")
        assert run.machine_counters()["inspection_state.reset"] == (
            TransitionCounters()
        )


def test_a_signal_admitted_before_stop_finishes_before_the_run_is_done():
    journal = Journal()
    held = journal.hold("ack", "first")
    gate = threading.Event()
    definition = inspection(handlers=(on_ack("ack"), on_ack("later", "async")))
    run = start(definition, journal, {"cam_a": [0.9, gate]})
    eventually(lambda: approved(run))
    raised: List[BaseException] = []

    def signal() -> None:
        try:
            run.signal("ack", source_id="cam_a", note="first")
        except BaseException as error:
            raised.append(error)

    caller = threading.Thread(target=signal)
    caller.start()
    assert journal.started("ack", "first").wait(WAIT)
    run.stop()
    gate.set()
    with pytest.raises(EventEmissionError, match="rejected"):
        run.signal("ack", source_id="cam_a", note="second")
    assert not run.wait(0.05)
    held.set()
    caller.join(WAIT)
    assert run.wait(WAIT)

    assert raised == []
    assert journal.values("ack") == ["first"]
    assert journal.values("later") == ["first"]
    later = run.reaction_counters["$handlers.later"]
    assert (later.completed, later.discarded) == (1, 0)
    assert run.state == "finished"


def test_a_sync_signal_handler_cannot_wait_for_its_own_run():
    journal = Journal()
    gate = threading.Event()
    runs: List[Any] = []
    attempts: List[BaseException] = []

    def wait_for_run(label: str, value: Any) -> None:
        if label == "ack":
            try:
                runs[0].wait(0.01)
            except BaseException as error:
                attempts.append(error)

    journal.on_record = wait_for_run
    runs.append(
        start(inspection(handlers=(on_ack("ack"),)), journal, {"cam_a": [0.9, gate]})
    )
    run = runs[0]
    eventually(lambda: approved(run))
    run.signal("ack", source_id="cam_a", note="ok")
    gate.set()
    assert run.wait(WAIT)

    (error,) = attempts
    assert isinstance(error, ContractError) and "wait for itself" in str(error)


def test_a_failing_sync_signal_handler_raises_to_the_caller_and_the_run_continues():
    journal = Journal()
    journal.failing.add("ack")
    gate = threading.Event()
    run = start(inspection(handlers=(on_ack("ack"),)), journal, {"cam_a": [0.9, gate]})
    eventually(lambda: approved(run))
    with pytest.raises(ReactionError, match="ack fails"):
        run.signal("ack", source_id="cam_a", note="ok")
    gate.set()
    assert run.wait(WAIT)

    assert run.state == "finished"
    assert run.machine_state("inspection_state", source_id="cam_a") == ("idle", 3)


# Abort settlement: accepted host-thread work finishes before done ---------


def signal_on_thread(run: Any, note: str) -> Tuple[threading.Thread, list]:
    """Call ``run.signal("ack", note=note)`` on a new thread; collect errors."""
    raised: List[BaseException] = []

    def signal() -> None:
        try:
            run.signal("ack", source_id="cam_a", note=note)
        except BaseException as error:
            raised.append(error)

    caller = threading.Thread(target=signal, name=f"host-{note}")
    caller.start()

    return caller, raised


def feed(end: str, gate: threading.Event) -> list:
    """Approve, then wait for ``gate``; a ``"failure"`` feed then fails its read."""
    items = [0.9, gate] if end == "cancel" else [0.9, gate, "boom"]

    return items


def abort(run: Any, end: str, gate: threading.Event) -> None:
    """``cancel()``, or let the feed reach its failing item (``"boom"``)."""
    if end == "cancel":
        run.cancel()
    gate.set()


def settled(run: Any, end: str) -> None:
    """The run is done with ``end``'s attribution and kept nothing of the run."""
    if end == "cancel":
        assert run.wait(WAIT)
        assert run.state == "cancelled"
    else:
        with pytest.raises(ActiveRunError, match="boom"):
            run.wait(WAIT)
        assert run.state == "failed"
    reactions = run._reactions
    assert reactions._in_flight == 0
    assert all(
        state.thread is None or not state.thread.is_alive()
        for state in reactions._handlers.values()
    )
    assert _ACTIVE_RUNS.get(run.session) is None
    with pytest.raises(EventEmissionError, match="rejected"):
        run.signal("ack", source_id="cam_a", note="after")


def discarded_outcomes(run: Any, selector: str) -> int:
    outcomes = run.reaction_outcomes()
    count = sum(
        1 for o in outcomes if o.handler == selector and o.status == "discarded"
    )

    return count


@pytest.mark.parametrize("end", ["cancel", "failure"])
def test_an_aborted_run_is_done_only_after_an_accepted_sync_signal_returned(end):
    journal = Journal()
    worker_gate = journal.hold("later", "busy")
    caller_gate = journal.hold("ack", "first")
    gate = threading.Event()
    definition = inspection(handlers=(on_ack("later", "async"), on_ack("ack")))
    run = start(definition, journal, {"cam_a": feed(end, gate)})
    eventually(lambda: approved(run))
    run.signal("ack", source_id="cam_a", note="busy")
    assert journal.started("later", "busy").wait(WAIT)
    caller, raised = signal_on_thread(run, "first")
    assert journal.started("ack", "first").wait(WAIT)
    abort(run, end, gate)
    eventually(lambda: run.reaction_counters["$handlers.later"].discarded == 1)

    with pytest.raises(EventEmissionError, match="rejected"):
        run.signal("ack", source_id="cam_a", note="late")
    assert not run.wait(0.05)
    assert run.state == "stopping"
    assert run._reactions._in_flight == 2  # the signal and the held worker event
    worker_gate.set()
    eventually(lambda: run.reaction_counters["$handlers.later"].running == 0)
    assert not run.wait(0.05)
    assert caller.is_alive()
    caller_gate.set()
    caller.join(WAIT)
    settled(run, end)

    assert raised == []
    assert journal.values("ack") == ["busy", "first"]
    assert journal.values("later") == ["busy"]
    later = run.reaction_counters["$handlers.later"]
    assert (later.completed, later.discarded, later.running) == (1, 1, 0)
    assert discarded_outcomes(run, "$handlers.later") == 1


@pytest.mark.parametrize("end", ["cancel", "failure"])
def test_an_aborted_run_is_done_only_after_a_host_thread_overflow_run_returned(end):
    journal = Journal()
    worker_gate = journal.hold("later", "e0")
    caller_gate = journal.hold("later", "e2")
    gate = threading.Event()
    overflow = on_ack("later", "async")
    overflow["execution"]["queue"] = {"max_depth": 1, "overflow": "synchronous"}
    run = start(inspection(handlers=(overflow,)), journal, {"cam_a": feed(end, gate)})
    eventually(lambda: approved(run))
    run.signal("ack", source_id="cam_a", note="e0")
    assert journal.started("later", "e0").wait(WAIT)
    run.signal("ack", source_id="cam_a", note="e1")
    caller, raised = signal_on_thread(run, "e2")
    eventually(lambda: run.reaction_counters["$handlers.later"].blocked == 1)
    worker_gate.set()
    assert journal.started("later", "e2").wait(WAIT)
    run.signal("ack", source_id="cam_a", note="e3")
    abort(run, end, gate)
    eventually(lambda: run.reaction_counters["$handlers.later"].discarded == 1)

    with pytest.raises(EventEmissionError, match="rejected"):
        run.signal("ack", source_id="cam_a", note="late")
    assert not run.wait(0.05)
    assert run.state == "stopping"
    assert run._reactions._in_flight == 2  # e2's signal and its inline event
    assert caller.is_alive()
    caller_gate.set()
    caller.join(WAIT)
    settled(run, end)

    assert raised == []
    assert journal.values("later") == ["e0", "e1", "e2"]
    (inline,) = [item for item in journal.entries("later") if item[1] == "e2"]
    assert inline[3] == "host-e2"
    later = run.reaction_counters["$handlers.later"]
    assert (later.completed, later.inline, later.discarded) == (3, 1, 1)
    assert (later.running, later.pending, later.blocked) == (0, 0, 0)
    assert discarded_outcomes(run, "$handlers.later") == 1


# System events ------------------------------------------------------------


def lifecycle(definition: dict, *, ended: str = "async") -> dict:
    """Adds started/ended handlers and a global machine following them."""
    definition["handlers"] += [
        {
            "name": event,
            "on": f"$system.events.{event}",
            "execution": {"mode": mode},
            "bindings": {"value": event},
            "workflow": recorder(event),
        }
        for event, mode in (("started", "sync"), ("ended", ended))
    ]
    definition["state_machines"].append(
        {
            "name": "lifecycle",
            "scope": "global",
            "initial_state": "idle",
            "states": ["idle", "running", "finished"],
            "transitions": [
                {"name": "begin", "from": ["idle"], "to": "running",
                 "on": "$system.events.started"},
                {"name": "end", "from": ["running"], "to": "finished",
                 "on": "$system.events.ended"},
            ],
        }
    )  # fmt: skip

    return definition


def test_started_comes_before_any_read_and_ended_after_the_last_pulse():
    journal = Journal()
    held = journal.hold("ended", "ended")
    run = start(lifecycle(inspection()), journal, {"cam_a": [0.9, 0.8]})
    assert journal.started("ended", "ended").wait(WAIT)
    assert not run.wait(0.05)
    held.set()
    assert run.wait(WAIT)

    labels = [item[0] for item in journal.items]
    assert labels[0] == "started"
    assert labels.index("ended") > max(
        position for position, label in enumerate(labels) if label == "inspect"
    )
    assert run.machine_state("lifecycle") == ("finished", 2)
    (cause,) = [item[2] for item in journal.entries("ended")]
    assert cause.origin_selector == "$system.events.ended" and cause.sample is None


def test_a_cancelled_run_publishes_no_ended():
    journal = Journal()
    gate = threading.Event()
    run = start(lifecycle(inspection()), journal, {"cam_a": [0.9, gate]})
    eventually(lambda: approved(run))
    run.cancel()
    gate.set()
    assert run.wait(WAIT)

    assert journal.values("started") == ["started"]
    assert journal.values("ended") == []
    assert run.machine_state("lifecycle") == ("running", 1)


def test_a_failing_started_handler_fails_start_before_any_source_is_read():
    journal = Journal()
    journal.failing.add("started")
    with pytest.raises(ActiveRunError, match="started fails") as raised:
        start(lifecycle(inspection()), journal, {"cam_a": [0.9]})

    assert raised.value.stage == "start"
    assert journal.values("inspect") == []


def test_a_failing_ended_handler_fails_the_run():
    journal = Journal()
    journal.failing.add("ended")
    run = start(lifecycle(inspection(), ended="sync"), journal, {"cam_a": [0.9]})
    with pytest.raises(ActiveRunError, match="ended fails") as raised:
        run.wait(WAIT)

    assert raised.value.stage == "handler"
    assert run.state == "failed"


def test_a_source_machine_cannot_follow_source_less_system_events():
    definition = lifecycle(inspection())
    definition["state_machines"][-1]["scope"] = "source"
    with pytest.raises(WorkflowCompileError, match="source"):
        compile_workflow(definition, catalogue=CATALOGUE)
