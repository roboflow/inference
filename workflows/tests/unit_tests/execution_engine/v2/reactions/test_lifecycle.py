"""Lifecycle of reactions inside active runs: drain, cancel, self-wait, failure.

Each scenario holds a handler on a gate the test controls, checks what the
run reports while it is held, then releases it. Waits are bounded.
"""

import threading
import weakref

import numpy as np
import pytest
from roboflow_workflows.execution_engine.v2.errors import (
    ActiveRunError,
    ContractError,
    EventEmissionError,
    ReactionCycleError,
    ReactionError,
)
from roboflow_workflows.execution_engine.v2.observer import ReactionObserver
from roboflow_workflows.execution_engine.v2.reactions import PlannedHandlerGroup
from roboflow_workflows.execution_engine.v2.reactions.dispatch import EventCause
from roboflow_workflows.execution_engine.v2.reactions.runtime import ReactionRuntime
from roboflow_workflows.execution_engine.v2.reactions.snapshots import snapshot_fields

from tests.unit_tests.execution_engine.v2.reactions.test_runtime import (
    NOTIFY,
    PASSIVE,
    PAYLOAD_HANDLER,
    WAIT,
    Probe,
    active_definition,
    eventually,
    handler,
    reacting,
    start,
)


def test_end_of_source_completes_only_after_queued_events_ran():
    probe = Probe()
    gate = probe.hold(0.0)
    plan = reacting(active_definition(), handler(depth=4))
    run = start(plan, probe, {"cam_a": [0, 1, 2]})

    probe.wait_for(lambda: len(probe.values("after")) == 3)
    # The main flow is done; the run is not, and says why.
    assert run.wait(0.05) is False
    counters = run.reaction_counters[NOTIFY]
    assert (counters.running, counters.pending) == (1, 2)
    gate.set()
    assert run.wait(WAIT)

    assert run.state == "finished"
    assert probe.values("reacted") == [0.0, 1.0, 2.0]
    assert not [thread for thread in threading.enumerate() if "reaction" in thread.name]


def test_stop_drains_events_of_admitted_pulses():
    probe = Probe()
    gate = probe.hold(0.0)
    hang = threading.Event()
    plan = reacting(active_definition(), handler(depth=4))
    run = start(plan, probe, {"cam_a": [0, 1, 2, hang, 3]})

    probe.wait_for(lambda: len(probe.values("after")) == 3)
    run.stop()
    hang.set()
    gate.set()
    assert run.wait(WAIT)

    assert run.state == "finished"
    assert probe.values("reacted") == [0.0, 1.0, 2.0]
    counters = run.reaction_counters[NOTIFY]
    assert (counters.emitted, counters.completed, counters.discarded) == (3, 3, 0)


def test_cancel_discards_queued_events_and_lets_the_running_one_finish():
    probe = Probe()
    gate = probe.hold(0.0)
    plan = reacting(active_definition(), handler(depth=4))
    run = start(plan, probe, {"cam_a": [0, 1, 2]})

    probe.wait_for(lambda: len(probe.values("after")) == 3)
    probe.started(0.0).wait(WAIT)
    run.cancel()
    counters = run.reaction_counters[NOTIFY]
    assert (counters.discarded, counters.pending, counters.running) == (2, 0, 1)
    gate.set()
    assert run.wait(WAIT)

    assert run.state == "cancelled"
    assert probe.values("reacted") == [0.0]
    statuses = [
        (item.status, item.cause.pulse.sequence) for item in run.reaction_outcomes()
    ]
    assert statuses == [("discarded", 1), ("discarded", 2), ("completed", 0)]


def test_cancel_wakes_an_emitter_waiting_under_synchronous_overflow():
    probe = Probe()
    gate = probe.hold(0.0)
    plan = reacting(active_definition(), handler(depth=1, overflow="synchronous"))
    run = start(plan, probe, {"cam_a": [0, probe.started(0.0), 1, 2]})

    eventually(lambda: run.reaction_counters[NOTIFY].blocked == 1)
    run.cancel()
    # The emitter wakes without waiting for the held handler.
    eventually(lambda: run.reaction_counters[NOTIFY].blocked == 0)
    assert probe.values("emitted") == [0.0, 1.0]
    gate.set()
    assert run.wait(WAIT)

    assert run.state == "cancelled"
    assert run.failure is None
    assert probe.values("react") == [0.0]
    counters = run.reaction_counters[NOTIFY]
    # Queued value 1 and waiting emitter's value 2 never ran: both discarded.
    assert (counters.completed, counters.discarded, counters.overflow_sync) == (1, 2, 1)
    assert counters.emitted == counters.completed + counters.discarded
    statuses = sorted(
        (item.status, item.cause.pulse.sequence) for item in run.reaction_outcomes()
    )
    assert statuses == [("completed", 0), ("discarded", 1), ("discarded", 2)]


def test_wait_from_a_handler_raises_and_stop_from_a_handler_works():
    probe = Probe()
    caught = []
    plan = reacting(active_definition(), handler(depth=4))
    feeds = {"cam_a": [0, threading.Event(), 1]}
    session = plan.create_session(resources={"probe": probe, "feeds": feeds})
    holder = {}
    original = probe.react

    def react(value, *, context, payload=None):
        original(value, context=context, payload=payload)
        try:
            holder["run"].wait()
        except ContractError as error:
            caught.append(error)
        holder["run"].stop()
        feeds["cam_a"][1].set()

    probe.react = react
    holder["run"] = run = session.start()
    assert run.wait(WAIT)

    assert len(caught) == 1 and "wait for itself" in str(caught[0])
    assert run.state == "finished"
    assert probe.values("reacted") == [0.0]


def test_sync_handler_failure_fails_an_active_run_with_attribution():
    probe = Probe()
    probe.failing.add(1.0)
    plan = reacting(active_definition(), handler(mode="sync"))
    run = start(plan, probe, {"cam_a": [0, 1, 2]})

    with pytest.raises(ActiveRunError) as caught:
        run.wait(WAIT)

    assert caught.value.stage == "step"
    assert caught.value.step_path == ("emitter",)
    assert isinstance(caught.value.__cause__.__cause__, ReactionError)
    assert probe.values("after") == [0.0]


def test_emission_after_the_reactions_closed_is_rejected():
    probe = Probe()
    plan = reacting(PASSIVE, handler(mode="sync"))
    session = plan.create_session(resources={"probe": probe})
    runtime = ReactionRuntime(plan, sessions=session.handler_sessions)
    cause = EventCause(
        event="seen",
        emitter=("emitter",),
        session_id=session.session_id,
        run_id="r",
        pulse=None,
        index=(),
        sample=None,
        temporal=None,
        sequence=0,
    )
    runtime.close()

    with pytest.raises(EventEmissionError, match="closed"):
        runtime.dispatch(
            plan.reactions.handlers, cause, fields={"value": 1.0}, snapshots={}
        )


def test_a_handler_reentered_on_its_own_thread_is_reported_not_deadlocked():
    probe = Probe()
    plan = reacting(PASSIVE, handler(mode="sync"))
    session = plan.create_session(resources={"probe": probe})
    runtime = ReactionRuntime(plan, sessions=session.handler_sessions)
    cause = EventCause(
        "seen", ("emitter",), session.session_id, "r", None, (), None, None, 0
    )
    errors = []

    def reenter(value, *, context, payload=None):
        try:
            runtime.dispatch(
                plan.reactions.handlers, cause, fields={"value": 2.0}, snapshots={}
            )
        except ReactionCycleError as error:
            errors.append(error)

    probe.react = reenter
    runtime.dispatch(
        plan.reactions.handlers, cause, fields={"value": 1.0}, snapshots={}
    )

    assert len(errors) == 1 and errors[0].handler == NOTIFY


def test_failed_async_handlers_retain_no_payload_or_traceback():
    class Payload:
        def __workflows_snapshot__(self):
            copy = Payload()
            copies.append(weakref.ref(copy))
            return copy

    copies = []
    probe = Probe()
    probe.failing.update({0.0, 1.0})
    probe.payloads.update({0.0: Payload(), 1.0: Payload()})
    bindings = {"value": "value", "payload": "payload"}
    plan = reacting(
        active_definition(),
        handler(depth=4, bindings=bindings, workflow=PAYLOAD_HANDLER),
    )
    run = start(plan, probe, {"cam_a": [0, 1]})
    assert run.wait(WAIT)
    probe.seen.clear()

    assert run.reaction_counters[NOTIFY].failed == 2
    assert len(copies) == 2 and all(ref() is None for ref in copies)


def test_an_event_arriving_while_an_emitter_waits_queues_behind_it():
    """E0 done, E1 running, E2 waiting on its emitter, E3 queued: 0, 1, 2, 3."""
    probe = Probe()
    gates = {value: probe.hold(value) for value in (0.0, 1.0)}
    plan = reacting(active_definition(), handler(depth=1, overflow="synchronous"))
    session = plan.create_session(resources={"probe": probe, "feeds": {}})
    runtime = ReactionRuntime(
        plan, sessions=session.handler_sessions, active_run_id="run"
    )
    (planned,) = plan.reactions.handlers

    def emit(value: float) -> None:
        cause = EventCause("seen", ("emitter",), "s", "r", None, (), None, None, 0)
        snapshot = snapshot_fields({"value": value}, where="test")
        runtime.dispatch(
            (planned,), cause, fields={}, snapshots={planned.path: snapshot}
        )

    emit(0.0)
    probe.started(0.0).wait(WAIT)
    emit(1.0)
    waiting = threading.Thread(target=emit, args=(2.0,), name="emitter-2")
    waiting.start()
    eventually(lambda: runtime.counters[NOTIFY].blocked == 1)
    gates[0.0].set()
    probe.started(1.0).wait(WAIT)
    emit(3.0)
    assert runtime.counters[NOTIFY].pending == 1
    gates[1.0].set()
    waiting.join(WAIT)
    runtime.close()
    assert runtime.join(WAIT)

    assert probe.values("reacted") == [0.0, 1.0, 2.0, 3.0]
    threads = probe.threads("react")
    assert threads[2.0] == "emitter-2"
    assert threads[0.0] == threads[1.0] == threads[3.0] != "emitter-2"
    assert probe.max_active == 1


class CallbackAction(ReactionObserver):
    """On the first outcome, the observer or the group callback acts."""

    def __init__(self, action: str) -> None:
        self.action = action
        self.run = None
        self.seen = []
        self.done = False

    def on_reaction_finished(self, *, outcome, error, result):
        self.seen.append(outcome.status)
        if self.done or self.action.startswith("group"):
            return
        self.act()

    def deliver(self, result) -> None:
        if not self.done and self.action.startswith("group"):
            self.act()

    def act(self) -> None:
        self.done = True
        if self.action in ("observer raises", "group raises"):
            raise RuntimeError("callback fails")
        if self.action == "cancels":
            self.run.cancel()
            self.seen.append("cancel returned")
        else:
            self.seen.append(len(self.run.reaction_outcomes()))


@pytest.mark.parametrize(
    "action", ["observer raises", "group raises", "cancels", "reads outcomes"]
)
def test_a_callback_that_fails_cancels_or_reads_outcomes_does_not_deadlock(action):
    probe = Probe()
    gate = probe.hold(0.0)
    callbacks = CallbackAction(action)
    group = PlannedHandlerGroup(
        name="notifications", handler=("notify",), fields={"out": "out"}
    )
    plan = reacting(active_definition(), handler(depth=2), groups=[group])
    run = start(
        plan,
        probe,
        {"cam_a": [0, 1]},
        observer=callbacks,
        handlers={"notifications": callbacks.deliver},
    )
    callbacks.run = run
    eventually(lambda: run.reaction_counters[NOTIFY].pending == 1)
    gate.set()
    if action.endswith("raises"):
        with pytest.raises(ActiveRunError, match="callback fails") as raised:
            run.wait(WAIT)
        assert raised.value.stage == ("observer" if "observer" in action else "handler")
    else:
        assert run.wait(WAIT)

    counters = run.reaction_counters[NOTIFY]
    if action == "reads outcomes":
        assert callbacks.seen == ["completed", 1, "completed"]
        assert run.state == "finished"
        return
    assert (counters.completed, counters.discarded) == (1, 1)
    statuses = [item.status for item in run.reaction_outcomes()]
    assert statuses == ["completed", "discarded"]
    if action == "cancels":
        # The discard is delivered right after the callback that caused it.
        assert callbacks.seen == ["completed", "cancel returned", "discarded"]
        assert run.state == "cancelled"


def test_cancel_discards_every_waiting_emitter_and_queued_event_exactly_once():
    runtime = ReactionRuntime(
        reacting(active_definition(), handler(depth=1, overflow="synchronous")),
        sessions={},
        active_run_id="run",
    )
    state = runtime._handlers[("notify",)]
    running = _Running(state)
    causes = [EventCause("seen", ("emitter",), "s", "r", None, (), None, None, k)
              for k in range(4)]  # fmt: skip
    snapshots = [
        snapshot_fields({"value": np.array([k])}, where="test") for k in range(4)
    ]
    released = [weakref.ref(item.fields["value"]) for item in snapshots]
    state.admit(causes[1], snapshots[1])  # queued behind the running event
    errors = []

    def wait_in_line(position: int) -> None:
        try:
            # Both waiters read snapshots before the blocked check and deletion.
            state.admit(causes[position], snapshots[position])  # noqa: F821
        except BaseException as error:
            errors.append(type(error).__name__)

    waiters = [threading.Thread(target=wait_in_line, args=(k,)) for k in (2, 3)]
    for waiter in waiters:
        waiter.start()
    eventually(lambda: state.counters.blocked == 2)
    del snapshots
    runtime.cancel()
    for waiter in waiters:
        waiter.join(WAIT)
    running.finish()

    assert sorted(errors) == ["RunAborted", "RunAborted"]
    counters = runtime.counters["$handlers.notify"]
    assert (counters.emitted, counters.discarded, counters.blocked) == (3, 3, 0)
    statuses = sorted((item.status, item.cause.sequence) for item in runtime.outcomes())
    assert statuses == [("discarded", 1), ("discarded", 2), ("discarded", 3)]
    assert all(reference() is None for reference in released[1:])


class _Running:
    """Marks a handler line as running an event, as a worker would."""

    def __init__(self, state) -> None:
        self.state = state
        with state.condition:
            state.running = True

    def finish(self) -> None:
        with self.state.condition:
            self.state.running = False
            self.state.condition.notify_all()
