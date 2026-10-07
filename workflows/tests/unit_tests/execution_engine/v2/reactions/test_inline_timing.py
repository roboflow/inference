"""Inline timing counts the emitter's wait for its turn, not only execution.

A handler is held on a gate while another emitter waits for its turn; the
test keeps it held for ``HELD`` seconds after the wait is visible in the
counters. The waiting emitter's own run is instant, so only a span that
starts before the wait reaches ``HELD``.
"""

import threading
import time

from roboflow_workflows.execution_engine.v2.reactions.dispatch import EventCause
from roboflow_workflows.execution_engine.v2.reactions.runtime import ReactionRuntime

from tests.unit_tests.execution_engine.v2.reactions.test_runtime import (
    NOTIFY,
    WAIT,
    Probe,
    active_definition,
    eventually,
    handler,
    reacting,
    start,
)

HELD = 0.2
"""Seconds the held handler keeps a waiting emitter waiting."""


def test_waiting_for_a_sync_handler_lock_counts_as_inline_time():
    probe = Probe()
    gate = probe.hold(0.0)
    plan = reacting(active_definition(), handler(mode="sync"))
    session = plan.create_session(resources={"probe": probe, "feeds": {}})
    runtime = ReactionRuntime(
        plan, sessions=session.handler_sessions, active_run_id="run"
    )
    (planned,) = plan.reactions.handlers

    def emit(value: float) -> None:
        cause = EventCause("seen", ("emitter",), "s", "r", None, (), None, None, 0)
        runtime.dispatch((planned,), cause, fields={"value": value}, snapshots={})

    holding = threading.Thread(target=emit, args=(0.0,), name="emitter-0")
    waiting = threading.Thread(target=emit, args=(1.0,), name="emitter-1")
    holding.start()
    assert probe.started(0.0).wait(WAIT)
    waiting.start()
    # Counted as emitted before it waits for the lock held by emitter-0.
    eventually(lambda: runtime.counters[NOTIFY].emitted == 2)
    time.sleep(HELD)
    gate.set()
    holding.join(WAIT)
    waiting.join(WAIT)
    runtime.close()
    assert runtime.join(WAIT)

    assert probe.values("reacted") == [0.0, 1.0]
    assert probe.threads("react") == {0.0: "emitter-0", 1.0: "emitter-1"}
    counters = runtime.counters[NOTIFY]
    assert (counters.emitted, counters.completed, counters.inline) == (2, 2, 2)
    # Both runs span at least HELD: the shorter one is emitter-1's lock wait.
    assert counters.inline_seconds - counters.inline_max_seconds >= HELD


def test_waiting_behind_earlier_events_under_synchronous_overflow_counts():
    probe = Probe()
    gate = probe.hold(0.0)
    feeds = {"cam_a": [0, probe.started(0.0), 1, 2, 3]}
    plan = reacting(active_definition(), handler(depth=1, overflow="synchronous"))
    run = start(plan, probe, feeds)

    # E0 runs (held), E1 is queued, E2's emitter waits behind both.
    eventually(lambda: run.reaction_counters[NOTIFY].blocked == 1)
    time.sleep(HELD)
    gate.set()
    assert run.wait(WAIT)

    assert probe.values("react") == [0.0, 1.0, 2.0, 3.0]
    assert probe.threads("react")[2.0] == probe.threads("emit")[2.0]
    counters = run.reaction_counters[NOTIFY]
    assert (counters.completed, counters.overflow_sync, counters.inline) == (4, 1, 1)
    # E2's only inline run includes its wait behind the held E0 and E1.
    assert counters.inline_seconds == counters.inline_max_seconds >= HELD


def test_an_emitter_cancelled_while_waiting_adds_no_inline_time():
    probe = Probe()
    gate = probe.hold(0.0)
    plan = reacting(active_definition(), handler(depth=1, overflow="synchronous"))
    run = start(plan, probe, {"cam_a": [0, probe.started(0.0), 1, 2]})

    eventually(lambda: run.reaction_counters[NOTIFY].blocked == 1)
    time.sleep(HELD)
    run.cancel()
    eventually(lambda: run.reaction_counters[NOTIFY].blocked == 0)
    gate.set()
    assert run.wait(WAIT)

    assert run.state == "cancelled"
    counters = run.reaction_counters[NOTIFY]
    assert (counters.completed, counters.discarded, counters.overflow_sync) == (1, 2, 1)
    assert (counters.inline, counters.inline_seconds) == (0, 0.0)
    assert counters.inline_max_seconds == 0.0
