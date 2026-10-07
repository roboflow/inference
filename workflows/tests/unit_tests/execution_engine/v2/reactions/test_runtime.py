"""Reaction runtime in real engine runs: sync handlers, async queues, groups.

Every scenario runs compiled plans through ``session.run`` or
``session.start``. Ordering is forced with gates the test controls (a feed
item waits until a handler started; a handler waits on a gate); waits are
bounded by ``WAIT`` and poll engine state, never sleep for luck.

The helpers here (blocks, the gated feed source, ``reacting``) are shared by
the other reaction runtime test modules.
"""

import dataclasses
import threading
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import pytest
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Output,
    Ref,
)
from roboflow_workflows.execution_engine.v2.errors import (
    ReactionError,
    StepExecutionError,
)
from roboflow_workflows.execution_engine.v2.events import Event
from roboflow_workflows.execution_engine.v2.kinds import FLOAT_KIND, WILDCARD_KIND
from roboflow_workflows.execution_engine.v2.observer import ReactionObserver
from roboflow_workflows.execution_engine.v2.pipelining.options import PipelineOptions
from roboflow_workflows.execution_engine.v2.plan import PulseKey
from roboflow_workflows.execution_engine.v2.reactions import (
    EventOrigin,
    PlannedHandler,
    PlannedHandlerGroup,
    QueuePolicy,
    ReactionPlan,
)
from roboflow_workflows.execution_engine.v2.sources import (
    Emission,
    Source,
    SourceOutput,
    SourceParams,
)

WAIT = 10.0
"""Upper bound of every wait, in seconds; the tests finish far earlier."""

NOTIFY = "$handlers.notify"

SEEN = Event({"value": FLOAT_KIND, "payload": []}, description="A value passed.")


class Probe:
    """Shared by blocks and handlers: a log, gates and concurrency tracking.

    ``hold(value)`` makes the handler wait for that value until released;
    ``started(value)`` is set once the handler entered for it, so a feed can
    wait for it. ``payloads`` supplies the emitted ``payload`` field per value.
    """

    def __init__(self) -> None:
        self.changed = threading.Condition()
        self.log: List[tuple] = []
        self.gates: Dict[float, threading.Event] = {}
        self.entered: Dict[float, threading.Event] = {}
        self.failing: set = set()
        self.payloads: Dict[float, Any] = {}
        self.seen: List[tuple] = []
        self.active = 0
        self.max_active = 0
        self.after_emit: Optional[Callable[[float, Any], None]] = None

    def hold(self, value: float) -> threading.Event:
        return self.gates.setdefault(float(value), threading.Event())

    def started(self, value: float) -> threading.Event:
        return self.entered.setdefault(float(value), threading.Event())

    def note(self, kind: str, value: float) -> None:
        with self.changed:
            self.log.append((kind, value, threading.current_thread().name))
            self.changed.notify_all()

    def values(self, kind: str) -> List[float]:
        with self.changed:
            return [value for item, value, _ in self.log if item == kind]

    def threads(self, kind: str) -> Dict[float, str]:
        with self.changed:
            return {value: thread for item, value, thread in self.log if item == kind}

    def wait_for(self, predicate: Callable[[], bool]) -> None:
        with self.changed:
            assert self.changed.wait_for(predicate, timeout=WAIT), list(self.log)

    def react(self, value: float, *, context: Any, payload: Any = None) -> None:
        with self.changed:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            self.note("react", value)
            self.seen.append(
                (value, context.cause, context.sample_at(context.indices[0]), payload)
            )
            self.started(value).set()
            gate = self.gates.get(value)
            if gate is not None:
                assert gate.wait(WAIT)
            if value in self.failing:
                raise RuntimeError(f"handler rejects {value}")
            self.note("reacted", value)
        finally:
            with self.changed:
                self.active -= 1


class Emitter(Block):
    """Emits ``seen`` once per call, then returns its value."""

    type = "test/emitter@v1"
    outputs = {"value": Output(FLOAT_KIND)}
    events = {"seen": SEEN}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)

    def __init__(self, *, probe: Probe) -> None:
        self.probe = probe

    def run(self, value):
        payload = self.probe.payloads.get(value)
        self.probe.note("emit", value)
        self.emit("seen", value=value, payload=payload)
        self.probe.note("emitted", value)
        if self.probe.after_emit is not None:
            self.probe.after_emit(value, payload)
        return {"value": value}


class After(Block):
    """A downstream step of the emitter."""

    type = "test/after@v1"
    outputs = {"value": Output(FLOAT_KIND)}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)

    def __init__(self, *, probe: Probe) -> None:
        self.probe = probe

    def run(self, value):
        self.probe.note("after", value)
        return {"value": value}


class React(Block):
    """The handler workflow's step: records, may wait on a gate or fail."""

    type = "test/react@v1"
    outputs = {"value": Output(FLOAT_KIND)}

    class Params(BlockParams):
        value: float | Ref(FLOAT_KIND)

    def __init__(self, *, probe: Probe) -> None:
        self.probe = probe

    def run(self, value):
        self.probe.react(value, context=self.execution_context)
        return {"value": value * 10}


class ReactWithPayload(Block):
    """A handler step that also receives the payload field."""

    type = "test/react_payload@v1"
    outputs = {"value": Output(FLOAT_KIND)}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)
        payload: Ref(WILDCARD_KIND)

    def __init__(self, *, probe: Probe) -> None:
        self.probe = probe

    def run(self, value, payload):
        self.probe.react(value, context=self.execution_context, payload=payload)
        return {"value": value * 10}


class Feed(Source):
    """Emits the values of ``feeds[feed]``; an ``Event`` item waits until set."""

    type = "test/feed@v1"
    outputs = {"value": SourceOutput(FLOAT_KIND)}

    class Params(SourceParams):
        feed: str

    def __init__(self, *, feeds: Dict[str, list]) -> None:
        self.feeds = feeds
        self.items = iter(())

    def open(self, *, feed) -> None:
        self.items = iter(self.feeds[feed])

    def read(self) -> Optional[Emission]:
        item = next(self.items, None)
        while isinstance(item, threading.Event):
            assert item.wait(WAIT)
            item = next(self.items, None)
        if item is None:
            return None
        return Emission({"value": float(item)})

    def close(self) -> None:
        pass


CATALOGUE = Catalogue([Emitter, After, React, ReactWithPayload], sources=[Feed])


def active_definition(*sources: str) -> dict:
    """``emitter`` then ``after`` over every source's ``value``."""
    names = sources or ("cam_a",)
    definition = {
        "version": "2.0",
        "inputs": [],
        "sources": [{"type": Feed.type, "name": name, "feed": name} for name in names],
        "steps": [
            {
                "type": Emitter.type,
                "name": "emitter",
                "value": f"$sources.{names[0]}.value",
            },
            {"type": After.type, "name": "after", "value": "$steps.emitter.value"},
        ],
        "outputs": [],
    }

    return definition


PASSIVE = {
    "version": "2.0",
    "inputs": [{"type": "WorkflowParameter", "name": "value"}],
    "steps": [
        {"type": Emitter.type, "name": "emitter", "value": "$inputs.value"},
        {"type": After.type, "name": "after", "value": "$steps.emitter.value"},
    ],
    "outputs": [
        {"type": "JsonField", "name": "value", "selector": "$steps.after.value"}
    ],
}

HANDLER = {
    "version": "2.0",
    "inputs": [{"type": "WorkflowParameter", "name": "value"}],
    "steps": [{"type": React.type, "name": "react", "value": "$inputs.value"}],
    "outputs": [{"type": "JsonField", "name": "out", "selector": "$steps.react.value"}],
}

PAYLOAD_HANDLER = {
    "version": "2.0",
    "inputs": [
        {"type": "WorkflowParameter", "name": "value"},
        {"type": "WorkflowParameter", "name": "payload"},
    ],
    "steps": [
        {
            "type": ReactWithPayload.type,
            "name": "react",
            "value": "$inputs.value",
            "payload": "$inputs.payload",
        }
    ],
    "outputs": [{"type": "JsonField", "name": "out", "selector": "$steps.react.value"}],
}


def handler(
    name: str = "notify",
    *,
    mode: str = "async",
    depth: int = 1,
    overflow: str = "synchronous",
    workflow: Optional[dict] = None,
    bindings: Optional[Mapping[str, str]] = None,
) -> PlannedHandler:
    """A handler of ``$steps.emitter.events.seen`` with a hand-built plan."""
    planned = PlannedHandler(
        path=(name,),
        origin=EventOrigin(kind="step", event="seen", path=("emitter",)),
        event=SEEN,
        plan=compile_workflow(workflow or HANDLER, catalogue=CATALOGUE),
        bindings=bindings or {"value": "value"},
        mode=mode,
        queue=(
            QueuePolicy(max_depth=depth, overflow=overflow) if mode == "async" else None
        ),
    )

    return planned


def reacting(
    definition: dict,
    *handlers: PlannedHandler,
    groups: Sequence[PlannedHandlerGroup] = (),
) -> Any:
    """Compile ``definition`` and attach handlers (until the JSON schema lands)."""
    plan = compile_workflow(definition, catalogue=CATALOGUE)
    reactions = ReactionPlan(handlers=tuple(handlers), groups=tuple(groups))
    reacting_plan = dataclasses.replace(plan, reactions=reactions)

    return reacting_plan


def start(
    plan: Any,
    probe: Probe,
    feeds: Dict[str, list],
    *,
    observer: Optional[ReactionObserver] = None,
    **options: Any,
) -> Any:
    session = plan.create_session(
        resources={"probe": probe, "feeds": feeds}, reaction_observer=observer
    )
    run = session.start(**options)

    return run


def eventually(predicate: Callable[[], bool]) -> None:
    """Poll engine state until ``predicate`` holds; fail after ``WAIT``."""
    deadline = time.monotonic() + WAIT
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached"
        time.sleep(0.001)


class Outcomes(ReactionObserver):
    """Records outcomes and checks that callbacks never overlap."""

    def __init__(self) -> None:
        self.items: List[tuple] = []
        self.threads: List[str] = []
        self.active = 0
        self.overlapped = False
        self.lock = threading.Lock()

    def on_reaction_finished(self, *, outcome, error, result):
        with self.lock:
            self.active += 1
            self.overlapped |= self.active > 1
        time.sleep(0.001)
        self.items.append((outcome.status, outcome.cause.pulse, error, result))
        self.threads.append(threading.current_thread().name)
        with self.lock:
            self.active -= 1


# Synchronous handlers -------------------------------------------------------


def test_sync_handler_completes_inside_emit_before_the_step_continues():
    probe = Probe()
    plan = reacting(PASSIVE, handler(mode="sync"))
    session = plan.create_session(resources={"probe": probe})

    result = session.run({"value": 2.0})

    assert [item[:2] for item in probe.log] == [
        ("emit", 2.0),
        ("react", 2.0),
        ("reacted", 2.0),
        ("emitted", 2.0),
        ("after", 2.0),
    ]
    assert len({thread for _, _, thread in probe.log}) == 1
    assert result.rows()[0]["value"] == 2.0


def test_sync_handler_failure_fails_the_emitting_step_with_attribution():
    probe = Probe()
    probe.failing.add(3.0)
    plan = reacting(PASSIVE, handler(mode="sync"))
    session = plan.create_session(resources={"probe": probe})

    with pytest.raises(StepExecutionError) as caught:
        session.run({"value": 3.0})

    assert caught.value.step_path == ("emitter",)
    reaction = caught.value.__cause__
    assert isinstance(reaction, ReactionError)
    assert reaction.handler == NOTIFY
    assert reaction.emitter == ("emitter",)
    assert reaction.event == "seen"
    assert reaction.step_path == ("react",)
    assert "handler rejects 3.0" in str(reaction)
    assert probe.values("emitted") == [] and probe.values("after") == []


def test_sync_handler_holds_the_emitting_stage_in_a_pipelined_run():
    probe = Probe()
    gate = probe.hold(0.0)
    plan = reacting(active_definition(), handler(mode="sync"))
    run = start(
        plan, probe, {"cam_a": [0, 1, 2]}, pipeline=PipelineOptions(max_in_flight=3)
    )

    probe.started(0.0).wait(WAIT)
    # Pulse 0 holds the emitter's turn: later pulses wait, pulse 0 waits too.
    assert probe.values("emit") == [0.0]
    assert probe.values("after") == []
    gate.set()
    assert run.wait(WAIT)

    emitted = [
        (kind, value) for kind, value, _ in probe.log if kind in ("emit", "reacted")
    ]
    assert emitted == [
        ("emit", 0.0), ("reacted", 0.0),
        ("emit", 1.0), ("reacted", 1.0),
        ("emit", 2.0), ("reacted", 2.0),
    ]  # fmt: skip
    threads = probe.threads("react")
    assert all(
        name.startswith("workflows-v2-pipeline") or "worker" in name or "v2" in name
        for name in threads.values()
    )
    assert probe.threads("react") == {
        value: thread for value, thread in probe.threads("emit").items()
    }
    counters = run.reaction_counters[NOTIFY]
    assert counters.completed == counters.inline == 3
    assert 0 < counters.inline_max_seconds <= counters.inline_seconds
    assert counters.snapshots == 0
    assert probe.max_active == 1


# Asynchronous queues ----------------------------------------------------------


def test_synchronous_overflow_finishes_earlier_events_first_then_runs_on_the_emitter():
    probe = Probe()
    gate = probe.hold(0.0)
    feeds = {"cam_a": [0, probe.started(0.0), 1, 2, 3]}
    plan = reacting(active_definition(), handler(depth=1, overflow="synchronous"))
    run = start(plan, probe, feeds)

    # E0 runs (held), E1 is queued, E2's emitter waits for its turn.
    eventually(lambda: run.reaction_counters[NOTIFY].blocked == 1)
    counters = run.reaction_counters[NOTIFY]
    assert (counters.running, counters.pending, counters.overflow_sync) == (1, 1, 1)
    assert probe.values("emitted") == [0.0, 1.0]
    assert probe.values("react") == [0.0]
    gate.set()
    assert run.wait(WAIT)

    assert probe.values("react") == [0.0, 1.0, 2.0, 3.0]
    assert probe.max_active == 1
    threads = probe.threads("react")
    emitter_thread = probe.threads("emit")[2.0]
    assert threads[2.0] == emitter_thread
    assert (
        threads[0.0]
        == threads[1.0]
        == threads[3.0]
        == f"workflows-v2-reaction-{NOTIFY}"
    )
    counters = run.reaction_counters[NOTIFY]
    assert (counters.emitted, counters.completed, counters.overflow_sync) == (4, 4, 1)
    assert (counters.pending, counters.running, counters.blocked) == (0, 0, 0)
    assert counters.inline == 1 and counters.inline_seconds > 0
    assert (counters.snapshots, counters.snapshot_unknown_sizes) == (4, 0)
    assert counters.max_pending == 1


def test_leaky_overflow_drops_the_oldest_queued_event_and_keeps_the_newest():
    probe = Probe()
    gate = probe.hold(0.0)
    outcomes = Outcomes()
    feeds = {"cam_a": [0, probe.started(0.0), 1, 2]}
    plan = reacting(active_definition(), handler(depth=1, overflow="leaky"))
    run = start(plan, probe, feeds, observer=outcomes)

    probe.wait_for(lambda: probe.values("emitted") == [0.0, 1.0, 2.0])
    assert run.reaction_counters[NOTIFY].dropped == 1
    gate.set()
    assert run.wait(WAIT)

    assert probe.values("react") == [0.0, 2.0]
    counters = run.reaction_counters[NOTIFY]
    assert (counters.emitted, counters.completed, counters.dropped) == (3, 2, 1)
    statuses = [
        (item.status, item.cause.pulse.sequence) for item in run.reaction_outcomes()
    ]
    assert statuses == [("dropped", 1), ("completed", 0), ("completed", 2)]
    assert [status for status, *_ in outcomes.items] == [
        "dropped",
        "completed",
        "completed",
    ]
    assert not outcomes.overlapped


def test_async_handler_runs_beside_the_main_flow_on_its_own_worker():
    probe = Probe()
    gate = probe.hold(0.0)
    plan = reacting(active_definition(), handler(depth=4))
    run = start(plan, probe, {"cam_a": [0, 1, 2]})

    # The main flow finishes every pulse while the handler holds event 0.
    probe.wait_for(lambda: len(probe.values("after")) == 3)
    assert probe.values("react") == [0.0]
    assert not run.done
    gate.set()
    assert run.wait(WAIT)

    assert probe.values("reacted") == [0.0, 1.0, 2.0]
    assert set(probe.threads("react").values()) == {f"workflows-v2-reaction-{NOTIFY}"}


def test_handler_keeps_the_original_cause_source_and_time_with_a_scalar_binding():
    probe = Probe()
    plan = reacting(active_definition("cam_a"), handler(depth=4))
    run = start(plan, probe, {"cam_a": [5]})
    assert run.wait(WAIT)

    ((value, cause, sample, _),) = probe.seen
    assert value == 5.0
    assert cause.event == "seen" and cause.emitter == ("emitter",)
    assert cause.pulse == PulseKey(run.run_id, "cam_a", 0)
    assert cause.index == ()
    assert cause.source_id == "cam_a" and cause.temporal is not None
    assert sample is not None and sample.source_id == "cam_a"
    assert cause.describe()["pulse"] == {"source": "cam_a", "sequence": 0}


def test_handler_output_group_is_delivered_with_its_cause():
    probe = Probe()
    group = PlannedHandlerGroup(
        name="notifications", handler=("notify",), fields={"tenfold": "out"}
    )
    plan = reacting(active_definition(), handler(depth=4), groups=[group])
    results: List[Any] = []
    run = start(
        plan, probe, {"cam_a": [1, 2]}, handlers={"notifications": results.append}
    )
    assert run.wait(WAIT)

    assert [result.rows()[0] for result in results] == [
        {"tenfold": 10.0},
        {"tenfold": 20.0},
    ]
    assert [result.source for result in results] == [NOTIFY, NOTIFY]
    assert [result.pulse.sequence for result in results] == [0, 1]
    assert [result.causes for result in results] == [
        (PulseKey(run.run_id, "cam_a", 0),),
        (PulseKey(run.run_id, "cam_a", 1),),
    ]


def test_async_handler_failure_is_reported_and_the_run_continues():
    probe = Probe()
    probe.failing.add(1.0)
    outcomes = Outcomes()
    plan = reacting(active_definition(), handler(depth=4))
    run = start(plan, probe, {"cam_a": [0, 1, 2]}, observer=outcomes)

    assert run.wait(WAIT)
    assert run.state == "finished"
    counters = run.reaction_counters[NOTIFY]
    assert (counters.completed, counters.failed) == (2, 1)
    failed = [item for item in run.reaction_outcomes() if item.status == "failed"]
    assert len(failed) == 1
    assert failed[0].step == ("react",)
    assert "handler rejects 1.0" in failed[0].error
    errors = [error for status, _, error, _ in outcomes.items if status == "failed"]
    assert isinstance(errors[0], StepExecutionError)
    assert set(outcomes.threads) == {f"workflows-v2-reaction-{NOTIFY}"}


def test_observer_callbacks_of_handler_threads_are_serialized():
    probe = Probe()
    outcomes = Outcomes()
    plan = reacting(
        active_definition(),
        handler("notify", depth=32),
        handler("audit", depth=32),
    )
    run = start(plan, probe, {"cam_a": list(range(20))}, observer=outcomes)

    assert run.wait(WAIT)
    assert len(outcomes.items) == 40
    assert not outcomes.overlapped
    assert {
        "workflows-v2-reaction-$handlers.notify",
        "workflows-v2-reaction-$handlers.audit",
    } == set(outcomes.threads)


def test_sync_handler_serializes_passive_pipeline_submissions():
    probe = Probe()
    gate = probe.hold(0.0)
    plan = reacting(PASSIVE, handler(mode="sync"))
    session = plan.create_session(resources={"probe": probe})

    with session.pipeline(options=PipelineOptions(max_in_flight=3)) as pipeline:
        futures = [pipeline.submit({"value": float(value)}) for value in range(3)]
        probe.started(0.0).wait(WAIT)
        assert probe.values("react") == [0.0]
        gate.set()
        rows = [future.result(WAIT).rows()[0]["value"] for future in futures]

    assert rows == [0.0, 1.0, 2.0]
    assert probe.values("react") == [0.0, 1.0, 2.0]
    assert probe.max_active == 1


def test_a_held_sync_handler_does_not_hold_other_sources_back():
    probe = Probe()
    gate = probe.hold(0.0)
    definition = active_definition("cam_a", "cam_b")
    definition["steps"].append(
        {"type": After.type, "name": "other", "value": "$sources.cam_b.value"}
    )
    plan = reacting(definition, handler(mode="sync"))
    feeds = {"cam_a": [0, 1], "cam_b": [10, 11, 12]}
    run = start(plan, probe, feeds, pipeline=PipelineOptions(max_in_flight=4))

    # cam_a's pulse 0 holds the emitter; cam_b's own route completes meanwhile.
    probe.wait_for(lambda: probe.values("after") == [10.0, 11.0, 12.0])
    assert probe.values("emit") == [0.0]
    gate.set()
    assert run.wait(WAIT)

    assert probe.values("reacted") == [0.0, 1.0]


def test_overflow_waiting_emitter_and_handler_callbacks_do_not_deadlock():
    probe = Probe()
    gate = probe.hold(0.0)
    outcomes = Outcomes()
    group = PlannedHandlerGroup(
        name="notifications", handler=("notify",), fields={"out": "out"}
    )
    plan = reacting(
        active_definition(), handler(depth=1, overflow="synchronous"), groups=[group]
    )
    delivered: List[Any] = []
    feeds = {"cam_a": [0, probe.started(0.0), 1, 2, 3]}
    run = start(
        plan,
        probe,
        feeds,
        observer=outcomes,
        handlers={"notifications": delivered.append},
        pipeline=PipelineOptions(max_in_flight=2),
    )

    eventually(lambda: run.reaction_counters[NOTIFY].blocked == 1)
    gate.set()
    assert run.wait(WAIT)

    assert [result.rows()[0]["out"] for result in delivered] == [0.0, 10.0, 20.0, 30.0]
    assert [status for status, *_ in outcomes.items] == ["completed"] * 4
    assert not outcomes.overlapped


CHILD = {
    "version": "2.0",
    "inputs": [{"type": "WorkflowParameter", "name": "x"}],
    "steps": [{"type": Emitter.type, "name": "emitter", "value": "$inputs.x"}],
    "outputs": [{"type": "JsonField", "name": "y", "selector": "$steps.emitter.value"}],
    "handlers": [
        {
            "name": "notify",
            "on": "$steps.emitter.events.seen",
            "execution": {"mode": "sync"},
            "bindings": {"value": "$event.value"},
            "workflow": HANDLER,
        }
    ],
}


def test_nested_handlers_keep_distinct_identities_on_diamond_reuse():
    definition = {
        "version": "2.0",
        "inputs": [],
        "sources": [{"type": Feed.type, "name": "cam_a", "feed": "cam_a"}],
        "steps": [
            {
                "type": "roboflow_core/inner_workflow@v1",
                "name": side,
                "workflow_definition": CHILD,
                "parameter_bindings": {"x": "$sources.cam_a.value"},
            }
            for side in ("left", "right")
        ],
        "handlers": [
            {
                "name": "audit",
                "on": "$steps.left/emitter.events.seen",
                "execution": {
                    "mode": "async",
                    "queue": {"max_depth": 4, "overflow": "leaky"},
                },
                "bindings": {"value": "$event.value"},
                "workflow": HANDLER,
            }
        ],
        "outputs": [
            {
                "type": "OutputGroup",
                "name": "audits",
                "anchor": "$handlers.audit",
                "outputs": [
                    {
                        "type": "JsonField",
                        "name": "out",
                        "selector": "$handlers.audit.out",
                    }
                ],
            }
        ],
    }
    plan = compile_workflow(definition, catalogue=CATALOGUE)
    probe = Probe()
    delivered: List[Any] = []
    run = start(plan, probe, {"cam_a": [1, 2]}, handlers={"audits": delivered.append})
    assert run.wait(WAIT)

    counters = run.reaction_counters
    assert sorted(counters) == [
        "$handlers.audit",
        "$handlers.left/notify",
        "$handlers.right/notify",
    ]
    assert all(item.completed == 2 for item in counters.values())
    emitters = sorted((value, cause.emitter) for value, cause, _, _ in probe.seen)
    assert emitters == [
        (1.0, ("left", "emitter")),
        (1.0, ("left", "emitter")),
        (1.0, ("right", "emitter")),
        (2.0, ("left", "emitter")),
        (2.0, ("left", "emitter")),
        (2.0, ("right", "emitter")),
    ]  # fmt: skip
    assert [result.rows()[0]["out"] for result in delivered] == [10.0, 20.0]
    assert {result.source for result in delivered} == {"$handlers.audit"}
