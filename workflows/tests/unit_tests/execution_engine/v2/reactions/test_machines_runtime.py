"""State machines in real engine runs: fixed and handler-selected transitions.

Every scenario compiles a JSON definition with ``state_machines`` and runs it
through ``session.start`` (or ``session.run``). Ordering is forced with gates
the test controls; waits are bounded by ``WAIT``.

The helpers here (journal, blocks, ``inspection``) are shared by
``test_signals.py`` and ``test_cascades.py``.
"""

import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pytest
from roboflow_workflows.execution_engine.v2.blocks.state_machine import (
    StateMachineSetBlock,
)
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Output,
    Ref,
)
from roboflow_workflows.execution_engine.v2.errors import ContractError
from roboflow_workflows.execution_engine.v2.events import Event
from roboflow_workflows.execution_engine.v2.kinds import (
    FLOAT_KIND,
    STRING_KIND,
    WILDCARD_KIND,
)
from roboflow_workflows.execution_engine.v2.reactions import EventOrigin
from roboflow_workflows.execution_engine.v2.reactions.dispatch import EventCause
from roboflow_workflows.execution_engine.v2.reactions.machines import (
    MachineRuntime,
    MachineStamp,
    TransitionCounters,
)
from roboflow_workflows.execution_engine.v2.reactions.runtime import session_reactions
from roboflow_workflows.execution_engine.v2.reactions.testing import block_call
from roboflow_workflows.execution_engine.v2.state import ManagedState

from .test_runtime import WAIT, Feed, eventually

REVIEW = Event({"score": FLOAT_KIND, "frame": []}, description="Needs a review.")


class Journal:
    """What blocks saw, plus gates: ``hold(label, value)`` parks that call."""

    def __init__(self) -> None:
        self.changed = threading.Condition()
        self.items: List[Tuple[str, Any, Optional[EventCause], str]] = []
        self.gates: Dict[Tuple[str, Any], threading.Event] = {}
        self.entered: Dict[Tuple[str, Any], threading.Event] = {}
        self.frames: Dict[float, Any] = {}
        self.subscribed: List[bool] = []
        self.failing: set = set()
        self.on_record: Optional[Callable[[str, Any], None]] = None

    def hold(self, label: str, value: Any) -> threading.Event:
        return self.gates.setdefault((label, value), threading.Event())

    def started(self, label: str, value: Any) -> threading.Event:
        return self.entered.setdefault((label, value), threading.Event())

    def visit(self, label: str, value: Any, context: Any) -> None:
        with self.changed:
            self.items.append(
                (label, value, context.cause, threading.current_thread().name)
            )
            self.changed.notify_all()
        if isinstance(value, (str, float)):
            self.started(label, value).set()
            gate = self.gates.get((label, value))
            if gate is not None:
                assert gate.wait(WAIT)
        if self.on_record is not None:
            self.on_record(label, value)
        if label in self.failing:
            raise RuntimeError(f"{label} fails")

    def entries(self, label: str) -> List[Tuple[str, Any, Optional[EventCause], str]]:
        with self.changed:
            return [item for item in self.items if item[0] == label]

    def values(self, label: str) -> List[Any]:
        return [item[1] for item in self.entries(label)]


class Inspect(Block):
    """Emits ``review_requested`` per score; mutates an array frame afterwards."""

    type = "test/inspect@v1"
    outputs = {"score": Output(FLOAT_KIND)}
    events = {"review_requested": REVIEW}

    class Params(BlockParams):
        score: Ref(FLOAT_KIND)

    def __init__(self, *, journal: Journal) -> None:
        self.journal = journal

    def run(self, score):
        frame = self.journal.frames.get(score, score)
        self.journal.subscribed.append(self.has_subscribers("review_requested"))
        self.journal.visit("inspect", score, self.execution_context)
        self.emit("review_requested", score=score, frame=frame)
        if isinstance(frame, np.ndarray):
            frame[...] = -1
        return {"score": score}


class Pick(Block):
    """Decides a review: approved from 0.5, rejected below, ``bogus`` if negative."""

    type = "test/pick@v1"
    outputs = {"decision": Output(STRING_KIND)}

    class Params(BlockParams):
        score: Ref(FLOAT_KIND)

    def __init__(self, *, journal: Journal) -> None:
        self.journal = journal

    def run(self, score):
        self.journal.visit("pick", score, self.execution_context)
        if score < 0:
            return {"decision": "bogus"}
        return {"decision": "approved" if score >= 0.5 else "rejected"}


class Record(Block):
    """Records its value under ``label``; may wait on a gate."""

    type = "test/record@v1"
    outputs = {"value": Output(WILDCARD_KIND)}

    class Params(BlockParams):
        label: str
        value: Ref(WILDCARD_KIND)

    def __init__(self, *, journal: Journal) -> None:
        self.journal = journal

    def run(self, label, value):
        self.journal.visit(label, value, self.execution_context)
        return {"value": value}


CATALOGUE = Catalogue([Inspect, Pick, Record, StateMachineSetBlock], sources=[Feed])

DECIDED = "$state_machines.inspection_state.events.decided"
DECISION_REQUESTED = "$state_machines.inspection_state.events.decision_requested"


def recorder(label: str, field: str = "value", kind: str = "*") -> dict:
    """A handler workflow recording its one input under ``label``."""
    workflow = {
        "inputs": [{"name": field, "kind": [kind]}],
        "steps": [
            {
                "type": Record.type,
                "name": "record",
                "label": label,
                "value": f"$inputs.{field}",
            }
        ],
        "outputs": [
            {"type": "JsonField", "name": "out", "selector": "$steps.record.value"}
        ],
    }

    return workflow


def execution(mode: str) -> dict:
    return {"mode": mode} if mode == "sync" else {"mode": "async"}


def inspection(
    *,
    sources: Tuple[str, ...] = ("cam_a",),
    scope: str = "source",
    decide: str = "sync",
    audit: str = "async",
    handlers: Tuple[dict, ...] = (),
    decide_steps: Tuple[dict, ...] = (),
) -> dict:
    """Design draft 05: inspect -> review -> decide -> decided -> audit.

    ``reset`` (on ``$signals.ack``) and ``abort`` (on ``$signals.abort``)
    return the machine to ``idle``.
    """
    definition = {
        "version": "2.0",
        "inputs": [],
        "sources": [
            {"type": Feed.type, "name": name, "feed": name} for name in sources
        ],
        "signals": [
            {"name": "ack", "fields": {"note": ["string"]}},
            {"name": "abort", "fields": {}},
        ],
        "steps": [
            {
                "type": Inspect.type,
                "name": "inspection",
                "score": f"$sources.{sources[0]}.value",
            }
        ],
        "outputs": [],
        "state_machines": [
            {
                "name": "inspection_state",
                "scope": scope,
                "initial_state": "idle",
                "states": ["idle", "reviewing", "approved", "rejected"],
                "transitions": [
                    {
                        "name": "start_review",
                        "from": ["idle"],
                        "to": "reviewing",
                        "on": "$steps.inspection.events.review_requested",
                        "emit": {
                            "name": "decision_requested",
                            "fields": {
                                "score": "$event.score",
                                "frame": "$event.frame",
                            },
                        },
                    },
                    {
                        "name": "finish_review",
                        "from": ["reviewing"],
                        "to": ["approved", "rejected"],
                        "handler": "decide",
                        "emit": {
                            "name": "decided",
                            "fields": {"state": "$transition.to", "by": "decide"},
                        },
                    },
                    {
                        "name": "reset",
                        "from": ["approved", "rejected"],
                        "to": "idle",
                        "on": "$signals.ack",
                    },
                    {
                        "name": "abort",
                        "from": ["reviewing"],
                        "to": "idle",
                        "on": "$signals.abort",
                    },
                ],
            }
        ],
        "handlers": [
            {
                "name": "decide",
                "on": DECISION_REQUESTED,
                "execution": execution(decide),
                "bindings": {"score": "$event.score"},
                "workflow": {
                    "inputs": [{"name": "score", "kind": ["float"]}],
                    "steps": [
                        *decide_steps,
                        {"type": Pick.type, "name": "pick", "score": "$inputs.score"},
                        {
                            "type": StateMachineSetBlock.type,
                            "name": "finish",
                            "machine": "inspection_state",
                            "transition": "finish_review",
                            "next_state": "$steps.pick.decision",
                        },
                    ],
                    "outputs": [
                        {
                            "type": "JsonField",
                            "name": "outcome",
                            "selector": "$steps.finish.outcome",
                        }
                    ],
                },
            },
            {
                "name": "audit",
                "on": DECIDED,
                "execution": execution(audit),
                "bindings": {"state": "$event.state"},
                "workflow": recorder("audit", "state", "string"),
            },
            *handlers,
        ],
    }

    return definition


def start(definition: dict, journal: Journal, feeds: Dict[str, list], **options):
    plan = compile_workflow(definition, catalogue=CATALOGUE)
    session = plan.create_session(resources={"journal": journal, "feeds": feeds})
    run = session.start(**options)

    return run


def chain(cause: Optional[EventCause]) -> List[str]:
    """Origin selectors from ``cause`` up its parents."""
    selectors = []
    while cause is not None:
        selectors.append(cause.origin_selector)
        cause = cause.parent

    return selectors


# Fixed and handler-selected transitions -------------------------------------


@pytest.mark.parametrize("decide", ["sync", "async"])
def test_draft_inspection_cascade_reaches_the_audit_with_its_full_cause(decide):
    journal = Journal()
    run = start(inspection(decide=decide), journal, {"cam_a": [0.9]})
    assert run.wait(WAIT)

    assert run.machine_state("inspection_state", source_id="cam_a") == ("approved", 2)
    counters = run.machine_counters()
    assert counters["inspection_state.start_review"] == TransitionCounters(applied=1)
    assert counters["inspection_state.finish_review"] == TransitionCounters(applied=1)
    ((_, state, cause, audit_thread),) = journal.entries("audit")
    assert state == "approved"
    assert chain(cause) == [
        DECIDED,
        DECISION_REQUESTED,
        "$steps.inspection.events.review_requested",
    ]
    assert cause.stamp == MachineStamp(
        machine=("inspection_state",),
        source_id="cam_a",
        transition="finish_review",
        from_state="reviewing",
        to_state="approved",
        version=2,
    )
    assert cause.parent.stamp.transition == "start_review"
    assert cause.source_id == "cam_a" and cause.pulse.source == "cam_a"
    (inspect_thread,) = [item[3] for item in journal.entries("inspect")]
    (pick_thread,) = [item[3] for item in journal.entries("pick")]
    assert (pick_thread == inspect_thread) == (decide == "sync")
    assert audit_thread != inspect_thread
    assert run.reaction_counters["$handlers.audit"].completed == 1


def test_a_step_only_a_machine_listens_to_keeps_the_fields_its_transition_copies():
    journal = Journal()
    journal.frames[0.9] = np.arange(4.0)
    look = {
        "name": "look",
        "on": DECISION_REQUESTED,
        "execution": {"mode": "async"},
        "bindings": {"frame": "$event.frame"},
        "workflow": recorder("look", "frame"),
    }
    run = start(inspection(handlers=(look,)), journal, {"cam_a": [0.9]})
    assert run.wait(WAIT)

    assert journal.subscribed == [True]
    ((_, frame, _, _),) = journal.entries("look")
    np.testing.assert_array_equal(frame, np.arange(4.0))
    assert run.reaction_counters["$handlers.look"].snapshots == 1


def test_one_event_applies_at_most_one_transition_of_each_machine():
    journal = Journal()
    gate = threading.Event()
    definition = inspection()
    definition["state_machines"].append(
        {
            "name": "ladder",
            "scope": "global",
            "initial_state": "zero",
            "states": ["zero", "one", "two"],
            "transitions": [
                {
                    "name": name,
                    "from": [source],
                    "to": target,
                    "on": "$steps.inspection.events.review_requested",
                }
                for name, source, target in (
                    ("up1", "zero", "one"),
                    ("up2", "one", "two"),
                )
            ],
        }
    )
    run = start(definition, journal, {"cam_a": [0.9, gate, 0.9]})
    eventually(lambda: journal.values("inspect") == [0.9])
    eventually(lambda: run.machine_state("ladder") == ("one", 1))
    gate.set()
    assert run.wait(WAIT)

    assert run.machine_state("ladder") == ("two", 2)
    counters = run.machine_counters()
    assert counters["ladder.up1"] == TransitionCounters(applied=1, ignored=1)
    assert counters["ladder.up2"] == TransitionCounters(applied=1, ignored=1)


def test_the_controller_picks_one_transition_for_one_event_of_a_compiled_plan():
    definition = inspection()
    definition["state_machines"][0]["transitions"].append(
        {
            "name": "skip",
            "from": ["reviewing"],
            "to": "approved",
            "on": "$steps.inspection.events.review_requested",
        }
    )
    plan = compile_workflow(definition, catalogue=CATALOGUE)
    emitted: List[MachineStamp] = []
    machines = MachineRuntime(
        plan.reactions,
        ManagedState(),
        emit=lambda origin, fields, *, parent, stamp: emitted.append(stamp),
    )
    origin = EventOrigin(kind="step", event="review_requested", path=("inspection",))
    cause = _cause(origin, source_id="cam_a")
    machines.on_event(origin, cause, {"score": 0.9, "frame": 0.9})
    assert machines.current(("inspection_state",), source_id="cam_a") == (
        "reviewing",
        1,
    )
    machines.on_event(origin, cause, {"score": 0.9, "frame": 0.9})

    assert machines.current(("inspection_state",), source_id="cam_a") == (
        "approved",
        2,
    )
    assert [stamp.transition for stamp in emitted] == ["start_review"]


@pytest.mark.parametrize(
    "scope, expected",
    [
        ("source", {"cam_a": ("on", 1), "cam_b": ("off", 2)}),
        ("global", {None: ("on", 3)}),
    ],
)
def test_source_machines_are_per_source_and_global_machines_are_shared(scope, expected):
    journal = Journal()
    steps = [
        {
            "type": Inspect.type,
            "name": f"inspect_{name}",
            "score": f"$sources.{name}.value",
        }
        for name in ("cam_a", "cam_b")
    ]
    transitions = [
        {
            "name": f"{verb}_{name}",
            "from": [source],
            "to": target,
            "on": f"$steps.inspect_{name}.events.review_requested",
        }
        for name in ("cam_a", "cam_b")
        for verb, source, target in (("on", "off", "on"), ("off", "on", "off"))
    ]
    definition = {
        "version": "2.0",
        "inputs": [],
        "sources": [
            {"type": Feed.type, "name": name, "feed": name}
            for name in ("cam_a", "cam_b")
        ],
        "steps": steps,
        "outputs": [],
        "state_machines": [
            {
                "name": "toggle",
                "scope": scope,
                "initial_state": "off",
                "states": ["off", "on"],
                "transitions": transitions,
            }
        ],
    }
    run = start(definition, journal, {"cam_a": [1.0], "cam_b": [1.0, 2.0]})
    assert run.wait(WAIT)

    for source_id, state in expected.items():
        assert run.machine_state("toggle", source_id=source_id) == state


def test_a_stamped_decision_is_stale_after_the_machine_went_a_b_a():
    journal = Journal()
    held = journal.hold("pick", 0.9)
    gate = threading.Event()
    run = start(inspection(decide="async"), journal, {"cam_a": [0.9, gate, 0.8]})
    assert journal.started("pick", 0.9).wait(WAIT)
    run.signal("abort", source_id="cam_a")  # reviewing -> idle, v2
    gate.set()  # 0.8: idle -> reviewing again, v3
    eventually(
        lambda: run.machine_state("inspection_state", source_id="cam_a")
        == ("reviewing", 3)
    )
    held.set()
    assert run.wait(WAIT)

    counters = run.machine_counters()
    assert counters["inspection_state.finish_review"] == TransitionCounters(
        applied=1, stale=1
    )
    assert run.machine_state("inspection_state", source_id="cam_a") == ("approved", 4)
    ((_, state, cause, _),) = journal.entries("audit")
    assert cause.parent.stamp.version == 3
    assert state == "approved"


def test_a_target_outside_the_to_states_fails_the_handler_and_keeps_the_state():
    journal = Journal()
    run = start(inspection(decide="async"), journal, {"cam_a": [-1.0]})
    assert run.wait(WAIT)

    (outcome,) = [
        item for item in run.reaction_outcomes() if item.handler == "$handlers.decide"
    ]
    assert outcome.status == "failed" and "bogus" in outcome.error
    assert run.machine_state("inspection_state", source_id="cam_a") == ("reviewing", 1)
    assert run.machine_counters()["inspection_state.finish_review"] == (
        TransitionCounters()
    )
    assert journal.values("audit") == []


def test_a_handler_step_named_like_a_main_flow_step_does_not_trigger_its_transitions():
    journal = Journal()
    twin = {"type": Inspect.type, "name": "inspection", "score": "$inputs.score"}
    run = start(inspection(decide_steps=(twin,)), journal, {"cam_a": [0.9]})
    assert run.wait(WAIT)

    assert journal.values("inspect") == [0.9, 0.9]
    assert journal.subscribed == [True, False]
    counters = run.machine_counters()
    assert counters["inspection_state.start_review"] == TransitionCounters(applied=1)
    assert run.machine_state("inspection_state", source_id="cam_a") == ("approved", 2)


def test_set_machine_state_outside_a_handler_run_is_rejected():
    with block_call(StateMachineSetBlock()):
        with pytest.raises(ContractError, match="state machines"):
            StateMachineSetBlock().run(
                trigger=None, machine="m", transition="t", next_state="s"
            )


GATED_CHILD = {
    "version": "2.0",
    "inputs": [{"type": "WorkflowParameter", "name": "x"}],
    "steps": [{"type": Inspect.type, "name": "inspection", "score": "$inputs.x"}],
    "outputs": [
        {"type": "JsonField", "name": "y", "selector": "$steps.inspection.score"}
    ],
    "state_machines": [
        {
            "name": "gate",
            "scope": "global",
            "initial_state": "closed",
            "states": ["closed", "open"],
            "transitions": [
                {
                    "name": "open",
                    "from": ["closed"],
                    "to": "open",
                    "on": "$steps.inspection.events.review_requested",
                    "emit": {"name": "opened", "fields": {"to": "$transition.to"}},
                }
            ],
        }
    ],
}


def test_nested_machines_keep_distinct_records_on_diamond_reuse():
    journal = Journal()
    definition = {
        "version": "2.0",
        "inputs": [],
        "sources": [{"type": Feed.type, "name": "cam_a", "feed": "cam_a"}],
        "steps": [
            {
                "type": "roboflow_core/inner_workflow@v1",
                "name": side,
                "workflow_definition": GATED_CHILD,
                "parameter_bindings": {"x": "$sources.cam_a.value"},
            }
            for side in ("left", "right")
        ],
        "outputs": [],
        "handlers": [
            {
                "name": "audit",
                "on": "$state_machines.left/gate.events.opened",
                "execution": {"mode": "sync"},
                "bindings": {"to": "$event.to"},
                "workflow": recorder("audit", "to", "string"),
            }
        ],
    }
    run = start(definition, journal, {"cam_a": [1.0]})
    assert run.wait(WAIT)

    counters = run.machine_counters()
    assert counters["left/gate.open"] == TransitionCounters(applied=1)
    assert counters["right/gate.open"] == TransitionCounters(applied=1)
    assert run.machine_state("left/gate") == ("open", 1)
    assert run.machine_state("right/gate") == ("open", 1)
    ((_, state, cause, _),) = journal.entries("audit")
    assert state == "open" and cause.emitter == ("left", "gate")
    with pytest.raises(ContractError, match="gate"):
        run.machine_state("gate")


def test_a_passive_session_applies_fixed_transitions_and_runs_machine_handlers():
    journal = Journal()
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowParameter", "name": "score"}],
        "steps": [
            {"type": Inspect.type, "name": "inspection", "score": "$inputs.score"}
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "score",
                "selector": "$steps.inspection.score",
            }
        ],
        "state_machines": [
            {
                "name": "toggle",
                "scope": "global",
                "initial_state": "off",
                "states": ["off", "on"],
                "transitions": [
                    {
                        "name": name,
                        "from": [source],
                        "to": target,
                        "on": "$steps.inspection.events.review_requested",
                        "emit": {"name": "flipped", "fields": {"to": "$transition.to"}},
                    }
                    for name, source, target in (
                        ("on", "off", "on"),
                        ("off", "on", "off"),
                    )
                ],
            }
        ],
        "handlers": [
            {
                "name": "audit",
                "on": "$state_machines.toggle.events.flipped",
                "bindings": {"to": "$event.to"},
                "workflow": recorder("audit", "to", "string"),
            }
        ],
    }
    plan = compile_workflow(definition, catalogue=CATALOGUE)
    session = plan.create_session(resources={"journal": journal})
    session.run({"score": 1.0})
    session.run({"score": 2.0})

    assert journal.values("audit") == ["on", "off"]
    assert session_reactions(session).machine_state("toggle") == ("off", 2)


def _cause(origin: EventOrigin, *, source_id: str) -> EventCause:
    from roboflow_workflows.execution_engine.v2.data import SampleContext

    cause = EventCause(
        event=origin.event,
        emitter=origin.path,
        session_id="s",
        run_id="r",
        pulse=None,
        index=(),
        sample=SampleContext(source_id=source_id),
        temporal=None,
        sequence=0,
        kind=origin.kind,
    )

    return cause
