"""Catalogue, journal and inspection helpers shared by machine runtime tests."""

import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from roboflow_workflows.v2.blocks.state.state_machine_set.v1 import StateMachineSetBlock
from roboflow_workflows.v2.data.kinds import (
    FLOAT_KIND,
    STRING_KIND,
    WILDCARD_KIND,
)
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    Output,
    Ref,
)
from roboflow_workflows.v2.engine.events import Event
from roboflow_workflows.v2.engine.reactions.dispatch import EventCause

from tests.v2.support.reactions.feeds import WAIT, Feed

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
