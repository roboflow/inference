"""Fixture blocks for the live-control tests (``test_controls.py``).

Every block records its calls in ``self.calls`` (one instance per step and
session), so a test reads exact call counts. ``Tracker`` is the stateful,
resettable member; ``HeldPainter`` is the phased, prunable member a test can
hold between its phases to build deterministic barriers.
"""

import threading
from typing import Any, Dict, List

from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Output,
    Ref,
)
from roboflow_workflows.execution_engine.v2.kinds import (
    FLOAT_KIND,
    INTEGER_KIND,
    STRING_KIND,
)
from roboflow_workflows.execution_engine.v2.phases import phase

WAIT = 10.0


class Recording(Block):
    """Abstract base: remembers every call's arguments and ``wants`` answers."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.asked: List[Dict[str, Any]] = []

    def remember(self, **arguments: Any) -> None:
        self.calls.append(arguments)


class Tracker(Recording):
    """Stateful, not prunable, resettable: counts calls since the last reset."""

    type = "test/m7c_tracker@v1"
    outputs = {"ticks": Output(INTEGER_KIND), "value": Output(FLOAT_KIND)}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)

    def __init__(self) -> None:
        super().__init__()
        self.ticks = 0
        self.resets = 0

    def reset_state(self) -> None:
        self.ticks = 0
        self.resets += 1

    def run(self, *, value) -> dict:
        self.remember(value=value)
        self.ticks += 1
        self.asked.append({"value": self.wants("value"), "ticks": self.wants("ticks")})
        return {"ticks": self.ticks, "value": value}


class Thresholder(Recording):
    """Prunable; reads a (controllable) threshold and reports what it saw."""

    type = "test/m7c_thresholder@v1"
    prunable = True
    outputs = {"above": Output(INTEGER_KIND), "threshold": Output(FLOAT_KIND)}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)
        threshold: float | Ref(FLOAT_KIND) = 0.5

    def run(self, *, value, threshold) -> dict:
        self.remember(value=value, threshold=threshold)
        return {"above": int(value >= threshold), "threshold": threshold}


class HeldPainter(Recording):
    """Prunable, phased; ``first`` may block on a gate until the test releases it.

    Both phases ask ``wants("extra")`` and read ``threshold``, so a test can
    assert that one call never mixes two control versions even when an update
    is published while the call is held between its phases.
    """

    type = "test/m7c_held_painter@v1"
    prunable = True
    outputs = {"value": Output(FLOAT_KIND), "extra": Output(STRING_KIND)}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)
        threshold: float | Ref(FLOAT_KIND) = 0.5
        hold: float = -1.0

    def __init__(self, *, gates: Dict[str, threading.Event]) -> None:
        super().__init__()
        self.gates = gates
        self.phases: List[Dict[str, Any]] = []

    @phase
    def first(self, *, value, threshold, hold):
        self.remember(value=value, threshold=threshold)
        self.phases.append(
            {
                "phase": "first",
                "value": value,
                "extra": self.wants("extra"),
                "threshold": threshold,
            }
        )
        if value == hold:
            self.gates["entered"].set()
            self.gates["release"].wait(WAIT)
        return value * 10

    @phase
    def second(self, *, first, threshold):
        wanted = self.wants("extra")
        self.phases.append(
            {
                "phase": "second",
                "value": first / 10,
                "extra": wanted,
                "threshold": threshold,
            }
        )
        result: Dict[str, Any] = {"value": first}
        if wanted:
            result["extra"] = f"extra:{first:g}"
        return result

    def run(self, *, value, threshold, hold):
        return self.second(
            first=self.first(value=value, threshold=threshold, hold=hold),
            threshold=threshold,
        )


class Remembering(Recording):
    """Not prunable; requests managed state (never reset by a control)."""

    type = "test/m7c_remembering@v1"
    outputs = {"value": Output(FLOAT_KIND)}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)

    def __init__(self, *, managed_state) -> None:
        super().__init__()
        self.state = managed_state

    def run(self, *, value) -> dict:
        self.remember(value=value)
        return {"value": value}


def step(block: type, name: str, **params: Any) -> dict:
    """Build a step of one fixture block."""
    return {"type": block.type, "name": name, **params}


def controls(**declared: dict) -> dict:
    """Build a root ``controls`` section."""
    return dict(declared)


def enable(*steps: str, **options: Any) -> dict:
    """Build one ``enable`` control."""
    return {"type": "enable", "steps": list(steps), **options}


def input_control(name: str, **options: Any) -> dict:
    """Build one ``input`` control."""
    return {"type": "input", "input": f"$inputs.{name}", **options}
