"""Blocks, a source and definition builders for graph update tests.

Every block takes a ``log`` resource and records its construction there, so a
test sees exactly which instances an update built. ``Count`` and ``Scale``
also take the ``model`` resource, which tests pass as a session ``Factory``
to check that an update never creates it again.
"""

import threading
from typing import Any, Dict, List, Optional

from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Output,
    Ref,
)
from roboflow_workflows.execution_engine.v2.implementations import Implementation
from roboflow_workflows.execution_engine.v2.kinds import (
    FLOAT_KIND,
    INTEGER_KIND,
    STRING_KIND,
)
from roboflow_workflows.execution_engine.v2.sources import (
    Emission,
    Source,
    SourceOutput,
    SourceParams,
)

WAIT = 10.0
"""Upper bound of every wait, in seconds; the tests finish far earlier."""


class Count(Block):
    """Stateful: counts its calls over the lifetime of its instance."""

    type = "test/update_count@v1"
    outputs = {"count": Output(INTEGER_KIND)}

    class Params(BlockParams):
        value: Ref()

    def __init__(self, *, log: List[Any], model: Any) -> None:
        log.append(("init", "count", id(self)))
        self.model = model
        self.count = 0

    def run(self, *, value) -> dict:
        self.count += 1
        return {"count": self.count}


class Scale(Block):
    """Prunable: multiplies a value by ``factor``."""

    type = "test/update_scale@v1"
    outputs = {"scaled": Output(FLOAT_KIND)}
    prunable = True

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)
        factor: float | Ref(FLOAT_KIND) = 2.0

    def __init__(self, *, log: List[Any], model: Any) -> None:
        log.append(("init", "scale", id(self)))
        self.model = model

    def run(self, *, value, factor) -> dict:
        return {"scaled": value * factor}


class Accurate(Implementation):
    name = "accurate"
    quality = ("accurate",)

    def run(self, *, value) -> dict:
        return {"label": "accurate"}


class Fast(Implementation):
    name = "fast"
    quality = ("fast",)

    def run(self, *, value) -> dict:
        return {"label": "fast"}


class Label(Block):
    """Two implementations chosen by quality label."""

    type = "test/update_label@v1"
    outputs = {"label": Output(STRING_KIND)}
    implementations = (Accurate, Fast)

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)


class Broken(Block):
    """Its constructor always fails."""

    type = "test/update_broken@v1"
    outputs = {"value": Output()}

    class Params(BlockParams):
        value: Ref()

    def __init__(self, *, log: List[Any]) -> None:
        log.append(("init", "broken", id(self)))
        raise RuntimeError("cannot load")

    def run(self, *, value) -> dict:
        return {"value": value}


class Mutator(Block):
    """Declares an in-place mutation of its payload."""

    type = "test/update_mutator@v1"
    outputs = {"payload": Output()}
    mutates = ("payload",)

    class Params(BlockParams):
        payload: Ref()

    def run(self, *, payload) -> dict:
        return {"payload": payload}


class Stateful(Block):
    """Asks for managed state."""

    type = "test/update_stateful@v1"
    outputs = {"value": Output()}

    class Params(BlockParams):
        value: Ref()

    def __init__(self, *, managed_state: Any) -> None:
        self.state = managed_state

    def run(self, *, value) -> dict:
        return {"value": value}


class Hold(Block):
    """Signals ``entered`` and waits for ``release`` inside ``run``."""

    type = "test/update_hold@v1"
    outputs = {"value": Output()}

    class Params(BlockParams):
        value: Ref()

    def __init__(self, *, gates: Dict[str, threading.Event]) -> None:
        self.gates = gates
        self.inside: Optional[Any] = None

    def run(self, *, value) -> dict:
        self.gates["entered"].set()
        inside = self.gates.get("inside")
        if inside is not None:
            self.inside = inside()
        assert self.gates["release"].wait(WAIT)
        return {"value": value}


class Ticks(Source):
    """Emits ``count`` values; waits on ``gates[after]`` after ``after`` of them."""

    type = "test/update_ticks@v1"
    outputs = {"value": SourceOutput(FLOAT_KIND)}

    class Params(SourceParams):
        count: int = 3
        after: int = -1

    def __init__(self, *, gates: Dict[str, threading.Event], log: List[Any]) -> None:
        self.gates = gates
        self.log = log
        self.emitted = 0
        self.count = 0
        self.after = -1

    def open(self, *, count, after) -> None:
        self.log.append(("open", "ticks", id(self)))
        self.count = count
        self.after = after

    def read(self) -> Optional[Emission]:
        if self.emitted == self.after:
            self.gates["paused"].set()
            assert self.gates["resume"].wait(WAIT)
        if self.emitted >= self.count:
            return None
        self.emitted += 1
        return Emission({"value": float(self.emitted)})

    def close(self) -> None:
        self.log.append(("close", "ticks", id(self)))


CATALOGUE = Catalogue(
    [Count, Scale, Label, Broken, Mutator, Stateful, Hold], sources=[Ticks]
)


def step(block: type, name: str, **params: Any) -> dict:
    return {"type": block.type, "name": name, **params}


def passive(steps: List[dict], outputs: Dict[str, str], **sections: Any) -> dict:
    """A passive definition with one float parameter ``value``."""
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowParameter", "name": "value", "kind": ["float"]}],
        "steps": steps,
        "outputs": [
            {"type": "JsonField", "name": name, "selector": selector}
            for name, selector in outputs.items()
        ],
        **sections,
    }

    return definition


def group(name: str, **fields: str) -> dict:
    return {
        "type": "OutputGroup",
        "name": name,
        "anchor": "$sources.ticks.value",
        "outputs": [
            {"type": "JsonField", "name": field, "selector": selector}
            for field, selector in fields.items()
        ],
    }


def active(steps: List[dict], groups: List[dict], **source_params: Any) -> dict:
    """An active definition reading the ``Ticks`` source named ``ticks``."""
    definition = {
        "version": "2.0",
        "inputs": [],
        "sources": [{"type": Ticks.type, "name": "ticks", **source_params}],
        "steps": steps,
        "outputs": groups,
    }

    return definition


def child(steps: List[dict], outputs: Dict[str, str], inputs: List[dict]) -> dict:
    definition = {
        "version": "2.0",
        "inputs": inputs,
        "steps": steps,
        "outputs": [
            {"type": "JsonField", "name": name, "selector": selector}
            for name, selector in outputs.items()
        ],
    }

    return definition


def nested(name: str, definition: dict, **bindings: Any) -> dict:
    return {
        "type": "roboflow_core/inner_workflow@v1",
        "name": name,
        "workflow_definition": definition,
        "parameter_bindings": bindings,
    }
