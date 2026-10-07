"""Small class-owned blocks for the demand (``prunable``, ``wants``) and quality tests.

Every block records each call's arguments in ``self.calls``; a session holds
one instance per step, so a test reads ``session.instances[path].calls`` for
exact call counts. Blocks that ask ``wants`` record the answers in ``asked``.
"""

from typing import Any, Dict, List

from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Output,
    Ref,
    Select,
    StepRef,
    Stop,
)
from roboflow_workflows.execution_engine.v2.events import Event
from roboflow_workflows.execution_engine.v2.implementations import Implementation
from roboflow_workflows.execution_engine.v2.kinds import (
    FLOAT_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
)
from roboflow_workflows.execution_engine.v2.phases import phase

SEEN = Event({"value": FLOAT_KIND}, description="A value passed the emitter.")


class Recording(Block):
    """Abstract base: remembers every call's arguments and ``wants`` answers."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.asked: List[Dict[str, bool]] = []

    def remember(self, **arguments: Any) -> None:
        self.calls.append(arguments)


class Double(Recording):
    """Pure, prunable: doubles a value."""

    type = "test/m7_double@v1"
    prunable = True
    outputs = {"doubled": Output(FLOAT_KIND)}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)

    def run(self, *, value) -> dict:
        self.remember(value=value)
        return {"doubled": value * 2}


class Painter(Recording):
    """Prunable; asks ``wants`` and paints the overlay only when wanted."""

    type = "test/m7_painter@v1"
    prunable = True
    outputs = {"overlay": Output(STRING_KIND), "count": Output(INTEGER_KIND)}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)

    def run(self, *, value) -> dict:
        self.remember(value=value)
        wanted = self.wants("overlay")
        self.asked.append({"overlay": wanted})
        result: Dict[str, Any] = {"count": 1}
        if wanted:
            result["overlay"] = f"paint:{value:g}"
        return result


class Forgetful(Recording):
    """Leaves ``overlay`` out without asking: always an error."""

    type = "test/m7_forgetful@v1"
    outputs = {"overlay": Output(STRING_KIND), "count": Output(INTEGER_KIND)}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)

    def run(self, *, value) -> dict:
        self.remember(value=value)
        return {"count": 1}


class Stubborn(Recording):
    """Asks ``wants`` but leaves ``overlay`` out whatever the answer."""

    type = "test/m7_stubborn@v1"
    outputs = {"overlay": Output(STRING_KIND), "count": Output(INTEGER_KIND)}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)

    def run(self, *, value) -> dict:
        self.remember(value=value)
        self.asked.append({"overlay": self.wants("overlay")})
        return {"count": 1}


class Curious(Recording):
    """Asks about an output it does not declare."""

    type = "test/m7_curious@v1"
    outputs = {"value": Output(FLOAT_KIND)}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)

    def run(self, *, value) -> dict:
        self.wants("nope")
        return {"value": value}


class Counter(Recording):
    """Stateful and not prunable: counts the calls of its session."""

    type = "test/m7_counter@v1"
    outputs = {"count": Output(INTEGER_KIND)}

    class Params(BlockParams):
        value: Ref()

    def run(self, *, value) -> dict:
        self.remember(value=value)
        return {"count": len(self.calls)}


class Sink(Recording):
    """Output-free effect block (not prunable by default)."""

    type = "test/m7_sink@v1"

    class Params(BlockParams):
        value: Ref()

    def run(self, *, value) -> None:
        self.remember(value=value)


class Emitter(Recording):
    """Emits ``seen`` for every value; events forbid pruning."""

    type = "test/m7_emitter@v1"
    outputs = {"value": Output(FLOAT_KIND)}
    events = {"seen": SEEN}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)

    def run(self, *, value) -> dict:
        self.remember(value=value)
        self.emit("seen", value=value)
        return {"value": value}


class Reactor(Recording):
    """Handler workflow step: records the handled value."""

    type = "test/m7_reactor@v1"
    outputs = {"value": Output(FLOAT_KIND)}

    class Params(BlockParams):
        value: float | Ref(FLOAT_KIND)

    def run(self, *, value) -> dict:
        self.remember(value=value)
        return {"value": value}


class Join(Recording):
    """Prunable recovery join: the first present side wins."""

    type = "test/m7_join@v1"
    prunable = True
    accepts_empty = True
    outputs = {"value": Output()}

    class Params(BlockParams):
        left: Ref()
        right: Ref()

    def run(self, *, left, right) -> dict:
        self.remember(left=left, right=right)
        return {"value": left if left is not None else right}


class Gate(Recording):
    """Prunable control block: continues to the targets when the value is truthy."""

    type = "test/m7_gate@v1"
    prunable = True

    class Params(BlockParams):
        value: Ref()
        next_steps: List[StepRef]

    def run(self, *, value, next_steps):
        self.remember(value=value)
        return Select(next_steps) if value else Stop()


class PhasedPainter(Recording):
    """Prunable, phased: ``first`` asks ``wants``, ``second`` honours the answer."""

    type = "test/m7_phased_painter@v1"
    prunable = True
    outputs = {"value": Output(FLOAT_KIND), "extra": Output(STRING_KIND)}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)

    @phase
    def first(self, *, value):
        wanted = self.wants("extra")
        self.asked.append({"extra": wanted, "phase": "first"})
        return value * 10

    @phase
    def second(self, *, first):
        wanted = self.wants("extra")
        self.asked.append({"extra": wanted, "phase": "second"})
        result: Dict[str, Any] = {"value": first}
        if wanted:
            result["extra"] = f"extra:{first:g}"
        return result

    def run(self, *, value):
        self.remember(value=value)
        return self.second(first=self.first(value=value))


class Items(Recording):
    """Prunable; returns an empty list for negative values, ``None`` for zero."""

    type = "test/m7_items@v1"
    prunable = True
    outputs = {"items": Output(LIST_OF_VALUES_KIND)}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)

    def run(self, *, value) -> dict:
        self.remember(value=value)
        if value == 0:
            return {"items": None}
        return {"items": [] if value < 0 else [value]}


# Quality fixtures -----------------------------------------------------------


class Accurate(Implementation):
    name = "accurate"
    quality = ("accurate", "balanced")

    def run(self, *, value) -> dict:
        return {"label": f"accurate@{self.execution_context.quality}"}


class Fast(Implementation):
    name = "fast"
    quality = ("fast",)

    def run(self, *, value) -> dict:
        return {"label": f"fast@{self.execution_context.quality}"}


class CudaFast(Implementation):
    name = "cuda-fast"
    requires = ("cuda",)
    quality = ("fast",)

    def run(self, *, value) -> dict:
        return {"label": "cuda-fast"}


class Model(Block):
    """Two implementations: ``accurate`` (declared first) and ``fast``."""

    type = "test/m7_model@v1"
    outputs = {"label": Output(STRING_KIND)}
    implementations = (Accurate, Fast)

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)


class GpuModel(Block):
    """``fast`` only on CUDA; ``accurate`` everywhere."""

    type = "test/m7_gpu_model@v1"
    outputs = {"label": Output(STRING_KIND)}
    implementations = (CudaFast, Accurate)

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)


class Legacy(Block):
    """Ordinary block without quality labels: behaves as before."""

    type = "test/m7_legacy@v1"
    outputs = {"label": Output(STRING_KIND)}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)

    def run(self, *, value) -> dict:
        return {"label": f"legacy@{self.execution_context.quality}"}


class Labeled(Block):
    """Ordinary block whose single implementation serves ``fast``."""

    type = "test/m7_labeled@v1"
    quality = ("fast",)
    outputs = {"label": Output(STRING_KIND)}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)

    def run(self, *, value) -> dict:
        return {"label": f"labeled@{self.execution_context.quality}"}


BLOCKS = (
    Double,
    Painter,
    Forgetful,
    Stubborn,
    Curious,
    Counter,
    Sink,
    Emitter,
    Reactor,
    Join,
    Gate,
    PhasedPainter,
    Items,
    Model,
    GpuModel,
    Legacy,
    Labeled,
)

CATALOGUE = Catalogue(BLOCKS, namespace="test")


def step(block: type, name: str, **params: Any) -> dict:
    """Build a step of one fixture block."""
    return {"type": block.type, "name": name, **params}


def definition(
    steps: List[dict],
    outputs: Dict[str, str],
    *,
    inputs: List[dict] = (),
    **sections: Any,
) -> dict:
    """Build a passive definition with one float parameter ``value`` by default."""
    definition = {
        "version": "2.0",
        "inputs": list(inputs)
        or [{"type": "WorkflowParameter", "name": "value", "kind": ["float"]}],
        "steps": steps,
        "outputs": [
            {"type": "JsonField", "name": name, "selector": selector}
            for name, selector in outputs.items()
        ],
        **sections,
    }

    return definition


def nested(name: str, definition: dict, **bindings: Any) -> dict:
    """Build a nested workflow step over an inline child definition."""
    return {
        "type": "roboflow_core/inner_workflow@v1",
        "name": name,
        "workflow_definition": definition,
        "parameter_bindings": bindings,
    }


HANDLER = {
    "version": "2.0",
    "inputs": [{"type": "WorkflowParameter", "name": "value"}],
    "steps": [step(Reactor, "react", value="$inputs.value")],
    "outputs": [{"type": "JsonField", "name": "out", "selector": "$steps.react.value"}],
}
