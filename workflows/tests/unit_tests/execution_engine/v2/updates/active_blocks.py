"""Sources, blocks, an operator and builders for active graph update tests.

Every wait a test needs is an event the test controls: the ``Feed`` source
reads scripted items (emissions, gates, the end), ``Hold`` blocks a pulse
until released, and ``Pair`` keeps a partial window across the boundary.
Constructions and lifecycle calls go to the ``log`` resource.
"""

import threading
from typing import Any, Callable, Dict, List, Optional

from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
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
from roboflow_workflows.execution_engine.v2.operators import (
    Operator,
    OperatorParams,
    OperatorPort,
    OperatorPulse,
)
from roboflow_workflows.execution_engine.v2.sources import (
    Emission,
    Source,
    SourceOutput,
    SourceParams,
)

WAIT = 10.0
"""Upper bound of every wait, in seconds; the tests finish far earlier."""

END = None
"""Feed item: the source ends."""


class Log(list):
    """Append-only list a test can wait on."""

    def __init__(self) -> None:
        super().__init__()
        self.changed = threading.Condition()

    def append(self, item: Any) -> None:
        with self.changed:
            super().append(item)
            self.changed.notify_all()

    def wait_for(self, predicate: Callable[[], bool]) -> None:
        with self.changed:
            assert self.changed.wait_for(predicate, timeout=WAIT), list(self)

    def count(self, kind: str) -> int:
        return sum(1 for item in self if item[0] == kind)


class Feed(Source):
    """Reads scripted items of ``feeds[name]``: emissions, gates, exceptions, END.

    A ``threading.Event`` item blocks ``read`` until the test sets it or the
    run stops (then the source ends). ``hook`` (a resource) is called inside
    ``read`` once per emission, so a test can run code on the reader thread.
    """

    type = "test/update_feed@v1"
    outputs = {"value": SourceOutput(FLOAT_KIND)}

    class Params(SourceParams):
        pass

    def __init__(
        self,
        *,
        feeds: Dict[str, list],
        log: Log,
        hook: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.feeds = feeds
        self.log = log
        self.hook = hook
        self.name = ""
        self.items = iter(())

    def open(self) -> None:
        self.name = self.source_name
        self.items = iter(self.feeds[self.name])
        self.log.append(("open", self.name, id(self)))

    def read(self) -> Optional[Emission]:
        item = next(self.items, END)
        while isinstance(item, threading.Event):
            self.log.append(("wait", self.name))
            while not item.wait(0.02):
                if self.stop_event.is_set():
                    return END
            item = next(self.items, END)
        if isinstance(item, Exception):
            raise item
        if item is not END and self.hook is not None:
            self.hook(self.name)
        self.log.append(("read", self.name, None if item is END else item.data))

        return item

    def close(self) -> None:
        self.log.append(("close", self.name, id(self)))


class Count(Block):
    """Stateful: counts its calls over the lifetime of its instance."""

    type = "test/update_active_count@v1"
    outputs = {"count": Output(INTEGER_KIND)}

    class Params(BlockParams):
        value: Ref()

    def __init__(self, *, log: Log) -> None:
        log.append(("init", "count", id(self)))
        self.count = 0

    def run(self, *, value) -> dict:
        self.count += 1
        return {"count": self.count}


class Scale(Block):
    """Prunable: multiplies by ``factor``; the typical attached consumer."""

    type = "test/update_active_scale@v1"
    outputs = {"scaled": Output(FLOAT_KIND)}
    prunable = True

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)
        factor: float = 2.0

    def __init__(self, *, log: Log) -> None:
        log.append(("init", "scale", id(self)))

    def run(self, *, value, factor) -> dict:
        return {"scaled": value * factor}


class Hold(Block):
    """Blocks every call until ``holds[name].release`` is set; logs each entry.

    ``holds`` is a resource: step name to ``{"entered": Event, "release":
    Event}``. ``entered`` is set on every call; the test clears it to wait
    for the next one. A call raises when ``holds[name]["fail"]`` is set.
    ``holds[name]["hook"]``, when present, is called on the block's thread
    once the hold is released.
    """

    type = "test/update_active_hold@v1"
    outputs = {"value": Output(FLOAT_KIND)}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)

    def __init__(self, *, holds: Dict[str, Dict[str, Any]]) -> None:
        self.holds = holds
        self.calls = 0

    def run(self, *, value) -> dict:
        gates = self.holds[self.execution_context.step_path[-1]]
        self.calls += 1
        gates["entered"].set()
        assert gates["release"].wait(WAIT), "the test never released the hold"
        failing = gates.get("fail")
        if failing is not None and failing.is_set():
            raise RuntimeError("the held step failed")
        hook = gates.get("hook")
        if hook is not None:
            hook()
        return {"value": value}


class Tracker(Block):
    """Stateful member of a ``reset_on_enable`` control."""

    type = "test/update_active_tracker@v1"
    outputs = {"ticks": Output(INTEGER_KIND)}

    class Params(BlockParams):
        value: Ref()

    def __init__(self, *, log: Log) -> None:
        log.append(("init", "tracker", id(self)))
        self.ticks = 0
        self.resets = 0

    def reset_state(self) -> None:
        self.ticks = 0
        self.resets += 1

    def run(self, *, value) -> dict:
        self.ticks += 1
        return {"ticks": self.ticks}


class Echo(Block):
    """Prunable (pure) consumer of any value: joins a control's closure by reading."""

    type = "test/update_active_echo@v1"
    outputs = {"out": Output()}
    prunable = True

    class Params(BlockParams):
        value: Ref()

    def run(self, *, value) -> dict:
        return {"out": value}


class Record(Block):
    """Handler step: appends its value to ``log`` and may hold like ``Hold``."""

    type = "test/update_active_record@v1"
    outputs = {"out": Output()}

    class Params(BlockParams):
        value: Ref()

    def __init__(
        self, *, log: Log, holds: Dict[str, Dict[str, threading.Event]]
    ) -> None:
        self.log = log
        self.holds = holds

    def run(self, *, value) -> dict:
        gates = self.holds.get("record")
        if gates is not None:
            gates["entered"].set()
            assert gates["release"].wait(WAIT), "the test never released the hold"
        self.log.append(("record", value))
        return {"out": value}


class Pair(Operator):
    """Emits one pulse per two arrivals, carrying both causes; drops a partial pair."""

    type = "test/update_pair@v1"
    input_roles = ("input",)

    class Params(OperatorParams):
        pass

    @classmethod
    def plan_ports(cls, name, params, inputs):
        return {"first": OperatorPort(*inputs[0].kinds)}

    def __init__(self, **arguments: Any) -> None:
        super().__init__(**arguments)
        self.buffer: List[Any] = []
        self.lifecycle: List[Any] = []

    def push(self, arrivals):
        pulses = []
        for arrival in arrivals:
            self.buffer.append(arrival)
            if len(self.buffer) == 2:
                first, second = self.buffer
                self.buffer = []
                pulses.append(
                    OperatorPulse(
                        ports={"first": first.entry},
                        causes=(first.pulse, second.pulse),
                    )
                )
        return pulses

    def end_input(self, name):
        self.lifecycle.append(("end_input", name))
        return []

    def finish(self, reason):
        self.lifecycle.append(("finish", reason, len(self.buffer)))
        return []

    def close(self):
        self.lifecycle.append(("close",))


CATALOGUE = Catalogue(
    [Count, Scale, Hold, Tracker, Echo, Record],
    sources=[Feed],
    operators=[Pair],
    kinds=[STRING_KIND],
)


def emit(value: float) -> Emission:
    return Emission({"value": float(value)})


def step(block: type, name: str, **params: Any) -> dict:
    return {"type": block.type, "name": name, **params}


def group(name: str, anchor: str, **fields: str) -> dict:
    return {
        "type": "OutputGroup",
        "name": name,
        "anchor": anchor,
        "outputs": [
            {"type": "JsonField", "name": field, "selector": selector}
            for field, selector in fields.items()
        ],
    }


def active(
    sources: List[str], steps: List[dict], groups: List[dict], **sections: Any
) -> dict:
    """An active definition whose ``Feed`` sources read ``feeds[<name>]``."""
    definition = {
        "version": "2.0",
        "inputs": [],
        "sources": [{"type": Feed.type, "name": name} for name in sources],
        "steps": steps,
        "outputs": groups,
        **sections,
    }

    return definition


def handler(name: str, on: str, *, mode: str = "async") -> dict:
    """An event handler whose child workflow runs one ``Record`` step."""
    definition = {
        "name": name,
        "on": on,
        "execution": {"mode": mode},
        "bindings": {"value": "$event.note"},
        "workflow": {
            "inputs": [{"name": "value", "kind": ["*"]}],
            "steps": [step(Record, "record", value="$inputs.value")],
            "outputs": [
                {"type": "JsonField", "name": "out", "selector": "$steps.record.out"}
            ],
        },
    }

    return definition


def compiled(definition: dict) -> Any:
    plan = compile_workflow(definition, catalogue=CATALOGUE)

    return plan


class Collector:
    """Handlers keeping every result per group, in delivery order."""

    def __init__(self) -> None:
        self.results: Dict[str, List[Any]] = {}
        self.changed = threading.Condition()

    def handler(self, name: str) -> Callable[[Any], None]:
        def handle(result: Any) -> None:
            with self.changed:
                self.results.setdefault(name, []).append(result)
                self.changed.notify_all()

        return handle

    def handlers(self, *names: str) -> Dict[str, Callable[[Any], None]]:
        return {name: self.handler(name) for name in names}

    def wait_for(self, name: str, count: int) -> List[Any]:
        with self.changed:
            assert self.changed.wait_for(
                lambda: len(self.results.get(name, ())) >= count, timeout=WAIT
            ), {k: len(v) for k, v in self.results.items()}
            results = list(self.results[name])

        return results

    def sequences(self, name: str) -> List[int]:
        return [result.pulse.sequence for result in self.results.get(name, ())]

    def versions(self, name: str) -> List[int]:
        return [result.graph_version for result in self.results.get(name, ())]


class Updater:
    """Runs ``run.apply_update`` on its own thread; the test joins it with a bound."""

    def __init__(self, run: Any, update: Any, **options: Any) -> None:
        self.run = run
        self.update = update
        self.options = options
        self.receipt: Any = None
        self.error: Optional[BaseException] = None
        self.finished = threading.Event()
        self.thread = threading.Thread(target=self._apply, daemon=True)
        self.thread.start()

    def _apply(self) -> None:
        try:
            self.receipt = self.run.apply_update(self.update, **self.options)
        except BaseException as raised:
            self.error = raised
        finally:
            self.finished.set()

    def result(self) -> Any:
        """Join; return the receipt or raise what ``apply_update`` raised."""
        assert self.finished.wait(WAIT), "apply_update did not return"
        if self.error is not None:
            raise self.error

        return self.receipt
