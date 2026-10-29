"""Catalogue, blocks and definition builders shared by recording and update tests."""

import threading
from fractions import Fraction
from typing import Any, Dict, List, Optional

import pytest
from roboflow_workflows.v2.data.batch import Timestamp
from roboflow_workflows.v2.data.kinds import FLOAT_KIND, INTEGER_KIND
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    Output,
    Ref,
    Select,
    StepRef,
    Stop,
)
from roboflow_workflows.v2.engine.operators.alignment import Align
from roboflow_workflows.v2.engine.operators.window import Window
from roboflow_workflows.v2.engine.sources import (
    Emission,
    Source,
    SourceOutput,
    SourceParams,
)

from tests.v2.support.execution.blocks import (
    ContinueIf,
    Counter,
    Echo,
    Expand,
)

WAIT = 10.0


MEDIA = Fraction(1, 1000)


TRIPWIRE: Dict[str, int] = {}


def media(milliseconds: int) -> Timestamp:
    return Timestamp(milliseconds, MEDIA, "media")


class Feed(Source):
    """Emits a scripted list of emissions; ``threading.Event`` items block."""

    type = "test/recorded_feed@v1"
    outputs = {"value": SourceOutput(FLOAT_KIND)}

    class Params(SourceParams):
        feed: str

    def __init__(self, *, feeds: Dict[str, list]):
        TRIPWIRE["source_constructed"] = TRIPWIRE.get("source_constructed", 0) + 1
        self.feeds = feeds

    def open(self, *, feed) -> None:
        self.items = iter(self.feeds[feed])

    def read(self) -> Optional[Emission]:
        item = next(self.items, None)
        while isinstance(item, threading.Event):
            item.wait(WAIT)
            item = next(self.items, None)
        if isinstance(item, Exception):
            raise item

        return item


class Model(Block):
    """Stands in for an expensive model: counts constructions and calls."""

    type = "test/recorded_model@v1"
    outputs = {"score": Output(FLOAT_KIND)}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)

    def __init__(self) -> None:
        TRIPWIRE["model_constructed"] = TRIPWIRE.get("model_constructed", 0) + 1

    def run(self, *, value) -> dict:
        TRIPWIRE["model_calls"] = TRIPWIRE.get("model_calls", 0) + 1
        return {"score": value / 10}


class Unrecordable(Block):
    """Outputs a payload type no recording codec handles."""

    type = "test/unrecordable@v1"
    outputs = {"thing": Output()}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)

    def run(self, *, value) -> dict:
        return {"thing": object()}


class MutateList(Block):
    """Appends to a list payload in place (a later causal mutator)."""

    type = "test/mutate_list@v1"
    outputs = {"items": Output()}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)

    def run(self, *, value) -> dict:
        return {"items": [value]}


class Nothing(Block):
    """Outputs ``None``: a delivered payload, neither empty nor filtered."""

    type = "test/nothing@v1"
    outputs = {"value": Output()}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)

    def run(self, *, value) -> dict:
        return {"value": None}


class Above(Block):
    """Control block: continue when ``value > threshold``; the threshold may be selected."""

    type = "test/recorded_above@v1"

    class Params(BlockParams):
        value: Ref()
        threshold: float | Ref(FLOAT_KIND) = 0.0
        next_steps: List[StepRef]

    def run(self, *, value, threshold, next_steps):
        return Select(next_steps) if value > threshold else Stop()


class StateTally(Block):
    """Counts calls in the session's managed state."""

    type = "test/state_tally@v1"
    outputs = {"tally": Output(INTEGER_KIND)}

    class Params(BlockParams):
        value: Ref()

    def __init__(self, *, managed_state) -> None:
        self.state = managed_state

    def run(self, *, value) -> dict:
        return {"tally": self.state.global_.incr("seen")}


CATALOGUE = Catalogue(
    [
        Above,
        ContinueIf,
        Counter,
        Echo,
        Expand,
        Model,
        Unrecordable,
        MutateList,
        Nothing,
        StateTally,
    ],
    sources=[Feed],
    operators=[Align, Window],
)


@pytest.fixture(autouse=True)
def _reset_tripwire():
    TRIPWIRE.clear()
    yield


def field(name: str, selector: str) -> Dict[str, str]:
    return {"type": "JsonField", "name": name, "selector": selector}


def group(name: str, anchor: str, *fields: Dict[str, str]) -> Dict[str, Any]:
    return {
        "type": "OutputGroup",
        "name": name,
        "anchor": anchor,
        "outputs": list(fields),
    }


def primary_definition(**sections: Any) -> Dict[str, Any]:
    """Feed ``cam`` -> model, gated children with partial filtering, a window.

    Groups: ``frames`` (value, score, children: gated per pulse, kept:
    filtered per child, empty: always an empty group) and ``clips`` (a
    two-pulse window of values, a time axis).
    """
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowParameter", "name": "capture_dir"}],
        "sources": [{"type": "test/recorded_feed@v1", "name": "cam", "feed": "cam"}],
        "operators": [
            {
                "type": "roboflow_core/window@v1",
                "name": "w",
                "size": 2,
                "collect": {"v": "$sources.cam.value"},
            }
        ],
        "steps": [
            {
                "type": "test/recorded_model@v1",
                "name": "model",
                "value": "$sources.cam.value",
            },
            {
                "type": "test/continue_if@v1",
                "name": "gate",
                "value": "$sources.cam.value",
                "threshold": 1.5,
                "next_steps": ["$steps.expand"],
            },
            {
                "type": "test/expand@v1",
                "name": "expand",
                "value": "$sources.cam.value",
                "count": 3,
            },
            {
                "type": "test/continue_if@v1",
                "name": "keep",
                "value": "$steps.expand.children",
                "threshold": 2.5,
                "next_steps": ["$steps.kept"],
            },
            {"type": "test/echo@v1", "name": "kept", "value": "$steps.expand.children"},
            {
                "type": "test/expand@v1",
                "name": "none",
                "value": "$sources.cam.value",
                "count": 0,
            },
        ],
        "outputs": [
            group(
                "frames",
                "$sources.cam.value",
                field("value", "$sources.cam.value"),
                field("score", "$steps.model.score"),
                field("children", "$steps.expand.children"),
                field("kept", "$steps.kept.value"),
                field("empty", "$steps.none.children"),
            ),
            group("clips", "$operators.w.v", field("clip", "$operators.w.v")),
        ],
        "recording": {
            "type": "file",
            "directory": "$inputs.capture_dir",
            "groups": ["frames", "clips"],
        },
    }
    definition.update(sections)

    return definition


def feed_values(*values: Optional[float]) -> List[Emission]:
    """One emission per value at 100 ms steps; ``None`` is a filtered pulse."""
    emissions = []
    for position, value in enumerate(values):
        data = {} if value is None else {"value": value}
        emissions.append(
            Emission(data, media=media(100 * position), source_metadata={"n": position})
        )

    return emissions
