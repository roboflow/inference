"""Small class-owned fixture blocks for executor tests.

Every block records the keyword arguments of each call in ``self.calls``.
The session constructs one instance per step, so a test reads
``session.instances[("name",)].calls`` for exact call counts and arguments.
"""

from concurrent.futures import Future
from typing import Any, Dict, List, Optional, Tuple

from pydantic import Field, field_validator, model_validator
from roboflow_workflows.execution_engine.v2.context import get_execution_context
from roboflow_workflows.execution_engine.v2.data import Batch
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Group,
    Output,
    Ref,
    Select,
    StepRef,
    Stop,
)
from roboflow_workflows.execution_engine.v2.kinds import (
    DICTIONARY_KIND,
    FLOAT_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
    Kind,
)


class Recording(Block):
    """Abstract base: remembers every call's arguments."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def remember(self, **arguments: Any) -> None:
        self.calls.append(arguments)


class Scale(Recording):
    """Literal, default or selector factor."""

    type = "test/scale@v1"
    outputs = {"scaled": Output(FLOAT_KIND)}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)
        # Shared constraint: applies to literals and selected values alike.
        factor: float | Ref(FLOAT_KIND) = Field(2.0, ge=0)

    def run(self, *, value, factor) -> dict:
        self.remember(value=value, factor=factor)
        return {"scaled": value * factor}


class Echo(Recording):
    """Returns its value unchanged (any kind)."""

    type = "test/echo@v1"
    outputs = {"value": Output()}

    class Params(BlockParams):
        value: Ref()

    def run(self, *, value) -> dict:
        self.remember(value=value)
        return {"value": value}


class Expand(Recording):
    """Adds a child axis: ``value + k`` for ``k < count``, plus the count."""

    type = "test/expand@v1"
    outputs = {
        "children": Output(FLOAT_KIND, expand="items"),
        "count": Output(INTEGER_KIND),
    }

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)
        count: int | Ref(INTEGER_KIND) = 2

    def run(self, *, value, count) -> dict:
        self.remember(value=value, count=count)
        children = Batch.of([value + position for position in range(count)])
        return {"children": children, "count": count}


class Source(Recording):
    """Input-free source creating its own top-level axis."""

    type = "test/source@v1"
    outputs = {"generated": Output(FLOAT_KIND, expand="generated")}

    class Params(BlockParams):
        values: List[float] = [1.0, 2.0]

    def run(self, *, values) -> dict:
        self.remember(values=values)
        return {"generated": Batch.of(values)}


class Sum(Recording):
    """Reducer over a group of children."""

    type = "test/sum@v1"
    outputs = {"total": Output(FLOAT_KIND)}

    class Params(BlockParams):
        values: Group(FLOAT_KIND)

    def run(self, *, values) -> dict:
        self.remember(values=values)
        return {"total": sum(values)}


class SumAcceptingEmpty(Sum):
    """Reducer that also runs on missing or fully filtered groups."""

    type = "test/sum_accepting_empty@v1"
    accepts_empty = True


class FirstNonEmpty(Recording):
    """Recovery join over alternative branches at one invocation level."""

    type = "test/first_non_empty@v1"
    accepts_empty = True
    outputs = {"value": Output()}

    class Params(BlockParams):
        data: List[Ref()]
        default: Any = None

    def run(self, *, data, default) -> dict:
        self.remember(data=data, default=default)
        chosen = next((item for item in data if item is not None), default)
        return {"value": chosen}


class ContinueIf(Recording):
    """Control block: continue when ``value > threshold``."""

    type = "test/continue_if@v1"

    class Params(BlockParams):
        value: Ref()
        threshold: float = 0.0
        next_steps: List[StepRef]

    def run(self, *, value, threshold, next_steps):
        self.remember(value=value, threshold=threshold, next_steps=next_steps)
        return Select(next_steps) if value > threshold else Stop()


class Route(Recording):
    """Control block choosing one named branch."""

    type = "test/route@v1"

    class Params(BlockParams):
        value: Ref()
        cases: Dict[str, StepRef]

    def run(self, *, value, cases):
        self.remember(value=value, cases=cases)
        return Select(cases[value]) if value in cases else Stop()


class BadControl(Recording):
    """Control block returning an invalid decision (for error tests)."""

    type = "test/bad_control@v1"

    class Params(BlockParams):
        answer: str
        next_steps: List[StepRef]

    def run(self, *, answer, next_steps):
        self.remember(answer=answer)
        if answer == "mapping":
            return {"next": next_steps}
        return Select("$steps.elsewhere")


class Notice(Recording):
    """Literal-only, output-free side-effect sink."""

    type = "test/notice@v1"

    class Params(BlockParams):
        message: str = "found"

    def run(self, *, message) -> None:
        self.remember(message=message)


class Counter(Recording):
    """Stateful block: counts calls over the lifetime of its session."""

    type = "test/counter@v1"
    outputs = {"count": Output(INTEGER_KIND)}

    class Params(BlockParams):
        value: Ref()

    def run(self, *, value) -> dict:
        self.remember(value=value)
        return {"count": len(self.calls)}


class Deferred(Recording):
    """Returns futures: in a mapping, inside a list payload and in a Batch."""

    type = "test/deferred@v1"
    outputs = {
        "doubled": Output(FLOAT_KIND),
        "pair": Output(),
        "children": Output(FLOAT_KIND, expand="items"),
    }

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)
        fail: bool = False

    def run(self, *, value, fail) -> dict:
        self.remember(value=value)
        doubled, first, child = Future(), Future(), Future()
        if fail:
            doubled.set_exception(RuntimeError(f"remote failure on {value}"))
        else:
            doubled.set_result(value * 2)
        first.set_result(value)
        child.set_result(value + 0.5)
        return {
            "doubled": doubled,
            "pair": [first, "fixed"],
            "children": Batch.of([child]),
        }


class Failing(Recording):
    """Raises for negative values."""

    type = "test/failing@v1"
    outputs = {"value": Output()}

    class Params(BlockParams):
        value: Ref()

    def run(self, *, value) -> dict:
        self.remember(value=value)
        if value < 0:
            raise ValueError(f"negative {value}")
        return {"value": value}


class Mutate(Recording):
    """Declared in-place mutation of a dictionary payload."""

    type = "test/mutate@v1"
    mutates = ("payload",)
    outputs = {"payload": Output()}

    class Params(BlockParams):
        payload: Ref()

    def run(self, *, payload) -> dict:
        self.remember(payload=payload)
        payload["count"] = payload.get("count", 0) + 1
        return {"payload": payload}


class InvertMany(Recording):
    """Batch-delivering block: one call for every invocation."""

    type = "test/invert_many@v1"
    outputs = {"inverted": Output(FLOAT_KIND)}

    class Params(BlockParams):
        values: Ref(FLOAT_KIND, batch="always")
        offset: float | Ref(FLOAT_KIND) = 0.0

    def run(self, *, values, offset) -> list:
        self.remember(values=values, offset=offset)
        return [{"inverted": offset - value} for value in values]


class Consensus(Recording):
    """List of batch selectors: receives ``list[Batch]``."""

    type = "test/consensus@v1"
    outputs = {"votes": Output()}

    class Params(BlockParams):
        predictions: List[Ref(batch="always")]

    def run(self, *, predictions) -> list:
        self.remember(predictions=predictions)
        return [{"votes": list(row)} for row in zip(*predictions)]


class Csv(Recording):
    """Mixed dict: only varying selector leaves arrive as ``Batch``."""

    type = "test/csv@v1"
    outputs = {"row": Output()}

    class Params(BlockParams):
        columns: Dict[str, str | float | None | Ref(batch="if_varying")]

    def run(self, *, columns):
        self.remember(columns=columns)
        batches = {
            key: value for key, value in columns.items() if isinstance(value, Batch)
        }
        if not batches:
            return {"row": dict(columns)}

        size = len(next(iter(batches.values())))
        return [
            {
                "row": {
                    key: value[position] if isinstance(value, Batch) else value
                    for key, value in columns.items()
                }
            }
            for position in range(size)
        ]


class StitchAndTranslate(Recording):
    """Parent plus child group: a parent-level and a child-level output."""

    type = "test/stitch_and_translate@v1"
    outputs = {
        "stitched": Output(FLOAT_KIND, source="parent"),
        "translated": Output(FLOAT_KIND, preserve="children"),
    }

    class Params(BlockParams):
        parent: Ref(FLOAT_KIND)
        children: Group(FLOAT_KIND)

    def run(self, *, parent, children) -> dict:
        self.remember(parent=parent, children=children)
        translated = Batch(
            [parent + child for child in children],
            indices=children.indices,
            parent_index=children.parent_index,
        )
        return {"stitched": parent + sum(children), "translated": translated}


class NamedGroups(Recording):
    """Compound group: dict of child groups of one parent."""

    type = "test/named_groups@v1"
    outputs = {"sizes": Output()}

    class Params(BlockParams):
        parent: Ref()
        groups: Dict[str, Group()]

    def run(self, *, parent, groups) -> dict:
        self.remember(parent=parent, groups=groups)
        return {"sizes": {name: list(group) for name, group in groups.items()}}


class Collect(Recording):
    """Whole-field group; returns the members as one list payload."""

    type = "test/collect@v1"
    outputs = {"members": Output()}

    class Params(BlockParams):
        values: Group()

    def run(self, *, values) -> dict:
        self.remember(values=values)
        return {"members": list(values)}


class Describe(Recording):
    """Optional parameter that may legitimately be None."""

    type = "test/describe@v1"
    outputs = {"text": Output(STRING_KIND)}

    class Params(BlockParams):
        value: Ref()
        suffix: Optional[str] | Ref(STRING_KIND) = None

    def run(self, *, value, suffix) -> dict:
        self.remember(value=value, suffix=suffix)
        return {"text": f"{value}{suffix or ''}"}


class Bad(Recording):
    """Returns whatever the ``result`` literal says (for result validation)."""

    type = "test/bad@v1"
    outputs = {"value": Output(INTEGER_KIND), "children": Output(expand="items")}

    class Params(BlockParams):
        value: Ref()
        mode: str = "ok"

    def run(self, *, value, mode):
        self.remember(value=value, mode=mode)
        results = {
            "ok": {"value": 1, "children": Batch.of([1])},
            "omitted": {"value": 1},
            "unknown": {"value": 1, "children": Batch.of([1]), "extra": 2},
            "not_mapping": [1],
            "list_as_children": {"value": 1, "children": [1, 2]},
            "wrong_kind": {"value": "one", "children": Batch.of([1])},
            "batch_as_payload": {"value": Batch.of([1]), "children": Batch.of([1])},
        }
        return results[mode]


def tagged_kind() -> Kind:
    """Kind with codecs: payloads are ``("tag", value)`` tuples."""
    return Kind(
        name="tagged",
        validate=lambda payload: isinstance(payload, tuple) and payload[0] == "tag",
        deserialize=lambda value: ("tag", value),
        serialize=lambda payload: {"tagged": payload[1]},
        convert_output=lambda payload, options: (
            "tag",
            payload[1] * options.get("scale", 1),
        ),
    )


class SplitMany(Recording):
    """Batch-delivering expansion: two parts per sample in one call."""

    type = "test/split_many@v1"
    outputs = {"parts": Output(FLOAT_KIND, expand="parts")}

    class Params(BlockParams):
        values: Ref(FLOAT_KIND, batch="always")

    def run(self, *, values) -> list:
        self.remember(values=values)
        return [{"parts": Batch.of([value, value + 0.5])} for value in values]


class SumMany(Recording):
    """Batch-delivering reducer: receives a Batch of groups."""

    type = "test/sum_many@v1"
    accepts_empty = True
    outputs = {"total": Output(FLOAT_KIND)}

    class Params(BlockParams):
        values: Group(FLOAT_KIND, batch="always")

    def run(self, *, values) -> list:
        self.remember(values=values)
        return [{"total": sum(group)} for group in values]


class EchoMany(Recording):
    """Batch-delivering, empty-accepting echo."""

    type = "test/echo_many@v1"
    accepts_empty = True
    outputs = {"value": Output()}

    class Params(BlockParams):
        values: Ref(batch="always")

    def run(self, *, values) -> list:
        self.remember(values=values)
        return [{"value": value} for value in values]


class Crop(Recording):
    """Two expand outputs sharing one child axis."""

    type = "test/crop@v1"
    outputs = {
        "crops": Output(FLOAT_KIND, expand="crops"),
        "boxes": Output(expand="crops"),
    }

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)
        mismatch: bool = False

    def run(self, *, value, mismatch) -> dict:
        self.remember(value=value)
        boxes = Batch.of(["box0", "box2"], indices=[(0,), (2,)] if mismatch else None)
        return {"crops": Batch.of([value, value * 2]), "boxes": boxes}


class Relabel(Recording):
    """Preserves the group it receives (cast groups included)."""

    type = "test/relabel@v1"
    outputs = {"labels": Output(preserve="values")}

    class Params(BlockParams):
        values: Group()
        prefix: str = "#"

    def run(self, *, values, prefix) -> dict:
        self.remember(values=values)
        return {"labels": Batch.of([f"{prefix}{value}" for value in values])}


class Probe(Recording):
    """Parameters named like observer arguments."""

    type = "test/probe@v1"
    outputs = {"result": Output()}

    class Params(BlockParams):
        step: Ref()
        index: int = 0
        arguments: str = "a"

    def run(self, *, step, index, arguments) -> dict:
        self.remember(step=step, index=index, arguments=arguments)
        return {"result": (step, index, arguments)}


class Pair(Recording):
    """Empty-accepting parent/children join; children may be absent (None)."""

    type = "test/pair@v1"
    accepts_empty = True
    outputs = {"pair": Output()}

    class Params(BlockParams):
        parent: Ref()
        children: Group()

    def run(self, *, parent, children) -> dict:
        self.remember(parent=parent, children=children)
        members = None if children is None else list(children)
        return {"pair": [parent, members]}


class PairMany(Recording):
    """Batch-delivering version of ``Pair``: one call across all parents."""

    type = "test/pair_many@v1"
    accepts_empty = True
    outputs = {"pair": Output()}

    class Params(BlockParams):
        parent: Ref(batch="always")
        children: Group(batch="always")

    def run(self, *, parent, children) -> list:
        self.remember(parent=parent, children=children)
        return [
            {"pair": [item, None if group is None else list(group)]}
            for item, group in zip(parent, children)
        ]


class Region(Recording):
    """Literal-or-selector tuple field; selected lists are normalized."""

    type = "test/region@v1"
    outputs = {"region": Output()}

    class Params(BlockParams):
        region: Tuple[int, int] | Ref()

    def run(self, *, region) -> dict:
        self.remember(region=region)
        return {"region": region}


class AppendInPlace(Recording):
    """Declares in-place mutation of a literal-or-selector list field."""

    type = "test/append_in_place@v1"
    mutates = ("items",)
    outputs = {"items": Output()}

    class Params(BlockParams):
        items: List[int] | Ref()

    def run(self, *, items) -> dict:
        self.remember(items=items)
        items.append(len(items))
        return {"items": items}


class EvenGateMany(Recording):
    """Batch-delivering control block: admits even values."""

    type = "test/even_gate_many@v1"

    class Params(BlockParams):
        values: Ref(batch="always")
        next_steps: List[StepRef]

    def run(self, *, values, next_steps) -> list:
        self.remember(values=values)
        return [Select(next_steps) if value % 2 == 0 else Stop() for value in values]


class DeferredMany(Recording):
    """Batch-delivering block returning a future per invocation."""

    type = "test/deferred_many@v1"
    outputs = {"value": Output()}

    class Params(BlockParams):
        values: Ref(batch="always")

    def run(self, *, values) -> list:
        self.remember(values=values)
        results = []
        for value in values:
            future = Future()
            future.set_result(value * 10)
            results.append({"value": future})
        return results


class LiteralGroups(Recording):
    """Group positions that also accept literals."""

    type = "test/literal_groups@v1"
    outputs = {"members": Output()}

    class Params(BlockParams):
        parent: Ref()
        groups: Dict[str, int | Group()]

    def run(self, *, parent, groups) -> dict:
        self.remember(parent=parent, groups=groups)
        return {"members": {name: list(group) for name, group in groups.items()}}


class HistoryAppend(Recording):
    """Mutates a literal dict nested in a compound field."""

    type = "test/history_append@v1"
    outputs = {"length": Output()}

    class Params(BlockParams):
        records: Dict[str, dict | Ref()]

    def run(self, *, records) -> dict:
        self.remember(records=records)
        records["literal"]["history"].append("called")
        return {"length": len(records["literal"]["history"])}


class ContextProbe(Recording):
    """Records the execution context seen inside ``run``."""

    type = "test/context_probe@v1"
    outputs = {"value": Output()}

    class Params(BlockParams):
        value: Ref()
        fail: bool = False

    def run(self, *, value, fail) -> dict:
        self.remember(context=get_execution_context(), value=value)
        if fail:
            raise ValueError("probe failure")
        return {"value": value}


class ContextProbeMany(Recording):
    """Batch-delivering context probe."""

    type = "test/context_probe_many@v1"
    outputs = {"value": Output()}

    class Params(BlockParams):
        values: Ref(batch="always")

    def run(self, *, values) -> list:
        self.remember(context=get_execution_context())
        return [{"value": value} for value in values]


class ContextFuture(Future):
    """A completed future that reads the execution context when resolved."""

    def result(self, timeout=None):
        return get_execution_context().step_path


class DeferredContext(Recording):
    """Returns a future resolved by the engine."""

    type = "test/deferred_context@v1"
    outputs = {"step": Output()}

    class Params(BlockParams):
        value: Ref()

    def run(self, *, value) -> dict:
        future = ContextFuture()
        future.set_result(None)
        return {"step": future}


class RunsInner(Recording):
    """Runs another session inside its own call (a nested engine call)."""

    type = "test/runs_inner@v1"
    outputs = {"inner": Output()}

    class Params(BlockParams):
        value: Ref()

    def __init__(self, *, inner) -> None:
        super().__init__()
        self.inner = inner

    def run(self, *, value) -> dict:
        before = get_execution_context()
        rows = self.inner.run({"value": value}).rows()
        after = get_execution_context()
        self.remember(before=before, after=after)
        return {"inner": rows}


class Thresholds(Recording):
    """Field and model validators that apply to selected values too."""

    type = "test/thresholds@v1"
    outputs = {"span": Output()}

    class Params(BlockParams):
        low: float | Ref(FLOAT_KIND) = 0.0
        high: float | Ref(FLOAT_KIND) = 1.0

        @field_validator("high")
        @classmethod
        def _high_is_at_most_hundred(cls, value):
            if isinstance(value, (int, float)) and value > 100:
                raise ValueError("high must be at most 100")
            return value

        @model_validator(mode="after")
        def _ordered(self):
            if isinstance(self.low, (int, float)) and isinstance(
                self.high, (int, float)
            ):
                if self.low > self.high:
                    raise ValueError("low must not exceed high")
            return self

    def run(self, *, low, high) -> dict:
        self.remember(low=low, high=high)
        return {"span": high - low}


class TextOrMapping(Recording):
    """Literal alternatives unrelated to the selected kinds."""

    type = "test/text_or_mapping@v1"
    outputs = {"payload": Output(), "items": Output()}

    class Params(BlockParams):
        payload: str | Ref(DICTIONARY_KIND) = "none"
        items: Tuple[int, int] | Ref(LIST_OF_VALUES_KIND) = (0, 0)

    def run(self, *, payload, items) -> dict:
        self.remember(payload=payload, items=items)
        return {"payload": payload, "items": items}


class StringGroups(Recording):
    """Group of strings that also accepts an integer literal."""

    type = "test/string_groups@v1"
    outputs = {"members": Output()}

    class Params(BlockParams):
        parent: Ref()
        groups: Dict[str, int | Group(STRING_KIND)]

    def run(self, *, parent, groups) -> dict:
        self.remember(groups=groups)
        return {"members": {name: list(group) for name, group in groups.items()}}


DECODED = []


def counting_kind() -> Kind:
    """Kind whose decoder records each call: ``"7"`` → ``{"decoded": 7}``."""

    def decode(value):
        DECODED.append(value)
        return {"decoded": int(value)}

    return Kind(
        name="counted",
        validate=lambda payload: isinstance(payload, dict) and "decoded" in payload,
        deserialize=decode,
        serialize=lambda payload: f"#{payload['decoded']}",
    )
