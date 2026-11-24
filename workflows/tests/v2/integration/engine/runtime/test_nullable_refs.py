"""A present ``None`` reaches an opted-in leaf; an unavailable value never does.

Compiled V2 workflows run fixture blocks that record ``(tag, value)`` per call.
The tag travels on a separate strict leaf, so every assertion names the row
that ran, not only how many calls happened or what the rows rendered. A
skipped invocation also renders ``None``; the recorded calls and skip reasons
tell the cases apart.

    present None  + allow_none   -> invoked with None
    present None  (default)      -> skipped, empty_value
    filtered / absent            -> skipped, filtered_input   (flag or not)
    denied gate / denied ancestor -> skipped or never planned  (flag or not)

The flag only changes varying leaves. A constant leaf (a parameter, or an
active source's ungrouped port, one value per pulse) never skipped on None,
with or without the flag; the active tests pin that existing behaviour.
"""

import threading
from typing import Any, Dict, List, Optional, Tuple

import pytest
from pydantic import StrictStr, model_validator
from roboflow_workflows.v2.data.batch import Axis, Batch, EntryLayout
from roboflow_workflows.v2.data.kinds import STRING_KIND
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    Group,
    Output,
    Ref,
)
from roboflow_workflows.v2.engine.errors import (
    ResolvedParameterError,
    StepExecutionError,
)
from roboflow_workflows.v2.engine.operators.alignment import Align
from roboflow_workflows.v2.engine.operators.window import Window
from roboflow_workflows.v2.engine.plan import CompileOptions, ExecutionObserver
from roboflow_workflows.v2.engine.sources import SourceOutput

from tests.v2.support.compilation.fixture_blocks import (
    Gate,
    Merge,
    batch_input,
    gate,
    nested,
    parameter,
    workflow,
)
from tests.v2.support.runtime.active import (
    WAIT,
    Collector,
    FeedSource,
    Log,
    Scripted,
    active,
    emit,
    group,
    source,
)

CALLS: List[Tuple[str, Any, Any]] = []
"""``(step type, tag, value)`` per call, in call order."""


class Take(Block):
    """Opted in: runs with a present None."""

    type = "nullable/take@v1"
    outputs = {"out": Output(STRING_KIND, source="value")}

    class Params(BlockParams):
        value: Ref(STRING_KIND, allow_none=True)
        tag: Ref(STRING_KIND)

    def run(self, *, value, tag) -> dict:
        CALLS.append(("take", tag, value))
        return {"out": f"{tag}:{value}"}


class Strict(Block):
    """The same block with the default leaf."""

    type = "nullable/strict@v1"
    outputs = {"out": Output(STRING_KIND, source="value")}

    class Params(BlockParams):
        value: Ref(STRING_KIND)
        tag: Ref(STRING_KIND)

    def run(self, *, value, tag) -> dict:
        CALLS.append(("strict", tag, value))
        return {"out": f"{tag}:{value}"}


class Maybe(Block):
    """Emits a present None for the text ``"none"``."""

    type = "nullable/maybe@v1"
    outputs = {"out": Output(STRING_KIND)}

    class Params(BlockParams):
        value: Ref(STRING_KIND)

    def run(self, *, value) -> dict:
        CALLS.append(("maybe", None, value))
        return {"out": None if value == "none" else value}


class Spread(Block):
    """Expands a list into children, tagged ``<tag>.<k>``, with an even flag."""

    type = "nullable/spread@v1"
    outputs = {
        "child": Output(expand="child"),
        "label": Output(STRING_KIND, expand="child"),
        "even": Output(expand="child"),
    }

    class Params(BlockParams):
        items: Ref()
        tag: Ref(STRING_KIND)

    def run(self, *, items, tag) -> dict:
        CALLS.append(("spread", tag, items))
        return {
            "child": Batch.of(list(items)),
            "label": Batch.of([f"{tag}.{k}" for k in range(len(items))]),
            "even": Batch.of([k % 2 == 0 for k in range(len(items))]),
        }


class Guarded(Block):
    """Opted in, but the author's validator rejects None under 'strict'."""

    type = "nullable/guarded@v1"
    outputs = {"out": Output()}

    class Params(BlockParams):
        value: Optional[StrictStr] | Ref(STRING_KIND, allow_none=True) = None
        policy: str = "lenient"

        @model_validator(mode="after")
        def _strict_needs_a_value(self) -> "Guarded.Params":
            if self.policy == "strict" and self.value is None:
                raise ValueError("a strict policy needs a value")

            return self

    def run(self, *, value, policy) -> dict:
        CALLS.append(("guarded", policy, value))
        return {"out": value}


class Batched(Block):
    """Opted in with batch delivery: admitted rows arrive as one Batch."""

    type = "nullable/batched@v1"
    outputs = {"out": Output(STRING_KIND, source="value")}

    class Params(BlockParams):
        value: Ref(STRING_KIND, batch="always", allow_none=True)  # noqa: F821
        tag: Ref(STRING_KIND, batch="always")  # noqa: F821

    def run(self, *, value, tag) -> list:
        CALLS.append(("batched", list(tag), list(value)))
        return [{"out": f"{t}:{v}"} for t, v in zip(tag, value)]


class Collect(Block):
    """Records the children of a time window (Group child kinds unchanged)."""

    type = "nullable/collect@v1"
    outputs = {"seen": Output()}

    class Params(BlockParams):
        values: Group()

    def run(self, *, values) -> dict:
        CALLS.append(("collect", None, list(values)))
        return {"seen": len(values)}


ITEMS = EntryLayout((Axis(id="items", kind="dynamic_nesting"),))


class Items(FeedSource):
    """Two grouped ports over one source-local axis: varying per item."""

    type = "nullable/items@v1"
    outputs = {
        "label": SourceOutput(STRING_KIND, layout=ITEMS),
        "tag": SourceOutput(STRING_KIND, layout=ITEMS),
    }


BLOCKS = [Take, Strict, Maybe, Spread, Guarded, Batched, Collect, Gate, Merge]
CATALOGUE = Catalogue(BLOCKS, kinds=[STRING_KIND])
ACTIVE_CATALOGUE = Catalogue(
    BLOCKS, kinds=[STRING_KIND], sources=[Scripted, Items], operators=[Align, Window]
)
MODES = ("run", "phases")


@pytest.fixture(autouse=True)
def _reset_calls():
    CALLS.clear()
    yield
    CALLS.clear()


class Skips(ExecutionObserver):
    """Records ``(step, index, reason)`` of every skipped invocation."""

    def __init__(self) -> None:
        self.skipped: List[Tuple[str, tuple, str]] = []

    def on_invocation_skipped(self, *, step, index, reason):
        self.skipped.append(("/".join(step), tuple(index), reason))


def step(block: type, name: str, **params: Any) -> dict:
    return {"type": block.type, "name": name, **params}


def strings(*names: str) -> List[dict]:
    return [batch_input(name, kind=["string"]) for name in names]


def run(definition: dict, inputs: dict, *, mode: str = "run") -> Tuple[list, Skips]:
    plan = compile_workflow(
        definition, catalogue=CATALOGUE, options=CompileOptions(block_execution=mode)
    )
    skips = Skips()
    rows = plan.create_session(observer=skips).run(inputs).rows()

    return rows, skips


def calls(block: str) -> List[Tuple[Any, Any]]:
    return [(tag, value) for name, tag, value in CALLS if name == block]


# --- admission truth table ------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_a_present_none_reaches_only_the_opted_in_leaf(mode: str) -> None:
    definition = workflow(
        [
            step(Take, "take", value="$inputs.value", tag="$inputs.tag"),
            step(Strict, "strict", value="$inputs.value", tag="$inputs.tag"),
        ],
        {"take": "$steps.take.out", "strict": "$steps.strict.out"},
        inputs=strings("value", "tag"),
    )

    rows, skips = run(
        definition,
        {"value": [None, "a", None], "tag": ["r0", "r1", "r2"]},
        mode=mode,
    )

    assert calls("take") == [("r0", None), ("r1", "a"), ("r2", None)]
    assert calls("strict") == [("r1", "a")]
    assert skips.skipped == [
        ("strict", (0,), "empty_value"),
        ("strict", (2,), "empty_value"),
    ]
    assert rows == [
        {"take": "r0:None", "strict": None},
        {"take": "r1:a", "strict": "r1:a"},
        {"take": "r2:None", "strict": None},
    ]


def test_batch_delivery_keeps_admitted_rows_with_none_in_order() -> None:
    definition = workflow(
        [
            gate("keep", "$inputs.keep", ["batched"]),
            step(Batched, "batched", value="$inputs.value", tag="$inputs.tag"),
        ],
        {"out": "$steps.batched.out"},
        inputs=strings("value", "tag") + [batch_input("keep")],
    )

    rows, skips = run(
        definition,
        {
            "value": [None, "b", None, "d"],
            "tag": ["r0", "r1", "r2", "r3"],
            "keep": [True, True, False, True],
        },
    )

    assert calls("batched") == [(["r0", "r1", "r3"], [None, "b", "d"])]
    assert skips.skipped == [("batched", (2,), "denied_by_gate")]
    assert [row["out"] for row in rows] == ["r0:None", "r1:b", None, "r3:d"]


def test_a_filtered_value_still_skips_an_opted_in_leaf() -> None:
    definition = workflow(
        [
            gate("keep", "$inputs.keep", ["maybe"]),
            step(Maybe, "maybe", value="$inputs.value"),
            step(Take, "take", value="$steps.maybe.out", tag="$inputs.tag"),
        ],
        {"out": "$steps.take.out"},
        inputs=strings("value", "tag") + [batch_input("keep")],
    )

    rows, skips = run(
        definition,
        {
            "value": ["none", "x", "y"],
            "tag": ["r0", "r1", "r2"],
            "keep": [True, False, True],
        },
    )

    assert calls("maybe") == [(None, "none"), (None, "y")]
    assert calls("take") == [("r0", None), ("r2", "y")]
    assert ("take", (1,), "filtered_input") in skips.skipped
    assert [row["out"] for row in rows] == ["r0:None", None, "r2:y"]


def test_a_missing_sibling_value_is_not_revived_by_the_other_leaf() -> None:
    # value is a present None (opted in); tag comes from a gated producer.
    definition = workflow(
        [
            gate("keep", "$inputs.keep", ["maybe"]),
            step(Maybe, "maybe", value="$inputs.tag"),
            step(Take, "take", value="$inputs.value", tag="$steps.maybe.out"),
        ],
        {"out": "$steps.take.out"},
        inputs=strings("value", "tag") + [batch_input("keep")],
    )

    rows, skips = run(
        definition,
        {"value": [None, None], "tag": ["r0", "r1"], "keep": [False, True]},
    )

    assert calls("take") == [("r1", None)]
    assert ("take", (0,), "filtered_input") in skips.skipped
    assert [row["out"] for row in rows] == [None, "r1:None"]


# --- gates -------------------------------------------------------------------


def test_a_denied_gate_wins_over_allow_none() -> None:
    definition = workflow(
        [
            gate("keep", "$inputs.keep", ["take"]),
            step(Take, "take", value="$inputs.value", tag="$inputs.tag"),
        ],
        {"out": "$steps.take.out"},
        inputs=strings("value", "tag") + [batch_input("keep")],
    )

    rows, skips = run(
        definition,
        {"value": [None, None], "tag": ["r0", "r1"], "keep": [False, True]},
    )

    assert calls("take") == [("r1", None)]
    assert skips.skipped == [("take", (0,), "denied_by_gate")]
    assert [row["out"] for row in rows] == [None, "r1:None"]


def test_two_gates_are_a_conjunction_for_an_opted_in_step() -> None:
    definition = workflow(
        [
            gate("first", "$inputs.a", ["take"]),
            gate("second", "$inputs.b", ["take"]),
            step(Take, "take", value="$inputs.value", tag="$inputs.tag"),
        ],
        {"out": "$steps.take.out"},
        inputs=strings("value", "tag") + [batch_input("a"), batch_input("b")],
    )

    rows, skips = run(
        definition,
        {
            "value": [None] * 4,
            "tag": ["r0", "r1", "r2", "r3"],
            "a": [True, True, False, False],
            "b": [True, False, True, False],
        },
    )

    assert calls("take") == [("r0", None)]
    assert [reason for *_, reason in skips.skipped] == ["denied_by_gate"] * 3
    assert [row["out"] for row in rows] == ["r0:None", None, None, None]


def test_a_whole_child_gate_governs_an_opted_in_step_inside_the_child() -> None:
    child = workflow(
        [step(Take, "take", value="$inputs.x", tag="$inputs.t")],
        {"out": "$steps.take.out"},
        inputs=[parameter("x", kind=["string"]), parameter("t", kind=["string"])],
    )
    definition = workflow(
        [
            gate("keep", "$inputs.keep", ["child"]),
            nested(
                "child",
                workflow_definition=child,
                parameter_bindings={"x": "$inputs.value", "t": "$inputs.tag"},
            ),
        ],
        {"out": "$steps.child.out"},
        inputs=strings("value", "tag") + [batch_input("keep")],
    )

    rows, _ = run(
        definition,
        {
            "value": [None, None, "c"],
            "tag": ["r0", "r1", "r2"],
            "keep": [True, False, True],
        },
    )

    assert calls("take") == [("r0", None), ("r2", "c")]
    assert [row["out"] for row in rows] == ["r0:None", None, "r2:c"]


def test_a_denied_ancestor_and_a_sparse_expansion_keep_their_skips() -> None:
    # Parent gate on the expansion; child gate admits even children only.
    definition = workflow(
        [
            gate("parents", "$inputs.keep", ["spread"]),
            step(Spread, "spread", items="$inputs.items", tag="$inputs.tag"),
            gate("children", "$steps.spread.even", ["take"]),
            step(Take, "take", value="$steps.spread.child", tag="$steps.spread.label"),
        ],
        {"out": "$steps.take.out"},
        inputs=[
            batch_input("items"),
            batch_input("tag", kind=["string"]),
            batch_input("keep"),
        ],
    )

    rows, skips = run(
        definition,
        {
            "items": [[None, "b"], [None], ["c", None, None]],
            "tag": ["r0", "r1", "r2"],
            "keep": [True, False, True],
        },
    )

    assert calls("spread") == [("r0", [None, "b"]), ("r2", ["c", None, None])]
    assert calls("take") == [("r0.0", None), ("r2.0", "c"), ("r2.2", None)]
    assert sorted(index for name, index, _ in skips.skipped if name == "take") == [
        (0, 1),
        (2, 1),
    ]
    # The denied parent has no children: its list renders empty.
    assert [row["out"] for row in rows] == [
        ["r0.0:None", None],
        [],
        ["r2.0:c", None, "r2.2:None"],
    ]


def test_an_empty_accepting_join_keeps_its_recovery_behaviour() -> None:
    definition = workflow(
        [
            gate("keep", "$inputs.keep", ["maybe"]),
            step(Maybe, "maybe", value="$inputs.value"),
            {"type": Merge.type, "name": "merge", "values": ["$steps.maybe.out"]},
        ],
        {"out": "$steps.merge.value"},
        inputs=strings("value") + [batch_input("keep")],
    )

    rows, skips = run(definition, {"value": ["none", "x"], "keep": [True, False]})

    assert skips.skipped == [("maybe", (1,), "denied_by_gate")]
    assert [row["out"] for row in rows] == [None, None]


# --- validation -----------------------------------------------------------------


def test_the_authors_validator_still_rejects_a_selected_none() -> None:
    definition = workflow(
        [step(Guarded, "guarded", value="$inputs.value", policy="strict")],
        {"out": "$steps.guarded.out"},
        inputs=strings("value"),
    )

    with pytest.raises(StepExecutionError) as caught:
        run(definition, {"value": ["a", None]})

    assert isinstance(caught.value.__cause__, ResolvedParameterError)
    assert "strict policy needs a value" in str(caught.value.__cause__)
    assert CALLS == []  # validation precedes every call of the step


def test_a_wrong_non_none_kind_still_fails_an_opted_in_leaf() -> None:
    definition = workflow(
        [step(Take, "take", value="$inputs.value", tag="$inputs.tag")],
        {"out": "$steps.take.out"},
        inputs=[batch_input("value"), batch_input("tag", kind=["string"])],
    )

    with pytest.raises(StepExecutionError) as caught:
        run(definition, {"value": [None, 7], "tag": ["r0", "r1"]})

    assert isinstance(caught.value.__cause__, ResolvedParameterError)
    assert caught.value.__cause__.field_path == ("value",)
    assert CALLS == []


# --- active sources and operators -------------------------------------------------


def _start(definition: dict, feeds: Dict[str, list], *groups: str) -> Collector:
    started = threading.Event()
    started.set()
    plan = compile_workflow(definition, catalogue=ACTIVE_CATALOGUE)
    session = plan.create_session(
        resources={"feeds": feeds, "log": Log(), "started": started}
    )
    collected = Collector()
    active_run = session.start(handlers=collected.handlers(*groups))
    assert active_run.wait(WAIT)
    assert active_run.state == "finished"
    session.close()

    return collected


def test_an_emitted_none_is_a_payload_and_read_none_ends_the_source() -> None:
    # Pulse 2 emits no label (absent); after pulse 3 read() returns None (end).
    # An ungrouped port is one constant value per pulse: as before this
    # change, None reaches the default block too; only absence skips.
    feeds = {
        "s": [
            emit(label=None, extra="p0"),
            emit(label="x", extra="p1"),
            emit(value=1.0, extra="p2"),
            emit(label=None, extra="p3"),
        ]
    }
    definition = active(
        [source("s")],
        [
            step(Take, "take", value="$sources.s.label", tag="$sources.s.extra"),
            step(Strict, "strict", value="$sources.s.label", tag="$sources.s.extra"),
        ],
        [
            group(
                "G",
                "$sources.s.extra",
                take="$steps.take.out",
                strict="$steps.strict.out",
            )
        ],
    )

    collected = _start(definition, feeds, "G")

    assert calls("take") == [("p0", None), ("p1", "x"), ("p3", None)]
    assert calls("strict") == [("p0", None), ("p1", "x"), ("p3", None)]
    assert collected.rows("G") == [
        {"take": "p0:None", "strict": "p0:None"},
        {"take": "p1:x", "strict": "p1:x"},
        {"take": None, "strict": None},
        {"take": "p3:None", "strict": "p3:None"},
    ]


def test_an_emitted_none_item_reaches_only_the_opted_in_step() -> None:
    feeds = {
        "s": [
            emit(label=Batch.of([None, "x", None]), tag=Batch.of(["t0", "t1", "t2"])),
            emit(label=Batch.of([None]), tag=Batch.of(["u0"])),
        ]
    }
    definition = active(
        [{"type": Items.type, "name": "s", "feed": "s"}],
        [
            step(Take, "take", value="$sources.s.label", tag="$sources.s.tag"),
            step(Strict, "strict", value="$sources.s.label", tag="$sources.s.tag"),
        ],
        [
            group(
                "G",
                "$sources.s.label",
                take="$steps.take.out",
                strict="$steps.strict.out",
            )
        ],
    )

    collected = _start(definition, feeds, "G")

    assert calls("take") == [("t0", None), ("t1", "x"), ("t2", None), ("u0", None)]
    assert calls("strict") == [("t1", "x")]
    first, second = collected.results["G"]
    assert first.rows() == [
        {"take": "t0:None", "strict": None},
        {"take": "t1:x", "strict": "t1:x"},
        {"take": "t2:None", "strict": None},
    ]
    assert second.rows() == [{"take": "u0:None", "strict": None}]


def test_operator_outputs_carry_a_present_none_to_opted_in_steps() -> None:
    feeds = {
        "a": [emit(label=None, pts=0), emit(label="x", pts=100)],
        "b": [emit(extra="b0", pts=0), emit(extra="b1", pts=100)],
    }
    definition = active(
        [source("a"), source("b")],
        [
            step(Take, "take", value="$operators.pair.a", tag="$operators.pair.b"),
            step(Strict, "strict", value="$operators.pair.a", tag="$operators.pair.b"),
            step(Collect, "collect", values="$operators.clip.x"),
        ],
        [
            group("P", "$operators.pair.a", take="$steps.take.out"),
            group("W", "$operators.clip.x", seen="$steps.collect.seen"),
        ],
    )
    definition["operators"] = [
        {
            "type": Align.type,
            "name": "pair",
            "inputs": {"a": "$sources.a.label", "b": "$sources.b.extra"},
            "clock": "media",
        },
        {
            "type": Window.type,
            "name": "clip",
            "size": 2,
            "collect": {"x": "$sources.a.label"},
        },
    ]

    collected = _start(definition, feeds, "P", "W")

    # Aligned ports are constant per pulse, like an ungrouped source port.
    assert calls("take") == [("b0", None), ("b1", "x")]
    assert calls("strict") == [("b0", None), ("b1", "x")]
    assert collected.rows("P") == [{"take": "b0:None"}, {"take": "b1:x"}]
    # A time window keeps the None payload; Group children are unchanged.
    assert calls("collect") == [(None, [None, "x"])]
