"""Integrated runs of child input boundaries: compile, create a session, run, rows.

Every block here records its calls, so each test asserts what ran as well as
what came out.
"""

from typing import Any, List

import pytest
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
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
from roboflow_workflows.execution_engine.v2.errors import WorkflowExecutionError
from roboflow_workflows.execution_engine.v2.kinds import (
    DICTIONARY_KIND,
    FLOAT_KIND,
    INTEGER_KIND,
    Kind,
)

from tests.unit_tests.execution_engine.v2.compilation.fixture_blocks import (
    batch_input,
    nested,
    parameter,
    workflow,
)

CALLS: List[tuple] = []
DECODED: List[Any] = []


def _decode(value: Any) -> dict:
    DECODED.append(value)
    return {"decoded": int(value)}


# A kind whose JSON form is a string and whose payload is a mutable dict.
CODED = Kind(
    name="coded",
    validate=lambda payload: isinstance(payload, dict) and "decoded" in payload,
    deserialize=_decode,
)
LABEL = Kind(name="label", validate=lambda payload: isinstance(payload, bytes))


class Record(Block):
    """Returns its value and records the call."""

    type = "runs/record@v1"
    outputs = {"value": Output(source="value")}

    class Params(BlockParams):
        value: Ref()

    def run(self, *, value) -> dict:
        CALLS.append(("record", value))
        return {"value": value}


class Pair(Block):
    """Receives two values and reports whether they are one object."""

    type = "runs/pair@v1"
    outputs = {"same": Output()}

    class Params(BlockParams):
        left: Ref()
        right: Ref()

    def run(self, *, left, right) -> dict:
        CALLS.append(("pair", left, right))
        return {"same": left is right}


class Children(Block):
    """Expands an integer n into children 10n, 10n+1, ... (n children)."""

    type = "runs/children@v1"
    outputs = {"child": Output(INTEGER_KIND, expand="child")}

    class Params(BlockParams):
        value: Ref(INTEGER_KIND)

    def run(self, *, value) -> dict:
        return {"child": Batch.of([10 * value + k for k in range(value)])}


class Keep(Block):
    """Continues to the targets when the value is truthy."""

    type = "runs/keep@v1"

    class Params(BlockParams):
        value: Ref()
        next_steps: List[StepRef]

    def run(self, *, value, next_steps):
        return Select(next_steps) if value else Stop()


class Collect(Block):
    """Collapses a group into a list and records the group indices."""

    type = "runs/collect@v1"
    outputs = {"values": Output()}

    class Params(BlockParams):
        data: Group()

    def run(self, *, data) -> dict:
        CALLS.append(("collect", list(data.indices)))
        return {"values": list(data)}


class Labelled(Block):
    """Group of labels whose literal alternative is a different type."""

    type = "runs/labelled@v1"
    outputs = {"labels": Output()}

    class Params(BlockParams):
        labels: str | Group(LABEL) = "default"

    def run(self, *, labels) -> dict:
        CALLS.append(("labelled", list(labels), list(labels.indices)))
        return {"labels": list(labels)}


class Bump(Block):
    """Mutates a decoded dictionary in place."""

    type = "runs/bump@v1"
    mutates = ("value",)
    outputs = {"value": Output(DICTIONARY_KIND, source="value")}

    class Params(BlockParams):
        value: Ref(DICTIONARY_KIND, CODED)

    def run(self, *, value) -> dict:
        value["decoded"] += 100
        CALLS.append(("bump", value["decoded"]))
        return {"value": value}


CATALOGUE = Catalogue(
    [Record, Pair, Children, Keep, Collect, Labelled, Bump],
    kinds=[CODED, LABEL, FLOAT_KIND, INTEGER_KIND],
)


@pytest.fixture(autouse=True)
def _reset_records():
    CALLS.clear()
    DECODED.clear()


def block(name: str, step_name: str, **params) -> dict:
    return {"type": f"runs/{name}@v1", "name": step_name, **params}


def _plan(steps, outputs, *, inputs=None):
    plan = compile_workflow(
        workflow(steps, outputs, inputs=[] if inputs is None else inputs),
        catalogue=CATALOGUE,
    )

    return plan


def test_direct_forwarding_of_defaults_literals_and_selected_values() -> None:
    forward = workflow(
        [block("record", "unrelated", value="$inputs.other")],
        {"value": "$inputs.value"},
        inputs=[
            parameter("value", default=7, kind=["integer"]),
            parameter("other", default=0),
        ],
    )
    plan = _plan(
        [
            nested("defaulted", workflow_definition=forward),
            nested(
                "literal", workflow_definition=forward, parameter_bindings={"value": 8}
            ),
            nested(
                "selected",
                workflow_definition=forward,
                parameter_bindings={"value": "$inputs.anything"},
            ),
        ],
        {
            "defaulted": "$steps.defaulted.value",
            "literal": "$steps.literal.value",
            "selected": "$steps.selected.value",
        },
        inputs=[parameter("anything")],
    )

    rows = plan.create_session().run({"anything": 9}).rows()

    assert rows == [{"defaulted": 7, "literal": 8, "selected": 9}]


def test_wildcard_to_typed_value_is_rejected_before_any_child_call_at_every_depth() -> (
    None
):
    leaf = workflow(
        [block("record", "echo", value="$inputs.x")],
        {"y": "$steps.echo.value"},
        inputs=[parameter("x", kind=["integer"])],
    )
    middle = workflow(
        [
            nested(
                "leaf", workflow_definition=leaf, parameter_bindings={"x": "$inputs.m"}
            )
        ],
        {"y": "$steps.leaf.y"},
        inputs=[parameter("m")],
    )
    plan = _plan(
        [
            nested(
                "middle",
                workflow_definition=middle,
                parameter_bindings={"m": "$inputs.anything"},
            )
        ],
        {"y": "$steps.middle.y"},
        inputs=[parameter("anything")],
    )
    assert plan.create_session().run({"anything": 4}).rows() == [{"y": 4}]
    with pytest.raises(WorkflowExecutionError) as info:
        plan.create_session().run({"anything": "wrong-type"})

    assert CALLS == [("record", 4)], "the invalid run made no child call"
    assert "$steps.middle/leaf" in str(info.value) and "'integer'" in str(info.value)


def test_directly_forwarded_child_input_is_checked_only_when_typed() -> None:
    def forward_only(kind):
        return workflow(
            [block("record", "unrelated", value="$inputs.other")],
            {"value": "$inputs.value"},
            inputs=[
                parameter("value", kind=None if kind is None else [kind]),
                parameter("other", default=0),
            ],
        )

    def plan_for(kind):
        plan = _plan(
            [
                nested(
                    "child",
                    workflow_definition=forward_only(kind),
                    parameter_bindings={"value": "$inputs.anything"},
                )
            ],
            {"value": "$steps.child.value"},
            inputs=[parameter("anything")],
        )
        return plan

    assert plan_for(None).create_session().run({"anything": "text"}).rows() == [
        {"value": "text"}
    ]
    with pytest.raises(
        WorkflowExecutionError, match=r"\$steps\.child: \$inputs\.value"
    ):
        plan_for("integer").create_session().run({"anything": "text"})


def test_constant_is_decoded_once_per_run_and_shared_by_all_consumers() -> None:
    child = workflow(
        [
            block("pair", "pair", left="$inputs.coded", right="$inputs.coded"),
            block("record", "record", value="$inputs.coded"),
        ],
        {"same": "$steps.pair.same", "direct": "$inputs.coded"},
        inputs=[parameter("coded", default="7", kind=["coded"])],
    )
    plan = _plan(
        [
            nested("defaulted", workflow_definition=child),
            nested(
                "literal", workflow_definition=child, parameter_bindings={"coded": "8"}
            ),
        ],
        {
            "same": "$steps.defaulted.same",
            "direct": "$steps.defaulted.direct",
            "literal": "$steps.literal.direct",
        },
    )
    assert DECODED == [], "compilation decodes nothing"

    session = plan.create_session()
    first = session.run({}).rows()
    second = session.run({}).rows()
    fresh = plan.create_session().run({}).rows()

    assert (
        first
        == second
        == fresh
        == [{"same": True, "direct": {"decoded": 7}, "literal": {"decoded": 8}}]
    )
    assert DECODED == ["7", "8"] * 3, "one decode per boundary per run"
    pair_calls = [call for call in CALLS if call[0] == "pair"]
    assert all(call[1] is call[2] for call in pair_calls)
    assert pair_calls[0][1] is not pair_calls[2][1], "no reuse across runs"


def test_mutating_a_decoded_constant_is_seen_within_a_run_and_fresh_next_run() -> None:
    child = workflow(
        [
            block("bump", "bump", value="$inputs.coded"),
            block("record", "after", value="$steps.bump.value"),
        ],
        {"after": "$steps.after.value", "direct": "$inputs.coded"},
        inputs=[parameter("coded", default="1", kind=["coded"])],
    )
    plan = _plan(
        [nested("child", workflow_definition=child)],
        {"after": "$steps.child.after", "direct": "$steps.child.direct"},
    )
    session = plan.create_session()

    rows = [session.run({}).rows(), session.run({}).rows()]

    assert rows == [[{"after": {"decoded": 101}, "direct": {"decoded": 101}}]] * 2
    assert [call for call in CALLS if call[0] == "bump"] == [("bump", 101)] * 2


def test_filtered_and_grouped_values_match_the_flattened_workflow() -> None:
    upstream = [
        block("keep", "keep", value="$inputs.values", next_steps=["$steps.expand"]),
        block("children", "expand", value="$inputs.values"),
    ]
    child = workflow(
        [
            block("record", "echo", value="$inputs.children"),
            block("collect", "collect", data="$inputs.children"),
        ],
        {"echo": "$steps.echo.value", "collected": "$steps.collect.values"},
        inputs=[batch_input("children", kind=["integer"])],
    )
    nested_plan = _plan(
        upstream
        + [
            nested(
                "child",
                workflow_definition=child,
                parameter_bindings={"children": "$steps.expand.child"},
            )
        ],
        {"echo": "$steps.child.echo", "collected": "$steps.child.collected"},
        inputs=[batch_input("values", kind=["integer"])],
    )
    flat_plan = _plan(
        upstream
        + [
            block("record", "echo", value="$steps.expand.child"),
            block("collect", "collect", data="$steps.expand.child"),
        ],
        {"echo": "$steps.echo.value", "collected": "$steps.collect.values"},
        inputs=[batch_input("values", kind=["integer"])],
    )

    nested_rows = nested_plan.create_session().run({"values": [2, 0, 1]}).rows()
    nested_calls = list(CALLS)
    CALLS.clear()
    flat_rows = flat_plan.create_session().run({"values": [2, 0, 1]}).rows()

    assert nested_rows == flat_rows
    assert nested_calls == CALLS
    assert ("collect", [(0, 0), (0, 1)]) in CALLS and ("collect", [(2, 0)]) in CALLS
    assert [row["collected"] for row in nested_rows] == [[20, 21], None, [10]]


def test_whole_child_control_still_gates_steps_that_read_boundaries() -> None:
    child = workflow(
        [block("record", "echo", value="$inputs.x")],
        {"y": "$steps.echo.value"},
        inputs=[batch_input("x", kind=["integer"])],
    )
    plan = _plan(
        [
            block("keep", "keep", value="$inputs.values", next_steps=["$steps.child"]),
            nested(
                "child",
                workflow_definition=child,
                parameter_bindings={"x": "$inputs.values"},
            ),
        ],
        {"y": "$steps.child.y"},
        inputs=[batch_input("values", kind=["integer"])],
    )

    rows = plan.create_session().run({"values": [1, 0, 3]}).rows()

    assert rows == [{"y": 1}, {"y": None}, {"y": 3}]
    assert CALLS == [("record", 1), ("record", 3)]


def test_literal_group_with_a_disjoint_literal_type_is_cast_not_kind_checked() -> None:
    plan = _plan(
        [block("labelled", "literal", labels="plain")],
        {"labels": "$steps.literal.labels"},
    )

    rows = plan.create_session().run({}).rows()

    assert rows == [{"labels": ["plain"]}]
    assert CALLS == [("labelled", ["plain"], [(0,)])]


class Offset(Block):
    """Output context comes from a field that may also be a literal."""

    type = "runs/offset@v1"
    outputs = {"shifted": Output(INTEGER_KIND, source="base")}

    class Params(BlockParams):
        base: int | Ref(INTEGER_KIND)
        delta: int | Ref(INTEGER_KIND) = 1

    def run(self, *, base, delta) -> dict:
        return {"shifted": base + delta}


def test_literal_context_field_compiles_and_contributes_no_context() -> None:
    catalogue = Catalogue.merge(CATALOGUE, Catalogue([Offset]))
    definition = workflow(
        [{"type": "runs/offset@v1", "name": "offset", "base": 5, "delta": "$inputs.d"}],
        {"shifted": "$steps.offset.shifted"},
        inputs=[batch_input("d", kind=["integer"])],
    )

    plan = compile_workflow(definition, catalogue=catalogue)
    rows = plan.create_session().run({"d": [1, 2]}).rows()

    offset = plan.step(("offset",))
    assert offset.outputs["shifted"].source_field == "base"
    assert offset.bindings_for("base") == ()
    assert rows == [{"shifted": 6}, {"shifted": 7}]
