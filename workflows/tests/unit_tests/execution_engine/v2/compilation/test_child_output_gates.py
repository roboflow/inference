"""Whole-child gates govern values a child forwards past its steps (decision 026).

A child output of ``$inputs.x``, a literal or a default passes no child step,
so it is planned as a gated ``ChildOutputPort``. These tests compile, run and
read rows through the public API and assert which blocks ran, and in what
order. The IDs in comments refer to tasks/m1r-gate-planner/report.md.
"""

import dataclasses
from typing import List

import pytest
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Group,
    Output,
    Ref,
)
from roboflow_workflows.execution_engine.v2.errors import (
    ContractError,
    CycleError,
    LineageError,
    WorkflowCompileError,
)
from roboflow_workflows.execution_engine.v2.plan import (
    ChildInputPort,
    ChildOutputPort,
    StepPort,
)

from tests.unit_tests.execution_engine.v2.compilation import test_boundary_runs as runs
from tests.unit_tests.execution_engine.v2.compilation.fixture_blocks import (
    axes_of,
    batch_input,
    nested,
    parameter,
    workflow,
)
from tests.unit_tests.execution_engine.v2.compilation.test_boundary_runs import (
    CALLS,
    DECODED,
    block,
)


class Rejoin(Block):
    """Empty-accepting join: the first available value, recorded."""

    type = "runs/rejoin@v1"
    accepts_empty = True
    outputs = {"value": Output()}

    class Params(BlockParams):
        values: List[Ref()]

    def run(self, *, values) -> dict:
        CALLS.append(("rejoin", list(values)))
        return {"value": next((item for item in values if item is not None), None)}


class Gather(Block):
    """Empty-accepting reducer recording where it ran and the indices it got."""

    type = "runs/gather@v1"
    accepts_empty = True
    outputs = {"count": Output()}

    class Params(BlockParams):
        data: Group()

    def run(self, *, data) -> dict:
        CALLS.append(("gather", data.parent_index, list(data.indices)))
        return {"count": len(data)}


CATALOGUE = Catalogue.merge(runs.CATALOGUE, Catalogue([Rejoin, Gather]))


@pytest.fixture(autouse=True)
def _reset_records():
    CALLS.clear()
    DECODED.clear()


def forwarding_child(*, value_kind=None, default=None):
    """A nonempty child: its body records ``x``, its output forwards ``x``."""
    child = workflow(
        [block("record", "body", value="$inputs.x")],
        {"out": "$inputs.x"},
        inputs=[parameter("x", default=default, kind=value_kind)],
    )

    return child


def forwarding_only_child():
    """A nonempty child whose body does not read ``x``; its output forwards ``x``."""
    child = workflow(
        [block("record", "body", value="$inputs.tag")],
        {"out": "$inputs.x"},
        inputs=[parameter("x"), parameter("tag", default="tag")],
    )

    return child


def gate(name: str, value: str, target: str) -> dict:
    return block("keep", name, value=value, next_steps=[f"$steps.{target}"])


def _plan(steps, outputs, *, inputs):
    plan = compile_workflow(
        workflow(steps, outputs, inputs=inputs), catalogue=CATALOGUE
    )

    return plan


def _paths(plan) -> List[tuple]:
    return [step.path for step in plan.steps]


@pytest.mark.parametrize(
    "mask, rows, calls",
    [
        ([False, False, False], [None, None, None], []),
        ([False, True, False], [None, 20, None], [("record", 20), ("record", 20)]),
        (
            [True, True, True],
            [10, 20, 30],
            [("record", value) for value in (10, 20, 30)] * 2,
        ),
    ],
)
def test_forwarded_input_obeys_the_whole_child_gate_and_orders_its_consumer(
    mask, rows, calls
) -> None:
    # G01: the root repro, consumer declared before its controller.
    plan = _plan(
        [
            block("record", "consumer", value="$steps.child.out"),
            nested(
                "child",
                workflow_definition=forwarding_child(),
                parameter_bindings={"x": "$inputs.x"},
            ),
            gate("gate", "$inputs.keep", "child"),
        ],
        {"forwarded": "$steps.child.out", "consumed": "$steps.consumer.value"},
        inputs=[batch_input("x"), batch_input("keep")],
    )

    result = plan.create_session().run({"x": [10, 20, 30], "keep": mask})

    assert _paths(plan).index(("gate",)) < _paths(plan).index(("consumer",))
    assert plan.step(("consumer",)).dependencies == (("gate",),)
    assert [row["forwarded"] for row in result.rows()] == rows
    assert [row["consumed"] for row in result.rows()] == rows
    assert sorted(CALLS) == sorted(calls)


def test_scalar_literal_and_default_broadcast_only_over_admitted_indices() -> None:
    # G02: scalar sources under a batch gate; the coded default decodes once
    # per run although a direct output and a consumer both read it.
    child = workflow(
        [block("record", "body", value="$inputs.coded")],
        {
            "coded": "$inputs.coded",
            "literal": "$inputs.literal",
            "root": "$inputs.root",
        },
        inputs=[
            parameter("coded", default="7", kind=["coded"]),
            parameter("literal"),
            parameter("root"),
        ],
    )
    plan = _plan(
        [
            gate("gate", "$inputs.keep", "child"),
            nested(
                "child",
                workflow_definition=child,
                parameter_bindings={"literal": 8, "root": "$inputs.root"},
            ),
            block(
                "pair", "pair", left="$steps.child.coded", right="$steps.child.coded"
            ),
        ],
        {
            "coded": "$steps.child.coded",
            "literal": "$steps.child.literal",
            "root": "$steps.child.root",
            "same": "$steps.pair.same",
        },
        inputs=[batch_input("keep"), parameter("root")],
    )
    session = plan.create_session()

    rows = session.run({"keep": [False, True, False], "root": "r"}).rows()
    denied = session.run({"keep": [False, False, False], "root": "r"})

    port = plan.outputs[0].source
    assert isinstance(port, ChildOutputPort)
    assert axes_of(plan.child_output(port).layout) == ["inputs"], "the gate's axis"
    assert rows == [
        {"coded": None, "literal": None, "root": None, "same": None},
        {"coded": {"decoded": 7}, "literal": 8, "root": "r", "same": True},
        {"coded": None, "literal": None, "root": None, "same": None},
    ]
    assert DECODED == ["7", "7"], "once per run, shared by the output and the pair"
    assert [call for call in CALLS if call[0] == "pair"][0][1] is not None
    assert denied.statuses["coded"] == "filtered"
    assert (
        denied.rows()
        == [{"coded": None, "literal": None, "root": None, "same": None}] * 3
    )


def test_nested_axes_keep_original_indices_under_ancestor_and_descendant_gates() -> (
    None
):
    # G03 + G05: a deeper forwarded value under an ancestor gate, and a
    # parent-level value broadcast under a child-level gate, with a genuine
    # empty group and conjunction of gates at two depths.
    deep = workflow(
        [block("record", "body", value="$inputs.x")],
        {"out": "$inputs.x", "stepped": "$steps.body.value"},
        inputs=[batch_input("x", kind=["integer"])],
    )
    plan = _plan(
        [
            block("children", "expand", value="$inputs.values"),
            gate("parent_gate", "$inputs.keep", "deep"),
            nested(
                "deep",
                workflow_definition=deep,
                parameter_bindings={"x": "$steps.expand.child"},
            ),
            gate("child_gate", "$steps.expand.child", "shallow"),
            gate("also_parent", "$inputs.keep", "shallow"),
            nested(
                "shallow",
                workflow_definition=forwarding_only_child(),
                parameter_bindings={"x": "$inputs.values"},
            ),
        ],
        {
            "deep": "$steps.deep.out",
            "stepped": "$steps.deep.stepped",
            "shallow": "$steps.shallow.out",
        },
        inputs=[batch_input("values", kind=["integer"]), batch_input("keep")],
    )

    result = plan.create_session().run(
        {"values": [2, 0, 1], "keep": [True, True, False]}
    )
    rows = result.rows()

    shallow = plan.child_output(plan.outputs[2].source)
    assert axes_of(shallow.layout) == ["inputs", "expand:child"]
    assert [gate.controller for gate in shallow.gates] == [
        ("child_gate",),
        ("also_parent",),
    ]
    assert rows == [
        {"deep": [20, 21], "stepped": [20, 21], "shallow": [2, 2]},
        {"deep": [], "stepped": [], "shallow": []},
        {"deep": [], "stepped": [None], "shallow": []},
    ], "a filtered group renders as [] in rows (existing row convention)"
    assert result.filtered_paths["deep"] == ((2,),), "the denied group itself"
    assert result.filtered_paths["shallow"] == ((2,),)


@pytest.mark.parametrize(
    "keep, denied, gathered",
    [
        pytest.param(
            [False, True, True],
            (0,),
            [((1,), []), ((2,), [(2, 0)])],
            id="denied-nonempty-group",
        ),
        pytest.param(
            [True, False, True],
            (1,),
            [((0,), [(0, 0), (0, 1)]), ((2,), [(2, 0)])],
            id="denied-genuine-empty-group",
        ),
    ],
)
def test_a_denied_group_is_absent_even_for_an_empty_accepting_reducer(
    keep, denied, gathered
) -> None:
    # CF-02 / RF-C2: an ancestor gate filters the group node itself, so a
    # denied (possibly genuinely empty) group never reaches a reducer, while
    # an admitted genuine empty group still does.
    deep = workflow(
        [block("record", "body", value="$inputs.x")],
        {"out": "$inputs.x"},
        inputs=[batch_input("x", depth=2)],
    )
    plan = _plan(
        [
            gate("gate", "$inputs.keep", "child"),
            nested(
                "child", workflow_definition=deep, parameter_bindings={"x": "$inputs.x"}
            ),
            block("gather", "gather", data="$steps.child.out"),
        ],
        {"out": "$steps.child.out"},
        inputs=[batch_input("x", depth=2), batch_input("keep")],
    )

    result = plan.create_session().run({"x": [[10, 11], [], [30]], "keep": keep})

    assert result.filtered_paths["out"] == (denied,), "the group node, not its leaves"
    assert [(call[1], call[2]) for call in CALLS if call[0] == "gather"] == gathered


def test_chained_forwarding_through_nested_children_combines_their_gates() -> None:
    # G04: leaf forwards, middle forwards the leaf's output; the inner gate
    # lives in middle, the outer one in the root. A sibling child reads the
    # forwarded value into its own body.
    middle = workflow(
        [
            gate("inner_gate", "$inputs.inner", "leaf"),
            {
                "type": "roboflow_core/inner_workflow@v1",
                "name": "leaf",
                "workflow_definition": forwarding_child(),
                "parameter_bindings": {"x": "$inputs.x"},
            },
        ],
        {"out": "$steps.leaf.out"},
        inputs=[batch_input("x"), batch_input("inner")],
    )
    reader = workflow(
        [block("record", "read", value="$inputs.y")], inputs=[batch_input("y")]
    )
    plan = _plan(
        [
            gate("outer_gate", "$inputs.outer", "middle"),
            nested(
                "middle",
                workflow_definition=middle,
                parameter_bindings={"x": "$inputs.x", "inner": "$inputs.inner"},
            ),
            nested(
                "reader",
                workflow_definition=reader,
                parameter_bindings={"y": "$steps.middle.out"},
            ),
        ],
        {"out": "$steps.middle.out"},
        inputs=[batch_input("x"), batch_input("inner"), batch_input("outer")],
    )

    rows = (
        plan.create_session()
        .run(
            {"x": [1, 2, 3], "inner": [True, False, True], "outer": [True, True, False]}
        )
        .rows()
    )

    outer = plan.child_output(plan.outputs[0].source)
    assert outer.source == ChildOutputPort(("middle", "leaf"), "out")
    assert plan.child_input(ChildInputPort(("reader",), "y")).source == outer.port
    assert rows == [{"out": 1}, {"out": None}, {"out": None}]
    assert sorted(CALLS) == [("record", 1), ("record", 1)], (
        "the leaf body and the sibling reader run only at index 0, where the inner "
        "and the outer gate both admit"
    )


def test_parent_source_and_sibling_uses_are_unaffected_and_payloads_are_shared() -> (
    None
):
    # G06: the same input feeds an ungated root consumer and two differently
    # gated child copies; filtering is per use, payload identity is shared.
    payloads = [{"n": 0}, {"n": 1}]
    plan = _plan(
        [
            gate("gate_a", "$inputs.a", "first"),
            gate("gate_b", "$inputs.b", "second"),
            nested(
                "first",
                workflow_definition=forwarding_child(),
                parameter_bindings={"x": "$inputs.x"},
            ),
            nested(
                "second",
                workflow_definition=forwarding_child(),
                parameter_bindings={"x": "$inputs.x"},
            ),
            block("pair", "same", left="$inputs.x", right="$steps.first.out"),
        ],
        {
            "root": "$inputs.x",
            "first": "$steps.first.out",
            "second": "$steps.second.out",
        },
        inputs=[batch_input("x"), batch_input("a"), batch_input("b")],
    )

    rows = (
        plan.create_session()
        .run({"x": payloads, "a": [True, False], "b": [False, True]})
        .rows()
    )

    assert rows == [
        {"root": {"n": 0}, "first": {"n": 0}, "second": None},
        {"root": {"n": 1}, "first": None, "second": {"n": 1}},
    ]
    [pair] = [call for call in CALLS if call[0] == "pair"]
    assert pair[1] is pair[2] is payloads[0], "borrowed payload, never copied"


def test_recovery_join_sees_denied_forwarded_data_as_missing() -> None:
    # G07: an empty-accepting join recovers the sibling; an ordinary consumer
    # skips the denied position.
    plan = _plan(
        [
            gate("gate", "$inputs.keep", "child"),
            nested(
                "child",
                workflow_definition=forwarding_child(),
                parameter_bindings={"x": "$inputs.x"},
            ),
            block("rejoin", "join", values=["$steps.child.out", "$inputs.fallback"]),
            block("record", "ordinary", value="$steps.child.out"),
        ],
        {"joined": "$steps.join.value"},
        inputs=[batch_input("x"), batch_input("keep"), batch_input("fallback")],
    )

    rows = (
        plan.create_session()
        .run({"x": [1, 2], "keep": [True, False], "fallback": ["f1", "f2"]})
        .rows()
    )

    assert rows == [{"joined": 1}, {"joined": "f2"}]
    assert ("rejoin", [None, "f2"]) in CALLS
    assert [call for call in CALLS if call[0] == "record"] == [
        ("record", 1),
        ("record", 1),
    ]


def test_gate_reading_the_output_it_gates_is_a_compile_time_cycle() -> None:
    # G08: self-gating and mutual gating through forwarded outputs.
    own = [
        nested(
            "child",
            workflow_definition=forwarding_child(),
            parameter_bindings={"x": "$inputs.x"},
        ),
        gate("gate", "$steps.child.out", "child"),
    ]
    mutual = [
        nested(
            "a",
            workflow_definition=forwarding_child(),
            parameter_bindings={"x": "$inputs.x"},
        ),
        nested(
            "b",
            workflow_definition=forwarding_child(),
            parameter_bindings={"x": "$inputs.x"},
        ),
        gate("gate_a", "$steps.b.out", "a"),
        gate("gate_b", "$steps.a.out", "b"),
    ]

    for steps in (own, mutual):
        with pytest.raises(CycleError):
            _plan(steps, {}, inputs=[batch_input("x")])
    assert CALLS == []


def test_step_outputs_and_wildcards_of_a_child_keep_their_existing_path() -> None:
    # G09: a child step's output is gated by the step itself; it gets no
    # extra boundary, so the mask applies once. Bodyless children still fail.
    child = workflow(
        [block("record", "body", value="$inputs.x")],
        {"stepped": "$steps.body.value", "all": "$steps.body.*"},
        inputs=[batch_input("x")],
    )
    plan = _plan(
        [
            gate("gate", "$inputs.keep", "child"),
            nested(
                "child",
                workflow_definition=child,
                parameter_bindings={"x": "$inputs.x"},
            ),
        ],
        {"stepped": "$steps.child.stepped", "all": "$steps.child.all"},
        inputs=[batch_input("x"), batch_input("keep")],
    )

    rows = plan.create_session().run({"x": [1, 2], "keep": [False, True]}).rows()

    assert [output.source for output in plan.outputs] == [
        StepPort(("child", "body"), "value"),
        StepPort(("child", "body"), "*"),
    ]
    assert plan.child_outputs == ()
    assert rows == [
        {"stepped": None, "all": None},
        {"stepped": 2, "all": {"value": 2}},
    ]
    with pytest.raises(WorkflowCompileError, match="nested workflow has no steps"):
        _plan(
            [
                nested(
                    "empty",
                    workflow_definition=workflow(
                        [], {"x": "$inputs.x"}, inputs=[batch_input("x")]
                    ),
                    parameter_bindings={"x": "$inputs.x"},
                )
            ],
            {},
            inputs=[batch_input("x")],
        )


def test_mutation_through_a_forwarded_output_conflicts_with_the_original_reader() -> (
    None
):
    # G11 (compile side): the gated output is a view, not a copy.
    child = workflow(
        [block("record", "body", value="$inputs.p")],
        {"out": "$inputs.p"},
        inputs=[parameter("p", kind=["dictionary"])],
    )
    plan = compile_workflow(
        workflow(
            [
                gate("gate", "$inputs.keep", "child"),
                nested(
                    "child",
                    workflow_definition=child,
                    parameter_bindings={"p": "$inputs.p"},
                ),
                block("bump", "bump", value="$steps.child.out"),
                block("record", "reader", value="$inputs.p"),
            ],
            inputs=[parameter("p", kind=["dictionary"]), parameter("keep")],
        ),
        catalogue=CATALOGUE,
    )

    assert any(
        warning.startswith("$steps.bump value ($steps.child.out)")
        and "$steps.reader value ($inputs.p)" in warning
        for warning in plan.warnings
    )


def test_hand_built_plans_cannot_skip_boundary_validation() -> None:
    # G11 (plan side): the plan constructor rejects inconsistent boundaries.
    plan = _plan(
        [
            gate("gate", "$inputs.keep", "child"),
            nested(
                "child",
                workflow_definition=forwarding_child(),
                parameter_bindings={"x": "$inputs.x"},
            ),
            block("record", "consumer", value="$steps.child.out"),
        ],
        {},
        inputs=[batch_input("x"), batch_input("keep")],
    )
    [output] = plan.child_outputs
    looping = dataclasses.replace(output, source=output.port)
    ungoverning = dataclasses.replace(
        output, gates=(dataclasses.replace(output.gates[0], target="$steps.consumer"),)
    )

    cases = [
        (dict(child_outputs=()), "unknown"),
        (dict(child_outputs=(looping,)), "cycle"),
        (dict(child_outputs=(ungoverning,)), "does not govern"),
        (dict(steps=plan.steps[::-1]), "earlier step"),
    ]
    for changes, fragment in cases:
        with pytest.raises(ContractError, match=fragment):
            dataclasses.replace(plan, **changes)


def test_unrelated_gate_and_forwarded_axes_are_rejected_at_compile_time() -> None:
    # G12: equal sizes never make unrelated axes correspond.
    with pytest.raises(LineageError, match="unrelated axes") as info:
        _plan(
            [
                block("children", "left", value="$inputs.values"),
                block("children", "right", value="$inputs.values"),
                gate("gate", "$steps.right.child", "child"),
                nested(
                    "child",
                    workflow_definition=forwarding_only_child(),
                    parameter_bindings={"x": "$steps.left.child"},
                ),
            ],
            {"out": "$steps.child.out"},
            inputs=[batch_input("values", kind=["integer"])],
        )

    assert info.value.step_path == ("child",)
    assert info.value.field_path == ("outputs", "out")


def test_a_default_forwarded_through_a_gated_output_keeps_its_own_identity() -> None:
    # Planner caveat: a child input over a constant is its own payload origin
    # (decoded once per run); a child output passes that origin on.
    child = workflow(
        [block("record", "body", value="$inputs.p")],
        {"out": "$inputs.p"},
        inputs=[parameter("p", default={"n": 0}, kind=["dictionary"])],
    )
    plan = _plan(
        [
            gate("gate", "$inputs.keep", "child"),
            nested("child", workflow_definition=child),
            block("bump", "bump", value="$steps.child.out"),
            block("record", "reader", value="$steps.child.out"),
        ],
        {},
        inputs=[parameter("keep")],
    )

    readers = {
        warning.split(", and ")[1].split(" value ")[0]
        for warning in plan.warnings
        if warning.startswith("$steps.bump value")
    }
    assert readers == {
        "$steps.child/body",
        "$steps.reader",
    }, "the body and the parent reader share the one decoded default"


def test_a_filtered_source_prefix_stays_filtered_under_a_deeper_gate_broadcast() -> (
    None
):
    # CF-02 follow-up: a source filtered upstream at (0,) is broadcast under an
    # admitting [N, C] child gate; its group stays absent instead of becoming
    # filtered children (and an empty group for an empty-accepting reducer).
    plan = _plan(
        [
            gate("upstream", "$inputs.pass", "pre"),
            block("record", "pre", value="$inputs.values"),
            block("children", "expand", value="$inputs.values"),
            gate("child_gate", "$steps.expand.child", "child"),
            nested(
                "child",
                workflow_definition=forwarding_only_child(),
                parameter_bindings={"x": "$steps.pre.value"},
            ),
            block("gather", "gather", data="$steps.child.out"),
        ],
        {"out": "$steps.child.out", "original": "$inputs.values"},
        inputs=[batch_input("values", kind=["integer"]), batch_input("pass")],
    )

    result = plan.create_session().run({"values": [2, 1], "pass": [False, True]})

    assert axes_of(plan.child_output(plan.outputs[0].source).layout) == [
        "inputs",
        "expand:child",
    ]
    assert result.filtered_paths["out"] == ((0,),)
    assert result.rows() == [{"out": [], "original": 2}, {"out": [1], "original": 1}]
    assert [(call[1], call[2]) for call in CALLS if call[0] == "gather"] == [
        ((1,), [(1, 0)])
    ]


@pytest.mark.parametrize("keep", [False, True])
@pytest.mark.parametrize("forwarded", [False, True])
@pytest.mark.parametrize("include_original", [False, True])
def test_scalar_child_gate_preserves_known_rows(
    keep: bool, forwarded: bool, include_original: bool
) -> None:
    child = forwarding_child()
    if not forwarded:
        child["outputs"][0]["selector"] = "$steps.body.value"
    outputs = {"out": "$steps.child.out"}
    if include_original:
        outputs["original"] = "$inputs.x"
    plan = _plan(
        [
            nested(
                "child",
                workflow_definition=child,
                parameter_bindings={"x": "$inputs.x"},
            ),
            gate("gate", "$inputs.keep", "child"),
        ],
        outputs,
        inputs=[batch_input("x"), parameter("keep")],
    )
    values = [10, 20, 30]
    result = plan.create_session().run({"x": values, "keep": keep})
    expected = [{"out": value if keep else None} for value in values]
    if include_original:
        for row, value in zip(expected, values):
            row["original"] = value

    assert result.rows() == expected
    assert result.rows(serialize=True) == expected
    assert result.input_row_count == 3
    body_calls = [call for call in CALLS if call[0] == "record"]
    assert len(body_calls) == (3 if keep else 0)
    if not keep:
        assert result.statuses["out"] == "filtered"
        assert "out" not in result.outputs.data
        if forwarded:
            assert result.filtered_paths["out"] == ((),)
