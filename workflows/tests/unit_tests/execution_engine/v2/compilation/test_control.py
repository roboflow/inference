"""Control targets, gates, control-derived invocation domains and ordering."""

import pytest
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.errors import (
    CycleError,
    LineageError,
    SelectorError,
)

from tests.unit_tests.execution_engine.v2.compilation.fixture_blocks import (
    CATALOGUE,
    axes_of,
    batch_input,
    gate,
    nested,
    parameter,
    step,
    workflow,
)

ITEMS = [batch_input("items")]
EXPAND = step("expand", "expand", value="$inputs.values")


def _compile(steps, *, inputs=None, outputs=None):
    plan = compile_workflow(
        workflow(steps, outputs, inputs=inputs), catalogue=CATALOGUE
    )

    return plan


def _gates(plan, path):
    return [(gate.controller, gate.target) for gate in plan.step(path).gates]


def test_step_refs_become_control_edges_and_controllers_run_first() -> None:
    plan = _compile(
        [
            step("sink", "notice", payload="alert"),
            step("echo", "echo", value="$inputs.items"),
            gate("gate", "$inputs.items", ["notice", "echo"]),
        ],
        inputs=ITEMS,
    )

    assert [step.path for step in plan.steps] == [("gate",), ("notice",), ("echo",)]
    assert plan.step(("gate",)).control_targets == {
        "$steps.notice": (("notice",),),
        "$steps.echo": (("echo",),),
    }
    assert _gates(plan, ("notice",)) == [(("gate",), "$steps.notice")]
    assert plan.step(("notice",)).dependencies == (("gate",),)
    assert plan.step(("gate",)).bindings[0].field == "value", "no binding for StepRefs"


def test_control_supplies_the_domain_of_a_step_without_flowing_data() -> None:
    gated = _compile(
        [
            gate("gate", "$inputs.items", ["notice"]),
            step("sink", "notice", payload="alert"),
        ],
        inputs=ITEMS,
    )
    ungated = _compile([step("sink", "notice", payload="alert")], inputs=ITEMS)
    scalar_gate = _compile(
        [gate("gate", "$inputs.p", ["notice"]), step("sink", "notice")],
        inputs=[parameter("p")],
    )

    notice = gated.step(("notice",))
    assert notice.bindings == () and notice.outputs == {}
    assert axes_of(notice.invocation_layout) == ["inputs"]
    assert axes_of(ungated.step(("notice",)).invocation_layout) == []
    assert axes_of(scalar_gate.step(("notice",)).invocation_layout) == []


def test_several_gates_are_kept_for_conjunction() -> None:
    plan = _compile(
        [
            gate("above_zero", "$inputs.items", ["notice"]),
            gate("above_one", "$inputs.items", ["notice"]),
            step("sink", "notice", payload="both gates"),
        ],
        inputs=ITEMS,
    )

    assert _gates(plan, ("notice",)) == [
        (("above_zero",), "$steps.notice"),
        (("above_one",), "$steps.notice"),
    ]


def test_parent_and_child_gates_give_a_control_only_step_the_child_domain() -> None:
    plan = _compile(
        [
            EXPAND,
            gate("parent_gate", "$inputs.values", ["notice"]),
            gate("child_gate", "$steps.expand.child", ["notice"]),
            step("sink", "notice", payload="selected child"),
        ]
    )

    notice = plan.step(("notice",))
    assert axes_of(notice.invocation_layout) == ["inputs", "expand:child"]
    assert [axes_of(gate.controller_layout) for gate in notice.gates] == [
        ["inputs"],
        ["inputs", "expand:child"],
    ]


def test_ancestor_gate_over_deeper_data_is_planned_by_prefix() -> None:
    plan = _compile(
        [
            EXPAND,
            gate("parent_gate", "$inputs.values", ["selected"]),
            step("echo", "selected", value="$steps.expand.child"),
        ]
    )

    selected = plan.step(("selected",))
    assert axes_of(selected.invocation_layout) == ["inputs", "expand:child"]
    assert axes_of(selected.gates[0].controller_layout) == ["inputs"]


@pytest.mark.parametrize(
    "steps",
    [
        pytest.param(
            [
                EXPAND,
                gate("child_gate", "$steps.expand.child", ["selected"]),
                step("echo", "selected", value="$inputs.values"),
            ],
            id="deeper-control-on-shallower-data",
        ),
        pytest.param(
            [
                EXPAND,
                step("expand", "other", value="$inputs.values"),
                gate("child_gate", "$steps.other.child", ["selected"]),
                step("echo", "selected", value="$steps.expand.child"),
            ],
            id="unrelated-control-lineage",
        ),
        pytest.param(
            [
                EXPAND,
                step("expand", "other", value="$inputs.values"),
                gate("a", "$steps.expand.child", ["notice"]),
                gate("b", "$steps.other.child", ["notice"]),
                step("sink", "notice"),
            ],
            id="unrelated-controls-without-data",
        ),
    ],
)
def test_incompatible_control_lineage_is_rejected(steps) -> None:
    with pytest.raises(LineageError) as info:
        _compile(steps)

    assert "decide" in str(info.value)


def test_dict_targets_route_independently_and_the_join_stays_ungated() -> None:
    plan = _compile(
        [
            step(
                "switch",
                "route",
                value="$inputs.items",
                cases={"1": "$steps.left", "2": "$steps.right"},
                default=["$steps.left"],
            ),
            step("echo", "left", value="$inputs.items"),
            step("echo", "right", value="$inputs.items"),
            step("merge", "merge", values=["$steps.left.value", "$steps.right.value"]),
        ],
        inputs=ITEMS,
    )

    assert plan.step(("route",)).control_targets == {
        "$steps.left": (("left",),),
        "$steps.right": (("right",),),
    }
    assert _gates(plan, ("left",)) == [(("route",), "$steps.left")]
    merge = plan.step(("merge",))
    assert merge.gates == (), "recovery join is not intersected with its branches"
    assert [b.field_path for b in merge.bindings] == [("values", 0), ("values", 1)]


def test_target_on_a_nested_workflow_gates_every_child_step() -> None:
    child = workflow(
        [
            step("echo", "echo", value="$inputs.message"),
            step("sink", "notice", payload="nested alert"),
            step("scale", "scale", value="$inputs.x"),
        ],
        {"message": "$steps.echo.value"},
        inputs=[
            parameter("message", default="child default"),
            batch_input("x", kind=["float"]),
        ],
    )
    plan = _compile(
        [
            EXPAND,
            gate("child_gate", "$steps.expand.child", ["child"]),
            nested(
                "child",
                workflow_definition=child,
                parameter_bindings={"x": "$steps.expand.child"},
            ),
        ]
    )

    children = [("child", "echo"), ("child", "notice"), ("child", "scale")]
    assert plan.step(("child_gate",)).control_targets == {
        "$steps.child": tuple(children)
    }
    for path in children:
        assert _gates(plan, path) == [(("child_gate",), "$steps.child")]
        assert axes_of(plan.step(path).invocation_layout) == ["inputs", "expand:child"]


def test_control_cycles_and_unknown_targets_are_rejected() -> None:
    with pytest.raises(CycleError) as through_data:
        _compile(
            [
                gate("g", "$steps.a.scaled", ["a"]),
                step("scale", "a", value="$inputs.values"),
            ]
        )
    with pytest.raises(CycleError, match="targets itself"):
        _compile([gate("g", "$inputs.values", ["g"])])
    with pytest.raises(SelectorError) as unknown:
        _compile([gate("g", "$inputs.values", ["missing"])])

    assert "$steps.g -> $steps.a -> $steps.g" in str(through_data.value)
    assert unknown.value.step_path == ("g",)
    assert unknown.value.field_path == ("next_steps", 0)


def test_target_outputs_are_ordered_before_their_users() -> None:
    plan = _compile(
        [
            step("scale", "after", value="$steps.gated.scaled"),
            gate("gate", "$inputs.values", ["gated"]),
            step("scale", "gated", value="$inputs.values"),
            step("reader", "unrelated", value="$inputs.payload"),
        ],
        inputs=[
            batch_input("values", kind=["float"]),
            parameter("payload", kind=["dictionary"]),
        ],
    )

    assert [step.path for step in plan.steps] == [
        ("gate",),
        ("gated",),
        ("after",),
        ("unrelated",),
    ]
    assert plan.step(("after",)).gates == (), "filtering propagates through data"


def test_a_target_repeated_in_one_controller_is_one_control_target() -> None:
    plan = _compile(
        [
            step(
                "switch",
                "route",
                value="$inputs.items",
                cases={"a": "$steps.echo", "b": "$steps.echo"},
            ),
            step("echo", "echo", value="$inputs.items"),
        ],
        inputs=ITEMS,
    )

    assert plan.step(("route",)).control_targets == {"$steps.echo": (("echo",),)}
    assert _gates(plan, ("echo",)) == [(("route",), "$steps.echo")]
