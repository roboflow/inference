"""Nested workflow inputs compile to located child input boundaries (decision 021).

These are plan-level checks. Run-time behaviour of the same boundaries is in
``test_boundary_runs.py``.
"""

import pytest
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.errors import SelectorError
from roboflow_workflows.execution_engine.v2.plan import (
    ChildInputPort,
    ChildOutputPort,
    Constant,
    InputPort,
    StepPort,
)

from tests.unit_tests.execution_engine.v2.compilation.fixture_blocks import (
    CATALOGUE,
    axes_of,
    batch_input,
    nested,
    origin_of,
    parameter,
    step,
    workflow,
)


def _compile(steps, outputs=None, *, inputs=None):
    plan = compile_workflow(
        workflow(steps, outputs, inputs=inputs), catalogue=CATALOGUE
    )

    return plan


def _boundaries(plan):
    return [(item.scope, item.name, item.source) for item in plan.child_inputs]


def forwarding_child(kind="integer", *, default=None):
    """Child whose only output forwards its input; its one step is a sink."""
    child = workflow(
        [step("sink", "s")],
        {"value": "$inputs.value"},
        inputs=[parameter("value", default=default, kind=[kind])],
    )

    return child


def test_direct_child_input_outputs_read_through_their_boundary() -> None:
    plan = _compile(
        [
            nested("defaulted", workflow_definition=forwarding_child(default=7)),
            nested(
                "literal",
                workflow_definition=forwarding_child(),
                parameter_bindings={"value": 8},
            ),
            nested(
                "selected",
                workflow_definition=forwarding_child(),
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

    names = ("defaulted", "literal", "selected")
    assert [output.source for output in plan.outputs] == [
        ChildOutputPort((name,), "value") for name in names
    ]
    assert [plan.child_output(output.source).source for output in plan.outputs] == [
        ChildInputPort((name,), "value") for name in names
    ]
    assert all(item.gates == () for item in plan.child_outputs), "ungated children"
    assert _boundaries(plan) == [
        (("defaulted",), "value", Constant(7)),
        (("literal",), "value", Constant(8)),
        (("selected",), "value", InputPort("anything")),
    ]
    assert {item.kinds for item in plan.child_inputs} == {("integer",)}


def test_deep_chains_list_outer_boundaries_first_and_keep_the_source_layout() -> None:
    leaf = workflow(
        [step("echo", "echo", value="$inputs.x")],
        {"y": "$steps.echo.value"},
        inputs=[batch_input("x", kind=["float"])],
    )
    middle = workflow(
        [
            nested(
                "leaf", workflow_definition=leaf, parameter_bindings={"x": "$inputs.m"}
            )
        ],
        {"y": "$steps.leaf.y", "m": "$inputs.m"},
        inputs=[batch_input("m")],
    )
    plan = _compile(
        [
            step("expand", "expand", value="$inputs.values"),
            nested(
                "middle",
                workflow_definition=middle,
                parameter_bindings={"m": "$steps.expand.child"},
            ),
        ],
        {"y": "$steps.middle.y", "m": "$steps.middle.m"},
    )

    assert _boundaries(plan) == [
        (("middle",), "m", StepPort(("expand",), "child")),
        (("middle", "leaf"), "x", ChildInputPort(("middle",), "m")),
    ]
    assert [axes_of(item.layout) for item in plan.child_inputs] == [
        ["inputs", "expand:child"],
    ] * 2
    echo = plan.step(("middle", "leaf", "echo"))
    assert echo.binding_for("value").source == ChildInputPort(("middle", "leaf"), "x")
    assert echo.dependencies == (("expand",),), "boundaries keep the producer edge"
    assert axes_of(echo.invocation_layout) == ["inputs", "expand:child"]
    assert plan.outputs[1].source == ChildOutputPort(("middle",), "m")
    assert plan.child_output(plan.outputs[1].source).source == ChildInputPort(
        ("middle",), "m"
    )


def test_consumers_of_one_child_input_share_one_boundary_and_unused_inputs_have_none() -> (
    None
):
    child = workflow(
        [
            step("scale", "a", value="$inputs.x"),
            step("scale", "b", value="$inputs.x", factor="$inputs.k"),
        ],
        inputs=[
            batch_input("x", kind=["float"]),
            parameter("k", default=3),
            parameter("unused", default="kept"),
        ],
    )
    plan = _compile(
        [
            nested(
                "child",
                workflow_definition=child,
                parameter_bindings={"x": "$inputs.values"},
            )
        ]
    )

    assert _boundaries(plan) == [
        (("child",), "x", InputPort("values")),
        (("child",), "k", Constant(3)),
    ]
    a, b = plan.step(("child", "a")), plan.step(("child", "b"))
    assert a.binding_for("value").source == b.binding_for("value").source


def test_literal_casts_are_direct_constants_not_child_inputs() -> None:
    from tests.unit_tests.execution_engine.v2.compilation.test_literal_casts import (
        LITERALS,
    )

    child = workflow(
        [{"type": "test/literal_batch@v1", "name": "cast", "value": 7}], inputs=[]
    )
    plan = compile_workflow(
        workflow([nested("child", workflow_definition=child)], inputs=[]),
        catalogue=LITERALS,
    )

    assert plan.child_inputs == ()
    assert plan.step(("child", "cast")).binding_for("value").source == Constant(7)


def test_mutation_analysis_follows_boundaries_to_their_origin() -> None:
    child = workflow(
        [step("increment", "change", value="$inputs.payload")],
        inputs=[parameter("payload", kind=["dictionary"])],
    )
    plan = _compile(
        [
            nested(
                "child",
                workflow_definition=child,
                parameter_bindings={"payload": "$inputs.payload"},
            ),
            step("reader", "read", value="$inputs.payload"),
        ],
        inputs=[parameter("payload", kind=["dictionary"])],
    )

    assert len(plan.warnings) == 1
    assert plan.warnings[0].startswith("$steps.child/change value ($inputs.payload)")
    assert "$steps.read value ($inputs.payload)" in plan.warnings[0]


def test_a_child_input_cannot_forward_a_whole_step_wildcard() -> None:
    child = workflow([step("sink", "s")], {"all": "$inputs.x"}, inputs=[parameter("x")])

    with pytest.raises(SelectorError, match="cannot forward every output"):
        _compile(
            [
                step("scale", "scale", value="$inputs.values"),
                nested(
                    "child",
                    workflow_definition=child,
                    parameter_bindings={"x": "$steps.scale.*"},
                ),
            ],
            {"all": "$steps.child.all"},
        )


def test_boundaries_do_not_change_whole_child_control() -> None:
    child = workflow(
        [step("echo", "echo", value="$inputs.x"), step("sink", "notice")],
        {"y": "$steps.echo.value"},
        inputs=[batch_input("x", kind=["float"])],
    )
    plan = _compile(
        [
            step("gate", "gate", value="$inputs.values", next_steps=["$steps.child"]),
            nested(
                "child",
                workflow_definition=child,
                parameter_bindings={"x": "$inputs.values"},
            ),
        ],
        {"y": "$steps.child.y"},
    )

    for path in (("child", "echo"), ("child", "notice")):
        assert [gate.controller for gate in plan.step(path).gates] == [("gate",)]
    assert origin_of(
        plan, plan.step(("child", "echo")).bindings[0].source
    ) == InputPort("values")
