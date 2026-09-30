"""Flat parameters: literals, defaults, selectors, compound leaves, kinds, cycles."""

import pytest
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.errors import (
    CycleError,
    KindMismatchError,
    ParamsValidationError,
    SelectorError,
)
from roboflow_workflows.execution_engine.v2.plan import InputPort, StepPort

from tests.unit_tests.execution_engine.v2.compilation.fixture_blocks import (
    CATALOGUE,
    batch_input,
    parameter,
    step,
    workflow,
)


def test_one_field_takes_a_literal_default_input_or_step_selector() -> None:
    definition = workflow(
        [
            step("scale", "literal", value="$inputs.values", factor=3),
            step("scale", "default", value="$inputs.values"),
            step("scale", "from_param", value="$inputs.values", factor="$inputs.f"),
            step(
                "scale",
                "from_step",
                value="$inputs.values",
                factor="$steps.literal.scaled",
            ),
        ],
        inputs=[batch_input("values", kind=["float"]), parameter("f", default=10)],
    )

    plan = compile_workflow(definition, catalogue=CATALOGUE)

    factors = {
        step.path[0]: [
            (binding.source, binding.mode) for binding in step.bindings_for("factor")
        ]
        for step in plan.steps
    }
    assert factors == {
        "literal": [],
        "default": [],
        "from_param": [(InputPort("f"), "constant")],
        "from_step": [(StepPort(("literal",), "scaled"), "element")],
    }
    assert plan.step(("literal",)).params.factor == 3
    assert plan.step(("default",)).params.factor == 2.0
    assert plan.step(("from_step",)).dependencies == (("literal",),)


def test_compound_list_and_dict_leaves_bind_individually_and_literals_stay() -> None:
    definition = workflow(
        [
            step(
                "compound",
                "c",
                params={"a": "$inputs.values", "b": 5, "c": "$inputs.f", "d": "text"},
                items=["$inputs.f", 9, "$inputs.values"],
                labels=["$inputs.values", "plain"],
            )
        ],
        inputs=[batch_input("values", kind=["float"]), parameter("f", default=7)],
    )

    plan = compile_workflow(definition, catalogue=CATALOGUE)
    compound = plan.step(("c",))

    assert [(binding.field_path, binding.mode) for binding in compound.bindings] == [
        (("params", "a"), "element"),
        (("params", "c"), "constant"),
        (("items", 0), "constant"),
        (("items", 2), "element"),
    ]
    assert compound.params.params["b"] == 5 and compound.params.items[1] == 9
    assert compound.params.labels == ["$inputs.values", "plain"], "list[str] is literal"
    assert compound.bindings_for("labels") == ()


def test_explicit_null_and_omitted_default_stay_distinct() -> None:
    definition = workflow(
        [
            step("sink", "omitted"),
            step("sink", "explicit_null", payload=None),
            step("sink", "literal", payload="literal"),
        ],
        inputs=[],
    )

    plan = compile_workflow(definition, catalogue=CATALOGUE)

    omitted, explicit, literal = (
        plan.step((name,)).params for name in ("omitted", "explicit_null", "literal")
    )
    assert omitted.payload == "default" and "payload" not in omitted.model_fields_set
    assert explicit.payload is None and "payload" in explicit.model_fields_set
    assert literal.payload == "literal"
    assert all(step.bindings == () and step.outputs == {} for step in plan.steps)


def test_required_parameter_missing_names_the_field() -> None:
    with pytest.raises(ParamsValidationError) as info:
        compile_workflow(workflow([step("scale", "s")]), catalogue=CATALOGUE)

    assert info.value.step_path == ("s",)
    assert info.value.field_path == ("value",)


@pytest.mark.parametrize(
    "steps, error, text",
    [
        pytest.param(
            [step("scale", "s", value="$inputs.missing")],
            SelectorError,
            "unknown workflow input 'missing'",
            id="unknown-input",
        ),
        pytest.param(
            [step("scale", "s", value="$steps.nope.scaled")],
            SelectorError,
            "unknown step 'nope'",
            id="unknown-step",
        ),
        pytest.param(
            [
                step("scale", "a", value="$inputs.values"),
                step("scale", "b", value="$steps.a.missing"),
            ],
            SelectorError,
            "has no output 'missing'; its outputs are ['scaled']",
            id="unknown-output",
        ),
        pytest.param(
            [
                step("parse_fields", "p", raw="x", expected_fields=["class-name"]),
                step("echo", "e", value="$steps.p.class_name"),
            ],
            SelectorError,
            "has no output 'class_name'",
            id="unknown-configured-output",
        ),
        pytest.param(
            [
                step("scale", "a", value="$inputs.values"),
                step("echo", "e", value="$steps.a.*"),
            ],
            SelectorError,
            "only valid in workflow outputs",
            id="wildcard-in-step",
        ),
    ],
)
def test_unresolvable_selectors_fail_with_the_field(steps, error, text) -> None:
    with pytest.raises(error) as info:
        compile_workflow(workflow(steps), catalogue=CATALOGUE)

    assert text in str(info.value)
    assert info.value.step_path


def test_kinds_must_overlap_unless_either_side_is_the_wildcard() -> None:
    mismatch = workflow(
        [step("scale", "s", value="$inputs.names")],
        inputs=[batch_input("names", kind=["string"])],
    )
    union = workflow(
        [step("parse_fields", "p", raw="$inputs.mixed", expected_fields=[])],
        inputs=[batch_input("mixed", kind=["float", "string"])],
    )
    wildcard = workflow(
        [step("scale", "s", value="$inputs.anything")],
        inputs=[batch_input("anything")],
    )

    with pytest.raises(KindMismatchError) as info:
        compile_workflow(mismatch, catalogue=CATALOGUE)
    compile_workflow(union, catalogue=CATALOGUE)
    compile_workflow(wildcard, catalogue=CATALOGUE)

    assert info.value.step_path == ("s",) and info.value.field_path == ("value",)
    assert "accepts kinds ['float']" in str(info.value)
    assert "provides ['string']" in str(info.value)


@pytest.mark.parametrize(
    "steps, chain",
    [
        pytest.param(
            [
                step("scale", "a", value="$steps.b.scaled"),
                step("scale", "b", value="$steps.a.scaled"),
            ],
            "$steps.a -> $steps.b -> $steps.a",
            id="direct",
        ),
        pytest.param(
            [step("scale", "a", value="$steps.a.scaled")],
            "$steps.a -> $steps.a",
            id="self",
        ),
        pytest.param(
            [
                step("compound", "a", params={"x": "$steps.b.scaled"}),
                step("scale", "b", value="$inputs.values", factor="$steps.a.echo"),
            ],
            "$steps.a -> $steps.b -> $steps.a",
            id="dict-leaf",
        ),
        pytest.param(
            [
                step("compound", "a", items=[1, "$steps.b.scaled"]),
                step("scale", "b", value="$steps.a.echo"),
            ],
            "$steps.a -> $steps.b -> $steps.a",
            id="list-leaf",
        ),
    ],
)
def test_cycles_through_any_selector_position_are_rejected(steps, chain) -> None:
    with pytest.raises(CycleError) as info:
        compile_workflow(workflow(steps), catalogue=CATALOGUE)

    assert chain in str(info.value)


def test_configured_outputs_are_resolved_from_literal_fields() -> None:
    definition = workflow(
        [
            step(
                "parse_fields",
                "parse",
                raw="$inputs.text",
                expected_fields=["a", "b-2"],
            ),
            step("echo", "use", value="$steps.parse.b-2"),
        ],
        {"all": "$steps.parse.*", "text": "$inputs.text"},
        inputs=[batch_input("text", kind=["string"])],
    )

    plan = compile_workflow(definition, catalogue=CATALOGUE)

    assert list(plan.step(("parse",)).outputs) == ["a", "b-2", "error_status"]
    assert plan.step(("parse",)).outputs["error_status"].kinds == ("boolean",)
    assert [output.source for output in plan.outputs] == [
        StepPort(("parse",), "*"),
        InputPort("text"),
    ]
