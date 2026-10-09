"""``compare_plans``: which steps an update retains, adds or breaks, and why."""

import dataclasses

import numpy as np
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.plan import CompileOptions
from roboflow_workflows.execution_engine.v2.updates import compare_plans

from tests.unit_tests.execution_engine.v2.updates.blocks import (
    CATALOGUE,
    Count,
    Label,
    Mutator,
    Scale,
    Stateful,
    active,
    child,
    group,
    nested,
    passive,
    step,
)

COUNTER = [step(Count, "count", value="$inputs.value")]


def compiled(definition: dict, **options) -> object:
    plan = compile_workflow(
        definition, catalogue=CATALOGUE, options=CompileOptions(**options)
    )

    return plan


def reasons(diff) -> list:
    found = sorted((change.name, change.reason) for change in diff.breaking)

    return found


def test_recompiled_identical_definition_retains_every_step() -> None:
    definition = passive(
        [*COUNTER, step(Scale, "scale", value="$inputs.value")],
        {"count": "$steps.count.count", "scaled": "$steps.scale.scaled"},
    )

    diff = compare_plans(compiled(definition), compiled(definition))

    assert diff.compatible
    assert diff.retained == (("count",), ("scale",))
    assert diff.added == ()
    assert diff.changes == ()


def test_reordered_declarations_still_retain_every_step() -> None:
    steps = [*COUNTER, step(Scale, "scale", value="$inputs.value")]
    outputs = {"count": "$steps.count.count", "scaled": "$steps.scale.scaled"}
    reordered = dict(reversed(list(outputs.items())))

    diff = compare_plans(
        compiled(passive(steps, outputs)),
        compiled(passive(list(reversed(steps)), reordered)),
    )

    assert diff.compatible
    assert set(diff.retained) == {("count",), ("scale",)}


def test_added_consumer_and_output_are_compatible() -> None:
    old = compiled(passive(COUNTER, {"count": "$steps.count.count"}))
    new = compiled(
        passive(
            [*COUNTER, step(Scale, "scale", value="$inputs.value")],
            {"count": "$steps.count.count", "scaled": "$steps.scale.scaled"},
        )
    )

    diff = compare_plans(old, new)

    assert diff.compatible
    assert diff.retained == (("count",),)
    assert diff.added == (("scale",),)
    assert {(c.component, c.name, c.reason) for c in diff.changes} == {
        ("step", "$steps.scale", "added"),
        ("output", "scaled", "added"),
    }
    assert diff.describe()["added"] == ["$steps.scale"]


def test_parameter_edit_of_retained_step_is_breaking() -> None:
    def plan(factor: float):
        return compiled(
            passive(
                [step(Scale, "scale", value="$inputs.value", factor=factor)],
                {"scaled": "$steps.scale.scaled"},
            )
        )

    diff = compare_plans(plan(2.0), plan(3.0))

    assert not diff.compatible
    assert reasons(diff) == [("$steps.scale", "params_changed")]
    assert diff.retained == ()


def test_quality_change_selects_another_implementation_and_breaks() -> None:
    definition = passive(
        [step(Label, "label", value="$inputs.value")], {"label": "$steps.label.label"}
    )

    diff = compare_plans(compiled(definition), compiled(definition, quality="fast"))

    assert reasons(diff) == [
        ("$steps.label", "implementation_changed"),
        ("$steps.label", "quality_changed"),
    ]


def test_same_block_type_with_another_class_breaks() -> None:
    class OtherCount(Count):
        type = Count.type

    other = Catalogue(
        [OtherCount, Scale, Label, Mutator, Stateful],
        sources=[],
    )
    definition = passive(COUNTER, {"count": "$steps.count.count"})

    diff = compare_plans(
        compiled(definition),
        compile_workflow(definition, catalogue=other),
    )

    assert ("$steps.count", "block_changed") in reasons(diff)


def test_previously_pruned_step_is_newly_demanded_and_the_reverse_breaks() -> None:
    steps = [*COUNTER, step(Scale, "scale", value="$inputs.value")]
    pruned = compiled(passive(steps, {"count": "$steps.count.count"}))
    demanded = compiled(
        passive(steps, {"count": "$steps.count.count", "scaled": "$steps.scale.scaled"})
    )

    forward = compare_plans(pruned, demanded)
    backward = compare_plans(demanded, pruned)

    assert ("scale",) in pruned.demand.pruned
    assert forward.compatible
    assert [(c.name, c.reason) for c in forward.changes if c.component == "step"] == [
        ("$steps.scale", "newly_demanded")
    ]
    assert reasons(backward) == [("$steps.scale", "newly_pruned")]


def test_second_output_of_a_retained_step_is_compatible() -> None:
    steps = [step(Scale, "scale", value="$inputs.value")]
    old = compiled(passive(steps, {"scaled": "$steps.scale.scaled"}))
    new = compiled(
        passive(
            steps, {"scaled": "$steps.scale.scaled", "again": "$steps.scale.scaled"}
        )
    )

    diff = compare_plans(old, new)

    assert diff.compatible
    assert diff.retained == (("scale",),)


def test_nested_child_input_rebinding_breaks_although_child_step_text_is_same() -> None:
    inner = child(
        [step(Scale, "inner", value="$inputs.x", factor="$inputs.k")],
        {"y": "$steps.inner.scaled"},
        inputs=[
            {"type": "WorkflowParameter", "name": "x", "kind": ["float"]},
            {"type": "WorkflowParameter", "name": "k", "kind": ["float"]},
        ],
    )

    def plan(k: float, *, consumer: bool = False):
        steps = [nested("child", inner, x="$inputs.value", k=k)]
        outputs = {"y": "$steps.child.y"}
        if consumer:
            steps.append(step(Scale, "after", value="$steps.child.y"))
            outputs["after"] = "$steps.after.scaled"
        return compiled(passive(steps, outputs))

    rebound = compare_plans(plan(3.0), plan(4.0))
    attached = compare_plans(plan(3.0), plan(3.0, consumer=True))

    assert reasons(rebound) == [("child.k", "changed")]
    assert ("child", "inner") in rebound.retained, "the step's own bindings look equal"
    assert attached.compatible
    assert attached.retained == (("child", "inner"),)
    assert attached.added == (("after",),)


def test_new_mutation_alias_of_a_retained_payload_breaks_and_shows_the_warning() -> (
    None
):
    steps = [
        step(Scale, "scale", value="$inputs.value"),
        step(Count, "count", value="$steps.scale.scaled"),
    ]
    outputs = {"count": "$steps.count.count"}
    old = compiled(passive(steps, outputs))
    new = compiled(
        passive(
            [*steps, step(Mutator, "mutator", payload="$steps.scale.scaled")],
            {**outputs, "payload": "$steps.mutator.payload"},
        )
    )

    diff = compare_plans(old, new)

    (warning,) = diff.breaking
    assert (warning.component, warning.reason) == ("warning", "added")
    assert "$steps.mutator" in warning.detail and "$steps.count" in warning.detail


def test_new_managed_state_need_breaks() -> None:
    old = compiled(passive(COUNTER, {"count": "$steps.count.count"}))
    new = compiled(
        passive(
            [*COUNTER, step(Stateful, "stateful", value="$inputs.value")],
            {"count": "$steps.count.count", "state": "$steps.stateful.value"},
        )
    )

    assert reasons(compare_plans(old, new)) == [("managed_state", "added")]


def test_source_and_input_changes_break_and_new_groups_do_not() -> None:
    steps = [step(Count, "count", value="$sources.ticks.value")]
    base = compiled(active(steps, [group("counts", count="$steps.count.count")]))
    more_groups = compiled(
        active(
            steps,
            [
                group("counts", count="$steps.count.count"),
                group("values", value="$sources.ticks.value"),
            ],
        )
    )
    other_source = compiled(
        active(steps, [group("counts", count="$steps.count.count")], count=5)
    )

    assert compare_plans(base, more_groups).compatible
    assert reasons(compare_plans(base, other_source)) == [("$sources.ticks", "changed")]


def test_array_constants_are_equal_only_when_identical() -> None:
    definition = passive(
        [step(Scale, "scale", value="$inputs.value")], {"scaled": "$steps.scale.scaled"}
    )
    plan = compiled(definition)
    shared = np.array([2.0])

    def with_factor(factor):
        (scale,) = plan.steps
        params = scale.params.model_copy(update={"factor": factor})
        return dataclasses.replace(
            plan, steps=(dataclasses.replace(scale, params=params),)
        )

    same_object = compare_plans(with_factor(shared), with_factor(shared))
    equal_values = compare_plans(with_factor(shared), with_factor(np.array([2.0])))

    assert same_object.compatible
    assert reasons(equal_values) == [("$steps.scale", "params_changed")]


def test_prunable_consumer_below_a_controlled_step_extends_its_closure() -> None:
    def plan(*, consumer: bool, controlled: str):
        steps = [
            step(Scale, "scale", value="$inputs.value"),
            step(Count, "count", value="$inputs.value"),
        ]
        outputs = {"scaled": "$steps.scale.scaled", "count": "$steps.count.count"}
        if consumer:
            steps.append(step(Scale, "after", value="$steps.scale.scaled"))
            outputs["after"] = "$steps.after.scaled"
        controls = {"switch": {"type": "enable", "steps": [f"$steps.{controlled}"]}}
        return compiled(passive(steps, outputs, controls=controls))

    below = compare_plans(
        plan(consumer=False, controlled="scale"),
        plan(consumer=True, controlled="scale"),
    )
    beside = compare_plans(
        plan(consumer=False, controlled="count"),
        plan(consumer=True, controlled="count"),
    )
    relisted = compare_plans(
        plan(consumer=False, controlled="scale"),
        plan(consumer=False, controlled="count"),
    )

    # The prunable consumer joins the closure as a pure step: the control is
    # the same declaration, so the update is compatible. Listing other steps
    # changes the declaration.
    assert below.compatible and beside.compatible
    assert [change for change in below.changes if change.component == "control"] == []
    assert reasons(relisted) == [("controls.switch", "changed")]
