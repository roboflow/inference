"""Mutating a payload a block keeps: warn by default, reject when strict.

``retains_inputs`` and ``retains_outputs`` are alias declarations. Mutation
analysis treats a kept payload as read for as long as it is kept::

    lend ──▶ change (mutates)          change runs after lend: still reported
    change (mutates) ──▶ keep          the change finished first: valid
    change (mutates)   keep            unordered: the existing reader conflict
"""

import pytest
from roboflow_workflows.v2.data.kinds import BOOLEAN_KIND, DICTIONARY_KIND
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref
from roboflow_workflows.v2.engine.errors import MutationConflictError
from roboflow_workflows.v2.engine.plan import CompileOptions

from tests.v2.support.compilation.fixture_blocks import (
    ALL_BLOCKS,
    gate,
    nested,
    parameter,
    step,
    workflow,
)
from tests.v2.unit.engine.compiler.test_operator_compilation import (
    CATALOGUE as OPERATOR_CATALOGUE,
)
from tests.v2.unit.engine.compiler.test_operator_compilation import (
    definition as active_definition,
)
from tests.v2.unit.engine.compiler.test_operator_compilation import window

STRICT = CompileOptions(mutation_conflicts="error")
PAYLOAD = [parameter("payload", kind=["dictionary"])]
UNORDERED = "without an ordering dependency"


class Keep(Block):
    """Keeps its input after the call, like a cache setter."""

    type = "test/keep@v1"
    retains_inputs = ("value",)
    outputs = {"stored": Output(BOOLEAN_KIND)}

    class Params(BlockParams):
        value: Ref(DICTIONARY_KIND)

    def run(self, *, value) -> dict:
        return {"stored": True}


class Lend(Block):
    """Hands out an object it keeps, like a cache getter."""

    type = "test/lend@v1"
    retains_outputs = ("value",)
    prunable = True
    outputs = {"value": Output(DICTIONARY_KIND)}

    class Params(BlockParams):
        trigger: Ref()

    def run(self, *, trigger) -> dict:
        return {"value": {}}


class LendThrough(Block):
    """Keeps its output, which takes its context (and maybe the object) from ``value``."""

    type = "test/lend_through@v1"
    retains_outputs = ("value",)
    outputs = {"value": Output(DICTIONARY_KIND, source="value")}

    class Params(BlockParams):
        value: Ref(DICTIONARY_KIND)

    def run(self, *, value) -> dict:
        return {"value": value}


class KeepAndBump(Block):
    """Keeps its input and also mutates it in place."""

    type = "test/keep_and_bump@v1"
    retains_inputs = ("value",)
    mutates = ("value",)
    outputs = {"value": Output(DICTIONARY_KIND, source="value")}

    class Params(BlockParams):
        value: Ref(DICTIONARY_KIND)

    def run(self, *, value) -> dict:
        return {"value": value}


RETAINING = (Keep, Lend, LendThrough, KeepAndBump)
CATALOGUE = Catalogue(ALL_BLOCKS + RETAINING, namespace="test")
ACTIVE_CATALOGUE = Catalogue.merge(
    OPERATOR_CATALOGUE, Catalogue(RETAINING, namespace="kept")
)


def compiled(steps, *, inputs=PAYLOAD, options=None):
    plan = compile_workflow(
        workflow(steps, inputs=inputs),
        catalogue=CATALOGUE,
        options=options or CompileOptions(),
    )

    return plan


def lent_and_changed():
    """The cache repro (C5): a later step mutates the object a getter keeps."""
    steps = [
        step("lend", "lend", trigger="$inputs.payload"),
        step("increment", "change", value="$steps.lend.value"),
    ]

    return steps


# --- what is reported -----------------------------------------------------------


def test_a_later_step_mutating_a_kept_output_is_reported() -> None:
    plan = compiled(lent_and_changed())

    assert plan.warnings == (
        "$steps.change value ($steps.lend.value) is mutated in place after "
        "$steps.lend keeps ['output value'] beyond its call; that may be the same "
        "payload, so the kept value changes too. The engine adds no copy; copy the "
        "payload before mutating it",
    )


def test_strict_policy_rejects_a_mutated_kept_output() -> None:
    with pytest.raises(MutationConflictError) as info:
        compiled(lent_and_changed(), options=STRICT)

    assert info.value.step_path == ("change",)
    assert info.value.field_path == ("value",)
    assert "$steps.lend keeps ['output value']" in str(info.value)


def test_a_mutator_ordered_after_an_input_retainer_is_reported() -> None:
    steps = [
        step("keep", "keep", value="$inputs.payload"),
        gate("after_keep", "$steps.keep.stored", ["change"]),
        step("increment", "change", value="$inputs.payload"),
    ]

    [warning] = compiled(steps).warnings

    assert warning.startswith("$steps.change value ($inputs.payload) is mutated")
    assert "after $steps.keep keeps ['input value ($inputs.payload)']" in warning


def test_a_mutation_finished_before_the_retainer_starts_is_valid() -> None:
    steps = [
        step("increment", "change", value="$inputs.payload"),
        step("keep", "keep", value="$steps.change.value"),
    ]

    plan = compiled(steps, options=STRICT)

    assert plan.warnings == ()


def test_an_unordered_mutator_is_reported_once_as_a_reader_conflict() -> None:
    steps = [
        step("increment", "change", value="$inputs.payload"),
        step("keep", "keep", value="$inputs.payload"),
    ]

    [warning] = compiled(steps).warnings

    assert UNORDERED in warning and "$steps.keep value" in warning
    assert "keeps" not in warning


def test_a_kept_output_does_not_mean_the_input_is_kept() -> None:
    # lend reads payload only as a trigger; its kept output is its own object.
    steps = [
        step("lend", "lend", trigger="$inputs.payload"),
        gate("after_lend", "$steps.lend.value", ["change"]),
        step("increment", "change", value="$inputs.payload"),
    ]

    plan = compiled(steps, options=STRICT)

    assert plan.warnings == ()


def test_two_retainers_of_one_payload_are_each_reported_once() -> None:
    steps = [
        step("keep", "first", value="$inputs.payload"),
        step("keep", "second", value="$inputs.payload"),
        gate("after_keep", "$steps.first.stored", ["change"]),
        step("increment", "change", value="$inputs.payload"),
    ]

    warnings = compiled(steps).warnings

    # change is unordered with second: that is the reader conflict.
    assert len(warnings) == 2
    assert sum("after $steps.first keeps" in item for item in warnings) == 1
    assert sum(UNORDERED in item and "$steps.second" in item for item in warnings) == 1


# --- conservative aliasing --------------------------------------------------------


def test_a_kept_output_with_a_source_field_may_be_the_input_object() -> None:
    # Output(source="value") may return the input itself, so mutating the input
    # after the step is reported, as is mutating the output.
    steps = [
        step("lend_through", "lend", value="$inputs.payload"),
        gate("after_lend", "$steps.lend.value", ["by_input", "by_output"]),
        step("increment", "by_input", value="$inputs.payload"),
        step("increment", "by_output", value="$steps.lend.value"),
    ]

    warnings = compiled(steps).warnings
    kept = [item for item in warnings if "keeps ['output value']" in item]

    assert [item.split(" ")[0] for item in kept] == [
        "$steps.by_input",
        "$steps.by_output",
    ]


def test_a_retainer_mutating_what_it_keeps_is_reported_conservatively() -> None:
    plan = compiled([step("keep_and_bump", "bump", value="$inputs.payload")])

    assert plan.warnings == (
        "$steps.bump value ($inputs.payload) is mutated in place by the step that "
        "also keeps ['input value ($inputs.payload)'] beyond its call; that may be "
        "the same payload, so the kept value changes too. The engine adds no copy; "
        "copy the payload before mutating it",
    )
    with pytest.raises(MutationConflictError, match="by the step that also keeps"):
        compiled(
            [step("keep_and_bump", "bump", value="$inputs.payload")], options=STRICT
        )


def test_a_retainer_without_mutators_compiles_strictly() -> None:
    steps = [
        step("lend", "lend", trigger="$inputs.payload"),
        step("keep", "keep", value="$steps.lend.value"),
        step("reader", "read", value="$steps.lend.value"),
    ]

    plan = compiled(steps, options=STRICT)

    assert plan.warnings == ()


# --- nested workflows and operators -------------------------------------------------


def test_a_kept_output_forwarded_out_of_a_child_is_tracked() -> None:
    child = workflow(
        [step("lend", "lend", trigger="$inputs.x")],
        inputs=[parameter("x", kind=["dictionary"])],
    )
    child["outputs"] = [
        {"type": "JsonField", "name": "lent", "selector": "$steps.lend.value"}
    ]
    steps = [
        nested(
            "child",
            workflow_definition=child,
            parameter_bindings={"x": "$inputs.payload"},
        ),
        step("increment", "change", value="$steps.child.lent"),
    ]

    [warning] = compiled(steps).warnings

    assert warning.startswith("$steps.change value ($steps.child.lent) is mutated")
    assert "after $steps.child/lend keeps ['output value']" in warning


def test_a_kept_output_forwarded_into_a_child_is_tracked() -> None:
    child = workflow(
        [step("increment", "change", value="$inputs.state")],
        inputs=[parameter("state", kind=["dictionary"])],
    )
    steps = [
        step("lend", "lend", trigger="$inputs.payload"),
        nested(
            "child",
            workflow_definition=child,
            parameter_bindings={"state": "$steps.lend.value"},
        ),
    ]

    [warning] = compiled(steps).warnings

    assert warning.startswith("$steps.child/change value ($inputs.state) is mutated")
    assert "after $steps.lend keeps ['output value']" in warning


def test_a_kept_output_collected_by_an_operator_is_tracked() -> None:
    plan = compile_workflow(
        active_definition(
            [
                {"type": Lend.type, "name": "lend", "trigger": "$sources.cam.image"},
                step("increment", "change", value="$operators.clip.lent"),
            ],
            [window("clip", {"lent": "$steps.lend.value"})],
        ),
        catalogue=ACTIVE_CATALOGUE,
    )

    operator_port, kept = plan.warnings
    # Two different retainers keep the payload: the window and lend.
    assert "may retain or emit it again" in operator_port
    assert kept.startswith("$steps.change value ($operators.clip.lent) is mutated")
    assert "after $steps.lend keeps ['output value']" in kept


def test_planned_steps_describe_their_retention() -> None:
    plan = compiled(lent_and_changed())

    lend, change = plan.step(("lend",)).describe(), plan.step(("change",)).describe()

    assert (lend["retains_inputs"], lend["retains_outputs"]) == ([], ["value"])
    assert (change["retains_inputs"], change["retains_outputs"]) == ([], [])
