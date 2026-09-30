"""Declared in-place mutation: warn by default, reject when strict."""

import pytest
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.errors import MutationConflictError
from roboflow_workflows.execution_engine.v2.plan import CompileOptions

from tests.unit_tests.execution_engine.v2.compilation.fixture_blocks import (
    CATALOGUE,
    gate,
    nested,
    parameter,
    step,
    workflow,
)

PAYLOAD = [parameter("payload", kind=["dictionary"])]
STRICT = CompileOptions(mutation_conflicts="error")


def _warnings(steps, *, inputs=PAYLOAD):
    plan = compile_workflow(workflow(steps, inputs=inputs), catalogue=CATALOGUE)

    return plan.warnings


def test_unordered_reader_of_a_mutated_payload_warns_by_default() -> None:
    steps = [
        step("increment", "change", value="$inputs.payload"),
        step("reader", "read", value="$inputs.payload"),
    ]

    warnings = _warnings(steps)

    assert warnings == (
        "$steps.change value ($inputs.payload) is mutated in place, and $steps.read "
        "value ($inputs.payload) may see the same payload without an ordering "
        "dependency between the two steps",
    )


def test_strict_policy_rejects_with_step_and_field_paths() -> None:
    steps = [
        step("increment", "change", value="$inputs.payload"),
        step("reader", "read", value="$inputs.payload"),
    ]

    with pytest.raises(MutationConflictError) as info:
        compile_workflow(
            workflow(steps, inputs=PAYLOAD), catalogue=CATALOGUE, options=STRICT
        )

    assert info.value.step_path == ("change",)
    assert info.value.field_path == ("value",)
    assert "$steps.read value" in str(info.value)


@pytest.mark.parametrize(
    "steps",
    [
        pytest.param(
            [
                step("increment", "change", value="$inputs.payload"),
                step("reader", "read", value="$steps.change.value"),
            ],
            id="reader-after-mutator",
        ),
        pytest.param(
            [
                gate("check", "$inputs.payload", ["change"]),
                step("increment", "change", value="$inputs.payload"),
            ],
            id="reader-before-mutator-by-control",
        ),
        pytest.param(
            [
                step("increment", "change", value="$inputs.payload"),
                step("reader", "read", value="$inputs.other"),
            ],
            id="different-payloads",
        ),
        pytest.param(
            [
                step("compound", "copy", params={"p": "$inputs.payload"}),
                step("increment", "change", value="$steps.copy.echo"),
                step("reader", "read", value="$inputs.payload"),
            ],
            id="output-without-declared-source",
        ),
    ],
)
def test_causally_ordered_or_unshared_mutation_compiles_without_warning(steps) -> None:
    plan = compile_workflow(
        workflow(steps, inputs=PAYLOAD + [parameter("other", kind=["dictionary"])]),
        catalogue=CATALOGUE,
        options=STRICT,
    )

    assert plan.warnings == ()


def test_declared_output_source_carries_a_potential_alias() -> None:
    steps = [
        step("echo", "forward", value="$inputs.payload"),
        step("increment", "change", value="$steps.forward.value"),
        step("reader", "read", value="$inputs.payload"),
    ]

    warnings = _warnings(steps)

    assert len(warnings) == 1
    assert warnings[0].startswith("$steps.change value ($steps.forward.value)")
    assert "$steps.read value ($inputs.payload)" in warnings[0]


def test_two_unordered_mutators_are_reported_once() -> None:
    steps = [
        step("increment", "first", value="$inputs.payload"),
        step("increment", "second", value="$inputs.payload"),
    ]

    warnings = _warnings(steps)

    assert len(warnings) == 1
    assert "$steps.first" in warnings[0] and "$steps.second" in warnings[0]


def test_a_shared_nested_default_is_a_potential_alias() -> None:
    child = workflow(
        [
            step("increment", "change", value="$inputs.state"),
            step("reader", "read", value="$inputs.state"),
        ],
        inputs=[parameter("state", default={"count": 0}, kind=["dictionary"])],
    )
    scalar_child = workflow(
        [step("scale", "a", value="$inputs.x"), step("scale", "b", value="$inputs.x")],
        inputs=[parameter("x", default=1.0)],
    )

    warnings = _warnings(
        [
            nested("child", workflow_definition=child),
            nested("scalars", workflow_definition=scalar_child),
        ],
        inputs=[],
    )

    assert len(warnings) == 1
    assert warnings[0].startswith("$steps.child/change value ($inputs.state)")
