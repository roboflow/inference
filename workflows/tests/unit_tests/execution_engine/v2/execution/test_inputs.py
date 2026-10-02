"""Workflow input preparation: forms, broadcasting, explicit axes and codecs."""

from concurrent.futures import Future

import pytest
from roboflow_workflows.execution_engine.v2.data import (
    Axis,
    Batch,
    EntryLayout,
    EntryMetadata,
    InputValue,
    SampleContext,
)
from roboflow_workflows.execution_engine.v2.errors import (
    WorkflowExecutionError,
    WorkflowInputError,
)
from roboflow_workflows.execution_engine.v2.kinds import FLOAT_KIND, INTEGER_KIND, Kind

from .blocks import Echo, Scale, Sum, tagged_kind
from .plans import BATCH, SCALAR, PlanBuilder, nested


def echo_plan(*, layout=BATCH, kinds=("*",), extra_kinds=()):
    plan = (
        PlanBuilder(kinds=extra_kinds)
        .input("items", layout, kinds=kinds)
        .step(Echo, "echo", at=layout, value="$inputs.items")
        .output("items", "$steps.echo.value")
        .build()
    )

    return plan


def scale_plan():
    plan = (
        PlanBuilder()
        .input("a", BATCH)
        .input("b", BATCH)
        .step(Scale, "s", at=BATCH, value="$inputs.a", factor="$inputs.b")
        .output("s", "$steps.s.scaled")
        .build()
    )

    return plan


@pytest.mark.parametrize("b", [10, [10]])
def test_scalar_and_singleton_inputs_broadcast_to_the_batch_size(b) -> None:
    session = scale_plan().create_session()

    rows = session.run({"a": [1, 2, 3], "b": b}).rows()

    assert rows == [{"s": 10}, {"s": 20}, {"s": 30}]
    assert len(session.instances[("s",)].calls) == 3


@pytest.mark.parametrize("b", [[], ["x", "y", "z"]])
def test_inputs_of_one_axis_with_different_lengths_are_rejected(b) -> None:
    session = scale_plan().create_session()

    with pytest.raises(WorkflowInputError, match="same length, or length 1"):
        session.run({"a": [1, 2], "b": b})

    assert session.instances[("s",)].calls == []


def test_empty_top_level_batch_runs_nothing_and_gives_no_rows() -> None:
    session = scale_plan().create_session()

    result = session.run({"a": [], "b": 2})

    assert result.rows() == []
    assert result.statuses == {"s": "complete"}
    assert session.instances[("s",)].calls == []


def test_nested_lists_follow_declared_axes_with_ragged_and_empty_groups() -> None:
    plan = (
        PlanBuilder()
        .input("groups", nested("c"))
        .step(Sum, "sum", at=BATCH, values="$inputs.groups")
        .output("sum", "$steps.sum.total")
        .output("groups", "$inputs.groups")
        .build()
    )
    session = plan.create_session()

    rows = session.run({"groups": [[1, 2], [], [3]]}).rows()

    assert rows == [
        {"sum": 3, "groups": [1, 2]},
        {"sum": 0, "groups": []},
        {"sum": 3, "groups": [3]},
    ]


def test_explicit_batch_keeps_sparse_indices_and_metadata() -> None:
    context = SampleContext(source_id="camera-7")
    items = InputValue(
        Batch(["a", "c"], indices=[(0,), (2,)]),
        metadata=EntryMetadata(sample={(2,): context}),
    )
    session = echo_plan().create_session()

    result = session.run({"items": items})

    assert [call["value"] for call in session.instances[("echo",)].calls] == ["a", "c"]
    assert result.outputs.data["items"].indices == ((0,), (2,))
    assert result.outputs.metadata["items"].sample_at((2,)) == context
    assert result.rows() == [{"items": "a"}, {"items": None}, {"items": "c"}]


def test_metadata_must_address_existing_positions() -> None:
    items = InputValue(
        Batch.of(["a"]), metadata=EntryMetadata(sample={(5,): SampleContext("x")})
    )

    with pytest.raises(WorkflowInputError, match=r"'items'.*\(5,\)"):
        echo_plan().create_session().run({"items": items})


def test_unknown_missing_and_empty_grouped_inputs_are_rejected() -> None:
    session = echo_plan().create_session()

    with pytest.raises(WorkflowInputError, match=r"Missing: \['items'\]"):
        session.run({})
    with pytest.raises(WorkflowInputError, match=r"unknown: \['other'\]"):
        session.run({"items": [1], "other": 2})
    with pytest.raises(WorkflowInputError, match="no value was provided"):
        session.run({"items": None})


def test_optional_parameter_takes_default_and_explicit_none_is_a_payload() -> None:
    plan = (
        PlanBuilder()
        .input("value", SCALAR, required=False, default=7)
        .step(Echo, "echo", value="$inputs.value")
        .output("value", "$inputs.value")
        .build()
    )
    session = plan.create_session()

    assert session.run({}).rows() == [{"value": 7}]
    assert session.run({"value": None}).rows() == [{"value": None}]
    # A parameter is static: its None is delivered, it never suppresses a call.
    assert session.instances[("echo",)].calls == [{"value": 7}, {"value": None}]


def test_kind_deserializer_runs_per_leaf_and_validator_rejects_with_index() -> None:
    tagged = tagged_kind()
    plan = echo_plan(kinds=("tagged",), extra_kinds=(tagged,))
    session = plan.create_session()

    session.run({"items": [1, None]})

    assert session.instances[("echo",)].calls == [{"value": ("tag", 1)}]
    rejecting = echo_plan(kinds=("integer",), extra_kinds=(INTEGER_KIND,))
    with pytest.raises(WorkflowInputError, match=r"'items' at index \[1\]"):
        rejecting.create_session().run({"items": [1, "two"]})


def test_union_of_kinds_uses_the_first_deserializer_that_succeeds() -> None:
    strict = Kind(name="strict", deserialize=lambda value: int(value))
    loose = Kind(name="loose", deserialize=lambda value: f"loose:{value}")
    plan = echo_plan(kinds=("strict", "loose"), extra_kinds=(strict, loose))
    session = plan.create_session()

    session.run({"items": ["3", "x"]})

    assert [call["value"] for call in session.instances[("echo",)].calls] == [
        3,
        "loose:x",
    ]


def test_inputs_sharing_nested_axes_must_share_their_groups() -> None:
    plan = (
        PlanBuilder()
        .input("left", nested("c"))
        .input("right", nested("c"))
        .step(Scale, "s", at=nested("c"), value="$inputs.left", factor="$inputs.right")
        .build()
    )

    with pytest.raises(WorkflowInputError, match="groups at depth 1 differ"):
        plan.create_session().run({"left": [[1, 2]], "right": [[1]]})


def test_inputs_on_independent_axes_do_not_align_by_size() -> None:
    other = EntryLayout(axes=(Axis(id="M", kind="sample"),))
    plan = (
        PlanBuilder()
        .input("left", BATCH, kinds=("float",))
        .input("right", other, kinds=("float",))
        .step(Echo, "left_echo", at=BATCH, value="$inputs.left")
        .step(Echo, "right_echo", at=other, value="$inputs.right")
        .output("left", "$steps.left_echo.value")
        .output("right", "$steps.right_echo.value")
        .build()
    )

    result = plan.create_session().run({"left": [1, 2], "right": [3, 4, 5]})

    assert result.outputs.layout["right"].axis_ids == ("M",)
    with pytest.raises(WorkflowExecutionError, match="independent input axes"):
        result.rows()


def test_float_kind_accepts_integers_and_rejects_strings() -> None:
    plan = echo_plan(kinds=("float",), extra_kinds=(FLOAT_KIND,))

    plan.create_session().run({"items": [1, 2.5]})
    with pytest.raises(WorkflowInputError, match="not a valid"):
        plan.create_session().run({"items": ["x"]})


def ready(value):
    future = Future()
    future.set_result(value)

    return future


def test_future_inputs_are_resolved_before_checks_consumers_and_outputs() -> None:
    plan = (
        PlanBuilder(kinds=(INTEGER_KIND,))
        .input("value", SCALAR, kinds=("integer",))
        .input("items", BATCH)
        .step(Echo, "echo", value="$inputs.value")
        .step(Echo, "each", at=BATCH, value="$inputs.items")
        .output("direct", "$inputs.value")
        .output("echo", "$steps.echo.value")
        .output("each", "$steps.each.value")
        .build()
    )
    session = plan.create_session()
    kept = {"id": 1}

    rows = session.run({"value": ready(7), "items": [ready("a"), kept]}).rows()

    assert rows == [
        {"direct": 7, "echo": 7, "each": "a"},
        {"direct": 7, "echo": 7, "each": kept},
    ]
    assert session.instances[("each",)].calls[1]["value"] is kept


def test_failing_future_input_names_the_input() -> None:
    failed = Future()
    failed.set_exception(RuntimeError("camera offline"))

    with pytest.raises(WorkflowInputError, match="'items'.*camera offline"):
        echo_plan().create_session().run({"items": [failed]})
