"""Argument binding and call shapes of the V2 executor."""

import pytest
from roboflow_workflows.execution_engine.v2.data import Batch
from roboflow_workflows.execution_engine.v2.errors import StepExecutionError

from .blocks import (
    AppendInPlace,
    Collect,
    Consensus,
    Csv,
    Describe,
    Echo,
    HistoryAppend,
    InvertMany,
    LiteralGroups,
    NamedGroups,
    Notice,
    Region,
    Scale,
)
from .plans import BATCH, SCALAR, PlanBuilder, nested


def calls(session, name):
    return session.instances[(name,)].calls


def test_one_field_takes_literal_default_parameter_and_step_selector() -> None:
    plan = (
        PlanBuilder()
        .input("values", BATCH)
        .input("f", SCALAR, required=False, default=10)
        .step(Scale, "literal", at=BATCH, value="$inputs.values", factor=3)
        .step(Scale, "default", at=BATCH, value="$inputs.values")
        .step(Scale, "from_param", at=BATCH, value="$inputs.values", factor="$inputs.f")
        .step(
            Scale,
            "from_step",
            at=BATCH,
            value="$inputs.values",
            factor="$steps.literal.scaled",
        )
        .output("from_step", "$steps.from_step.scaled")
        .output("from_param", "$steps.from_param.scaled")
        .build()
    )
    session = plan.create_session()

    first = session.run({"values": [1, 2]}).rows()
    second = session.run({"values": [1, 2], "f": 100}).rows()

    assert first == [
        {"from_step": 3, "from_param": 10},
        {"from_step": 12, "from_param": 20},
    ]
    assert [row["from_param"] for row in second] == [100, 200]
    assert [call["factor"] for call in calls(session, "default")] == [2.0] * 4
    assert len(calls(session, "from_step")) == 4


def test_compound_list_of_batch_selectors_arrives_as_list_of_batches() -> None:
    plan = (
        PlanBuilder()
        .input("a", BATCH)
        .input("b", BATCH)
        .step(Consensus, "vote", at=BATCH, predictions=["$inputs.a", "$inputs.b"])
        .output("votes", "$steps.vote.votes")
        .build()
    )
    session = plan.create_session()

    rows = session.run({"a": [10, 20], "b": [100, 200]}).rows()

    (call,) = calls(session, "vote")
    assert isinstance(call["predictions"], list)
    assert [list(batch) for batch in call["predictions"]] == [[10, 20], [100, 200]]
    assert all(batch.indices == ((0,), (1,)) for batch in call["predictions"])
    assert rows == [{"votes": [10, 100]}, {"votes": [20, 200]}]


def test_batch_only_list_casts_scalar_leaf_beside_batch_leaf() -> None:
    plan = (
        PlanBuilder()
        .input("a", BATCH)
        .input("s", SCALAR)
        .step(Consensus, "vote", at=BATCH, predictions=["$inputs.a", "$inputs.s"])
        .output("votes", "$steps.vote.votes")
        .build()
    )
    session = plan.create_session()

    rows = session.run({"a": [10, 20], "s": 8}).rows()

    (call,) = calls(session, "vote")
    assert [list(batch) for batch in call["predictions"]] == [[10, 20], [8, 8]]
    assert rows == [{"votes": [10, 8]}, {"votes": [20, 8]}]


def test_batch_only_list_of_scalars_is_one_call_over_index_zero() -> None:
    plan = (
        PlanBuilder()
        .input("x", SCALAR)
        .input("y", SCALAR)
        .step(Consensus, "vote", predictions=["$inputs.x", "$inputs.y"])
        .output("votes", "$steps.vote.votes")
        .build()
    )
    session = plan.create_session()

    rows = session.run({"x": 7, "y": 8}).rows()

    (call,) = calls(session, "vote")
    assert [batch.indices for batch in call["predictions"]] == [((0,),), ((0,),)]
    assert rows == [{"votes": [7, 8]}]


def test_mixed_dict_with_only_static_leaves_is_a_scalar_call_returning_a_mapping() -> (
    None
):
    plan = (
        PlanBuilder()
        .input("value", SCALAR)
        .step(Csv, "csv", columns={"value": "$inputs.value", "label": "fixed"})
        .output("row", "$steps.csv.row")
        .build()
    )
    session = plan.create_session()

    rows = session.run({"value": 7}).rows()

    assert plan.steps[0].delivers_batches is False
    assert calls(session, "csv") == [{"columns": {"value": 7, "label": "fixed"}}]
    assert rows == [{"row": {"value": 7, "label": "fixed"}}]


def test_mixed_dict_with_varying_leaf_batches_it_and_keeps_literal_scalar() -> None:
    plan = (
        PlanBuilder()
        .input("values", BATCH)
        .step(
            Csv, "csv", at=BATCH, columns={"value": "$inputs.values", "label": "fixed"}
        )
        .output("row", "$steps.csv.row")
        .build()
    )
    session = plan.create_session()

    rows = session.run({"values": [10, 20]}).rows()

    (call,) = calls(session, "csv")
    assert isinstance(call["columns"]["value"], Batch)
    assert call["columns"]["label"] == "fixed"
    assert rows == [
        {"row": {"value": 10, "label": "fixed"}},
        {"row": {"value": 20, "label": "fixed"}},
    ]


def test_batch_delivery_over_nested_domain_is_one_call_with_full_indices() -> None:
    plan = (
        PlanBuilder()
        .input("values", nested("c"))
        .step(InvertMany, "invert", at=nested("c"), values="$inputs.values", offset=1)
        .output("inverted", "$steps.invert.inverted")
        .build()
    )
    session = plan.create_session()

    rows = session.run({"values": [[1, 2], [], [3]]}).rows()

    (call,) = calls(session, "invert")
    assert call["values"].indices == ((0, 0), (0, 1), (2, 0))
    assert call["offset"] == 1
    assert rows == [{"inverted": [0, -1]}, {"inverted": []}, {"inverted": [-2]}]


def test_literal_list_payload_is_not_a_logical_group() -> None:
    plan = (
        PlanBuilder()
        .input("item", SCALAR)
        .step(Echo, "echo", value="$inputs.item")
        .output("value", "$steps.echo.value")
        .build()
    )
    session = plan.create_session()

    rows = session.run({"item": [1, 2, 3]}).rows()

    assert calls(session, "echo") == [{"value": [1, 2, 3]}]
    assert rows == [{"value": [1, 2, 3]}]


def test_scalar_leaves_of_a_compound_group_are_cast_per_parent() -> None:
    plan = (
        PlanBuilder()
        .input("parents", BATCH)
        .input("label", SCALAR)
        .step(
            NamedGroups,
            "named",
            at=BATCH,
            parent="$inputs.parents",
            groups={"selected": "$inputs.label"},
        )
        .output("sizes", "$steps.named.sizes")
        .build()
    )
    session = plan.create_session()

    session.run({"parents": [10, 20], "label": "chosen"})

    groups = [call["groups"]["selected"] for call in calls(session, "named")]
    assert [group.indices for group in groups] == [((0, 0),), ((1, 0),)]
    assert [list(group) for group in groups] == [["chosen"], ["chosen"]]


def test_whole_field_scalar_group_at_scalar_level_is_a_singleton_batch() -> None:
    plan = (
        PlanBuilder()
        .input("value", SCALAR)
        .step(Collect, "collect", values="$inputs.value")
        .output("members", "$steps.collect.members")
        .build()
    )
    session = plan.create_session()

    rows = session.run({"value": "constant"}).rows()

    (call,) = calls(session, "collect")
    assert call["values"].indices == ((0,),)
    assert rows == [{"members": ["constant"]}]


def test_resolved_value_violating_literal_constraint_fails_before_any_call() -> None:
    plan = (
        PlanBuilder()
        .input("values", BATCH)
        .input("factors", BATCH)
        .step(
            Scale, "scale", at=BATCH, value="$inputs.values", factor="$inputs.factors"
        )
        .output("scaled", "$steps.scale.scaled")
        .build()
    )
    session = plan.create_session()

    with pytest.raises(StepExecutionError, match="factor") as caught:
        session.run({"values": [1, 2], "factors": [1, -1]})

    assert caught.value.index == (1,)
    assert calls(session, "scale") == []


def test_selected_value_of_wrong_kind_is_rejected_with_field_context() -> None:
    plan = (
        PlanBuilder()
        .input("values", BATCH)
        .step(Scale, "scale", at=BATCH, value="$inputs.values")
        .output("scaled", "$steps.scale.scaled")
        .build()
    )
    session = plan.create_session()

    with pytest.raises(StepExecutionError, match=r"parameter value is not a valid"):
        session.run({"values": [1, "two"]})

    assert calls(session, "scale") == []


def test_none_from_varying_binding_skips_default_block_but_static_none_does_not() -> (
    None
):
    plan = (
        PlanBuilder()
        .input("values", BATCH)
        .input("suffix", SCALAR, required=False, default=None)
        .step(
            Describe,
            "describe",
            at=BATCH,
            value="$inputs.values",
            suffix="$inputs.suffix",
        )
        .step(Describe, "literal_none", at=BATCH, value="$inputs.values", suffix=None)
        .output("text", "$steps.describe.text")
        .build()
    )
    session = plan.create_session()

    result = session.run({"values": ["a", None, "c"]})

    assert [call["suffix"] for call in calls(session, "describe")] == [None, None]
    assert len(calls(session, "literal_none")) == 2
    assert result.rows() == [{"text": "a"}, {"text": None}, {"text": "c"}]
    assert result.filtered_paths["text"] == ((1,),)


def test_literal_parameters_are_private_copies_per_invocation() -> None:
    plan = (
        PlanBuilder()
        .input("values", BATCH)
        .step(Csv, "csv", at=BATCH, columns={"value": "$inputs.values", "tag": "x"})
        .step(Notice, "notice", message="kept")
        .build()
    )
    session = plan.create_session()

    session.run({"values": [1]})
    (call,) = calls(session, "csv")
    call["columns"]["tag"] = "changed"
    session.run({"values": [1]})

    assert calls(session, "csv")[1]["columns"]["tag"] == "x"
    assert plan.steps[0].params.columns["tag"] == "x"


def test_selected_payload_keeps_identity_while_the_literal_is_parsed() -> None:
    # Decision 018: runtime validation checks, it never converts or copies.
    plan = (
        PlanBuilder()
        .input("region", SCALAR)
        .step(Region, "literal", region=[1, 2])
        .step(Region, "selected", region="$inputs.region")
        .output("literal", "$steps.literal.region")
        .output("selected", "$steps.selected.region")
        .build()
    )
    selected = [1, 2]

    result = plan.create_session().run({"region": selected})

    assert result.outputs.data["literal"] == (1, 2)
    assert result.outputs.data["selected"] is selected


def test_mutated_field_keeps_the_selected_object() -> None:
    plan = (
        PlanBuilder()
        .input("items", SCALAR)
        .step(AppendInPlace, "append", items="$inputs.items")
        .output("items", "$steps.append.items")
        .build()
    )
    items = [7]

    result = plan.create_session().run({"items": items})

    assert result.outputs.data["items"] is items
    assert items == [7, 1]


def test_literal_at_a_group_position_is_cast_like_a_selected_scalar() -> None:
    plan = (
        PlanBuilder()
        .input("parents", BATCH)
        .input("offset", SCALAR)
        .step(
            LiteralGroups,
            "groups",
            at=BATCH,
            parent="$inputs.parents",
            groups={"literal": 8, "selected": "$inputs.offset"},
        )
        .output("members", "$steps.groups.members")
        .build()
    )
    session = plan.create_session()

    rows = session.run({"parents": [2, 1], "offset": 9}).rows()

    groups = [call["groups"] for call in calls(session, "groups")]
    assert [group["literal"].indices for group in groups] == [((0, 0),), ((1, 0),)]
    assert [group["selected"].indices for group in groups] == [((0, 0),), ((1, 0),)]
    assert rows == [{"members": {"literal": [8], "selected": [9]}}] * 2


def test_literal_inside_a_compound_field_is_a_private_copy_per_run() -> None:
    plan = (
        PlanBuilder()
        .input("value", SCALAR)
        .step(
            HistoryAppend,
            "append",
            records={"literal": {"history": []}, "selected": "$inputs.value"},
        )
        .output("length", "$steps.append.length")
        .build()
    )
    session = plan.create_session()

    runs = [session.run({"value": {}}).rows() for _ in range(2)]

    assert runs == [[{"length": 1}], [{"length": 1}]]
    assert plan.steps[0].params.records["literal"] == {"history": []}
