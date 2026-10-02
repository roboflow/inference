"""Expansion, reduction, preserved groups, sparse indices and metadata."""

import pytest
from roboflow_workflows.execution_engine.v2.data import (
    Axis,
    Batch,
    EntryLayout,
    EntryMetadata,
    InputValue,
    SampleContext,
)
from roboflow_workflows.execution_engine.v2.errors import StepExecutionError

from .blocks import (
    Bad,
    ContinueIf,
    Expand,
    Scale,
    Source,
    StitchAndTranslate,
    Sum,
    SumAcceptingEmpty,
)
from .plans import BATCH, PlanBuilder, nested

GENERATED = EntryLayout(axes=(Axis(id="source/generated", kind="dynamic_nesting"),))


def calls(session, name):
    return session.instances[(name,)].calls


def expand_reduce_plan(reducer=Sum, *, gate_children=False):
    builder = (
        PlanBuilder()
        .input("values", BATCH)
        .input("counts", BATCH)
        .step(
            Expand, "expand", at=BATCH, value="$inputs.values", count="$inputs.counts"
        )
    )
    child_gates = ()
    if gate_children:
        builder.step(
            ContinueIf,
            "gate",
            at=nested("expand/items"),
            value="$steps.expand.children",
            threshold=10,
            next_steps=["$steps.scale"],
        )
        child_gates = (("gate", "$steps.scale"),)
    builder.step(
        Scale,
        "scale",
        at=nested("expand/items"),
        gates=child_gates,
        value="$steps.expand.children",
        factor=10,
    )
    builder.step(reducer, "sum", at=BATCH, values="$steps.scale.scaled")
    builder.output("scaled", "$steps.scale.scaled").output("sum", "$steps.sum.total")
    plan = builder.build()

    return plan


def test_ragged_expansion_scales_children_and_reduces_per_parent() -> None:
    session = expand_reduce_plan().create_session()

    result = session.run({"values": [1, 2, 3], "counts": [2, 0, 1]})

    assert result.rows() == [
        {"scaled": [10, 20], "sum": 30},
        {"scaled": [], "sum": 0},
        {"scaled": [30], "sum": 30},
    ]
    assert [call["values"].indices for call in calls(session, "sum")] == [
        ((0, 0), (0, 1)),
        (),
        ((2, 0),),
    ]
    assert calls(session, "sum")[1]["values"].parent_index == (1,)
    assert len(calls(session, "scale")) == 3


def test_all_filtered_group_is_skipped_but_accepting_reducer_gets_empty_batch() -> None:
    inputs = {"values": [1, 20], "counts": [2, 2]}
    default = expand_reduce_plan(gate_children=True).create_session()
    accepting = expand_reduce_plan(
        SumAcceptingEmpty, gate_children=True
    ).create_session()

    default_result = default.run(inputs)
    accepting_result = accepting.run(inputs)

    assert default_result.rows() == [
        {"scaled": [None, None], "sum": None},
        {"scaled": [200, 210], "sum": 410},
    ]
    assert len(calls(default, "sum")) == 1
    assert [len(call["values"]) for call in calls(accepting, "sum")] == [0, 2]
    assert calls(accepting, "sum")[0]["values"].parent_index == (0,)
    assert accepting_result.rows()[0] == {"scaled": [None, None], "sum": 0}


def test_genuine_empty_expansion_reaches_default_reducer() -> None:
    session = expand_reduce_plan().create_session()

    result = session.run({"values": [1, 20], "counts": [0, 0]})

    assert result.rows() == [{"scaled": [], "sum": 0}, {"scaled": [], "sum": 0}]
    assert len(calls(session, "sum")) == 2
    assert result.statuses == {"scaled": "complete", "sum": "complete"}


def test_group_block_keeps_parent_and_child_layouts_in_separate_outputs() -> None:
    plan = (
        PlanBuilder()
        .input("parents", BATCH)
        .step(Expand, "expand", at=BATCH, value="$inputs.parents", count=2)
        .step(
            StitchAndTranslate,
            "stitch",
            at=BATCH,
            parent="$inputs.parents",
            children="$steps.expand.children",
        )
        .step(
            Scale,
            "scale",
            at=nested("expand/items"),
            value="$steps.stitch.translated",
            factor="$steps.expand.children",
        )
        .output("stitched", "$steps.stitch.stitched")
        .output("translated", "$steps.stitch.translated")
        .output("product", "$steps.scale.scaled")
        .build()
    )
    session = plan.create_session()

    result = session.run({"parents": [10, 20]})

    assert result.outputs.layout["stitched"].axis_ids == ("N",)
    assert result.outputs.layout["translated"].axis_ids == ("N", "expand/items")
    assert result.rows() == [
        {"stitched": 31, "translated": [20, 21], "product": [200, 231]},
        {"stitched": 61, "translated": [40, 41], "product": [800, 861]},
    ]


def test_sparse_group_survivors_keep_indices_in_preserved_output() -> None:
    plan = (
        PlanBuilder()
        .input("values", BATCH)
        .step(Expand, "expand", at=BATCH, value="$inputs.values", count=3)
        .step(
            ContinueIf,
            "gate",
            at=nested("expand/items"),
            value="$steps.expand.children",
            threshold=10.5,
            next_steps=["$steps.scale"],
        )
        .step(
            Scale,
            "scale",
            at=nested("expand/items"),
            gates=(("gate", "$steps.scale"),),
            value="$steps.expand.children",
            factor=1,
        )
        .step(
            StitchAndTranslate,
            "stitch",
            at=BATCH,
            parent="$inputs.values",
            children="$steps.scale.scaled",
        )
        .output("translated", "$steps.stitch.translated")
        .build()
    )
    session = plan.create_session()

    result = session.run({"values": [10, 20]})

    children = [call["children"] for call in calls(session, "stitch")]
    assert [group.indices for group in children] == [
        ((0, 1), (0, 2)),
        ((1, 0), (1, 1), (1, 2)),
    ]
    assert result.filtered_paths["translated"] == ((0, 0),)
    assert result.rows() == [
        {"translated": [None, 21, 22]},
        {"translated": [40, 41, 42]},
    ]


def test_source_step_creates_its_own_top_level_axis() -> None:
    plan = (
        PlanBuilder()
        .input("values", BATCH)
        .step(Source, "source", values=[5.0, 6.0, 7.0])
        .step(Scale, "scale", at=BATCH, value="$inputs.values")
        .step(Scale, "generated", at=GENERATED, value="$steps.source.generated")
        .output("scaled", "$steps.scale.scaled")
        .output("generated", "$steps.generated.scaled")
        .build()
    )
    session = plan.create_session()

    result = session.run({"values": [1, 2]})

    assert plan.axis_origin("source/generated").kind == "expand"
    assert result.rows() == [
        {"scaled": 2, "generated": [10, 12, 14]},
        {"scaled": 4, "generated": [10, 12, 14]},
    ]
    assert len(calls(session, "source")) == 1


def test_output_metadata_common_or_none_and_explicit_none_stops_inheritance() -> None:
    north, south = SampleContext(source_id="north"), SampleContext(source_id="south")
    left = InputValue(
        Batch.of([1, 2, 3]),
        metadata=EntryMetadata(sample={(): north}),
    )
    right = InputValue(
        Batch.of([4, 5, 6]),
        metadata=EntryMetadata(sample={(): north, (1,): south}),
    )
    plan = (
        PlanBuilder()
        .input("left", BATCH)
        .input("right", BATCH)
        .step(Scale, "both", at=BATCH, value="$inputs.left", factor="$inputs.right")
        .step(Scale, "one", at=BATCH, value="$inputs.left", factor=1)
        .output("both", "$steps.both.scaled")
        .output("one", "$steps.one.scaled")
        .build()
    )

    result = plan.create_session().run({"left": left, "right": right})

    both = result.outputs.metadata["both"]
    assert [both.sample_at((index,)) for index in range(3)] == [north, None, north]
    assert both.sample[(1,)] is None
    assert result.outputs.metadata["one"].sample == {(): north}


def test_output_source_field_selects_context_of_one_binding() -> None:
    parents = InputValue(
        Batch.of([10, 20]),
        metadata=EntryMetadata(
            sample={(0,): SampleContext("a"), (1,): SampleContext("b")}
        ),
    )
    plan = (
        PlanBuilder()
        .input("parents", BATCH)
        .step(Expand, "expand", at=BATCH, value="$inputs.parents", count=1)
        .step(
            StitchAndTranslate,
            "stitch",
            at=BATCH,
            parent="$inputs.parents",
            children="$steps.expand.children",
        )
        .output("stitched", "$steps.stitch.stitched")
        .output("children", "$steps.expand.children")
        .build()
    )

    result = plan.create_session().run({"parents": parents})

    stitched = result.outputs.metadata["stitched"]
    children = result.outputs.metadata["children"]
    assert [stitched.sample_at((index,)).source_id for index in range(2)] == ["a", "b"]
    assert children.sample_at((1, 0)).source_id == "b"


@pytest.mark.parametrize(
    "mode, message",
    [
        ("omitted", r"Omitted: \['children'\]"),
        ("unknown", r"unknown: \['extra'\]"),
        ("not_mapping", "return a mapping"),
        ("list_as_children", "a list is one payload"),
        ("wrong_kind", r"output 'value' is not a valid \['integer'\]"),
        ("batch_as_payload", "only expand/preserve outputs return a Batch"),
    ],
)
def test_result_violations_name_step_output_and_index(mode, message) -> None:
    plan = (
        PlanBuilder()
        .input("values", BATCH)
        .step(Bad, "bad", at=BATCH, value="$inputs.values", mode=mode)
        .output("value", "$steps.bad.value")
        .build()
    )

    with pytest.raises(StepExecutionError, match=message) as caught:
        plan.create_session().run({"values": [1]})

    assert caught.value.step_path == ("bad",)
    assert caught.value.index == (0,)


def test_same_values_reduce_twice_across_two_nested_levels() -> None:
    plan = (
        PlanBuilder()
        .input("values", nested("c", "d"))
        .step(Sum, "inner", at=nested("c"), values="$inputs.values")
        .step(Sum, "outer", at=BATCH, values="$steps.inner.total")
        .output("inner", "$steps.inner.total")
        .output("outer", "$steps.outer.total")
        .build()
    )

    rows = plan.create_session().run({"values": [[[1, 2], []], [[3]]]}).rows()

    assert rows == [{"inner": [3, 0], "outer": 3}, {"inner": [3], "outer": 3}]


def test_reducer_context_is_common_or_none_over_its_children() -> None:
    same, other = SampleContext("same"), SampleContext("other")
    values = InputValue(
        Batch(
            [
                Batch([1, 2], parent_index=(0,)),
                Batch([3, 4], parent_index=(1,)),
            ]
        ),
        metadata=EntryMetadata(sample={(): same, (1, 1): other}),
    )
    plan = (
        PlanBuilder()
        .input("values", nested("c"))
        .step(Sum, "sum", at=BATCH, values="$inputs.values")
        .output("sum", "$steps.sum.total")
        .build()
    )

    result = plan.create_session().run({"values": values})

    metadata = result.outputs.metadata["sum"]
    assert metadata.sample_at((0,)) == same
    assert metadata.sample_at((1,)) is None


def test_preserve_output_must_align_with_the_delivered_group() -> None:
    plan = (
        PlanBuilder()
        .input("parents", BATCH)
        .input("children", nested("c"))
        .step(
            StitchAndTranslate,
            "stitch",
            at=BATCH,
            parent="$inputs.parents",
            children="$inputs.children",
        )
        .build()
    )
    session = plan.create_session()
    session.instances[("stitch",)].run = lambda *, parent, children: {
        "stitched": 0,
        "translated": Batch([1], indices=[(0, 5)], parent_index=(0,)),
    }

    with pytest.raises(StepExecutionError, match="returned indices"):
        session.run({"parents": [1], "children": [[1, 2]]})


def test_denied_parent_filters_its_genuine_empty_group_too() -> None:
    plan = (
        PlanBuilder()
        .input("values", BATCH)
        .input("groups", nested("c"))
        .step(
            ContinueIf,
            "gate",
            at=BATCH,
            value="$inputs.values",
            next_steps=["$steps.sum"],
        )
        .step(
            Sum,
            "sum",
            at=BATCH,
            gates=(("gate", "$steps.sum"),),
            values="$inputs.groups",
        )
        .output("sum", "$steps.sum.total")
        .build()
    )
    session = plan.create_session()

    result = session.run({"values": [0, 1], "groups": [[], []]})

    assert result.rows() == [{"sum": None}, {"sum": 0}]
    assert len(calls(session, "sum")) == 1


def test_child_mask_of_one_lineage_does_not_reach_a_fresh_expansion() -> None:
    plan = (
        PlanBuilder()
        .input("values", BATCH)
        .step(Expand, "first", at=BATCH, value="$inputs.values", count=2)
        .step(
            ContinueIf,
            "gate",
            at=nested("first/items"),
            value="$steps.first.children",
            threshold=100,
            next_steps=["$steps.masked"],
        )
        .step(
            Scale,
            "masked",
            at=nested("first/items"),
            gates=(("gate", "$steps.masked"),),
            value="$steps.first.children",
        )
        .step(Expand, "second", at=BATCH, value="$inputs.values", count=2)
        .step(Scale, "fresh", at=nested("second/items"), value="$steps.second.children")
        .output("masked", "$steps.masked.scaled")
        .output("fresh", "$steps.fresh.scaled")
        .build()
    )

    rows = plan.create_session().run({"values": [1]}).rows()

    assert rows == [{"masked": [None, None], "fresh": [2, 4]}]
