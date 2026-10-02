"""Decision 016: absent groups versus present groups whose children were filtered.

A group exists once its parent produced it: genuinely empty, partly or fully
filtered. A group whose parent was filtered before expansion never existed:

    empty-accepting block, other binding supplies the index   receives None
    empty-accepting block, the group is its only source       not invoked there
    default block                                              skipped
"""

from .blocks import ContinueIf, Echo, Expand, Pair, PairMany, SumAcceptingEmpty
from .plans import BATCH, PlanBuilder, nested


def calls(session, name):
    return session.instances[(name,)].calls


def gated_expansion(builder, *, parent_threshold, child_threshold):
    """values → parent gate → expand (count 2) → child gate → echo children."""
    builder.input("values", BATCH)
    builder.step(
        ContinueIf,
        "parent_gate",
        at=BATCH,
        value="$inputs.values",
        threshold=parent_threshold,
        next_steps=["$steps.expand"],
    )
    builder.step(
        Expand,
        "expand",
        at=BATCH,
        gates=(("parent_gate", "$steps.expand"),),
        value="$inputs.values",
        count=2,
    )
    builder.step(
        ContinueIf,
        "child_gate",
        at=nested("expand/items"),
        value="$steps.expand.children",
        threshold=child_threshold,
        next_steps=["$steps.echo"],
    )
    builder.step(
        Echo,
        "echo",
        at=nested("expand/items"),
        gates=(("child_gate", "$steps.echo"),),
        value="$steps.expand.children",
    )

    return builder


def test_group_only_reducer_is_not_invoked_for_an_absent_group() -> None:
    builder = gated_expansion(PlanBuilder(), parent_threshold=0, child_threshold=15)
    plan = (
        builder.step(SumAcceptingEmpty, "sum", at=BATCH, values="$steps.echo.value")
        .output("sum", "$steps.sum.total")
        .build()
    )
    session = plan.create_session()

    result = session.run({"values": [0, 10, 20]})

    # parent 0: filtered before expansion (absent group) → no call
    # parent 10: children 10, 11 all filtered (present group) → empty Batch
    # parent 20: children 20, 21 survive
    assert [call["values"].parent_index for call in calls(session, "sum")] == [
        (1,),
        (2,),
    ]
    assert [list(call["values"]) for call in calls(session, "sum")] == [[], [20, 21]]
    assert result.rows() == [{"sum": None}, {"sum": 0}, {"sum": 41}]
    assert result.filtered_paths["sum"] == ((0,),)


def test_absent_group_arrives_as_none_when_an_item_supplies_the_index() -> None:
    builder = gated_expansion(PlanBuilder(), parent_threshold=0, child_threshold=15)
    plan = (
        builder.step(
            Pair,
            "pair",
            at=BATCH,
            parent="$inputs.values",
            children="$steps.echo.value",
        )
        .output("pair", "$steps.pair.pair")
        .build()
    )
    session = plan.create_session()

    rows = session.run({"values": [0, 10, 20]}).rows()

    assert [call["children"] for call in calls(session, "pair")][0] is None
    assert rows == [
        {"pair": [0, None]},
        {"pair": [10, []]},
        {"pair": [20, [20, 21]]},
    ]


def test_batch_delivered_absent_group_is_a_none_member() -> None:
    builder = gated_expansion(PlanBuilder(), parent_threshold=0, child_threshold=15)
    plan = (
        builder.step(
            PairMany,
            "pair",
            at=BATCH,
            parent="$inputs.values",
            children="$steps.echo.value",
        )
        .output("pair", "$steps.pair.pair")
        .build()
    )
    session = plan.create_session()

    rows = session.run({"values": [0, 20]}).rows()

    (call,) = calls(session, "pair")
    assert call["children"].indices == ((0,), (1,))
    assert call["children"][0] is None
    assert list(call["children"][1]) == [20, 21]
    assert rows == [{"pair": [0, None]}, {"pair": [20, [20, 21]]}]


def test_wholly_filtered_nested_output_keeps_its_row_shape() -> None:
    # Request R3: one input whose only child is filtered.
    builder = gated_expansion(PlanBuilder(), parent_threshold=-1, child_threshold=100)
    plan = builder.output("echo", "$steps.echo.value").build()

    result = plan.create_session().run({"values": [1]})

    assert result.statuses == {"echo": "filtered"}
    assert result.rows() == [{"echo": [None, None]}]
