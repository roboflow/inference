"""Control decisions: gates, conjunction, prefix masks, routing and recovery."""

import pytest
from roboflow_workflows.execution_engine.v2.errors import StepExecutionError

from .blocks import (
    BadControl,
    ContinueIf,
    Echo,
    Expand,
    FirstNonEmpty,
    Notice,
    Route,
    Scale,
)
from .plans import BATCH, SCALAR, PlanBuilder, nested


def calls(session, name):
    return session.instances[(name,)].calls


def gates_plan(*thresholds):
    builder = PlanBuilder().input("items", BATCH)
    gates = []
    for position, threshold in enumerate(thresholds):
        name = f"gate{position}"
        builder.step(
            ContinueIf,
            name,
            at=BATCH,
            value="$inputs.items",
            threshold=threshold,
            next_steps=["$steps.notice"],
        )
        gates.append((name, "$steps.notice"))
    builder.step(Notice, "notice", at=BATCH, gates=gates)
    plan = builder.build()

    return plan


def test_control_only_sink_runs_once_per_admitted_index() -> None:
    session = gates_plan(0).create_session()

    session.run({"items": [0, 1, 2, 0]})
    session.run({"items": [0, 0]})

    assert len(calls(session, "gate0")) == 6
    assert len(calls(session, "notice")) == 2


def test_ungated_literal_sink_runs_once_at_scalar_level() -> None:
    plan = PlanBuilder().input("items", BATCH).step(Notice, "notice").build()
    session = plan.create_session()

    session.run({"items": [0, 1, 2, 0]})

    assert calls(session, "notice") == [{"message": "found"}]


def test_multiple_gates_are_a_conjunction() -> None:
    session = gates_plan(0, 1).create_session()

    session.run({"items": [0, 1, 2]})

    assert len(calls(session, "notice")) == 1


def test_empty_mask_beside_nonempty_mask_denies_everything() -> None:
    # OBS-V1-01 correction: V1 ignores the empty mask and calls the sink 3 times.
    session = gates_plan(1, 100).create_session()

    session.run({"items": [0, 2, 3, 4]})

    assert len(calls(session, "gate0")) == 4
    assert calls(session, "notice") == []


def test_parent_and_child_gates_drive_a_control_only_sink_at_child_level() -> None:
    plan = (
        PlanBuilder()
        .input("items", BATCH)
        .step(Expand, "expand", at=BATCH, value="$inputs.items", count="$inputs.items")
        .step(
            ContinueIf,
            "parent_gate",
            at=BATCH,
            value="$inputs.items",
            threshold=1,
            next_steps=["$steps.notice"],
        )
        .step(
            ContinueIf,
            "child_gate",
            at=nested("expand/items"),
            value="$steps.expand.children",
            threshold=2.5,
            next_steps=["$steps.notice"],
        )
        .step(
            Notice,
            "notice",
            at=nested("expand/items"),
            gates=(("parent_gate", "$steps.notice"), ("child_gate", "$steps.notice")),
        )
        .build()
    )
    session = plan.create_session()

    session.run({"items": [1, 2, 3]})

    # children: [1], [2, 3], [3, 4, 5]; parent gate admits 2 and 3; child > 2.5
    assert len(calls(session, "child_gate")) == 6
    assert len(calls(session, "notice")) == 4


def test_shallow_gate_admits_deeper_data_by_prefix() -> None:
    # OBS-V1-03 correction: V1 makes zero calls here.
    plan = (
        PlanBuilder()
        .input("items", BATCH)
        .step(Expand, "expand", at=BATCH, value="$inputs.items", count="$inputs.items")
        .step(
            ContinueIf,
            "parent_gate",
            at=BATCH,
            value="$inputs.items",
            threshold=1,
            next_steps=["$steps.echo"],
        )
        .step(
            Echo,
            "echo",
            at=nested("expand/items"),
            gates=(("parent_gate", "$steps.echo"),),
            value="$steps.expand.children",
        )
        .output("selected", "$steps.echo.value")
        .build()
    )
    session = plan.create_session()

    result = session.run({"items": [1, 2, 0]})

    assert [call["value"] for call in calls(session, "echo")] == [2, 3]
    assert result.rows() == [
        {"selected": [None]},
        {"selected": [2, 3]},
        {"selected": []},
    ]


def test_scalar_gate_governs_the_whole_batch_domain() -> None:
    plan = (
        PlanBuilder()
        .input("flag", SCALAR)
        .input("values", BATCH)
        .step(ContinueIf, "gate", value="$inputs.flag", next_steps=["$steps.scale"])
        .step(
            Scale,
            "scale",
            at=BATCH,
            gates=(("gate", "$steps.scale"),),
            value="$inputs.values",
        )
        .output("scaled", "$steps.scale.scaled")
        .build()
    )
    session = plan.create_session()

    denied = session.run({"flag": 0, "values": [1, 2]})
    admitted = session.run({"flag": 1, "values": [1, 2]})

    assert denied.rows() == [{"scaled": None}, {"scaled": None}]
    assert denied.statuses == {"scaled": "filtered"}
    assert denied.filtered_paths == {"scaled": ((0,), (1,))}
    assert admitted.rows() == [{"scaled": 2}, {"scaled": 4}]
    assert len(calls(session, "scale")) == 2


def test_denied_invocations_propagate_downstream_without_placeholders() -> None:
    plan = (
        PlanBuilder()
        .input("values", BATCH)
        .step(
            ContinueIf,
            "gate",
            at=BATCH,
            value="$inputs.values",
            threshold=15,
            next_steps=["$steps.gated"],
        )
        .step(
            Scale,
            "gated",
            at=BATCH,
            gates=(("gate", "$steps.gated"),),
            value="$inputs.values",
        )
        .step(Scale, "after", at=BATCH, value="$steps.gated.scaled")
        .step(Scale, "ungated", at=BATCH, value="$inputs.values", factor=1)
        .output("gated", "$steps.gated.scaled")
        .output("after", "$steps.after.scaled")
        .output("ungated", "$steps.ungated.scaled")
        .build()
    )
    session = plan.create_session()

    rows = session.run({"values": [10, 20, 30]}).rows()

    assert rows == [
        {"gated": None, "after": None, "ungated": 10},
        {"gated": 40, "after": 80, "ungated": 20},
        {"gated": 60, "after": 120, "ungated": 30},
    ]
    assert [len(calls(session, name)) for name in ("gated", "after", "ungated")] == [
        2,
        2,
        3,
    ]


def test_routing_with_recovery_join_including_all_missing() -> None:
    plan = (
        PlanBuilder()
        .input("items", BATCH)
        .step(
            Route,
            "route",
            at=BATCH,
            value="$inputs.items",
            cases={"1": "$steps.left", "2": "$steps.right"},
        )
        .step(
            Echo,
            "left",
            at=BATCH,
            gates=(("route", "$steps.left"),),
            value="$inputs.items",
        )
        .step(
            Echo,
            "right",
            at=BATCH,
            gates=(("route", "$steps.right"),),
            value="$inputs.items",
        )
        .step(
            FirstNonEmpty,
            "merge",
            at=BATCH,
            data=["$steps.left.value", "$steps.right.value"],
            default="none",
        )
        .output("merged", "$steps.merge.value")
        .build()
    )
    session = plan.create_session()

    rows = session.run({"items": ["1", "2", "3"]}).rows()

    assert [call["data"] for call in calls(session, "merge")] == [
        ["1", None],
        [None, "2"],
        [None, None],
    ]
    assert rows == [{"merged": "1"}, {"merged": "2"}, {"merged": "none"}]
    assert len(calls(session, "left")) == 1 and len(calls(session, "right")) == 1


def test_recovery_join_runs_at_known_indices_when_every_branch_is_filtered() -> None:
    plan = (
        PlanBuilder()
        .input("items", BATCH)
        .step(
            ContinueIf,
            "gate",
            at=BATCH,
            value="$inputs.items",
            threshold=100,
            next_steps=["$steps.left"],
        )
        .step(
            Echo,
            "left",
            at=BATCH,
            gates=(("gate", "$steps.left"),),
            value="$inputs.items",
        )
        .step(FirstNonEmpty, "merge", at=BATCH, data=["$steps.left.value"], default=0)
        .output("merged", "$steps.merge.value")
        .build()
    )
    session = plan.create_session()

    rows = session.run({"items": [1, 2]}).rows()

    assert calls(session, "left") == []
    assert [call["data"] for call in calls(session, "merge")] == [[None], [None]]
    assert rows == [{"merged": 0}, {"merged": 0}]


def test_one_selection_fans_out_to_several_targets() -> None:
    plan = (
        PlanBuilder()
        .input("items", BATCH)
        .step(
            ContinueIf,
            "gate",
            at=BATCH,
            value="$inputs.items",
            next_steps=["$steps.notice", "$steps.echo"],
        )
        .step(Notice, "notice", at=BATCH, gates=(("gate", "$steps.notice"),))
        .step(
            Echo,
            "echo",
            at=BATCH,
            gates=(("gate", "$steps.echo"),),
            value="$inputs.items",
        )
        .output("echo", "$steps.echo.value")
        .build()
    )
    session = plan.create_session()

    rows = session.run({"items": [0, 1, 2]}).rows()

    assert rows == [{"echo": None}, {"echo": 1}, {"echo": 2}]
    assert len(calls(session, "notice")) == 2


def test_one_target_key_governing_every_step_of_a_nested_workflow() -> None:
    # A compiled nested-workflow target lists every child step; each is gated.
    plan = (
        PlanBuilder()
        .input("value", SCALAR)
        .step(
            ContinueIf,
            "gate",
            value="$inputs.value",
            next_steps=["$steps.inner"],
            targets={"$steps.inner": ["inner_first", "inner_second"]},
        )
        .step(Notice, "inner_first", gates=(("gate", "$steps.inner"),))
        .step(
            Notice, "inner_second", gates=(("gate", "$steps.inner"),), message="second"
        )
        .build()
    )
    session = plan.create_session()

    session.run({"value": 0})
    denied = [len(calls(session, name)) for name in ("inner_first", "inner_second")]
    session.run({"value": 1})
    admitted = [len(calls(session, name)) for name in ("inner_first", "inner_second")]

    assert denied == [0, 0]
    assert admitted == [1, 1]


@pytest.mark.parametrize(
    "answer, message",
    [
        ("mapping", r"must return Select\(\.\.\.\) or Stop\(\)"),
        ("unknown_target", r"selected \['\$steps.elsewhere'\], which are not targets"),
    ],
)
def test_control_result_must_be_a_selection_of_known_targets(answer, message) -> None:
    plan = (
        PlanBuilder()
        .step(BadControl, "control", answer=answer, next_steps=["$steps.notice"])
        .step(Notice, "notice", gates=(("control", "$steps.notice"),))
        .build()
    )
    session = plan.create_session()

    with pytest.raises(StepExecutionError, match=message) as caught:
        session.run({})

    assert caught.value.index == ()
    assert calls(session, "notice") == []
