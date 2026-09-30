"""Batch-delivering calls, shared expand axes, casts and parent broadcast."""

from concurrent.futures import Future

import pytest
from roboflow_workflows.execution_engine.v2.data import Batch
from roboflow_workflows.execution_engine.v2.errors import (
    ContractError,
    StepExecutionError,
)
from roboflow_workflows.execution_engine.v2.plan import resolve_futures

from .blocks import (
    ContinueIf,
    Crop,
    DeferredMany,
    Echo,
    EchoMany,
    EvenGateMany,
    Expand,
    Relabel,
    Scale,
    SplitMany,
    SumMany,
)
from .plans import BATCH, SCALAR, PlanBuilder, nested


def calls(session, name):
    return session.instances[(name,)].calls


def test_batch_delivering_block_expands_every_sample_in_one_call() -> None:
    plan = (
        PlanBuilder()
        .input("values", BATCH)
        .step(SplitMany, "split", at=BATCH, values="$inputs.values")
        .step(
            Scale,
            "scale",
            at=nested("split/parts"),
            value="$steps.split.parts",
            factor=10,
        )
        .output("parts", "$steps.scale.scaled")
        .build()
    )
    session = plan.create_session()

    rows = session.run({"values": [1, 2]}).rows()

    assert len(calls(session, "split")) == 1
    assert rows == [{"parts": [10, 15]}, {"parts": [20, 25]}]


def test_batch_delivered_group_is_a_batch_of_groups_including_empty_ones() -> None:
    plan = (
        PlanBuilder()
        .input("groups", nested("c"))
        .step(SumMany, "sum", at=BATCH, values="$inputs.groups")
        .output("sum", "$steps.sum.total")
        .build()
    )
    session = plan.create_session()

    rows = session.run({"groups": [[1, 2], [], [3]]}).rows()

    (call,) = calls(session, "sum")
    assert call["values"].indices == ((0,), (1,), (2,))
    assert [group.parent_index for group in call["values"]] == [(0,), (1,), (2,)]
    assert rows == [{"sum": 3}, {"sum": 0}, {"sum": 3}]


def test_empty_accepting_batch_call_receives_none_at_filtered_positions() -> None:
    plan = (
        PlanBuilder()
        .input("values", BATCH)
        .step(
            ContinueIf,
            "gate",
            at=BATCH,
            value="$inputs.values",
            threshold=1,
            next_steps=["$steps.echo"],
        )
        .step(
            Echo,
            "echo",
            at=BATCH,
            gates=(("gate", "$steps.echo"),),
            value="$inputs.values",
        )
        .step(EchoMany, "many", at=BATCH, values="$steps.echo.value")
        .output("value", "$steps.many.value")
        .build()
    )
    session = plan.create_session()

    rows = session.run({"values": [1, 2, 3]}).rows()

    (call,) = calls(session, "many")
    assert list(call["values"]) == [None, 2, 3]
    assert call["values"].indices == ((0,), (1,), (2,))
    assert rows == [{"value": None}, {"value": 2}, {"value": 3}]


def test_batch_step_is_not_called_for_an_empty_domain() -> None:
    plan = (
        PlanBuilder()
        .input("values", BATCH)
        .step(SplitMany, "split", at=BATCH, values="$inputs.values")
        .output("parts", "$steps.split.parts")
        .build()
    )
    session = plan.create_session()

    rows = session.run({"values": []}).rows()

    assert calls(session, "split") == []
    assert rows == []


def crop_plan(*, mismatch):
    plan = (
        PlanBuilder()
        .input("values", BATCH)
        .step(Crop, "crop", at=BATCH, value="$inputs.values", mismatch=mismatch)
        .output("crops", "$steps.crop.crops")
        .output("boxes", "$steps.crop.boxes")
        .build()
    )

    return plan


def test_expand_outputs_sharing_an_axis_align_by_index() -> None:
    plan = crop_plan(mismatch=False)

    rows = plan.create_session().run({"values": [3]}).rows()

    outputs = plan.steps[0].outputs
    assert outputs["crops"].layout == outputs["boxes"].layout
    assert rows == [{"crops": [3, 6], "boxes": ["box0", "box2"]}]


def test_expand_outputs_sharing_an_axis_must_return_the_same_children() -> None:
    plan = crop_plan(mismatch=True)

    with pytest.raises(StepExecutionError, match="share axis 'crop/crops'"):
        plan.create_session().run({"values": [3]})


def test_cast_group_is_preserved_under_its_own_cast_axis() -> None:
    plan = (
        PlanBuilder()
        .input("parents", BATCH)
        .input("label", SCALAR)
        .step(
            ContinueIf,
            "gate",
            at=BATCH,
            value="$inputs.parents",
            threshold=1,
            next_steps=["$steps.relabel"],
        )
        .step(
            Relabel,
            "relabel",
            at=BATCH,
            gates=(("gate", "$steps.relabel"),),
            values="$inputs.label",
        )
        .output("labels", "$steps.relabel.labels")
        .build()
    )

    result = plan.create_session().run({"parents": [1, 2], "label": "x"})

    assert result.outputs.layout["labels"].axis_ids == ("N", "relabel/values/cast")
    assert result.filtered_paths["labels"] == ((0,),)
    assert result.outputs.data["labels"][0].indices == ((1, 0),)
    assert result.rows() == [{"labels": []}, {"labels": ["#x"]}]


def test_layout_without_a_domain_source_is_rejected() -> None:
    plan = (
        PlanBuilder()
        .input("label", SCALAR)
        .step(Relabel, "relabel", at=BATCH, values="$inputs.label")
        .build()
    )

    with pytest.raises(ContractError, match="no varying binding or gate"):
        plan.create_session().run({"label": "x"})


def test_parent_value_is_broadcast_to_its_children() -> None:
    plan = (
        PlanBuilder()
        .input("values", BATCH)
        .step(Expand, "expand", at=BATCH, value="$inputs.values", count=2)
        .step(
            Scale,
            "scale",
            at=nested("expand/items"),
            value="$steps.expand.children",
            factor="$inputs.values",
        )
        .output("scaled", "$steps.scale.scaled")
        .build()
    )
    session = plan.create_session()

    rows = session.run({"values": [2, 3]}).rows()

    assert [call["factor"] for call in calls(session, "scale")] == [2, 2, 3, 3]
    assert rows == [{"scaled": [4, 6]}, {"scaled": [9, 12]}]


def test_block_parameters_named_like_observer_arguments_do_not_collide() -> None:
    from roboflow_workflows.execution_engine.v2.plan import ExecutionObserver

    from .blocks import Probe

    seen = []

    class Watcher(ExecutionObserver):
        def on_invocation(self, *, step, index, arguments, result):
            seen.append((step, index, dict(arguments)))

    plan = (
        PlanBuilder()
        .input("value", SCALAR)
        .step(Probe, "probe", step="$inputs.value", index=3, arguments="text")
        .output("result", "$steps.probe.result")
        .build()
    )

    rows = plan.create_session(observer=Watcher()).run({"value": "v"}).rows()

    assert rows == [{"result": ("v", 3, "text")}]
    assert seen == [(("probe",), (), {"step": "v", "index": 3, "arguments": "text"})]


def test_batch_views_expose_source_metadata_and_invocation_layout() -> None:
    plan = (
        PlanBuilder()
        .input("values", nested("c"))
        .step(EchoMany, "many", at=nested("c"), values="$inputs.values")
        .build()
    )
    session = plan.create_session()

    session.run({"values": [[1], [2, 3]]})

    (call,) = calls(session, "many")
    assert isinstance(call["values"], Batch)
    assert call["values"].layout.axis_ids == ("N", "c")
    assert call["values"].indices == ((0, 0), (1, 0), (1, 1))


def test_batch_delivering_control_and_consumer_are_one_call_each_across_parents() -> (
    None
):
    plan = (
        PlanBuilder()
        .input("values", nested("c"))
        .step(
            EvenGateMany,
            "gate",
            at=nested("c"),
            values="$inputs.values",
            next_steps=["$steps.echo"],
        )
        .step(
            EchoMany,
            "echo",
            at=nested("c"),
            gates=(("gate", "$steps.echo"),),
            values="$inputs.values",
        )
        .output("echo", "$steps.echo.value")
        .build()
    )
    session = plan.create_session()

    rows = session.run({"values": [[20, 21], [10]]}).rows()

    (gate_call,) = calls(session, "gate")
    (echo_call,) = calls(session, "echo")
    assert gate_call["values"].indices == ((0, 0), (0, 1), (1, 0))
    assert echo_call["values"].indices == ((0, 0), (1, 0))
    assert rows == [{"echo": [20, None]}, {"echo": [10]}]


def test_futures_in_a_cross_parent_batch_call_are_resolved_per_invocation() -> None:
    plan = (
        PlanBuilder()
        .input("values", nested("c"))
        .step(DeferredMany, "deferred", at=nested("c"), values="$inputs.values")
        .output("value", "$steps.deferred.value")
        .build()
    )

    rows = plan.create_session().run({"values": [[1, 2], [3]]}).rows()

    assert rows == [{"value": [10, 20]}, {"value": [30]}]


def test_future_resolution_keeps_a_flat_view_and_its_full_indices() -> None:
    future = Future()
    future.set_result("done")
    view = Batch(["kept", future], indices=[(0, 1), (2, 0)])

    resolved = resolve_futures(view)

    assert resolved.indices == ((0, 1), (2, 0))
    assert list(resolved) == ["kept", "done"]
    assert resolve_futures(resolved) is resolved
