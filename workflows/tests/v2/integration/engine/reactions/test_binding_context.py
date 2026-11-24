"""Real binding-specific context, isolated by field, index and invocation."""

import threading
from dataclasses import replace
from typing import Any, List

import pytest
from pydantic import Field
from roboflow_workflows.v2.data.batch import (
    EntryMetadata,
    InputValue,
    SampleContext,
    TemporalContext,
    Timestamp,
)
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.engine import Catalogue, compile_workflow
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    Group,
    Output,
    Ref,
)
from roboflow_workflows.v2.engine.errors import ContractError
from roboflow_workflows.v2.engine.phases import phase
from roboflow_workflows.v2.engine.pipelining.options import PipelineOptions
from roboflow_workflows.v2.engine.plan import CompileOptions


def contexts(context, index, fields):
    result = {"common": (context.sample_at(index), context.temporal_at(index))}
    for field in fields:
        sample = context.sample_at(index, field=field)
        temporal = context.temporal_at(index, field=field)
        assert context.sample_at(index, field=field) is sample
        if sample is not None:
            assert context.source_id_at(index, field=field) == sample.source_id
        result[field] = (sample, temporal)
    for field in ("missing", "display_label", "$inputs.image"):
        with pytest.raises(ContractError, match="unknown Params field"):
            context.sample_at(index, field=field)
    with pytest.raises(ContractError, match="names no index"):
        context.temporal_at((999,), field=fields[0])
    return result


class Probe(Block):
    type = "test/binding_context_probe@v1"
    outputs = {"out": Output()}

    class Params(BlockParams):
        image: Ref()
        other: Ref()
        label: str = Field(default="hello", alias="display_label")

    @phase
    def inspect(self, *, image, other, label):
        context = self.execution_context
        return {
            "out": contexts(context, context.indices[0], ("image", "other", "label"))
        }

    def run(self, *, image, other, label):
        return self.inspect(image=image, other=other, label=label)


class Pass(Block):
    type = "test/binding_context_pass@v1"
    outputs = {"out": Output(source="value")}

    class Params(BlockParams):
        image: Any | Ref()
        value: Any | Ref()

    def run(self, *, image, value):
        return {"out": value}


class Lists(Probe):
    type = "test/binding_context_list@v1"

    class Params(BlockParams):
        image: List[Any | Ref()]
        other: Ref()
        label: str = Field(default="hello", alias="display_label")


class Grouped(Probe):
    type = "test/binding_context_group@v1"

    class Params(BlockParams):
        image: Group()
        other: Any | Ref()
        label: str = Field(default="hello", alias="display_label")


class Batched(Probe):
    type = "test/binding_context_batch@v1"

    class Params(BlockParams):
        image: Ref(batch="always")  # noqa: F821 (batch mode is a value)
        other: Ref(batch="always")  # noqa: F821 (batch mode is a value)
        label: str = Field(default="hello", alias="display_label")

    def __init__(self, *, barrier):
        self.barrier = barrier

    @phase
    def inspect(self, *, image, other, label):
        self.barrier.wait(timeout=5)
        context = self.execution_context
        return [
            {"out": contexts(context, index, ("image", "other", "label"))}
            for index in context.indices
        ]


CATALOGUE = Catalogue([Probe, Pass, Lists, Grouped, Batched], kinds=[IMAGE_KIND])


def definition(block=Probe, *, nested=False, batch=False, image=None, other=None):
    inputs = [
        {"type": "WorkflowBatchInput" if batch else "WorkflowParameter", "name": name}
        for name in ("image", "other")
    ]
    body = {
        "version": "2.0",
        "inputs": inputs,
        "steps": [
            {
                "type": Pass.type,
                "name": "pass",
                "image": "$inputs.image",
                "value": "$inputs.other",
            },
            {
                "type": block.type,
                "name": "probe",
                "image": image or "$inputs.image",
                "other": other or "$steps.pass.out",
            },
        ],
        "outputs": [
            {"type": "JsonField", "name": "out", "selector": "$steps.probe.out"}
        ],
    }
    if not nested:
        return body
    return {
        "version": "2.0",
        "inputs": inputs,
        "steps": [
            {
                "type": "roboflow_core/inner_workflow@v1",
                "name": "child",
                "workflow_definition": body,
                "parameter_bindings": {
                    name: f"$inputs.{name}" for name in ("image", "other")
                },
            }
        ],
        "outputs": [
            {"type": "JsonField", "name": "out", "selector": "$steps.child.out"}
        ],
    }


def carried(value, source, ticks=1):
    sample = SampleContext(source_id=source)
    temporal = TemporalContext(Timestamp(ticks, 1, "clock"))
    return InputValue(
        value, metadata=EntryMetadata(sample={(): sample}, temporal={(): temporal})
    ), (sample, temporal)


@pytest.mark.parametrize("mode", ["run", "phases"])
@pytest.mark.parametrize("nested", [False, True])
def test_image_context_survives_context_free_varying_binding(mode, nested):
    import torch
    from roboflow_workflows.v2.data.image import ImageData

    d = definition(nested=nested)
    d["inputs"][0]["type"] = "WorkflowImage"
    plan = compile_workflow(
        d, catalogue=CATALOGUE, options=CompileOptions(block_execution=mode)
    )
    session = plan.create_session()
    image, expected = carried(
        ImageData.from_tensor(torch.zeros((3, 2, 2), dtype=torch.uint8)), "camera"
    )
    [row] = session.run({"image": image, "other": 0}).rows()
    assert row["out"] == {
        "common": (None, None),
        "image": expected,
        "other": (None, None),
        "label": (None, None),
    }
    session.close()


@pytest.mark.parametrize("mode", ["run", "phases"])
@pytest.mark.parametrize("same", [False, True])
def test_list_bound_leaves_merge_only_that_field(mode, same):
    d = definition(
        Lists, image=["$inputs.image", "$inputs.other"], other="$inputs.image"
    )
    session = compile_workflow(
        d, catalogue=CATALOGUE, options=CompileOptions(block_execution=mode)
    ).create_session()
    image, expected = carried(1, "a")
    other, _ = carried(2, "a" if same else "b", ticks=1 if same else 2)
    [row] = session.run({"image": image, "other": other}).rows()
    assert row["out"]["image"] == (expected if same else (None, None))
    assert row["out"]["other"] == expected
    session.close()


@pytest.mark.parametrize("mode", ["run", "phases"])
@pytest.mark.parametrize("sources", [("a", "a"), ("a", "b"), ("a", None)])
def test_group_merges_members_without_hiding_explicit_none(mode, sources):
    d = definition(Grouped, batch=True, other=7)
    session = compile_workflow(
        d, catalogue=CATALOGUE, options=CompileOptions(block_execution=mode)
    ).create_session()
    temporal = TemporalContext(Timestamp(1, 1, "clock"))
    metadata = EntryMetadata(
        sample={
            (i,): SampleContext(source_id=s) if s else None
            for i, s in enumerate(sources)
        },
        temporal={(i,): temporal for i in range(2)},
    )
    [row] = session.run(
        {"image": InputValue([1, 2], metadata=metadata), "other": [0, 0]}
    ).rows()
    expected = SampleContext(source_id="a") if sources == ("a", "a") else None
    assert row["out"]["image"] == (expected, temporal)
    session.close()


@pytest.mark.parametrize("mode", ["run", "phases"])
def test_empty_group_keeps_parent_sample_but_does_not_invent_a_time(mode):
    session = compile_workflow(
        definition(Grouped, batch=True, other=7),
        catalogue=CATALOGUE,
        options=CompileOptions(block_execution=mode),
    ).create_session()
    image, expected = carried([], "parent")
    [row] = session.run({"image": image, "other": []}).rows()
    assert row["out"]["image"] == (expected[0], None)
    session.close()


@pytest.mark.parametrize("mode", ["run", "phases"])
def test_batch_and_concurrent_parent_calls_do_not_share_lookup_caches(mode):
    d = definition(Batched, batch=True)
    # One phase is ordered across submissions. Two independent steps must
    # overlap on the parent's W2 team instead of relying on scheduler timing.
    d["steps"].append(dict(d["steps"][-1], name="probe2"))
    d["outputs"].append(
        {"type": "JsonField", "name": "out2", "selector": "$steps.probe2.out"}
    )
    session = compile_workflow(
        d, catalogue=CATALOGUE, options=CompileOptions(block_execution=mode)
    ).create_session(resources={"barrier": threading.Barrier(2)})
    with session.pipeline(
        options=PipelineOptions(max_in_flight=2, max_workflow_parallelism=2)
    ) as pipeline:
        futures = []
        for run in range(4):
            metadata = EntryMetadata(
                sample={(i,): SampleContext(source_id=f"{run}:{i}") for i in range(2)}
            )
            futures.append(
                pipeline.submit(
                    {"image": InputValue([1, 2], metadata=metadata), "other": [0, 0]}
                )
            )
        for run, future in enumerate(futures):
            rows = future.result(timeout=5).rows()
            assert [row["out"]["image"][0].source_id for row in rows] == [
                f"{run}:0",
                f"{run}:1",
            ]
            assert all(row["out"]["common"] == (None, None) for row in rows)
            assert all(row["out2"] == row["out"] for row in rows)
    session.close()


class Origin(Probe):
    type = "test/binding_context_origin@v1"

    class Params(BlockParams):
        image: Any | Ref()
        other: Any | Ref()
        label: str = Field(default="hello", alias="display_label")

    def __init__(self, *, probe):
        self.probe = probe

    @phase
    def inspect(self, *, image, other, label):
        context = self.execution_context
        result = contexts(context, context.indices[0], ("image", "other", "label"))
        self.probe.seen.append((result, context.cause))
        return {"out": result}


@pytest.mark.parametrize("mode", ["run", "phases"])
def test_known_constant_fields_use_handler_origin_when_they_carry_no_context(mode):
    from roboflow_workflows.v2.engine.reactions import ReactionPlan

    from tests.v2.integration.engine.reactions.test_runtime import (
        CATALOGUE as RUNTIME_CATALOGUE,
    )
    from tests.v2.integration.engine.reactions.test_runtime import WAIT
    from tests.v2.integration.engine.reactions.test_runtime import Probe as RuntimeProbe
    from tests.v2.integration.engine.reactions.test_runtime import (
        active_definition,
        handler,
        start,
    )

    d = definition(Origin)
    d["steps"] = [{"type": Origin.type, "name": "probe", "image": 1, "other": 2}]
    catalogue = Catalogue.merge(RUNTIME_CATALOGUE, Catalogue([Origin]))
    plan = compile_workflow(
        d, catalogue=catalogue, options=CompileOptions(block_execution=mode)
    )
    target = replace(
        handler(), plan=plan, bindings={"image": "value", "other": "value"}
    )
    probe = RuntimeProbe()
    parent = compile_workflow(active_definition("camera"), catalogue=catalogue)
    parent = replace(parent, reactions=ReactionPlan(handlers=(target,)))
    run = start(parent, probe, {"camera": [1]})
    assert run.wait(WAIT)
    [(result, cause)] = probe.seen
    for field in ("common", "image", "other", "label"):
        assert result[field] == (cause.sample, cause.temporal)
