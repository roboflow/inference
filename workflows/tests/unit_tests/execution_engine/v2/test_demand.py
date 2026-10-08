"""Demand: requested outputs, ``prunable`` steps and per-call ``wants``.

The plan keeps every step a requested output, a recorded group, an operator
or a non-prunable block needs, drops the rest, and tells each call which of
its outputs somebody reads. Unrequested results are absent keys, never
``None``; a filtered payload stays ``None`` and a genuine empty value stays
what it is.
"""

import threading
from dataclasses import replace
from typing import Any, Dict, List

import pytest
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.compilation.demand import (
    NOT_PRUNABLE,
    apply_demand,
    compute_demand,
)
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Output,
    Ref,
    spec_of,
)
from roboflow_workflows.execution_engine.v2.errors import (
    ContractError,
    DeclarationError,
    DemandError,
    EventEmissionError,
    StepExecutionError,
)
from roboflow_workflows.execution_engine.v2.kinds import FLOAT_KIND
from roboflow_workflows.execution_engine.v2.pipelining import PipelineOptions, passive
from roboflow_workflows.execution_engine.v2.plan import CompileOptions, StepPort
from roboflow_workflows.execution_engine.v2.reactions.testing import block_call
from roboflow_workflows.execution_engine.v2.sources import Emission

from tests.unit_tests.execution_engine.v2.compilation.fixture_blocks import (
    CATALOGUE as FIXTURE_CATALOGUE,
)
from tests.unit_tests.execution_engine.v2.compilation.fixture_blocks import (
    workflow as fixture_workflow,
)
from tests.unit_tests.execution_engine.v2.compilation.test_dynamic import (
    dynamic_block,
)
from tests.unit_tests.execution_engine.v2.m7_demand.blocks import (
    BLOCKS,
    CATALOGUE,
    HANDLER,
    Counter,
    Curious,
    Double,
    Emitter,
    Forgetful,
    Gate,
    Items,
    Join,
    Painter,
    PhasedPainter,
    Sink,
    Stubborn,
    definition,
    nested,
    step,
)
from tests.unit_tests.execution_engine.v2.test_active_runtime import (
    WAIT,
    Collector,
    Log,
    Scripted,
    active,
    emit,
    group,
    source,
)

ACTIVE_CATALOGUE = Catalogue(BLOCKS, sources=[Scripted], namespace="test")


def compiled(steps: List[dict], outputs: Dict[str, str], *, requested=None, **sections):
    options = CompileOptions(requested_outputs=requested)
    plan = compile_workflow(
        definition(steps, outputs, **sections), catalogue=CATALOGUE, options=options
    )

    return plan


def paths(plan) -> List[tuple]:
    return [step.path for step in plan.steps]


class ActiveHarness:
    """A compiled active plan over the scripted feed source, and one run."""

    def __init__(self, definition_: dict, feeds: Dict[str, list], **options: Any):
        self.log = Log()
        self.started = threading.Event()
        self.plan = compile_workflow(
            definition_, catalogue=ACTIVE_CATALOGUE, options=CompileOptions(**options)
        )
        self.session = self.plan.create_session(
            resources={"feeds": feeds, "log": self.log, "started": self.started}
        )

    def start(self, handlers: Dict[str, Any], **options: Any):
        self.started.clear()
        try:
            run = self.session.start(handlers=handlers, **options)
        finally:
            self.started.set()

        return run

    def instance(self, *path: str):
        return self.session.instances[tuple(path)]


# Requests and typo detection ----------------------------------------------


def test_unknown_requested_output_is_a_compile_error_listing_the_declared_names():
    steps = [step(Double, "double", value="$inputs.value")]

    with pytest.raises(DemandError, match="does not declare") as raised:
        compiled(steps, {"doubled": "$steps.double.doubled"}, requested=("dubled",))

    assert "'dubled'" in str(raised.value) and "['doubled']" in str(raised.value)


def test_unknown_group_or_field_request_is_a_compile_error():
    definition_ = active(
        [source("a")],
        [step(Double, "double", value="$sources.a.value")],
        [group("frames", "$sources.a.value", doubled="$steps.double.doubled")],
    )

    for requested in (("frame",), ("frames.doubld",), ("doubled",)):
        with pytest.raises(DemandError, match="does not declare") as raised:
            compile_workflow(
                definition_,
                catalogue=ACTIVE_CATALOGUE,
                options=CompileOptions(requested_outputs=requested),
            )
        assert "frames.doubled" in str(raised.value)


def test_compile_options_reject_malformed_requests():
    with pytest.raises(ContractError, match="collection of output names"):
        CompileOptions(requested_outputs="doubled")
    with pytest.raises(ContractError, match="repeats"):
        CompileOptions(requested_outputs=("a", "a"))


# Pruning and retention ------------------------------------------------------


def test_default_request_keeps_every_output_and_only_prunes_dead_prunable_steps():
    steps = [
        step(Double, "double", value="$inputs.value"),
        step(Double, "dead", value="$steps.double.doubled"),
        step(Counter, "counter", value="$steps.double.doubled"),
        step(Painter, "painter", value="$steps.double.doubled"),
    ]
    outputs = {"doubled": "$steps.double.doubled", "count": "$steps.painter.count"}

    plan = compiled(steps, outputs)
    result = plan.create_session().run({"value": 2.0})

    assert paths(plan) == [("double",), ("counter",), ("painter",)]
    assert plan.demand.requested is None
    assert dict(plan.demand.pruned)[("dead",)].block_type == Double.type
    assert plan.demand.retained[("counter",)] == NOT_PRUNABLE
    assert plan.demand.wanted[("painter",)] == frozenset({"count"})
    assert result.rows() == [{"doubled": 4.0, "count": 1}]
    assert set(result.statuses) == {"doubled", "count"}


def test_requested_subset_prunes_a_prunable_chain_and_omits_its_keys():
    steps = [
        step(Double, "double", value="$inputs.value"),
        step(Double, "again", value="$steps.double.doubled"),
        step(Painter, "painter", value="$steps.again.doubled"),
    ]
    outputs = {
        "doubled": "$steps.double.doubled",
        "overlay": "$steps.painter.overlay",
        "count": "$steps.painter.count",
    }

    plan = compiled(steps, outputs, requested=("doubled",))
    session = plan.create_session()
    result = session.run({"value": 1.5})

    assert paths(plan) == [("double",)]
    assert [item.describe() for item in plan.demand.outputs] == [
        {"status": "requested"},
        {"status": "omitted"},
        {"status": "omitted"},
    ]
    assert plan.demand.describe()["pruned"] == {
        "$steps.again": {
            "type": Double.type,
            "reason": plan.demand.pruned[("again",)].reason,
        },
        "$steps.painter": {
            "type": Painter.type,
            "reason": plan.demand.pruned[("painter",)].reason,
        },
    }
    assert result.rows() == [{"doubled": 3.0}]
    assert set(result.statuses) == {"doubled"} and set(result.selections) == {"doubled"}
    assert set(session.instances) == {("double",)}


def test_non_prunable_steps_and_their_upstream_stay_without_a_requested_output():
    steps = [
        step(Double, "double", value="$inputs.value"),
        step(Double, "feed", value="$steps.double.doubled"),
        step(Counter, "counter", value="$steps.feed.doubled"),
        step(Sink, "sink", value="$steps.feed.doubled"),
        step(Double, "dead", value="$steps.feed.doubled"),
    ]
    outputs = {"doubled": "$steps.double.doubled", "count": "$steps.counter.count"}

    plan = compiled(steps, outputs, requested=())
    session = plan.create_session()
    for value in (1.0, 2.0, 3.0):
        result = session.run({"value": value})

    assert paths(plan) == [("double",), ("feed",), ("counter",), ("sink",)]
    assert plan.demand.retained[("feed",)] in (
        "dependency of $steps.counter",
        "dependency of $steps.sink",
    )
    assert plan.demand.wanted[("double",)] == frozenset({"doubled"})
    assert plan.demand.wanted[("counter",)] == frozenset()
    assert result.rows() == [{}] and result.statuses == {}
    assert len(session.instances[("counter",)].calls) == 3
    assert [call["value"] for call in session.instances[("sink",)].calls] == [
        4.0,
        8.0,
        12.0,
    ]


def test_prunable_controller_whose_targets_are_all_pruned_goes_too():
    steps = [
        step(Double, "double", value="$inputs.value"),
        step(
            Gate, "gate", value="$steps.double.doubled", next_steps=["$steps.painter"]
        ),
        step(Painter, "painter", value="$steps.double.doubled"),
    ]
    outputs = {"doubled": "$steps.double.doubled", "count": "$steps.painter.count"}

    pruned = compiled(steps, outputs, requested=("doubled",))
    kept = compiled(steps, outputs, requested=("count",))
    rows = kept.create_session().run({"value": 1.0}).rows()

    assert paths(pruned) == [("double",)]
    assert paths(kept) == [("double",), ("gate",), ("painter",)]
    assert kept.demand.retained[("gate",)] == "dependency of $steps.painter"
    assert rows == [{"count": 1}]


def test_controller_keeps_its_target_keys_when_some_targets_are_pruned():
    steps = [
        step(Double, "double", value="$inputs.value"),
        step(
            Gate,
            "gate",
            value="$steps.double.doubled",
            next_steps=["$steps.painter", "$steps.dead"],
        ),
        step(Painter, "painter", value="$steps.double.doubled"),
        step(Double, "dead", value="$steps.double.doubled"),
    ]
    outputs = {"count": "$steps.painter.count", "dead": "$steps.dead.doubled"}

    plan = compiled(steps, outputs, requested=("count",))
    rows = plan.create_session().run({"value": 1.0}).rows()

    assert plan.step(("gate",)).control_targets == {
        "$steps.painter": (("painter",),),
        "$steps.dead": (),
    }
    assert rows == [{"count": 1}]


def test_unknown_dynamic_blocks_are_never_pruned():
    definition_ = fixture_workflow(
        [
            {"type": "CountingOffset", "name": "offset", "value": "$inputs.values"},
            {"type": "test/scale@v1", "name": "scale", "value": "$inputs.values"},
        ],
        {"total": "$steps.offset.total", "scaled": "$steps.scale.scaled"},
        dynamic_blocks=[dynamic_block("CountingOffset", outputs=["total", "calls"])],
    )

    plan = compile_workflow(
        definition_,
        catalogue=FIXTURE_CATALOGUE,
        options=CompileOptions(requested_outputs=("scaled",)),
    )

    assert paths(plan) == [("offset",), ("scale",)]
    assert plan.demand.retained[("offset",)] == NOT_PRUNABLE
    assert not spec_of(plan.step(("offset",)).spec.block_class).prunable


# wants() ------------------------------------------------------------------


def test_wants_answers_per_plan_and_lets_the_call_leave_the_output_out():
    steps = [step(Painter, "painter", value="$inputs.value")]
    outputs = {"overlay": "$steps.painter.overlay", "count": "$steps.painter.count"}

    with_overlay = compiled(steps, outputs, requested=("overlay", "count"))
    without = compiled(steps, outputs, requested=("count",))
    painted = with_overlay.create_session()
    skipped = without.create_session()
    full = painted.run({"value": 2.0})
    partial = skipped.run({"value": 2.0})

    assert full.rows() == [{"overlay": "paint:2", "count": 1}]
    assert partial.rows() == [{"count": 1}]
    assert painted.instances[("painter",)].asked == [{"overlay": True}]
    assert skipped.instances[("painter",)].asked == [{"overlay": False}]
    omitted = [
        (event["step"], event["outputs"])
        for result in (full, partial)
        for event in result.trace
        if event["event"] == "outputs_omitted"
    ]
    assert omitted == [(["painter"], ["overlay"])]


def test_leaving_an_output_out_without_asking_is_an_error():
    steps = [step(Forgetful, "forgetful", value="$inputs.value")]
    plan = compiled(steps, {"count": "$steps.forgetful.count"})

    with pytest.raises(StepExecutionError, match="never asked about") as raised:
        plan.create_session().run({"value": 1.0})

    assert "Omitted: ['overlay']" in str(raised.value)


def test_leaving_a_wanted_output_out_is_an_error_even_after_asking():
    steps = [step(Stubborn, "stubborn", value="$inputs.value")]
    plan = compiled(
        steps,
        {"overlay": "$steps.stubborn.overlay", "count": "$steps.stubborn.count"},
    )

    with pytest.raises(StepExecutionError, match="wanted by a reader") as raised:
        plan.create_session().run({"value": 1.0})

    assert "asked wants() about it" in str(raised.value)


def test_asking_about_an_undeclared_output_fails_clearly():
    plan = compiled(
        [step(Curious, "curious", value="$inputs.value")],
        {"value": "$steps.curious.value"},
    )

    with pytest.raises(StepExecutionError, match="declares no such output"):
        plan.create_session().run({"value": 1.0})


def test_wants_outside_a_call_and_inside_block_call_helper():
    painter = Painter()
    with pytest.raises(EventEmissionError, match="outside an engine call"):
        painter.wants("overlay")

    with block_call(Painter, source_id="cam"):
        assert painter.wants("overlay") is True
        with pytest.raises(ContractError, match="declares no such output"):
            painter.wants("nope")


def test_hand_built_context_without_demand_wants_everything():
    from roboflow_workflows.execution_engine.v2.plan import (
        Binding,
        CompiledWorkflow,
        EntryLayout,
        InputPort,
        PlannedInput,
        PlannedOutput,
        PlannedStep,
        PlannedWorkflowOutput,
    )

    spec = spec_of(Painter)
    planned = PlannedStep(
        path=("painter",),
        spec=spec,
        namespace="test",
        params=spec.validate_params({"value": "$inputs.value"}),
        bindings=(
            Binding(
                field="value",
                position=(),
                selector="$inputs.value",
                source=InputPort("value"),
                source_layout=EntryLayout(),
                mode="constant",
            ),
        ),
        invocation_layout=EntryLayout(),
        outputs={
            name: PlannedOutput(
                name=name, kinds=output.kind_names, layout=EntryLayout()
            )
            for name, output in spec.outputs.items()
        },
    )
    plan = CompiledWorkflow(
        inputs={
            "value": PlannedInput(name="value", kinds=("float",), layout=EntryLayout())
        },
        steps=(planned,),
        outputs=(
            PlannedWorkflowOutput(
                name="count",
                selector="$steps.painter.count",
                source=StepPort(("painter",), "count"),
            ),
        ),
        catalogue=CATALOGUE,
    )
    session = plan.create_session()

    rows = session.run({"value": 3.0}).rows()

    assert plan.demand is None
    assert plan.wanted_outputs(("painter",)) == frozenset({"overlay", "count"})
    assert rows == [{"count": 1}]
    assert session.instances[("painter",)].asked == [{"overlay": True}]


# Phases, pipelines, tickets --------------------------------------------------


def test_wants_is_consistent_across_overlapping_phases_of_a_pipelined_run():
    steps = [
        step(PhasedPainter, "phased", value="$inputs.value"),
        step(Double, "after", value="$steps.phased.value"),
    ]
    outputs = {"doubled": "$steps.after.doubled", "extra": "$steps.phased.extra"}
    options = PipelineOptions(max_in_flight=3)

    plans = {
        requested: compile_workflow(
            definition(steps, outputs),
            catalogue=CATALOGUE,
            options=CompileOptions(
                block_execution="phases", requested_outputs=requested
            ),
        )
        for requested in (("doubled",), ("doubled", "extra"))
    }
    results = {}
    for requested, plan in plans.items():
        session = plan.create_session()
        with passive.open_pipeline(session, options=options) as pipeline:
            futures = [pipeline.submit({"value": float(index)}) for index in range(6)]
            results[requested] = [
                future.result(timeout=WAIT).rows()[0] for future in futures
            ]
        asked = session.instances[("phased",)].asked
        expected = "extra" in requested
        assert len(asked) == 12 and all(item["extra"] is expected for item in asked)

    assert results[("doubled",)] == [{"doubled": 20.0 * index} for index in range(6)]
    assert results[("doubled", "extra")] == [
        {"doubled": 20.0 * index, "extra": f"extra:{10 * index:g}"}
        for index in range(6)
    ]


def test_pipelined_active_run_with_pruned_and_gate_denied_steps_completes():
    definition_ = active(
        [source("a")],
        [
            step(Double, "double", value="$sources.a.value"),
            step(Gate, "gate", value="$sources.a.value", next_steps=["$steps.painter"]),
            step(Painter, "painter", value="$steps.double.doubled"),
            step(Double, "dead", value="$steps.double.doubled"),
            step(Counter, "counter", value="$steps.double.doubled"),
        ],
        [
            group(
                "frames",
                "$sources.a.value",
                doubled="$steps.double.doubled",
                count="$steps.painter.count",
                dead="$steps.dead.doubled",
            ),
            group("extra", "$sources.a.value", overlay="$steps.painter.overlay"),
        ],
    )
    feeds = {"a": [emit(value=float(value)) for value in (1, 0, 2, 0, 3)]}
    harness = ActiveHarness(
        definition_, feeds, requested_outputs=("frames.doubled", "frames.count")
    )
    collected = Collector()

    run = harness.start(
        collected.handlers("frames"), pipeline=PipelineOptions(max_in_flight=3)
    )

    assert run.wait(WAIT)
    assert paths(harness.plan) == [("double",), ("gate",), ("painter",), ("counter",)]
    assert [item.name for item in harness.plan.output_groups] == ["frames"]
    assert collected.rows("frames") == [
        {"doubled": 2.0, "count": 1},
        {"doubled": 0.0, "count": None},
        {"doubled": 4.0, "count": 1},
        {"doubled": 0.0, "count": None},
        {"doubled": 6.0, "count": 1},
    ]
    assert collected.sequences("frames") == [0, 1, 2, 3, 4]
    assert len(harness.instance("counter").calls) == 5
    assert harness.instance("painter").asked == [{"overlay": False}] * 3
    assert run.counters["a"].ended


def test_registering_a_handler_for_an_omitted_group_is_an_error():
    definition_ = active(
        [source("a")],
        [step(Double, "double", value="$sources.a.value")],
        [
            group("frames", "$sources.a.value", doubled="$steps.double.doubled"),
            group("extra", "$sources.a.value", doubled="$steps.double.doubled"),
        ],
    )
    harness = ActiveHarness(definition_, {"a": []}, requested_outputs=("frames",))

    with pytest.raises(ContractError, match="unknown output groups \\['extra'\\]"):
        harness.start({"extra": lambda result: None})


# Nested demand, gates and joins ---------------------------------------------


CHILD = definition(
    [
        step(Double, "double", value="$inputs.value"),
        step(Painter, "painter", value="$steps.double.doubled"),
    ],
    {"doubled": "$steps.double.doubled", "overlay": "$steps.painter.overlay"},
)


def test_two_instances_of_one_child_are_demanded_independently():
    steps = [
        nested("left", CHILD, value="$inputs.value"),
        nested("right", CHILD, value="$inputs.value"),
    ]
    outputs = {
        "left_doubled": "$steps.left.doubled",
        "left_overlay": "$steps.left.overlay",
        "right_doubled": "$steps.right.doubled",
        "right_overlay": "$steps.right.overlay",
    }

    plan = compiled(steps, outputs, requested=("left_overlay", "right_doubled"))
    rows = plan.create_session().run({"value": 1.0}).rows()

    assert paths(plan) == [("left", "double"), ("left", "painter"), ("right", "double")]
    assert plan.demand.wanted[("left", "painter")] == frozenset({"overlay"})
    assert plan.demand.wanted[("right", "double")] == frozenset({"doubled"})
    assert rows == [{"left_overlay": "paint:2", "right_doubled": 2.0}]


GATED_CHILD = definition(
    [
        step(Gate, "gate", value="$inputs.keep", next_steps=["$steps.kept"]),
        step(Double, "kept", value="$inputs.value"),
        step(Double, "other", value="$inputs.value"),
    ],
    {"kept": "$steps.kept.doubled", "other": "$steps.other.doubled"},
    inputs=[
        {"type": "WorkflowParameter", "name": "value", "kind": ["float"]},
        {"type": "WorkflowParameter", "name": "keep"},
    ],
)


def test_child_gate_is_retained_for_the_parent_join_which_recovers_the_survivor():
    steps = [
        nested("child", GATED_CHILD, value="$inputs.value", keep="$inputs.keep"),
        step(Join, "join", left="$steps.child.kept", right="$steps.child.other"),
        step(Painter, "painter", value="$steps.child.other"),
    ]
    outputs = {"joined": "$steps.join.value", "overlay": "$steps.painter.overlay"}
    plan = compile_workflow(
        definition(
            steps,
            outputs,
            inputs=[
                {"type": "WorkflowParameter", "name": "value", "kind": ["float"]},
                {"type": "WorkflowParameter", "name": "keep"},
            ],
        ),
        catalogue=CATALOGUE,
        options=CompileOptions(requested_outputs=("joined",)),
    )
    session = plan.create_session()

    admitted = session.run({"value": 1.0, "keep": True}).rows()
    denied = session.run({"value": 1.0, "keep": False}).rows()

    assert paths(plan) == [
        ("child", "gate"),
        ("child", "kept"),
        ("child", "other"),
        ("join",),
    ]
    assert plan.demand.retained[("child", "gate")] == "dependency of $steps.child/kept"
    assert admitted == [{"joined": 2.0}]
    assert denied == [{"joined": 2.0}]
    assert [call["left"] for call in session.instances[("join",)].calls] == [2.0, None]
    assert len(session.instances[("child", "kept")].calls) == 1


# Effects, events and recording roots ---------------------------------------


def test_events_managed_state_mutation_or_no_outputs_forbid_prunable():
    with pytest.raises(DeclarationError, match="declares events"):

        class PrunableEmitter(Block):
            type = "test/m7_bad_emitter@v1"
            prunable = True
            outputs = {"value": Output(FLOAT_KIND)}
            events = Emitter.events

            class Params(BlockParams):
                value: Ref(FLOAT_KIND)

            def run(self, *, value):
                return {"value": value}

    with pytest.raises(DeclarationError, match="no outputs"):

        class PrunableSink(Block):
            type = "test/m7_bad_sink@v1"
            prunable = True

            class Params(BlockParams):
                value: Ref()

            def run(self, *, value):
                return None

    with pytest.raises(DeclarationError, match="managed_state"):

        class PrunableStateful(Block):
            type = "test/m7_bad_stateful@v1"
            prunable = True
            outputs = {"value": Output(FLOAT_KIND)}

            class Params(BlockParams):
                value: Ref(FLOAT_KIND)

            def __init__(self, *, managed_state):
                self.state = managed_state

            def run(self, *, value):
                return {"value": value}

    with pytest.raises(DeclarationError, match="mutates"):

        class PrunableMutator(Block):
            type = "test/m7_bad_mutator@v1"
            prunable = True
            mutates = ("value",)
            outputs = {"value": Output()}

            class Params(BlockParams):
                value: Ref()

            def run(self, *, value):
                return {"value": value}

    with pytest.raises(DeclarationError, match="prunable must be True or False"):

        class Maybe(Block):
            type = "test/m7_bad_maybe@v1"
            prunable = "yes"
            outputs = {"value": Output()}

            class Params(BlockParams):
                value: Ref()

            def run(self, *, value):
                return {"value": value}


def test_an_emitter_in_an_unrequested_branch_still_runs_and_its_handler_fires():
    steps = [
        step(Double, "double", value="$inputs.value"),
        step(Emitter, "emitter", value="$steps.double.doubled"),
        step(Double, "tail", value="$steps.emitter.value"),
    ]
    outputs = {"value": "$inputs.value", "tail": "$steps.tail.doubled"}
    handlers = [
        {
            "name": "notify",
            "on": "$steps.emitter.events.seen",
            "execution": {"mode": "sync"},
            "bindings": {"value": "$event.value"},
            "workflow": HANDLER,
        }
    ]

    plan = compiled(steps, outputs, requested=("value",), handlers=handlers)
    session = plan.create_session()
    rows = session.run({"value": 2.0}).rows()

    assert paths(plan) == [("double",), ("emitter",)]
    assert plan.demand.retained[("emitter",)] == NOT_PRUNABLE
    assert plan.demand.wanted[("emitter",)] == frozenset()
    assert rows == [{"value": 2.0}]
    assert session.instances[("emitter",)].calls == [{"value": 4.0}]
    (handler,) = session.handler_sessions.values()
    assert handler.instances[("react",)].calls == [{"value": 4.0}]


def test_recorded_groups_are_demanded_whole_even_when_unrequested(tmp_path):
    definition_ = active(
        [source("a")],
        [
            step(Double, "double", value="$sources.a.value"),
            step(Painter, "painter", value="$steps.double.doubled"),
        ],
        [
            group("frames", "$sources.a.value", doubled="$steps.double.doubled"),
            group(
                "archive",
                "$sources.a.value",
                overlay="$steps.painter.overlay",
                count="$steps.painter.count",
            ),
        ],
    )
    definition_["recording"] = {
        "type": "file",
        "directory": str(tmp_path),
        "groups": ["archive"],
    }

    harness = ActiveHarness(
        definition_, {"a": [emit(value=1.0)]}, requested_outputs=("frames",)
    )
    statuses = {item.name: item.describe() for item in harness.plan.demand.outputs}
    collected = Collector()
    run = harness.start(collected.handlers("frames", "archive"))

    assert run.wait(WAIT)
    assert statuses == {
        "frames": {"status": "requested", "fields": ["doubled"]},
        "archive": {"status": "recorded", "fields": ["overlay", "count"]},
    }
    assert harness.plan.demand.wanted[("painter",)] == frozenset({"overlay", "count"})
    assert harness.plan.recording.groups == ("archive",)
    assert collected.rows("archive") == [{"overlay": "paint:2", "count": 1}]
    assert collected.rows("frames") == [{"doubled": 2.0}]

    with pytest.raises(DemandError, match="records whole"):
        ActiveHarness(definition_, {"a": []}, requested_outputs=("archive.count",))


def test_operator_inputs_are_demand_roots():
    from roboflow_workflows.execution_engine.v2.operators.alignment import Align

    catalogue = Catalogue(
        BLOCKS, sources=[Scripted], operators=[Align], namespace="test"
    )
    definition_ = active(
        [source("a"), source("b")],
        [
            step(Double, "double_a", value="$sources.a.value"),
            step(Double, "double_b", value="$sources.b.value"),
            step(Double, "dead", value="$sources.a.value"),
        ],
        [
            group(
                "pairs",
                "$operators.pair.a",
                a="$operators.pair.a",
                b="$operators.pair.b",
            )
        ],
    )
    definition_["operators"] = [
        {
            "type": Align.type,
            "name": "pair",
            "inputs": {"a": "$steps.double_a.doubled", "b": "$steps.double_b.doubled"},
            "clock": "media",
        }
    ]

    plan = compile_workflow(definition_, catalogue=catalogue)

    assert paths(plan) == [("double_a",), ("double_b",)]
    assert plan.demand.retained[("double_a",)] == "$operators.pair input 'a'"


# Omitted versus filtered versus empty ---------------------------------------


def test_omitted_filtered_empty_and_none_results_are_distinguishable():
    steps = [
        step(Gate, "gate", value="$inputs.keep", next_steps=["$steps.items"]),
        step(Items, "items", value="$inputs.value"),
        step(Double, "double", value="$inputs.value"),
    ]
    outputs = {"items": "$steps.items.items", "doubled": "$steps.double.doubled"}
    inputs = [
        {"type": "WorkflowParameter", "name": "value", "kind": ["float"]},
        {"type": "WorkflowParameter", "name": "keep"},
    ]
    plan = compile_workflow(
        definition(steps, outputs, inputs=inputs),
        catalogue=CATALOGUE,
        options=CompileOptions(requested_outputs=("items",)),
    )
    session = plan.create_session()

    empty = session.run({"value": -1.0, "keep": True})
    none = session.run({"value": 0.0, "keep": True})
    denied = session.run({"value": 2.0, "keep": False})
    present = session.run({"value": 2.0, "keep": True})

    # A genuine empty value and an explicit ``None`` payload are complete
    # results; a gate denial is filtered; an unrequested output has no key.
    assert empty.statuses == {"items": "complete"} and empty.rows() == [{"items": []}]
    assert none.statuses == {"items": "complete"} and none.rows() == [{"items": None}]
    assert denied.statuses == {"items": "filtered"} and denied.rows() == [
        {"items": None}
    ]
    assert present.rows() == [{"items": [2.0]}]
    assert paths(plan) == [("gate",), ("items",)]
    for result in (empty, none, denied, present):
        assert "doubled" not in result.statuses
        assert "doubled" not in result.rows()[0]
        assert "doubled" not in result.selections


def test_group_fields_requested_by_name_are_delivered_alone():
    definition_ = active(
        [source("a")],
        [
            step(Items, "items", value="$sources.a.value"),
            step(Painter, "painter", value="$sources.a.value"),
        ],
        [
            group(
                "frames",
                "$sources.a.value",
                items="$steps.items.items",
                overlay="$steps.painter.overlay",
                count="$steps.painter.count",
            )
        ],
    )
    feeds = {"a": [emit(value=-1.0), emit(value=0.0), emit(value=1.0)]}
    harness = ActiveHarness(
        definition_, feeds, requested_outputs=("frames.items", "frames.count")
    )
    collected = Collector()

    run = harness.start(collected.handlers("frames"))

    assert run.wait(WAIT)
    (frames,) = harness.plan.output_groups
    assert [field.name for field in frames.outputs] == ["items", "count"]
    assert collected.rows("frames") == [
        {"items": [], "count": 1},
        {"items": None, "count": 1},
        {"items": [1.0], "count": 1},
    ]
    assert [result.statuses for result in collected.results["frames"]] == [
        {"items": "complete", "count": "complete"}
    ] * 3
    assert harness.instance("painter").asked == [{"overlay": False}] * 3
    assert {item.name: item.status for item in harness.plan.demand.outputs} == {
        "frames": "requested",
        "frames.items": "requested",
        "frames.overlay": "omitted",
        "frames.count": "requested",
    }


# The seam for live controls -------------------------------------------------


def test_compute_demand_is_reusable_with_a_narrower_root_set():
    steps = [
        step(Double, "double", value="$inputs.value"),
        step(Painter, "painter", value="$steps.double.doubled"),
        step(Counter, "counter", value="$steps.double.doubled"),
    ]
    outputs = {"overlay": "$steps.painter.overlay", "count": "$steps.painter.count"}
    plan = compiled(steps, outputs)

    narrowed = compute_demand(
        plan,
        root_sources=[(StepPort(("painter",), "count"), "live request")],
        always=[(("counter",), NOT_PRUNABLE)],
    )

    assert dict(narrowed.retained) == {
        ("double",): "dependency of $steps.painter",
        ("painter",): "live request",
        ("counter",): NOT_PRUNABLE,
    }
    assert narrowed.wanted[("painter",)] == frozenset({"count"})
    with pytest.raises(ContractError, match="without a demand record"):
        apply_demand(plan, requested=None)


@pytest.mark.parametrize(
    "roots",
    [list, iter, lambda items: (item for item in items)],
    ids=["list", "iterator", "generator"],
)
def test_compute_demand_consumes_root_sources_once(roots):
    plan = compiled(
        [step(Painter, "painter", value="$inputs.value")],
        {"overlay": "$steps.painter.overlay", "count": "$steps.painter.count"},
    )
    items = [(StepPort(("painter",), "overlay"), "request")]

    demand = compute_demand(plan, root_sources=roots(items), always=iter([]))

    assert dict(demand.retained) == {("painter",): "request"}
    assert dict(demand.wanted) == {("painter",): frozenset({"overlay"})}


# Hand-built or altered demand records ---------------------------------------


def with_wanted(plan, path, names):
    wanted = dict(plan.demand.wanted)
    wanted[path] = frozenset(names)

    return replace(plan, demand=replace(plan.demand, wanted=wanted))


def test_a_demand_record_must_want_every_selected_workflow_output():
    plan = compiled(
        [step(Painter, "painter", value="$inputs.value")],
        {"overlay": "$steps.painter.overlay", "count": "$steps.painter.count"},
    )

    with pytest.raises(ContractError, match=r"does not want outputs \['overlay'\]"):
        with_wanted(plan, ("painter",), {"count"})


def test_a_demand_record_must_want_what_a_retained_step_binds():
    plan = compiled(
        [
            step(Double, "double", value="$inputs.value"),
            step(Painter, "painter", value="$steps.double.doubled"),
        ],
        {"count": "$steps.painter.count"},
        requested=("count",),
    )

    with pytest.raises(ContractError, match=r"does not want outputs \['doubled'\]"):
        with_wanted(plan, ("double",), ())


def test_a_demand_record_must_want_what_a_nested_boundary_reads():
    plan = compiled(
        [nested("left", CHILD, value="$inputs.value")],
        {"left_overlay": "$steps.left.overlay"},
        requested=("left_overlay",),
    )

    with pytest.raises(ContractError, match=r"does not want outputs \['overlay'\]"):
        with_wanted(plan, ("left", "painter"), ())


def test_a_demand_record_must_want_what_a_kept_group_field_reads():
    definition_ = active(
        [source("a")],
        [step(Painter, "painter", value="$sources.a.value")],
        [group("frames", "$sources.a.value", overlay="$steps.painter.overlay")],
    )
    plan = compile_workflow(definition_, catalogue=ACTIVE_CATALOGUE)

    with pytest.raises(ContractError, match=r"does not want outputs \['overlay'\]"):
        with_wanted(plan, ("painter",), ())


def test_a_demand_record_must_want_what_an_operator_reads():
    from roboflow_workflows.execution_engine.v2.operators.alignment import Align

    catalogue = Catalogue(
        BLOCKS, sources=[Scripted], operators=[Align], namespace="test"
    )
    definition_ = active(
        [source("a"), source("b")],
        [
            step(Double, "double_a", value="$sources.a.value"),
            step(Double, "double_b", value="$sources.b.value"),
        ],
        [group("pairs", "$operators.pair.a", a="$operators.pair.a")],
    )
    definition_["operators"] = [
        {
            "type": Align.type,
            "name": "pair",
            "inputs": {"a": "$steps.double_a.doubled", "b": "$steps.double_b.doubled"},
            "clock": "media",
        }
    ]
    plan = compile_workflow(definition_, catalogue=catalogue)

    with pytest.raises(ContractError, match=r"does not want outputs \['doubled'\]"):
        with_wanted(plan, ("double_b",), ())


def test_a_demand_record_may_want_more_than_the_plan_reads():
    plan = compiled(
        [step(Painter, "painter", value="$inputs.value")],
        {"count": "$steps.painter.count"},
        requested=("count",),
    )
    generous = with_wanted(plan, ("painter",), {"overlay", "count"})

    assert generous.create_session().run({"value": 1.0}).rows() == [{"count": 1}]


def test_a_demand_record_must_match_the_plan_request_and_steps():
    plan = compiled(
        [step(Painter, "painter", value="$inputs.value")],
        {"overlay": "$steps.painter.overlay", "count": "$steps.painter.count"},
        requested=("count",),
    )

    with pytest.raises(ContractError, match="the plan's options request"):
        replace(plan, demand=replace(plan.demand, requested=None))
    with pytest.raises(ContractError, match="retention reasons for steps"):
        replace(plan, demand=replace(plan.demand, retained={}))


# Handler groups ---------------------------------------------------------------


def handler_definition():
    definition_ = active(
        [source("a")],
        [
            step(Double, "double", value="$sources.a.value"),
            step(Emitter, "emitter", value="$steps.double.doubled"),
        ],
        [
            group("frames", "$sources.a.value", doubled="$steps.double.doubled"),
            group("alerts", "$handlers.notify.out", out="$handlers.notify.out"),
        ],
    )
    definition_["handlers"] = [
        {
            "name": "notify",
            "on": "$steps.emitter.events.seen",
            "execution": {"mode": "sync"},
            "bindings": {"value": "$event.value"},
            "workflow": HANDLER,
        }
    ]

    return definition_


@pytest.mark.parametrize("requested", [None, ("frames",), ("frames", "alerts")])
def test_handler_groups_are_recorded_as_delivered_by_their_handler(requested):
    plan = compile_workflow(
        handler_definition(),
        catalogue=ACTIVE_CATALOGUE,
        options=CompileOptions(requested_outputs=requested),
    )

    assert plan.describe()["demand"]["outputs"]["alerts"] == {
        "status": "handler",
        "fields": ["out"],
        "reason": "delivered by its handler; requests do not narrow it",
    }


def test_a_handler_group_field_request_lists_handler_groups_as_whole_only():
    with pytest.raises(DemandError, match="does not declare") as raised:
        compile_workflow(
            handler_definition(),
            catalogue=ACTIVE_CATALOGUE,
            options=CompileOptions(requested_outputs=("alerts.out",)),
        )

    message = str(raised.value)
    assert "'alerts'" in message and "handler group names (whole only" in message
