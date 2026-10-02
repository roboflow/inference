"""Compilation of operators: records, one dependency graph, domains and layouts.

Nothing here opens a source, constructs an operator or runs a block. The graph
under test, with its pulse domains::

    $sources.cam ──┐                      ┌─▶ scale (domain pair) ─▶ [$operators.clip]
                   ├─▶ [$operators.pair] ─┤                            (domain pair)
    $sources.probe ┘   (domains cam+probe)└─▶ group "pairs"            └─▶ steps (domain clip)
"""

import json
from dataclasses import replace

import pytest
from roboflow_workflows.execution_engine.v2 import (
    Catalogue,
    Source,
    SourceOutput,
    compile_workflow,
)
from roboflow_workflows.execution_engine.v2.compilation import CompileOptions
from roboflow_workflows.execution_engine.v2.data import Axis, EntryLayout
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Group,
    Output,
)
from roboflow_workflows.execution_engine.v2.errors import (
    ContractError,
    CycleError,
    LineageError,
    MutationConflictError,
    NestedWorkflowError,
    OperatorInputError,
    ParamsValidationError,
    SelectorError,
    UnknownBlockError,
    WorkflowCompileError,
)
from roboflow_workflows.execution_engine.v2.introspection import (
    describe_workflow,
    discover_connections,
)
from roboflow_workflows.execution_engine.v2.kinds import DICTIONARY_KIND, FLOAT_KIND
from roboflow_workflows.execution_engine.v2.operators.alignment import Align
from roboflow_workflows.execution_engine.v2.operators.window import Window
from roboflow_workflows.execution_engine.v2.plan import (
    CompiledWorkflow,
    PlannedOperator,
    PlannedOperatorInput,
    PlannedSourceOutput,
    SourcePort,
    StepPort,
)

from tests.unit_tests.execution_engine.v2.compilation.fixture_blocks import (
    ALL_BLOCKS,
    axes_of,
    gate,
    nested,
    parameter,
    step,
)


class Frames(Source):
    """Scalar frames, stationary tiles, unstable tiles and a mutable dict."""

    type = "test/frames@v1"
    outputs = {
        "image": SourceOutput(FLOAT_KIND),
        "tiles": SourceOutput(
            FLOAT_KIND,
            layout=EntryLayout((Axis("tile", "sample", stationary=True),)),
        ),
        "loose": SourceOutput(FLOAT_KIND, layout=EntryLayout((Axis("any", "sample"),))),
        "meta": SourceOutput(DICTIONARY_KIND),
    }

    def open(self):
        raise AssertionError("compilation must not open a source")

    def read(self):
        raise AssertionError("compilation must not read a source")


class Probe(Source):
    """One scalar reading per pulse."""

    type = "test/probe@v1"
    outputs = {"value": SourceOutput(FLOAT_KIND)}

    def open(self):
        raise AssertionError("compilation must not open a source")

    def read(self):
        raise AssertionError("compilation must not read a source")


class TemporalMean(Block):
    """T-oriented collapse: the consumed last axis must be time."""

    type = "test/temporal_mean@v1"
    outputs = {"mean": Output(FLOAT_KIND, source="values", context_policy="last")}

    class Params(BlockParams):
        values: Group(FLOAT_KIND, temporal=True)

    def run(self, *, values):
        return {"mean": sum(values) / max(len(values), 1)}


class PickSome(Block):
    """A selected collection: replaces the collapsed axis with K."""

    type = "test/pick_some@v1"
    outputs = {
        "picked": Output(
            FLOAT_KIND,
            expand="picked",
            source="candidates",
            context_policy="selected",
        )
    }

    class Params(BlockParams):
        candidates: Group(FLOAT_KIND)

    def run(self, *, candidates):
        raise AssertionError("compilation must not run a block")


CATALOGUE = Catalogue(
    ALL_BLOCKS + (TemporalMean, PickSome),
    sources=[Frames, Probe],
    operators=[Align, Window],
    namespace="test",
)

CAM = {"type": "test/frames@v1", "name": "cam"}
PROBE = {"type": "test/probe@v1", "name": "probe"}
PAIR = {
    "type": "v2/align@v1",
    "name": "pair",
    "clock": "media",
    "tolerance_ms": 50,
    "inputs": {"left": "$sources.cam.image", "right": "$sources.probe.value"},
}


def window(name, collect, *, hold=None, size=3):
    declaration = {
        "type": "v2/window@v1",
        "name": name,
        "size": size,
        "collect": collect,
    }
    if hold is not None:
        declaration["hold"] = hold

    return declaration


def group(name, anchor, **fields):
    return {
        "type": "OutputGroup",
        "name": name,
        "anchor": anchor,
        "outputs": [
            {"type": "JsonField", "name": field, "selector": selector}
            for field, selector in fields.items()
        ],
    }


def definition(steps=(), operators=(), outputs=(), *, sources=(CAM, PROBE), inputs=()):
    return {
        "version": "2.0",
        "inputs": list(inputs),
        "sources": list(sources),
        "operators": list(operators),
        "steps": list(steps),
        "outputs": list(outputs),
    }


def compile_definition(*args, options=None, **kwargs):
    plan = compile_workflow(
        definition(*args, **kwargs),
        catalogue=CATALOGUE,
        options=options or CompileOptions(),
    )

    return plan


def _child(steps, outputs, inputs=({"type": "WorkflowParameter", "name": "x"},)):
    return {
        "version": "2.0",
        "inputs": list(inputs),
        "steps": steps,
        "outputs": [
            {"type": "JsonField", "name": name, "selector": selector}
            for name, selector in outputs.items()
        ],
    }


# --- records, domains and routes ---------------------------------------------


def test_an_alignment_becomes_a_domain_its_consumers_and_groups_follow() -> None:
    plan = compile_definition(
        [step("scale", "scale", value="$operators.pair.left")],
        [PAIR],
        [group("pairs", "$operators.pair.left", scaled="$steps.scale.scaled")],
    )

    [operator] = plan.operators.values()
    assert isinstance(operator, PlannedOperator)
    assert operator.spec.type == "v2/align@v1" and operator.namespace == "test"
    assert operator.params.tolerance_ms == 50
    assert [(item.name, item.role, item.domain) for item in operator.inputs] == [
        ("left", "input", "cam"),
        ("right", "input", "probe"),
    ]
    assert operator.upstream_domains == ("cam", "probe")
    assert {name: axes_of(port.layout) for name, port in operator.outputs.items()} == {
        "left": [],
        "right": [],
    }
    assert plan.step(("scale",)).domain == "pair"
    assert [step.path for step in plan.route("pair")] == [("scale",)]
    assert plan.route("cam") == ()
    assert [group.name for group in plan.groups_of("pair")] == ["pairs"]
    assert plan.consumers_of("cam") == (operator,) == plan.consumers_of("probe")
    assert plan.consumers_of("pair") == ()
    assert plan.domains == ("cam", "probe", "pair")
    assert plan.domain_ports("pair") is operator.outputs
    assert plan.output_groups[0].anchor == SourcePort("pair", "left", "operator")
    assert plan.output_groups[0].anchor.describe() == "$operators.pair.left"
    json.dumps(plan.describe())


def test_a_batch_alignment_plans_a_stationary_member_axis_with_an_operator_origin() -> (
    None
):
    plan = compile_definition(
        operators=[
            {
                **PAIR,
                "layout": "batch",
                "inputs": {"a": "$sources.cam.image", "b": "$sources.probe.value"},
            }
        ]
    )

    [axis] = plan.operator("pair").outputs["samples"].layout.axes
    assert (axis.id, axis.kind, axis.stationary) == (
        "operators.pair:samples",
        "sample",
        True,
    )
    assert plan.axis_origin(axis.id).describe() == "$operators.pair"


def test_forward_references_and_steps_between_operators_follow_one_graph() -> None:
    # Declaration order is the reverse of data flow: the consumer of the
    # window comes first, the window before the step it collects from.
    plan = compile_definition(
        [
            step("collapse", "late", data="$operators.clip.scaled"),
            step("scale", "scale", value="$operators.pair.left"),
        ],
        [window("clip", {"scaled": "$steps.scale.scaled"}), PAIR],
    )

    assert list(plan.operators) == ["pair", "clip"]
    assert [step.path for step in plan.steps] == [("scale",), ("late",)]
    assert plan.step(("scale",)).domain == "pair"
    assert plan.step(("late",)).domain == "clip"
    assert plan.operator("clip").inputs[0].domain == "pair"
    assert plan.step(("late",)).dependencies == ()
    assert axes_of(plan.operator("clip").outputs["scaled"].layout) == [
        "operators.clip:t"
    ]


def test_a_cycle_through_an_operator_and_a_step_is_a_cycle_error() -> None:
    with pytest.raises(CycleError, match=r"\$operators\.clip"):
        compile_definition(
            [step("scale", "scale", value="$operators.clip.again")],
            [window("clip", {"again": "$steps.scale.scaled"})],
        )


def test_static_steps_run_in_every_domain_including_operator_domains() -> None:
    plan = compile_definition(
        [
            step("constant", "fixed"),
            step("scale", "scale", value="$operators.pair.left"),
        ],
        [PAIR],
    )

    assert plan.step(("fixed",)).domain is None
    assert [step.path for step in plan.route("pair")] == [("fixed",), ("scale",)]
    assert [step.path for step in plan.route("cam")] == [("fixed",)]


def test_a_step_joining_an_operator_domain_and_a_source_is_a_lineage_error() -> None:
    with pytest.raises(
        LineageError,
        match=r"joins independent pulse domains \(operator 'pair' via "
        r"\$operators\.pair\.left; source 'probe' via \$sources\.probe\.value\); "
        r"source 'probe' feeds operator 'pair'",
    ):
        compile_definition(
            [
                step(
                    "scale",
                    "scale",
                    value="$operators.pair.left",
                    factor="$sources.probe.value",
                )
            ],
            [PAIR],
        )


def test_a_group_anchored_on_an_operator_rejects_fields_of_its_upstream_source() -> (
    None
):
    with pytest.raises(LineageError, match="comes from source 'cam'"):
        compile_definition(
            operators=[PAIR],
            outputs=[group("g", "$operators.pair.left", raw="$sources.cam.image")],
        )


# --- nesting and gates --------------------------------------------------------


def test_nested_workflows_feed_and_consume_operators_through_bindings() -> None:
    child = _child(
        [step("scale", "scale", value="$inputs.x")], {"y": "$steps.scale.scaled"}
    )
    plan = compile_definition(
        [
            nested(
                "before",
                workflow_definition=child,
                parameter_bindings={"x": "$sources.cam.image"},
            ),
            nested(
                "after",
                workflow_definition=child,
                parameter_bindings={"x": "$operators.clip.y"},
            ),
        ],
        [window("clip", {"y": "$steps.before.y"})],
    )

    assert plan.step(("before", "scale")).domain == "cam"
    assert plan.step(("after", "scale")).domain == "clip"
    assert plan.operator("clip").inputs[0].source == StepPort(
        ("before", "scale"), "scaled"
    )


def test_a_gated_child_output_feeding_an_operator_waits_for_the_gate() -> None:
    child = _child([step("scale", "scale", value="$inputs.x")], {"x": "$inputs.x"})
    plan = compile_definition(
        [
            gate("gate", "$sources.cam.image", ["child"]),
            nested(
                "child",
                workflow_definition=child,
                parameter_bindings={"x": "$sources.cam.image"},
            ),
            step("scale", "after", value="$operators.clip.forwarded"),
        ],
        [window("clip", {"forwarded": "$steps.child.x"})],
    )

    [item] = plan.operator("clip").inputs
    assert item.domain == "cam"
    [child_output] = plan.child_outputs
    assert item.source == child_output.port
    assert [gate.controller for gate in child_output.gates] == [("gate",)]


def test_a_nested_operator_declaration_is_an_explicit_root_only_error() -> None:
    child = {
        **_child(
            [step("scale", "scale", value="$inputs.x")], {"y": "$steps.scale.scaled"}
        )
    }
    child["operators"] = [window("inner", {"x": "$inputs.x"})]

    with pytest.raises(
        NestedWorkflowError, match="only the root workflow declares operators"
    ):
        compile_definition(
            [
                nested(
                    "child",
                    workflow_definition=child,
                    parameter_bindings={"x": "$sources.cam.image"},
                )
            ]
        )


def test_operator_ports_are_addressed_from_the_root_only() -> None:
    child = _child(
        [step("scale", "scale", value="$operators.clip.image")],
        {"y": "$steps.scale.scaled"},
    )

    with pytest.raises(SelectorError, match="operators are addressed from the root"):
        compile_definition(
            [nested("child", workflow_definition=child, parameter_bindings={"x": 1.0})],
            [window("clip", {"image": "$sources.cam.image"})],
        )


def test_gates_across_an_operator_boundary_join_domains() -> None:
    with pytest.raises(
        LineageError,
        match=r"\(source 'cam' via gate of \$steps\.gate; operator 'clip' via .*\); "
        r"source 'cam' feeds operator 'clip'.*hold input",
    ):
        compile_definition(
            [
                gate("gate", "$sources.cam.image", ["after"]),
                step("scale", "after", value="$operators.clip.image"),
            ],
            [window("clip", {"image": "$sources.cam.image"})],
        )


# --- windows and selection ----------------------------------------------------


@pytest.mark.parametrize(
    ("collected", "extra_steps", "expected"),
    [
        ("$sources.cam.image", [], ["operators.clip:t"]),
        ("$sources.cam.tiles", [], ["sources.cam:tile", "operators.clip:t"]),
        (
            "$steps.stable.tile",
            [step("stable_expand", "stable", value="$sources.cam.tiles")],
            ["sources.cam:tile", "stable:tiles", "operators.clip:t"],
        ),
    ],
    ids=["scalar", "stationary-source-axis", "stationary-crop"],
)
def test_windows_append_t_after_a_stationary_parent_path(
    collected, extra_steps, expected
) -> None:
    plan = compile_definition(extra_steps, [window("clip", {"frames": collected})])

    layout = plan.operator("clip").outputs["frames"].layout
    assert axes_of(layout) == expected
    assert layout.axes[-1].kind == "time"
    assert plan.axis_origin("operators.clip:t").describe() == "$operators.clip"


@pytest.mark.parametrize(
    ("collected", "extra_steps", "extra_operators", "axis_id", "context"),
    [
        (
            "$sources.cam.loose",
            [],
            [],
            "sources.cam:any",
            "axis 'sources.cam:any' comes from $sources.cam",
        ),
        (
            "$steps.expand.child",
            [step("expand", "expand", value="$sources.cam.image")],
            [],
            "expand:child",
            "axis 'expand:child' comes from $steps.expand.child",
        ),
        (
            "$operators.first.frames",
            [],
            [window("first", {"frames": "$sources.cam.image"})],
            "operators.first:t",
            "axis 'operators.first:t' comes from $operators.first",
        ),
        (
            "$steps.pick.picked",
            [step("pick_some", "pick", candidates="$sources.cam.tiles")],
            [],
            "pick:picked",
            "axis 'pick:picked' comes from $steps.pick.picked, a selected collection. "
            "Its K axis stays an axis even with one member",
        ),
    ],
    ids=["unstable-source-axis", "dynamic-crop", "second-t", "selected-k"],
)
def test_the_window_rejects_ineligible_collections_and_the_compiler_names_the_producer(
    collected, extra_steps, extra_operators, axis_id, context
) -> None:
    # The window class owns the rule; the compiler only adds the selector and
    # the producer of the offending axis.
    with pytest.raises(OperatorInputError) as raised:
        compile_definition(
            extra_steps, [*extra_operators, window("clip", {"frames": collected})]
        )

    error = raised.value
    assert (error.field_path, error.axis_id) == (("collect", "frames"), axis_id)
    assert error.step_path == ("$operators", "clip")
    assert f"collect.frames ({collected}): {context}" in str(error)


def test_a_selected_collection_stays_a_nonstationary_k_even_before_windowing() -> None:
    plan = compile_definition(
        [step("pick_some", "pick", candidates="$operators.clip.frames")],
        [window("clip", {"frames": "$sources.cam.image"})],
    )

    [axis] = plan.step(("pick",)).outputs["picked"].layout.axes
    assert (axis.kind, axis.stationary) == ("dynamic_nesting", False)
    assert plan.step(("pick",)).invocation_layout.axes == ()


def test_hold_must_be_a_parent_of_every_collected_field() -> None:
    plan = compile_definition(
        operators=[
            window(
                "clip",
                {"tiles": "$sources.cam.tiles"},
                hold={"reference": "$sources.cam.image"},
            )
        ]
    )
    assert axes_of(plan.operator("clip").outputs["reference"].layout) == []

    with pytest.raises(OperatorInputError) as raised:
        compile_definition(
            operators=[
                window(
                    "clip",
                    {"frame": "$sources.cam.image"},
                    hold={"reference": "$sources.cam.tiles"},
                )
            ]
        )
    assert raised.value.field_path == ("hold", "reference")
    assert "hold.reference ($sources.cam.tiles)" in str(raised.value)


def test_collect_and_hold_must_come_from_one_domain() -> None:
    with pytest.raises(LineageError, match="collects from several pulse domains"):
        compile_definition(
            operators=[
                window(
                    "clip",
                    {"frame": "$sources.cam.image"},
                    hold={"reading": "$sources.probe.value"},
                )
            ]
        )


def test_t_oriented_groups_require_a_final_time_axis_and_s_groups_do_not() -> None:
    plan = compile_definition(
        [
            step("temporal_mean", "mean", values="$operators.clip.tiles"),
            step("expand", "split", value="$operators.clip.frames"),
            step("collapse", "per_time", data="$steps.split.child"),
        ],
        [
            window(
                "clip", {"tiles": "$sources.cam.tiles", "frames": "$sources.cam.image"}
            )
        ],
    )
    assert axes_of(plan.step(("mean",)).outputs["mean"].layout) == ["sources.cam:tile"]
    assert axes_of(plan.step(("per_time",)).outputs["output"].layout) == [
        "operators.clip:t"
    ]

    with pytest.raises(
        LineageError,
        match=r"T-oriented Group.*last axis 'split:child' is dynamic_nesting, after "
        r"the time axis.*Collapse the axes after T first",
    ):
        compile_definition(
            [
                step("expand", "split", value="$operators.clip.frames"),
                step("temporal_mean", "mean", values="$steps.split.child"),
            ],
            [window("clip", {"frames": "$sources.cam.image"})],
        )
    with pytest.raises(
        LineageError, match="T-oriented Group.*ungrouped and has no time axis"
    ):
        compile_definition([step("temporal_mean", "mean", values="$sources.cam.image")])
    with pytest.raises(LineageError) as raised:
        compile_definition([step("temporal_mean", "mean", values="$sources.cam.tiles")])
    assert "no time axis at all. Collect the values over time first" in str(
        raised.value
    )
    assert "after T" not in str(raised.value)


# --- declaration errors -------------------------------------------------------


@pytest.mark.parametrize(
    ("operators", "sources", "error", "message"),
    [
        (
            [{**PAIR, "type": "v2/unknown@v1"}],
            (CAM, PROBE),
            UnknownBlockError,
            "unknown operator type 'v2/unknown@v1'",
        ),
        (
            [{**PAIR, "clock": "$inputs.clock"}],
            (CAM, PROBE),
            ParamsValidationError,
            "operator parameters are literals",
        ),
        (
            [window("clip", {"x": "$sources.cam.image"}, size=0)],
            (CAM, PROBE),
            ParamsValidationError,
            "size",
        ),
        (
            [{**PAIR, "inputs": {"only": "$sources.cam.image", "x": "$inputs.p"}}],
            (CAM, PROBE),
            LineageError,
            "reads a static value",
        ),
        (
            [
                {
                    "type": "v2/window@v1",
                    "name": "clip",
                    "size": 2,
                    "inputs": {"x": "$sources.cam.image"},
                }
            ],
            (CAM, PROBE),
            WorkflowCompileError,
            "does not accept input input 'x'",
        ),
        (
            [{**PAIR, "name": "cam"}],
            (CAM, PROBE),
            WorkflowCompileError,
            r"operators reuse source names \['cam'\]",
        ),
        (
            [{**PAIR, "inputs": {}}],
            (CAM, PROBE),
            WorkflowCompileError,
            "declares no inputs",
        ),
        (
            [
                {
                    **PAIR,
                    "inputs": {"a": "$sources.cam.nope", "b": "$sources.probe.value"},
                }
            ],
            (CAM, PROBE),
            SelectorError,
            "has no output 'nope'",
        ),
        (
            [
                window("clip", {"x": "$sources.cam.image"}),
                window("clip", {"x": "$sources.cam.image"}),
            ],
            (CAM, PROBE),
            WorkflowCompileError,
            "duplicate operator name 'clip'",
        ),
        ([PAIR], (), WorkflowCompileError, "need sources: session.run"),
    ],
    ids=[
        "unknown-type",
        "selector-parameter",
        "invalid-parameter",
        "static-input",
        "wrong-role",
        "name-clash",
        "no-inputs",
        "unknown-port",
        "duplicate-name",
        "passive",
    ],
)
def test_operator_declaration_errors(operators, sources, error, message) -> None:
    with pytest.raises(error, match=message):
        compile_definition(
            operators=operators,
            sources=sources,
            inputs=[parameter("p", default=1.0, kind=["float"])],
        )


def test_passive_inputs_may_carry_a_time_axis_for_ordinary_temporal_blocks() -> None:
    frames = {
        "name": "frames",
        "kind": ["float"],
        "axes": [
            {"id": "n", "kind": "sample", "stationary": True},
            {"id": "t", "kind": "time"},
        ],
    }
    plan = compile_workflow(
        {
            "version": "2.0",
            "inputs": [frames],
            "steps": [step("temporal_mean", "mean", values="$inputs.frames")],
            "outputs": [
                {"type": "JsonField", "name": "mean", "selector": "$steps.mean.mean"}
            ],
        },
        catalogue=CATALOGUE,
    )
    assert axes_of(plan.step(("mean",)).outputs["mean"].layout) == ["n"]

    unstable = {
        **frames,
        "axes": [{"id": "n", "kind": "sample"}, {"id": "t", "kind": "time"}],
    }
    with pytest.raises(WorkflowCompileError, match="must be stationary"):
        compile_workflow(
            {"version": "2.0", "inputs": [unstable], "steps": [], "outputs": []},
            catalogue=CATALOGUE,
        )


# --- mutation analysis ----------------------------------------------------------


def test_mutating_a_payload_an_operator_may_retain_is_reported() -> None:
    steps = [step("increment", "bump", value="$operators.clip.meta")]
    operators = [window("clip", {"meta": "$sources.cam.meta"})]

    plan = compile_definition(steps, operators)
    [warning] = plan.warnings
    assert "$steps.bump value ($operators.clip.meta) is mutated in place" in warning
    assert "['$operators.clip.meta']" in warning

    with pytest.raises(MutationConflictError, match="may retain or emit it again"):
        compile_definition(
            steps, operators, options=CompileOptions(mutation_conflicts="error")
        )


def test_mutation_origins_trace_operator_ports_back_to_their_inputs() -> None:
    plan = compile_definition(
        [
            step("increment", "bump", value="$sources.cam.meta"),
            step("reader", "read", value="$operators.pair.meta"),
        ],
        [
            {
                **PAIR,
                "inputs": {
                    "meta": "$sources.cam.meta",
                    "other": "$sources.probe.value",
                },
            }
        ],
    )

    # The operator reads the value after the whole pulse of cam, so the
    # reader, which depends on the operator, is ordered after the mutation.
    assert plan.warnings == ()


# --- directly constructed plans -----------------------------------------------


def _pipeline_plan():
    plan = compile_definition(
        [step("scale", "scale", value="$operators.pair.left")],
        [PAIR, window("clip", {"scaled": "$steps.scale.scaled"})],
    )

    return plan


def _rebuilt(plan, **overrides):
    fields = dict(
        inputs=plan.inputs,
        steps=plan.steps,
        outputs=plan.outputs,
        catalogue=plan.catalogue,
        sources=plan.sources,
        output_groups=plan.output_groups,
        operators=plan.operators,
    )
    fields.update(overrides)
    rebuilt = CompiledWorkflow(**fields)

    return rebuilt


def test_hand_built_plans_keep_operators_in_topological_order() -> None:
    plan = _pipeline_plan()
    _rebuilt(plan)

    reversed_order = dict(reversed(list(plan.operators.items())))
    with pytest.raises(
        ContractError, match="comes from operator 'pair', which is not listed before"
    ):
        _rebuilt(plan, operators=reversed_order)


def test_hand_built_operator_inputs_are_checked_against_derived_domains() -> None:
    plan = _pipeline_plan()
    clip = plan.operator("clip")
    [item] = clip.inputs

    wrong_domain = replace(clip, inputs=(replace(item, domain="cam"),))
    with pytest.raises(ContractError, match="records domain 'cam'"):
        _rebuilt(plan, operators={"pair": plan.operator("pair"), "clip": wrong_domain})

    own = replace(
        clip,
        inputs=(
            replace(
                item,
                source=SourcePort("clip", "scaled", "operator"),
                layout=clip.outputs["scaled"].layout,
                domain="clip",
            ),
        ),
    )
    with pytest.raises(ContractError, match="reads the operator's own pulses"):
        _rebuilt(plan, operators={"pair": plan.operator("pair"), "clip": own})


def test_hand_built_plans_reject_operators_without_sources_and_name_clashes() -> None:
    plan = _pipeline_plan()
    renamed = {"cam": replace(plan.operator("pair"), name="cam")}

    with pytest.raises(ContractError, match="also source names"):
        _rebuilt(plan, steps=(), operators=renamed)
    with pytest.raises(ContractError, match="need declared sources"):
        _rebuilt(
            plan,
            steps=(),
            sources={},
            output_groups=(),
            operators={"pair": plan.operator("pair")},
        )


def _stationary(layout: EntryLayout) -> EntryLayout:
    """The same axis ids, relabelled as stable nesting the producer never declared."""
    relabelled = EntryLayout(
        tuple(
            replace(
                axis,
                kind="sample" if axis.kind == "sample" else "static_nesting",
                stationary=True,
            )
            for axis in layout.axes
        )
    )

    return relabelled


def _windowed_plan(collected: str, steps=()):
    plan = compile_definition(list(steps), [window("clip", {"x": collected})])

    return plan


def test_hand_built_operator_inputs_cannot_forge_producer_stationarity() -> None:
    # C22-01: the recorded input layout must be the producer's own, axis kind
    # and stationarity included, not merely the same axis ids.
    plan = _windowed_plan("$sources.cam.tiles")
    clip = plan.operator("clip")
    loose = _stationary(plan.source("cam").outputs["loose"].layout)
    forged = replace(
        clip,
        inputs=(
            replace(
                clip.inputs[0],
                selector="$sources.cam.loose",
                source=SourcePort("cam", "loose"),
                layout=loose,
            ),
        ),
        outputs={
            "x": replace(
                clip.outputs["x"],
                layout=EntryLayout(loose.axes + clip.outputs["x"].layout.axes[-1:]),
            )
        },
    )

    with pytest.raises(ContractError, match=r"sources\.cam:any \(sample\)"):
        _rebuilt(plan, operators={"clip": forged})


def test_hand_built_selected_collections_cannot_be_relabelled_stationary() -> None:
    plan = compile_definition(
        [step("pick_some", "pick", candidates="$operators.clip.x")],
        [window("clip", {"x": "$sources.cam.image"})],
    )
    pick = plan.step(("pick",))
    k = pick.outputs["picked"].layout
    clip = plan.operator("clip")
    recollect = replace(
        clip,
        name="next",
        inputs=(
            replace(
                clip.inputs[0],
                selector="$steps.pick.picked",
                source=StepPort(("pick",), "picked"),
                layout=_stationary(k),
                domain="clip",
            ),
        ),
        outputs={
            "x": replace(
                clip.outputs["x"],
                layout=_stationary(k).append_axis(Axis("operators.next:t", "time")),
            )
        },
    )
    with pytest.raises(ContractError, match=r"pick:picked \(dynamic_nesting\)"):
        _rebuilt(plan, operators={"clip": clip, "next": recollect})

    with pytest.raises(ContractError, match="the block declares dynamic_nesting"):
        replace(
            pick,
            outputs={"picked": replace(pick.outputs["picked"], layout=_stationary(k))},
        )


def _relabel(layout: EntryLayout, position: int, **changes) -> EntryLayout:
    """``layout`` with one axis restated, keeping its id."""
    axes = list(layout.axes)
    axes[position] = replace(axes[position], **changes)
    relabelled = EntryLayout(tuple(axes))

    return relabelled


def _restated(planned_step, output_name: str, layout: EntryLayout):
    """Rebuild ``planned_step`` with one output restating its axes."""
    output = replace(planned_step.outputs[output_name], layout=layout)
    rebuilt = replace(
        planned_step, outputs={**planned_step.outputs, output_name: output}
    )

    return rebuilt


def test_a_forwarding_step_cannot_relabel_a_selected_k_as_stable() -> None:
    # C22-01 recheck: Window -> PickSome -> Echo; Echo's record must not turn
    # the selected collection's K into a stable axis another window collects.
    plan = compile_definition(
        [
            step("pick_some", "pick", candidates="$operators.clip.x"),
            step("echo", "forward", value="$steps.pick.picked"),
        ],
        [window("clip", {"x": "$sources.cam.image"})],
    )
    forward = plan.step(("forward",))
    k = forward.outputs["value"].layout
    assert (k.axes[0].kind, k.axes[0].stationary) == ("dynamic_nesting", False)

    with pytest.raises(
        ContractError, match=r"pick:picked \(static_nesting, stationary\)"
    ):
        _restated(forward, "value", _stationary(k))
    with pytest.raises(ContractError, match="runs over"):
        replace(
            forward,
            invocation_layout=_stationary(k),
            outputs={"value": replace(forward.outputs["value"], layout=_stationary(k))},
        )
    forged_binding = replace(forward.bindings[0], source_layout=_stationary(k))
    forged = replace(
        forward,
        bindings=(forged_binding,),
        invocation_layout=_stationary(k),
        outputs={"value": replace(forward.outputs["value"], layout=_stationary(k))},
    )
    with pytest.raises(ContractError, match=r"differ from the axes \['pick:picked"):
        _rebuilt(plan, steps=(plan.step(("pick",)), forged))


@pytest.mark.parametrize(
    "changes",
    [{"stationary": False}, {"kind": "static_nesting"}],
    ids=["stationary-flag", "axis-kind"],
)
def test_forwarded_and_parent_axes_keep_their_producer_declarations(changes) -> None:
    plan = compile_definition(
        [
            step("echo", "forward", value="$sources.cam.tiles"),
            step("stable_expand", "stable", value="$steps.forward.value"),
            step("collapse", "per_tile", data="$steps.stable.tile"),
        ],
        [window("clip", {"x": "$steps.forward.value"})],
    )
    # Valid forwarding of a stationary path is accepted and collected.
    assert _rebuilt(plan).operator("clip").inputs[0].layout.axes[0].stationary
    forward, stable, per_tile = (
        plan.step(path) for path in (("forward",), ("stable",), ("per_tile",))
    )

    # Same-layout forwarding, an expanded output's inherited parent and a
    # collapsed output's surviving parent all keep the parent declaration.
    for planned_step, output_name in (
        (forward, "value"),
        (stable, "tile"),
        (per_tile, "output"),
    ):
        layout = planned_step.outputs[output_name].layout
        with pytest.raises(ContractError, match="inconsistent with invocation axes"):
            _restated(planned_step, output_name, _relabel(layout, 0, **changes))
    with pytest.raises(ContractError, match="runs over"):
        replace(
            per_tile,
            invocation_layout=_relabel(per_tile.invocation_layout, 0, **changes),
        )


def test_hand_built_operator_ports_must_be_what_the_class_plans() -> None:
    # C22-02: ports, roles and parameters are re-derived from the class.
    plan = _windowed_plan("$sources.cam.image")
    clip = plan.operator("clip")
    assert _rebuilt(plan).operator("clip") == clip

    without_t = replace(
        clip, outputs={"x": replace(clip.outputs["x"], layout=EntryLayout())}
    )
    renamed = replace(clip, outputs={"y": replace(clip.outputs["x"], name="y")})
    other_kind = replace(
        clip, outputs={"x": replace(clip.outputs["x"], kinds=("dictionary",))}
    )
    for forged in (without_t, renamed, other_kind):
        with pytest.raises(
            ContractError, match="records ports .* but v2/window@v1 plans"
        ):
            _rebuilt(plan, operators={"clip": forged})

    invalid_params = replace(clip, params=type(clip.params).model_construct(size=0))
    with pytest.raises(ContractError, match="is not a valid v2/window@v1: .*size"):
        _rebuilt(plan, operators={"clip": invalid_params})

    pair_plan = compile_definition(operators=[PAIR])
    pair = pair_plan.operator("pair")
    collecting = replace(pair, inputs=(replace(pair.inputs[0], role="collect"),))
    with pytest.raises(ContractError, match="does not accept collect input 'left'"):
        _rebuilt(pair_plan, operators={"pair": collecting})


def test_planned_operator_records_validate_their_fields() -> None:
    port = PlannedSourceOutput("x", ("float",))
    item = PlannedOperatorInput(
        name="x",
        role="collect",
        selector="$sources.cam.image",
        source=SourcePort("cam", "image"),
        layout=EntryLayout(),
        domain="cam",
    )

    with pytest.raises(ContractError, match="unknown role"):
        replace(item, role="gather")
    with pytest.raises(ContractError, match="repeats input names"):
        PlannedOperator("w", None, "", None, (item, item), {"x": port})
    with pytest.raises(ContractError, match="consumes no inputs"):
        PlannedOperator("w", None, "", None, (), {"x": port})
    with pytest.raises(ContractError, match="origin must be one of"):
        SourcePort("cam", "image", "block")


# --- introspection ---------------------------------------------------------------


def test_introspection_shows_operators_their_inputs_and_edges() -> None:
    plan = compile_definition(
        [step("scale", "scale", value="$operators.pair.left")],
        [
            window("clip", {"scaled": "$steps.scale.scaled"}),
            PAIR,
        ],
        [group("clips", "$operators.clip.scaled", scaled="$operators.clip.scaled")],
    )

    described = describe_workflow(plan)
    json.dumps(described)
    clip = described["operators"]["clip"]
    assert clip["node_id"] == "$operators.clip"
    assert clip["parameters"]["size"] == 3
    assert clip["inputs"]["scaled"]["domain"] == "pair"
    assert clip["inputs"]["scaled"]["role"] == "collect"
    assert clip["outputs"]["scaled"]["axes"] == ["operators.clip:t"]
    assert clip["groups"] == ["clips"]
    assert described["operators"]["pair"]["consumers"] == ["$operators.clip"]
    assert described["operators"]["clip"]["consumers"] == []
    assert described["steps"][0]["domain"] == "$operators.pair"
    assert described["output_groups"]["clips"]["source"] == "$operators.clip"
    assert described["axes"]["operators.clip:t"] == {
        "kind": "time",
        "stationary": False,
        "origin": "$operators.clip",
    }

    edges = [
        (edge.kind, edge.source, edge.target, edge.field_path)
        for edge in discover_connections(plan)
    ]
    assert (
        "operator_input",
        "$sources.cam",
        "$operators.pair",
        ("inputs", "left"),
    ) in edges
    assert (
        "operator_input",
        "$steps.scale",
        "$operators.clip",
        ("collect", "scaled"),
    ) in edges
    assert ("data", "$operators.pair", "$steps.scale", ("value",)) in edges
    assert ("anchor", "$operators.clip", "$output_groups.clips", ()) in edges


def test_source_and_operator_namespaces_cannot_be_forged() -> None:
    with pytest.raises(SelectorError, match="unknown source 'pair'"):
        compile_definition([step("scale", "scale", value="$sources.pair.left")], [PAIR])
    with pytest.raises(SelectorError, match="unknown operator 'cam'"):
        compile_definition([step("scale", "scale", value="$operators.cam.image")])

    plan = _pipeline_plan()
    scale = plan.step(("scale",))
    [binding] = scale.bindings_for("value")
    forged = replace(
        scale,
        bindings=(replace(binding, source=SourcePort("pair", "left", "source")),)
        + tuple(item for item in scale.bindings if item is not binding),
    )
    with pytest.raises(ContractError, match="Plan has no source 'pair'"):
        _rebuilt(plan, steps=(forged,))
