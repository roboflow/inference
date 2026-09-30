"""Compilation of active definitions: sources, output groups and causal domains.

Nothing here opens a source or runs a block. The fixtures deliberately raise
in their constructors and lifecycle methods to prove that compilation and
session creation never touch them.
"""

import json
from dataclasses import replace
from types import MappingProxyType

import pytest
from roboflow_workflows.execution_engine.v2 import (
    Catalogue,
    PulseKey,
    Source,
    SourceOutput,
    SourceParams,
    compile_workflow,
)
from roboflow_workflows.execution_engine.v2.data import Axis, EntryLayout
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Output,
    Ref,
)
from roboflow_workflows.execution_engine.v2.errors import (
    ContractError,
    KindMismatchError,
    LineageError,
    NestedWorkflowError,
    ParamsValidationError,
    ResourceError,
    SelectorError,
    WorkflowCompileError,
    WorkflowInputError,
)
from roboflow_workflows.execution_engine.v2.kinds import FLOAT_KIND, STRING_KIND
from roboflow_workflows.execution_engine.v2.plan import (
    Binding,
    CompiledWorkflow,
    Constant,
    InputPort,
    PlannedInput,
    PlannedOutputGroup,
    PlannedSource,
    PlannedSourceOutput,
    PlannedStep,
    PlannedWorkflowOutput,
    SourcePort,
    StepPort,
)
from roboflow_workflows.execution_engine.v2.resources import Factory

from tests.unit_tests.execution_engine.v2.compilation.fixture_blocks import (
    ALL_BLOCKS,
    axes_of,
    gate,
    nested,
    parameter,
    step,
)


class Temperature(Source):
    """Scalar ports; a static ``path`` parameter and a selectable ``probe``."""

    type = "test/temperature@v1"
    outputs = {
        "temperature": SourceOutput(FLOAT_KIND),
        "label": SourceOutput(STRING_KIND),
    }

    class Params(SourceParams):
        path: str
        probe: str | Ref(STRING_KIND) = "probe-1"

    def __init__(self, *, clock):
        raise AssertionError("compilation must not construct a source")

    def open(self, *, path, probe):
        raise AssertionError("compilation must not open a source")

    def read(self):
        raise AssertionError("compilation must not read a source")


class Camera(Source):
    """A scalar image port and a grouped port over one local sample axis."""

    type = "test/camera@v1"
    outputs = {
        "image": SourceOutput(FLOAT_KIND),
        "tiles": SourceOutput(
            FLOAT_KIND, layout=EntryLayout((Axis("tile", "sample"),))
        ),
    }

    def open(self):
        raise AssertionError("compilation must not open a source")

    def read(self):
        raise AssertionError("compilation must not read a source")


class Tag(Block):
    """A literal-only field that looks like a selector stays a literal."""

    type = "test/tag@v1"
    outputs = {"tagged": Output(source="value")}

    class Params(BlockParams):
        value: Ref()
        label: str = ""

    def run(self, *, value, label):
        return {"tagged": (label, value)}


CATALOGUE = Catalogue(
    ALL_BLOCKS + (Tag,),
    sources=[Temperature, Camera],
    namespace="test",
    providers={"clock": "wall"},
)

TEMPERATURE = {"type": "test/temperature@v1", "name": "temp", "path": "temps.csv"}
CAMERA = {"type": "test/camera@v1", "name": "cam"}


def group(name, anchor, **fields):
    """Declare an ``OutputGroup`` with ``fields`` as name to selector."""
    return {
        "type": "OutputGroup",
        "name": name,
        "anchor": anchor,
        "outputs": [
            {"type": "JsonField", "name": field, "selector": selector}
            for field, selector in fields.items()
        ],
    }


def active(steps, outputs=(), *, sources=(TEMPERATURE, CAMERA), inputs=()):
    """Build an active definition."""
    return {
        "version": "2.0",
        "inputs": list(inputs),
        "sources": list(sources),
        "steps": steps,
        "outputs": list(outputs),
    }


def compile_active(steps, outputs=(), **kwargs):
    plan = compile_workflow(active(steps, outputs, **kwargs), catalogue=CATALOGUE)

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


# --- parsing and structure -------------------------------------------------


def test_sources_and_groups_become_plan_records_without_constructing_anything() -> None:
    plan = compile_active(
        [step("scale", "scale", value="$sources.temp.temperature")],
        [group("temps", "$sources.temp.temperature", t="$steps.scale.scaled")],
    )

    assert plan.is_active
    assert list(plan.sources) == ["temp", "cam"]
    temp = plan.source("temp")
    assert isinstance(temp, PlannedSource)
    assert temp.spec.type == "test/temperature@v1" and temp.namespace == "test"
    assert temp.params.path == "temps.csv" and temp.bindings == ()
    assert temp.outputs["temperature"] == PlannedSourceOutput(
        name="temperature", kinds=("float",), layout=EntryLayout()
    )
    assert temp.port("label") == SourcePort("temp", "label")
    assert temp.step_path == ("$sources", "temp")

    [temps] = plan.output_groups
    assert isinstance(temps, PlannedOutputGroup)
    assert temps.anchor == SourcePort("temp", "temperature") and temps.source == "temp"
    assert [output.name for output in temps.outputs] == ["t"]
    assert temps.outputs[0].source == StepPort(("scale",), "scaled")
    assert plan.outputs == ()


def test_a_passive_plan_is_unchanged() -> None:
    plan = compile_workflow(
        {
            "version": "2.0",
            "inputs": [{"type": "WorkflowBatchInput", "name": "values"}],
            "steps": [step("scale", "scale", value="$inputs.values")],
            "outputs": [
                {"type": "JsonField", "name": "s", "selector": "$steps.scale.scaled"}
            ],
        },
        catalogue=CATALOGUE,
    )

    assert not plan.is_active
    assert plan.sources == {} and plan.output_groups == ()
    assert all(item.domain is None for item in plan.steps)
    assert plan.describe()["sources"] == {} and plan.describe()["output_groups"] == {}
    with pytest.raises(WorkflowInputError, match="declares no sources"):
        plan.create_session().start()


@pytest.mark.parametrize(
    ("definition", "error", "message"),
    [
        (
            active([], sources=[TEMPERATURE, {**CAMERA, "name": "temp"}]),
            WorkflowCompileError,
            "duplicate source name 'temp'",
        ),
        (
            active([], sources=[{"type": "test/none@v1", "name": "x"}]),
            WorkflowCompileError,
            "unknown source type 'test/none@v1'; known source types",
        ),
        (
            active([], sources=[{"name": "x"}]),
            WorkflowCompileError,
            r"\$sources\.x\) must declare a type",
        ),
        (
            active(
                [],
                [group("g", "$sources.temp.label"), group("g", "$sources.temp.label")],
            ),
            WorkflowCompileError,
            "duplicate output group names",
        ),
        (
            active([], [group("g", "$steps.x.y")]),
            SelectorError,
            "anchor must select a source port",
        ),
        (
            active([], [group("g", "$sources.temp.missing")]),
            SelectorError,
            r"\$sources\.temp has no output 'missing'",
        ),
        (
            active([], [group("g", "$sources.nope.x")]),
            SelectorError,
            "unknown source 'nope'",
        ),
        (
            active([], [{**group("g", "$sources.temp.label"), "optional": True}]),
            WorkflowCompileError,
            r"unsupported keys \['optional'\]",
        ),
        (
            active(
                [],
                [
                    {
                        "type": "OutputGroup",
                        "name": "g",
                        "anchor": "$sources.temp.label",
                        "outputs": [
                            {
                                "type": "JsonField",
                                "name": "a",
                                "selector": "$sources.temp.label",
                                "optional": True,
                            }
                        ],
                    }
                ],
            ),
            WorkflowCompileError,
            r"unsupported keys \['optional'\]",
        ),
        (
            active(
                [],
                [
                    group("g", "$sources.temp.label", a="$sources.temp.label"),
                    {
                        "type": "JsonField",
                        "name": "b",
                        "selector": "$sources.temp.label",
                    },
                ],
            ),
            WorkflowCompileError,
            "mixes OutputGroup and JsonField",
        ),
        (
            active(
                [],
                [{"type": "JsonField", "name": "b", "selector": "$sources.temp.label"}],
            ),
            WorkflowCompileError,
            "flat outputs .* beside sources",
        ),
        (
            {**active([], sources=[]), "outputs": [group("g", "$sources.temp.label")]},
            WorkflowCompileError,
            "declares no sources; use flat JsonField outputs",
        ),
        (
            active([], inputs=[{"type": "WorkflowImage", "name": "image"}]),
            WorkflowCompileError,
            r"inputs \['image'\] are grouped",
        ),
        (
            active(
                [],
                [
                    {
                        "type": "OutputGroup",
                        "name": "g",
                        "anchor": "$sources.temp.label",
                        "outputs": [group("inner", "$sources.temp.label")],
                    }
                ],
            ),
            WorkflowCompileError,
            "groups do not nest",
        ),
    ],
)
def test_structural_rules_of_active_definitions(definition, error, message) -> None:
    with pytest.raises(error, match=message):
        compile_workflow(definition, catalogue=CATALOGUE)


def test_nested_workflows_declare_no_sources_and_no_groups() -> None:
    child_with_sources = {
        **_child([step("constant", "k")], {}, inputs=()),
        "sources": [CAMERA],
    }
    child_with_groups = {
        **_child([step("constant", "k")], {}, inputs=()),
        "outputs": [group("g", "$sources.cam.image")],
    }
    child_reading_a_source = _child(
        [step("scale", "k", value="$sources.temp.temperature")], {}, inputs=()
    )

    with pytest.raises(NestedWorkflowError, match="nested workflow declares sources"):
        compile_active([nested("child", workflow_definition=child_with_sources)])
    with pytest.raises(NestedWorkflowError, match="declares output groups"):
        compile_active([nested("child", workflow_definition=child_with_groups)])
    with pytest.raises(SelectorError, match="sources are addressed from the root"):
        compile_active([nested("child", workflow_definition=child_reading_a_source)])


# --- source parameters -------------------------------------------------------


def test_source_parameters_bind_ungrouped_inputs_as_constants() -> None:
    plan = compile_active(
        [],
        sources=[{**TEMPERATURE, "probe": "$inputs.probe"}],
        inputs=[parameter("probe", default="p-7", kind=["string"])],
    )

    [binding] = plan.source("temp").bindings
    assert binding == Binding(
        field="probe",
        position=(),
        selector="$inputs.probe",
        source=InputPort("probe"),
        source_layout=EntryLayout(),
        mode="constant",
    )
    assert plan.source("temp").describe()["bindings"][0]["selector"] == "$inputs.probe"


@pytest.mark.parametrize(
    ("params", "inputs", "error", "message"),
    [
        ({"probe": "$steps.k.value"}, (), SelectorError, "static configuration"),
        ({"probe": "$sources.cam.image"}, (), SelectorError, "static configuration"),
        ({"probe": "$inputs.nope"}, (), SelectorError, "unknown workflow input 'nope'"),
        (
            {"probe": "$inputs.probe"},
            (
                {
                    "name": "probe",
                    "kind": ["string"],
                    "axes": [{"id": "p", "kind": "sample"}],
                },
            ),
            WorkflowCompileError,
            "grouped",
        ),
        (
            {"probe": "$inputs.probe"},
            (parameter("probe", default=1, kind=["float"]),),
            KindMismatchError,
            r"accepts kinds \['string'\], but '\$inputs\.probe' provides \['float'\]",
        ),
        ({"path": 3}, (), ParamsValidationError, r"\$sources\.temp .*path"),
        ({"unknown": 1}, (), ParamsValidationError, "unknown"),
    ],
)
def test_source_parameter_rules(params, inputs, error, message) -> None:
    definition = active(
        [step("constant", "k")], sources=[{**TEMPERATURE, **params}], inputs=inputs
    )

    with pytest.raises(error, match=message):
        compile_workflow(definition, catalogue=CATALOGUE)


def test_grouped_ports_get_scoped_axes_with_a_source_origin() -> None:
    plan = compile_active([step("collapse", "tiles", data="$sources.cam.tiles")])

    port = plan.source_port(SourcePort("cam", "tiles"))
    assert axes_of(port.layout) == ["sources.cam:tile"]
    assert port.layout.axes[0].kind == "sample"
    origin = plan.axis_origin("sources.cam:tile")
    assert (origin.kind, origin.name, origin.describe()) == (
        "source",
        "cam",
        "$sources.cam",
    )
    tiles = plan.step(("tiles",))
    assert tiles.bindings[0].mode == "group" and axes_of(tiles.invocation_layout) == []
    assert plan.describe()["sources"]["cam"]["outputs"]["tiles"]["axes"] == [
        "sources.cam:tile"
    ]


# --- bindings ----------------------------------------------------------------


def test_scalar_source_ports_bind_like_constants_and_check_kinds() -> None:
    plan = compile_active(
        [
            step("scale", "scale", value="$sources.temp.temperature"),
            step("collapse", "cast", data="$sources.cam.image"),
        ]
    )

    value = plan.step(("scale",)).binding_for("value")
    assert (
        value.source == SourcePort("temp", "temperature") and value.mode == "constant"
    )
    assert axes_of(plan.step(("scale",)).invocation_layout) == []
    cast = plan.step(("cast",)).binding_for("data")
    assert cast.mode == "constant_group" and axes_of(cast.cast_layout) == [
        "cast/data/cast"
    ]

    with pytest.raises(KindMismatchError, match=r"accepts kinds \['float'\]"):
        compile_active([step("scale", "scale", value="$sources.temp.label")])
    with pytest.raises(SelectorError, match=r"\$sources\.temp has no output 'nope'"):
        compile_active([step("scale", "scale", value="$sources.temp.nope")])


def test_selector_looking_literals_stay_literal_and_dynamic_positions_accept_sources() -> (
    None
):
    plan = compile_active(
        [
            step(
                "tag",
                "tag",
                value="$sources.temp.temperature",
                label="$sources.cam.image",
            )
        ]
    )

    tag = plan.step(("tag",))
    assert [binding.field for binding in tag.bindings] == ["value"]
    assert tag.params.label == "$sources.cam.image"
    assert tag.domain == "temp"


def test_source_data_flows_through_nested_bindings_and_child_outputs() -> None:
    child = _child(
        [step("echo", "echo", value="$inputs.x")],
        {"out": "$steps.echo.value", "fwd": "$inputs.x"},
    )
    plan = compile_active(
        [
            gate("g", "$sources.cam.image", ["child"]),
            nested(
                "child",
                workflow_definition=child,
                parameter_bindings={"x": "$sources.cam.image"},
            ),
        ],
        [
            group(
                "frames",
                "$sources.cam.image",
                out="$steps.child.out",
                fwd="$steps.child.fwd",
            )
        ],
    )

    [child_input] = plan.child_inputs
    assert child_input.source == SourcePort("cam", "image")
    assert plan.origin(plan.child_outputs[0].port) == SourcePort("cam", "image")
    assert plan.step(("child", "echo")).domain == "cam"
    assert [gate_.controller for gate_ in plan.child_outputs[0].gates] == [("g",)]


# --- domains -----------------------------------------------------------------


def test_each_step_records_the_one_source_it_depends_on() -> None:
    child = _child(
        [step("echo", "echo", value="$inputs.x")], {"out": "$steps.echo.value"}
    )
    plan = compile_active(
        [
            step("scale", "scale", value="$sources.temp.temperature"),
            step("scale", "again", value="$steps.scale.scaled"),
            step("constant", "const"),
            step("sink", "effect", payload="fixed"),
            gate("g", "$sources.cam.image", ["gated", "child"]),
            step("sink", "gated", payload="fixed"),
            nested(
                "child",
                workflow_definition=child,
                parameter_bindings={"x": "$steps.const.value"},
            ),
        ]
    )

    domains = {"/".join(item.path): item.domain for item in plan.steps}
    assert domains == {
        "scale": "temp",
        "again": "temp",
        "const": None,
        "effect": None,
        "g": "cam",
        "gated": "cam",
        "child/echo": "cam",
    }
    assert plan.describe()["steps"][0]["domain"] == "temp"


def test_route_runs_the_source_domain_plus_static_steps_once_per_pulse() -> None:
    plan = compile_active(
        [
            step("scale", "scale", value="$sources.temp.temperature"),
            step("constant", "const"),
            step("sink", "effect", payload="fixed"),
            step("collapse", "tiles", data="$sources.cam.tiles"),
        ],
        [
            group(
                "temps",
                "$sources.temp.temperature",
                t="$steps.scale.scaled",
                c="$steps.const.value",
            ),
            group("labels", "$sources.temp.label", label="$sources.temp.label"),
            group("frames", "$sources.cam.image", tiles="$steps.tiles.output"),
        ],
    )

    assert [item.path for item in plan.route("temp")] == [
        ("scale",),
        ("const",),
        ("effect",),
    ]
    assert [item.path for item in plan.route("cam")] == [
        ("const",),
        ("effect",),
        ("tiles",),
    ]
    assert [item.name for item in plan.groups_of("temp")] == ["temps", "labels"]
    assert [item.name for item in plan.groups_of("cam")] == ["frames"]
    with pytest.raises(ContractError, match="no source 'mic'"):
        plan.route("mic")


@pytest.mark.parametrize(
    ("steps", "outputs", "message"),
    [
        (
            [
                step(
                    "scale",
                    "s",
                    value="$sources.temp.temperature",
                    factor="$sources.cam.image",
                )
            ],
            (),
            r"\$steps\.s joins independent sources \['cam', 'temp'\]",
        ),
        (
            [
                step("scale", "s", value="$sources.temp.temperature"),
                step(
                    "scale", "t", value="$steps.s.scaled", factor="$sources.cam.image"
                ),
            ],
            (),
            r"\$steps\.t joins independent sources",
        ),
        (
            [
                gate("g", "$sources.cam.image", ["s"]),
                step("scale", "s", value="$sources.temp.temperature"),
            ],
            (),
            r"'cam' via gate of \$steps\.g; 'temp' via \$sources\.temp\.temperature",
        ),
        (
            [
                gate("a", "$sources.cam.image", ["k"]),
                gate("b", "$sources.temp.temperature", ["k"]),
                step("sink", "k", payload="x"),
            ],
            (),
            r"\$steps\.k joins independent sources",
        ),
        (
            [
                gate("g", "$sources.cam.image", ["child"]),
                nested(
                    "child",
                    workflow_definition=_child(
                        [step("echo", "e", value="$inputs.x")], {"fwd": "$inputs.x"}
                    ),
                    parameter_bindings={"x": "$sources.temp.temperature"},
                ),
                step("echo", "reader", value="$steps.child.fwd"),
            ],
            (),
            r"\$steps\.child/e joins independent sources .*'cam' via gate of \$steps\.g; "
            r"'temp' via \$steps\.child: \$inputs\.x",
        ),
        (
            [step("scale", "s", value="$sources.temp.temperature")],
            [group("g", "$sources.cam.image", x="$steps.s.scaled")],
            "anchored on .* but its field 'x' .* comes from source 'temp'",
        ),
        (
            [],
            [group("g", "$sources.cam.image", x="$sources.temp.label")],
            "comes from source 'temp'",
        ),
    ],
)
def test_joining_independent_sources_is_rejected_at_compile_time(
    steps, outputs, message
) -> None:
    with pytest.raises(LineageError, match=message):
        compile_active(steps, outputs)


def test_static_fields_may_join_any_group() -> None:
    plan = compile_active(
        [
            step("constant", "const"),
            step("scale", "s", value="$sources.temp.temperature"),
        ],
        [
            group(
                "temps",
                "$sources.temp.temperature",
                c="$steps.const.value",
                s="$steps.s.scaled",
            ),
            group(
                "frames", "$sources.cam.image", c="$steps.const.value", p="$inputs.p"
            ),
        ],
        inputs=[parameter("p", default=1)],
    )

    assert [item.name for item in plan.output_groups] == ["temps", "frames"]


# --- plan consistency and description --------------------------------------


def _hand_built(**overrides):
    plan = compile_active([step("scale", "scale", value="$sources.temp.temperature")])
    fields = {
        "inputs": plan.inputs,
        "steps": plan.steps,
        "outputs": (),
        "catalogue": plan.catalogue,
        "sources": plan.sources,
        "output_groups": plan.output_groups,
    }
    fields.update(overrides)
    built = CompiledWorkflow(**fields)

    return built


def test_hand_built_plans_are_checked_for_source_references_and_domains() -> None:
    plan = _hand_built()
    [scale] = plan.steps

    with pytest.raises(
        ContractError, match="parameter value: Plan has no source 'temp'"
    ):
        _hand_built(sources={})
    with pytest.raises(
        ContractError, match="records domain 'cam', but .* derive 'temp'"
    ):
        _hand_built(steps=(PlannedStep(**{**_fields(scale), "domain": "cam"}),))
    with pytest.raises(
        ContractError, match="records domain 'temp', but the plan declares no sources"
    ):
        CompiledWorkflow(
            inputs={},
            steps=(
                PlannedStep(**{**_fields(scale), "bindings": (), "domain": "temp"}),
            ),
            outputs=(),
            catalogue=plan.catalogue,
        )
    with pytest.raises(ContractError, match="anchor: Plan has no source 'temp'"):
        _hand_built(
            sources={},
            steps=(),
            output_groups=(PlannedOutputGroup("g", SourcePort("temp", "label"), ()),),
        )
    with pytest.raises(ContractError, match="anchor.*no output 'nope'"):
        _hand_built(
            output_groups=(PlannedOutputGroup("g", SourcePort("temp", "nope"), ()),)
        )
    with pytest.raises(ContractError, match="uses output groups; flat outputs"):
        _hand_built(
            outputs=(
                PlannedWorkflowOutput(
                    "f", "$sources.temp.label", SourcePort("temp", "label")
                ),
            )
        )
    with pytest.raises(ContractError, match="accepts only ungrouped inputs"):
        _hand_built(
            inputs={
                "image": PlannedInput(
                    name="image",
                    kinds=("*",),
                    layout=EntryLayout((Axis("inputs", "sample"),)),
                )
            }
        )


def _fields(planned: PlannedStep) -> dict:
    return {
        "path": planned.path,
        "spec": planned.spec,
        "namespace": planned.namespace,
        "params": planned.params,
        "bindings": planned.bindings,
        "invocation_layout": planned.invocation_layout,
        "outputs": planned.outputs,
        "gates": planned.gates,
        "control_targets": planned.control_targets,
        "dependencies": planned.dependencies,
        "domain": planned.domain,
    }


def test_planned_source_and_group_records_validate_their_own_fields() -> None:
    spec = CATALOGUE.source_entry("test/temperature@v1").spec
    params = spec.validate_params({"path": "x"})

    with pytest.raises(ContractError, match="Source name must use"):
        PlannedSource(
            name="bad name",
            spec=spec,
            namespace="",
            params=params,
            bindings=(),
            outputs={},
        )
    with pytest.raises(ContractError, match="must read an ungrouped workflow input"):
        PlannedSource(
            name="temp",
            spec=spec,
            namespace="",
            params=params,
            bindings=(
                Binding(
                    field="path",
                    position=(),
                    selector="",
                    source=Constant(1),
                    source_layout=EntryLayout(),
                    mode="constant",
                ),
            ),
            outputs={},
        )
    with pytest.raises(ContractError, match="repeats field names"):
        PlannedOutputGroup(
            "g",
            SourcePort("temp", "label"),
            (
                PlannedWorkflowOutput(
                    "a", "$sources.temp.label", SourcePort("temp", "label")
                ),
                PlannedWorkflowOutput(
                    "a", "$sources.temp.label", SourcePort("temp", "label")
                ),
            ),
        )


def test_describe_is_json_friendly_and_includes_sources_groups_and_domains() -> None:
    plan = compile_active(
        [step("scale", "scale", value="$sources.temp.temperature")],
        [group("temps", "$sources.temp.temperature", t="$steps.scale.scaled")],
    )

    description = json.loads(json.dumps(plan.describe()))

    assert description["sources"]["temp"]["type"] == "test/temperature@v1"
    assert description["output_groups"] == {
        "temps": {
            "anchor": "$sources.temp.temperature",
            "outputs": {
                "t": {
                    "selector": "$steps.scale.scaled",
                    "source": "$steps.scale.scaled",
                    "options": {},
                }
            },
            "dependencies": ["$steps.scale"],
        }
    }
    assert description["steps"][0]["domain"] == "temp"
    assert (
        description["steps"][0]["bindings"][0]["source"] == "$sources.temp.temperature"
    )


def test_pulse_key_identifies_run_source_and_sequence() -> None:
    key = PulseKey(active_run_id="run-1", source="temp", sequence=0)

    assert key.lineage_id == "run:run-1/source:temp"
    assert key.pulse_id == 0
    assert key.run_id == "run-1:temp:0"
    assert PulseKey("run-2", "temp", 0) != key
    for kwargs in (
        {"active_run_id": "", "source": "temp", "sequence": 0},
        {"active_run_id": "r", "source": "", "sequence": 0},
        {"active_run_id": "r", "source": "temp", "sequence": -1},
        {"active_run_id": "r", "source": "temp", "sequence": True},
    ):
        with pytest.raises(ContractError):
            PulseKey(**kwargs)


# --- session creation --------------------------------------------------------


def test_create_session_resolves_source_resources_without_constructing_sources() -> (
    None
):
    plan = compile_active([step("scale", "scale", value="$sources.temp.temperature")])

    session = plan.create_session(resources={"test.clock": "monotonic"})

    assert dict(session.source_resources) == {
        "temp": MappingProxyType({"clock": session.source_resources["temp"]["clock"]}),
        "cam": MappingProxyType({}),
    }
    resolved = session.source_resources["temp"]["clock"]
    assert (resolved.value, resolved.source) == ("monotonic", "provided:test.clock")
    assert list(session.instances) == [("scale",)]
    with pytest.raises(WorkflowInputError, match="declares sources .*session.start"):
        session.run({})


def test_create_session_uses_catalogue_providers_and_reports_missing_resources() -> (
    None
):
    plan = compile_active([])
    calls = []

    session = plan.create_session(
        resources={"clock": Factory(lambda: calls.append("made") or "ticking")}
    )
    assert session.source_resources["temp"]["clock"].value == "ticking" and calls == [
        "made"
    ]
    assert plan.create_session().source_resources["temp"]["clock"].source == (
        "catalogue:test.clock"
    )

    bare = Catalogue(ALL_BLOCKS, sources=[Temperature, Camera], namespace="test")
    with pytest.raises(ResourceError, match=r"no value for required resource 'clock'"):
        compile_workflow(active([]), catalogue=bare).create_session()


# --- group readiness ---------------------------------------------------------


def test_groups_record_the_steps_their_fields_wait_for() -> None:
    child = _child(
        [step("echo", "echo", value="$inputs.x")],
        {"out": "$steps.echo.value", "fwd": "$inputs.x", "fixed": "$inputs.k"},
        inputs=(
            {"type": "WorkflowParameter", "name": "x"},
            {"type": "WorkflowParameter", "name": "k", "default_value": 7},
        ),
    )
    plan = compile_active(
        [
            step("scale", "scale", value="$sources.temp.temperature"),
            step("scale", "again", value="$steps.scale.scaled"),
            step("expand", "expand", value="$sources.cam.image"),
            gate("g", "$sources.cam.image", ["child"]),
            nested(
                "child",
                workflow_definition=child,
                parameter_bindings={"x": "$sources.cam.image"},
            ),
        ],
        [
            group("temps", "$sources.temp.temperature", t="$steps.again.scaled"),
            group("raw", "$sources.temp.label", label="$sources.temp.label"),
            group("frames", "$sources.cam.image", all="$steps.expand.*"),
            group("forwarded", "$sources.cam.image", fwd="$steps.child.fwd"),
            group("constant", "$sources.cam.image", fixed="$steps.child.fixed"),
            group("inner", "$sources.cam.image", out="$steps.child.out"),
        ],
    )

    dependencies = {item.name: item.dependencies for item in plan.output_groups}
    assert dependencies == {
        "temps": (("again",),),
        "raw": (),
        "frames": (("expand",),),
        "forwarded": (("g",),),
        "constant": (("g",),),
        "inner": (("child", "echo"),),
    }
    assert plan.describe()["output_groups"]["forwarded"]["dependencies"] == ["$steps.g"]


def test_group_dependencies_follow_plan_order_and_are_checked() -> None:
    plan = compile_active(
        [
            step("scale", "b", value="$sources.temp.temperature"),
            step("scale", "a", value="$sources.temp.temperature"),
        ],
        [
            group(
                "g",
                "$sources.temp.temperature",
                x="$steps.a.scaled",
                y="$steps.b.scaled",
            )
        ],
    )
    [planned_group] = plan.output_groups

    assert planned_group.dependencies == (("b",), ("a",))
    with pytest.raises(ContractError, match="records dependencies .* but its fields"):
        _hand_built(
            steps=plan.steps,
            output_groups=(
                PlannedOutputGroup(
                    "g", planned_group.anchor, planned_group.outputs, dependencies=()
                ),
            ),
        )


# --- session stop ------------------------------------------------------------


def test_session_stop_targets_the_current_active_run(monkeypatch) -> None:
    import roboflow_workflows.execution_engine.v2.active.runtime as runtime

    stopped = []
    monkeypatch.setattr(runtime, "stop_session", stopped.append, raising=False)
    session = compile_active([]).create_session()

    session.stop()
    session.stop()

    assert stopped == [session, session]
    with pytest.raises(WorkflowInputError, match="declares no sources"):
        compile_workflow(
            {
                "version": "2.0",
                "inputs": [],
                "steps": [step("constant", "k")],
                "outputs": [],
            },
            catalogue=CATALOGUE,
        ).create_session().stop()


# --- review corrections C1 and C2 --------------------------------------------


def test_unknown_source_port_in_a_selection_is_a_selector_error() -> None:
    with pytest.raises(
        SelectorError,
        match=r"outputs\[0\]\.outputs\[0\]: \$sources\.temp has no output 'nope'",
    ):
        compile_active([], [group("g", "$sources.temp.label", v="$sources.temp.nope")])
    with pytest.raises(SelectorError, match=r"\$sources\.temp has no output 'nope'"):
        compile_active([], [group("g", "$sources.temp.nope", v="$sources.temp.label")])

    forwarded = _child([step("constant", "k")], {"v": "$inputs.x"})
    with pytest.raises(KindMismatchError, match="not an output of \\$sources\\.temp"):
        compile_active(
            [
                nested(
                    "child",
                    workflow_definition=forwarded,
                    parameter_bindings={"x": "$sources.temp.nope"},
                )
            ],
            [group("g", "$sources.temp.label", v="$steps.child.v")],
        )


def test_group_field_joining_two_sources_through_a_gated_forward_is_a_lineage_error() -> (
    None
):
    forwarded = _child([step("constant", "k")], {"v": "$inputs.x"})

    with pytest.raises(
        LineageError, match=r"field 'v' \(\$steps\.child\.v\) joins independent sources"
    ):
        compile_active(
            [
                gate("g", "$sources.cam.image", ["child"]),
                nested(
                    "child",
                    workflow_definition=forwarded,
                    parameter_bindings={"x": "$sources.temp.temperature"},
                ),
            ],
            [group("out", "$sources.cam.image", v="$steps.child.v")],
        )


def test_hand_built_source_records_are_checked_against_their_declarations() -> None:
    plan = compile_active(
        [],
        sources=[{**TEMPERATURE, "probe": "$inputs.probe"}],
        inputs=[parameter("probe", default="p", kind=["string"])],
    )
    temp = plan.source("temp")
    [binding] = temp.bindings
    port = temp.outputs["temperature"]

    def rebuilt(**changes):
        return CompiledWorkflow(
            inputs=plan.inputs,
            steps=(),
            outputs=(),
            catalogue=plan.catalogue,
            sources={"temp": replace(temp, **changes)},
        )

    with pytest.raises(
        ContractError, match="recorded under key 'temp' but names itself 'other'"
    ):
        rebuilt(name="other")
    with pytest.raises(
        ContractError,
        match="records bindings at .* but its parameters select exactly once at",
    ):
        rebuilt(bindings=())
    with pytest.raises(
        ContractError, match=r"selects \'\$inputs\.probe\' but reads \$inputs\.missing"
    ):
        rebuilt(bindings=(replace(binding, source=InputPort("missing")),))
    with pytest.raises(ContractError, match="records selector '\\$inputs\\.other'"):
        rebuilt(bindings=(replace(binding, selector="$inputs.other"),))
    with pytest.raises(ContractError, match="records outputs \\['temperature'\\], but"):
        rebuilt(outputs={"temperature": port})
    with pytest.raises(
        ContractError, match="output 'temperature' records .* scoped to this source"
    ):
        rebuilt(
            outputs={**temp.outputs, "temperature": replace(port, kinds=("string",))}
        )
    with pytest.raises(ContractError, match="output 'temperature' records"):
        rebuilt(
            outputs={
                **temp.outputs,
                "temperature": replace(
                    port, layout=EntryLayout((Axis("sources.temp:t", "time"),))
                ),
            }
        )
    with pytest.raises(ContractError, match="accepts only ungrouped inputs"):
        CompiledWorkflow(
            inputs={
                "probe": PlannedInput(
                    name="probe",
                    kinds=("string",),
                    layout=EntryLayout((Axis("p", "sample"),)),
                )
            },
            steps=(),
            outputs=(),
            catalogue=plan.catalogue,
            sources={"temp": temp},
        )


# --- source plan invariants -------------------------------------------------


def test_source_bindings_must_read_the_input_their_selector_names() -> None:
    plan = compile_active(
        [],
        sources=[{**TEMPERATURE, "probe": "$inputs.probe"}],
        inputs=[
            parameter("probe", default="p", kind=["string"]),
            parameter("other", default="q", kind=["string"]),
        ],
    )
    temp = plan.source("temp")
    [binding] = temp.bindings

    def rebuilt(bindings):
        return CompiledWorkflow(
            inputs=plan.inputs,
            steps=(),
            outputs=(),
            catalogue=plan.catalogue,
            sources={"temp": replace(temp, bindings=bindings)},
        )

    with pytest.raises(
        ContractError,
        match=r"selects '\$inputs\.probe' but reads \$inputs\.other",
    ):
        rebuilt((replace(binding, source=InputPort("other")),))
    with pytest.raises(ContractError, match="select exactly once at"):
        rebuilt((binding, binding))
    assert rebuilt((binding,)).source("temp").bindings == (binding,)
    unknown = temp.spec.validate_params(
        {"path": "temps.csv", "probe": "$inputs.missing"}, source_name="temp"
    )
    with pytest.raises(ContractError, match="reads unknown workflow input 'missing'"):
        CompiledWorkflow(
            inputs=plan.inputs,
            steps=(),
            outputs=(),
            catalogue=plan.catalogue,
            sources={
                "temp": replace(
                    temp,
                    params=unknown,
                    bindings=(
                        replace(
                            binding,
                            selector="$inputs.missing",
                            source=InputPort("missing"),
                        ),
                    ),
                )
            },
        )


def test_output_groups_are_anchored_on_source_ports_only() -> None:
    field = PlannedWorkflowOutput("x", "$inputs.p", InputPort("p"))

    with pytest.raises(
        ContractError, match=r"anchored on a source port, got \$inputs\.p"
    ):
        PlannedOutputGroup("g", InputPort("p"), (field,))
    with pytest.raises(ContractError, match="anchored on a source port"):
        PlannedOutputGroup("g", StepPort(("s",), "out"), ())


@pytest.mark.parametrize("selector", ["$steps.probe.value", "$sources.probe.value"])
def test_hand_built_source_bindings_reject_non_input_selector_namespaces(
    selector,
) -> None:
    plan = compile_active(
        [],
        sources=[{**TEMPERATURE, "probe": "$inputs.probe"}],
        inputs=[parameter("probe", default="p", kind=["string"])],
    )
    source = plan.source("temp")
    [binding] = source.bindings
    params = source.spec.validate_params(
        {"path": "temps.csv", "probe": selector}, source_name="temp"
    )
    with pytest.raises(ContractError, match="must select a static workflow input"):
        CompiledWorkflow(
            inputs=plan.inputs,
            steps=(),
            outputs=(),
            catalogue=plan.catalogue,
            sources={
                "temp": replace(
                    source,
                    params=params,
                    bindings=(replace(binding, selector=selector),),
                )
            },
        )
