"""Nested workflow composition: aliases, bindings, references, limits, dynamics."""

import copy

import pytest
from roboflow_workflows.execution_engine.v2.compilation import (
    WorkflowReference,
    compile_workflow,
)
from roboflow_workflows.execution_engine.v2.compilation.composition import (
    compose_workflow,
)
from roboflow_workflows.execution_engine.v2.errors import (
    CycleError,
    KindMismatchError,
    LineageError,
    NestedWorkflowError,
    ParamsValidationError,
    SelectorError,
)
from roboflow_workflows.execution_engine.v2.plan import (
    ChildInputPort,
    CompileOptions,
    Constant,
    InputPort,
    StepPort,
)

from tests.unit_tests.execution_engine.v2.compilation.fixture_blocks import (
    CATALOGUE,
    axes_of,
    batch_input,
    gate,
    nested,
    origin_of,
    parameter,
    step,
    workflow,
)


def scaling_child(*, default=100):
    return workflow(
        [step("scale", "inner_scale", value="$inputs.x", factor="$inputs.k")],
        {"y": "$steps.inner_scale.scaled"},
        inputs=[batch_input("x", kind=["float"]), parameter("k", default=default)],
    )


ECHO_AND_NOTICE = workflow(
    [
        step("echo", "echo", value="$inputs.message"),
        step("sink", "notice", payload="nested alert"),
    ],
    {"message": "$steps.echo.value"},
    inputs=[parameter("message", default="child default")],
)


class RecordingResolver:
    """Local saved-workflow store that records every call."""

    def __init__(self, saved):
        self.saved = saved
        self.calls = []

    def __call__(self, reference: WorkflowReference):
        self.calls.append(reference)
        return self.saved[reference.workflow_id]


def saved(name, workflow_id, **fields):
    return nested(
        name, workflow_workspace_id="local", workflow_id=workflow_id, **fields
    )


def _compile(definition, **kwargs):
    plan = compile_workflow(definition, catalogue=CATALOGUE, **kwargs)

    return plan


def test_child_inputs_alias_parent_sources_and_keep_parent_lineage() -> None:
    definition = workflow(
        [
            step("expand", "expand", value="$inputs.values"),
            nested(
                "child",
                workflow_definition=scaling_child(),
                parameter_bindings={"x": "$steps.expand.child"},
            ),
            step("scale", "after_child", value="$steps.child.y", factor=1),
        ],
        {"child_y": "$steps.child.y"},
    )

    plan = _compile(definition)

    assert [step.path for step in plan.steps] == [
        ("expand",),
        ("child", "inner_scale"),
        ("after_child",),
    ]
    inner = plan.step(("child", "inner_scale"))
    value, factor = inner.binding_for("value"), inner.binding_for("factor")
    assert value.source == ChildInputPort(("child",), "x")
    assert factor.source == ChildInputPort(("child",), "k")
    x, k = plan.child_input(value.source), plan.child_input(factor.source)
    assert (x.source, x.kinds, axes_of(x.layout)) == (
        StepPort(("expand",), "child"),
        ("float",),
        ["inputs", "expand:child"],
    )
    assert (k.source, axes_of(k.layout)) == (Constant(100), [])
    assert axes_of(value.source_layout) == ["inputs", "expand:child"]
    assert inner.params.value == "$inputs.x", "selector text stays scope-local"
    assert axes_of(inner.invocation_layout) == ["inputs", "expand:child"]
    after = plan.step(("after_child",))
    assert after.binding_for("value").source == StepPort(
        ("child", "inner_scale"), "scaled"
    )
    assert plan.outputs[0].source == StepPort(("child", "inner_scale"), "scaled")


def test_literal_bindings_and_defaults_become_constants() -> None:
    plan = _compile(
        workflow(
            [
                nested(
                    "child",
                    workflow_definition=scaling_child(),
                    parameter_bindings={"x": "$inputs.values", "k": 3},
                )
            ],
        )
    )

    inner = plan.step(("child", "inner_scale"))
    assert origin_of(plan, inner.binding_for("factor").source) == Constant(3)
    assert origin_of(plan, inner.binding_for("value").source) == InputPort("values")


@pytest.mark.parametrize(
    "bindings, error, text",
    [
        pytest.param(
            {"k": 3},
            NestedWorkflowError,
            "miss required child inputs ['x']",
            id="missing",
        ),
        pytest.param(
            {"x": "$inputs.values", "z": 1},
            NestedWorkflowError,
            "unknown child inputs ['z']",
            id="unknown",
        ),
        pytest.param(
            {"x": "$inputs.values\n"},
            SelectorError,
            "malformed selector",
            id="malformed",
        ),
        pytest.param(
            {"x": "$inputs.nope"},
            SelectorError,
            "unknown workflow input 'nope'",
            id="unknown-parent-input",
        ),
    ],
)
def test_invalid_parameter_bindings_are_rejected(bindings, error, text) -> None:
    definition = workflow(
        [
            nested(
                "child",
                workflow_definition=scaling_child(),
                parameter_bindings=bindings,
            )
        ]
    )

    with pytest.raises(error) as info:
        _compile(definition)

    assert text in str(info.value)
    assert info.value.step_path == ("child",)


def test_binding_kinds_and_child_defaults_are_checked() -> None:
    string_child = workflow(
        [step("echo", "e", value="$inputs.x")],
        {"y": "$steps.e.value"},
        inputs=[batch_input("x", kind=["string"])],
    )

    with pytest.raises(KindMismatchError) as kinds:
        _compile(
            workflow(
                [
                    nested(
                        "child",
                        workflow_definition=string_child,
                        parameter_bindings={"x": "$inputs.values"},
                    )
                ]
            )
        )
    with pytest.raises(ParamsValidationError) as default:
        _compile(
            workflow(
                [
                    nested(
                        "child",
                        workflow_definition=scaling_child(default=-1),
                        parameter_bindings={"x": "$inputs.values"},
                    )
                ]
            )
        )

    assert kinds.value.step_path == ("child",)
    assert kinds.value.field_path == ("parameter_bindings", "x")
    assert default.value.step_path == ("child", "inner_scale")
    assert default.value.field_path == ("factor",)


def test_child_outputs_by_name_wildcards_and_unknown_names() -> None:
    wildcard_child = workflow(
        [step("parse_fields", "parse", raw="x", expected_fields=["a", "b"])],
        {"everything": "$steps.parse.*"},
        inputs=[],
    )
    base = [nested("child", workflow_definition=wildcard_child)]

    plan = _compile(workflow(base, {"all": "$steps.child.everything"}, inputs=[]))
    with pytest.raises(SelectorError) as unknown:
        _compile(workflow(base, {"x": "$steps.child.missing"}, inputs=[]))
    with pytest.raises(SelectorError) as star:
        _compile(workflow(base, {"x": "$steps.child.*"}, inputs=[]))
    with pytest.raises(SelectorError) as into_step:
        _compile(
            workflow(
                base + [step("echo", "e", value="$steps.child.everything")], inputs=[]
            )
        )

    assert plan.outputs[0].source == StepPort(("child", "parse"), "*")
    assert "has no output 'missing'; its outputs are ['everything']" in str(
        unknown.value
    )
    assert "nested workflow $steps.child" in str(unknown.value)
    assert unknown.value.field_path == ("outputs", "x")
    assert "select its outputs ['everything'] by name" in str(star.value)
    assert "only valid in workflow outputs" in str(into_step.value)


def test_repeated_children_get_independent_paths_and_axes() -> None:
    expanding = workflow(
        [step("expand", "expand", value="$inputs.x")],
        {"child": "$steps.expand.child"},
        inputs=[batch_input("x", kind=["float"])],
    )
    definition = workflow(
        [
            step("echo", "echo", value="$inputs.values"),
            nested("first", workflow_definition=ECHO_AND_NOTICE),
            nested("second", workflow_definition=ECHO_AND_NOTICE),
            nested(
                "left",
                workflow_definition=expanding,
                parameter_bindings={"x": "$inputs.values"},
            ),
            nested(
                "right",
                workflow_definition=expanding,
                parameter_bindings={"x": "$inputs.values"},
            ),
            step(
                "scale", "join", value="$steps.left.child", factor="$steps.left.child"
            ),
        ],
    )

    plan = _compile(definition)

    paths = [step.path for step in plan.steps]
    assert paths[:5] == [
        ("echo",),
        ("first", "echo"),
        ("first", "notice"),
        ("second", "echo"),
        ("second", "notice"),
    ]
    left = plan.step(("left", "expand")).outputs["child"].layout
    right = plan.step(("right", "expand")).outputs["child"].layout
    assert axes_of(left) == ["inputs", "left/expand:child"]
    assert axes_of(right) == ["inputs", "right/expand:child"]
    with pytest.raises(LineageError, match="never by size"):
        _compile(
            workflow(
                definition["steps"][:5]
                + [
                    step(
                        "scale",
                        "bad",
                        value="$steps.left.child",
                        factor="$steps.right.child",
                    )
                ]
            )
        )


def test_saved_references_are_fetched_once_and_copied_per_use() -> None:
    store = {"saved_child": ECHO_AND_NOTICE}
    snapshot = copy.deepcopy(store)
    resolver = RecordingResolver(store)
    definition = workflow(
        [
            saved("first", "saved_child"),
            saved("second", "saved_child"),
            saved("pinned", "saved_child", workflow_version_id="3"),
        ],
        {"first": "$steps.first.message", "second": "$steps.second.message"},
        inputs=[],
    )

    plan = _compile(definition, reference_resolver=resolver)

    assert resolver.calls == [
        WorkflowReference("local", "saved_child"),
        WorkflowReference("local", "saved_child", "3"),
    ]
    assert store == snapshot
    assert {step.path for step in plan.steps} >= {("first", "echo"), ("second", "echo")}
    first, second = (plan.step((name, "echo")) for name in ("first", "second"))
    assert first.binding_for("value").source is not second.binding_for("value").source


def test_saved_diamond_is_valid_and_cycles_name_the_reference_chain() -> None:
    leaf = workflow([step("sink", "s")], inputs=[])
    left = workflow([saved("leaf", "leaf")], inputs=[])
    right = workflow([saved("leaf", "leaf")], inputs=[])
    resolver = RecordingResolver({"leaf": leaf, "left": left, "right": right})

    plan = _compile(
        workflow([saved("a", "left"), saved("b", "right")], inputs=[]),
        reference_resolver=resolver,
    )
    self_cycle = RecordingResolver(
        {"loop": workflow([saved("again", "loop")], inputs=[])}
    )
    mutual = RecordingResolver(
        {
            "ping": workflow([saved("to_pong", "pong")], inputs=[]),
            "pong": workflow([saved("to_ping", "ping")], inputs=[]),
        }
    )

    assert [step.path for step in plan.steps] == [
        ("a", "leaf", "s"),
        ("b", "leaf", "s"),
    ]
    assert [call.workflow_id for call in resolver.calls] == ["left", "leaf", "right"]
    with pytest.raises(CycleError) as single:
        _compile(
            workflow([saved("root", "loop")], inputs=[]), reference_resolver=self_cycle
        )
    with pytest.raises(CycleError) as double:
        _compile(
            workflow([saved("root", "ping")], inputs=[]), reference_resolver=mutual
        )
    assert "local/loop -> local/loop" in str(single.value)
    assert "local/ping -> local/pong -> local/ping" in str(double.value)


def test_depth_and_count_limits_are_exact() -> None:
    leaf = workflow([step("sink", "s")], inputs=[])
    middle = workflow([nested("leaf", workflow_definition=leaf)], inputs=[])
    two_deep = workflow([nested("middle", workflow_definition=middle)], inputs=[])
    three_uses = workflow(
        [nested(name, workflow_definition=leaf) for name in "abc"], inputs=[]
    )

    _compile(two_deep, options=CompileOptions(max_nested_depth=2))
    _compile(three_uses, options=CompileOptions(max_nested_count=3))
    with pytest.raises(NestedWorkflowError, match="depth 2 exceeds the limit of 1"):
        _compile(two_deep, options=CompileOptions(max_nested_depth=1))
    with pytest.raises(
        NestedWorkflowError, match="3 nested workflow steps exceed the limit of 2"
    ):
        _compile(three_uses, options=CompileOptions(max_nested_count=2))


def test_reference_problems_are_nested_workflow_errors() -> None:
    definition = workflow([saved("child", "missing")], inputs=[])

    def failing(reference):
        raise KeyError(reference.workflow_id)

    with pytest.raises(NestedWorkflowError, match="received no reference_resolver"):
        _compile(definition)
    with pytest.raises(
        NestedWorkflowError, match="resolving saved workflow local/missing failed"
    ) as info:
        _compile(definition, reference_resolver=failing)
    with pytest.raises(NestedWorkflowError, match="nested workflow has no steps"):
        _compile(
            workflow(
                [nested("child", workflow_definition=workflow([], inputs=[]))],
                inputs=[],
            )
        )
    assert isinstance(info.value.__cause__, KeyError)


def test_child_within_child_bound_to_an_upstream_parent_step() -> None:
    middle = workflow(
        [
            nested(
                "leaf",
                workflow_definition=scaling_child(),
                parameter_bindings={"x": "$inputs.x"},
            )
        ],
        {"y": "$steps.leaf.y"},
        inputs=[batch_input("x", kind=["float"])],
    )
    definition = workflow(
        [
            step("scale", "double", value="$inputs.values", factor=2),
            step("echo", "leaf", value="$inputs.values"),
            nested(
                "middle",
                workflow_definition=middle,
                parameter_bindings={"x": "$steps.double.scaled"},
            ),
        ],
        {"y": "$steps.middle.y"},
    )

    plan = _compile(definition)

    inner = plan.step(("middle", "leaf", "inner_scale"))
    value = inner.binding_for("value").source
    assert value == ChildInputPort(("middle", "leaf"), "x")
    assert plan.child_input(value).source == ChildInputPort(("middle",), "x")
    assert origin_of(plan, value) == StepPort(("double",), "scaled")
    assert origin_of(plan, inner.binding_for("factor").source) == Constant(100)
    scopes = [(item.scope, item.name) for item in plan.child_inputs]
    assert scopes.index((("middle",), "x")) < scopes.index((("middle", "leaf"), "x"))
    assert plan.step(("leaf",)).path != inner.path, "tuple paths cannot collide"


def test_grouped_parent_input_keeps_its_depth_in_a_reused_child() -> None:
    gated_label = workflow(
        [
            gate("gate", "$inputs.values", ["echo"]),
            step("echo", "echo", value="$inputs.label"),
        ],
        {"result": "$steps.echo.value"},
        inputs=[batch_input("values"), parameter("label", default="default")],
    )
    definition = workflow(
        [
            nested(
                name,
                workflow_definition=gated_label,
                parameter_bindings={"values": "$inputs.values", "label": name},
            )
            for name in ("left", "right")
        ],
        inputs=[batch_input("values", depth=2)],
    )

    plan = _compile(definition)

    for name in ("left", "right"):
        echo = plan.step((name, "echo"))
        assert axes_of(echo.invocation_layout) == ["inputs", "inputs.values:1"]
        assert origin_of(plan, echo.binding_for("value").source) == Constant(name)
        assert [gate.controller for gate in echo.gates] == [(name, "gate")]


def test_alias_cycles_through_child_outputs_are_rejected() -> None:
    passthrough = workflow(
        [step("sink", "s")], {"y": "$inputs.x"}, inputs=[parameter("x")]
    )
    definition = workflow(
        [
            nested(
                "child",
                workflow_definition=passthrough,
                parameter_bindings={"x": "$steps.child.y"},
            )
        ],
        {"y": "$steps.child.y"},
        inputs=[],
    )

    with pytest.raises(CycleError, match="refers back to itself"):
        _compile(definition)


def test_dynamic_definitions_are_collected_root_first_with_first_wins() -> None:
    def dynamic(block_type, marker):
        return {
            "type": "DynamicBlockDefinition",
            "manifest": {"block_type": block_type, "marker": marker},
        }

    child = workflow(
        [step("sink", "s")],
        inputs=[],
        dynamic_blocks=[dynamic("B", 1), dynamic("A", "child")],
    )
    same = workflow(
        [step("sink", "s")], inputs=[], dynamic_blocks=[dynamic("A", "root")]
    )
    definition = workflow(
        [
            nested("first", workflow_definition=child),
            nested("second", workflow_definition=same),
        ],
        inputs=[],
        dynamic_blocks=[dynamic("A", "root"), {"manifest": {}}],
    )

    composition = compose_workflow(
        definition, options=CompileOptions(), reference_resolver=None
    )

    kept = [
        (item.definition.get("manifest", {}).get("block_type"), item.location)
        for item in composition.dynamic_definitions
    ]
    assert kept == [
        ("A", "dynamic_blocks_definitions[0]"),
        (None, "dynamic_blocks_definitions[1]"),
        ("B", "steps[0].workflow_definition.dynamic_blocks_definitions[0]"),
    ]
    assert composition.warnings == (
        "dynamic block definition at steps[0].workflow_definition."
        "dynamic_blocks_definitions[1] redefines the block type defined at "
        "dynamic_blocks_definitions[0]; the first definition is used",
    )


def _integer_child(*, default=None, steps=None, outputs=None):
    return workflow(
        steps or [step("echo", "echo", value="$inputs.value")],
        outputs if outputs is not None else {"value": "$steps.echo.value"},
        inputs=[parameter("value", default=default, kind=["integer"])],
    )


@pytest.mark.parametrize(
    "child, bindings, field_path",
    [
        pytest.param(
            _integer_child(),
            {"value": "wrong-type"},
            ("parameter_bindings", "value"),
            id="literal-binding",
        ),
        pytest.param(
            _integer_child(default="wrong-type"), {}, ("inputs", "value"), id="default"
        ),
        pytest.param(
            _integer_child(
                steps=[step("sink", "s")], outputs={"value": "$inputs.value"}
            ),
            {"value": "wrong-type"},
            ("parameter_bindings", "value"),
            id="passed-through-to-an-output",
        ),
        pytest.param(
            _integer_child(steps=[step("sink", "s")], outputs={}),
            {"value": "wrong-type"},
            ("parameter_bindings", "value"),
            id="unused-input",
        ),
    ],
)
def test_nested_constants_must_fit_the_declared_child_input_kinds(
    child, bindings, field_path
) -> None:
    definition = workflow(
        [nested("child", workflow_definition=child, parameter_bindings=bindings)],
        inputs=[],
    )

    with pytest.raises(KindMismatchError) as info:
        _compile(definition)

    assert info.value.step_path == ("child",)
    assert info.value.field_path == field_path
    assert "accepts kinds ['integer']" in str(info.value)
    assert "'wrong-type'" in str(info.value)


def test_nested_constants_are_checked_at_every_alias_level_and_by_consumers() -> None:
    middle = workflow(
        [
            nested(
                "leaf",
                workflow_definition=_integer_child(),
                parameter_bindings={"value": "$inputs.m"},
            )
        ],
        inputs=[parameter("m")],
    )
    to_float_field = workflow(
        [step("scale", "s", value="$inputs.anything")],
        inputs=[parameter("anything")],
    )

    with pytest.raises(KindMismatchError) as inner:
        _compile(
            workflow(
                [
                    nested(
                        "middle",
                        workflow_definition=middle,
                        parameter_bindings={"m": "text"},
                    )
                ],
                inputs=[],
            )
        )
    with pytest.raises(KindMismatchError) as consumer:
        _compile(
            workflow(
                [
                    nested(
                        "child",
                        workflow_definition=to_float_field,
                        parameter_bindings={"anything": "text"},
                    )
                ],
                inputs=[],
            )
        )
    valid = _compile(
        workflow(
            [nested("middle", workflow_definition=middle, parameter_bindings={"m": 7})],
            inputs=[],
        )
    )

    assert inner.value.step_path == ("middle", "leaf")
    assert inner.value.field_path == ("parameter_bindings", "value")
    assert consumer.value.step_path == ("child", "s")
    assert consumer.value.field_path == ("value",)
    leaf = valid.step(("middle", "leaf", "echo")).binding_for("value").source
    assert origin_of(valid, leaf) == Constant(7)


def test_constants_for_deserialized_kinds_are_left_to_run_time() -> None:
    from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
    from roboflow_workflows.execution_engine.v2.declaration import (
        Block,
        BlockParams,
        Output,
        Ref,
    )
    from roboflow_workflows.execution_engine.v2.kinds import Kind

    decoded = []
    encoded = Kind(
        name="encoded",
        validate=lambda payload: isinstance(payload, bytes),
        deserialize=lambda value: decoded.append(value) or value.encode(),
    )

    class Decode(Block):
        type = "test/decode@v1"
        outputs = {"size": Output()}

        class Params(BlockParams):
            data: Ref(encoded)

        def run(self, *, data) -> dict:
            return {"size": len(data)}

    child = workflow(
        [{"type": "test/decode@v1", "name": "decode", "data": "$inputs.blob"}],
        {"size": "$steps.decode.size"},
        inputs=[parameter("blob", kind=["encoded"])],
    )
    definition = workflow(
        [
            nested(
                "child", workflow_definition=child, parameter_bindings={"blob": "abc"}
            )
        ],
        inputs=[],
    )

    plan = compile_workflow(
        definition, catalogue=Catalogue.merge(CATALOGUE, Catalogue([Decode]))
    )

    port = plan.step(("child", "decode")).binding_for("data").source
    assert plan.child_input(port).kinds == ("encoded",)
    assert plan.child_input(port).source == Constant("abc"), "decoded at run time"
    assert decoded == [], "compilation never calls a deserializer"
