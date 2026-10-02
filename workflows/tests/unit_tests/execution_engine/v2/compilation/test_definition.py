"""Definition structure, input shorthand layouts, names and whole-string checks."""

import copy

import pytest
from roboflow_workflows.execution_engine.v2.compilation import (
    ROOT_AXIS,
    compile_workflow,
)
from roboflow_workflows.execution_engine.v2.errors import (
    ParamsValidationError,
    SelectorError,
    UnknownBlockError,
    WorkflowCompileError,
)

from tests.unit_tests.execution_engine.v2.compilation.fixture_blocks import (
    CATALOGUE,
    axes_of,
    batch_input,
    nested,
    parameter,
    step,
    workflow,
)


def test_v1_style_inputs_map_to_shared_root_and_private_nested_axes() -> None:
    definition = workflow(
        [],
        inputs=[
            {"type": "WorkflowImage", "name": "image"},
            batch_input("values", kind=["float"]),
            batch_input("tiles", depth=3),
            batch_input("others", depth=2),
            parameter("factor", default=2, kind=["float"]),
            parameter("label"),
        ],
    )

    plan = compile_workflow(definition, catalogue=_with_image_kind())

    layouts = {name: axes_of(item.layout) for name, item in plan.inputs.items()}
    assert layouts == {
        "image": ["inputs"],
        "values": ["inputs"],
        "tiles": ["inputs", "inputs.tiles:1", "inputs.tiles:2"],
        "others": ["inputs", "inputs.others:1"],
        "factor": [],
        "label": [],
    }
    assert plan.inputs["image"].kinds == ("image",)
    assert plan.inputs["tiles"].kinds == ("*",)
    assert plan.inputs["image"].layout.axes[0] == ROOT_AXIS
    assert plan.inputs["factor"].required is False
    assert plan.inputs["factor"].default == 2
    assert plan.inputs["label"].required is False, "V1: omitted parameter is None"
    assert plan.inputs["values"].required is True
    assert plan.axis_origin("inputs.tiles:2").name == "tiles"


def test_explicit_axes_keep_independent_lineages_and_reject_inconsistent_reuse() -> (
    None
):
    definition = workflow(
        [],
        inputs=[
            {"name": "a", "kind": "float", "axes": [{"id": "cams", "kind": "sample"}]},
            {
                "name": "b",
                "kind": ["float"],
                "axes": [
                    {"id": "cams", "kind": "sample"},
                    {"id": "tiles", "kind": "static_nesting"},
                ],
            },
            {"name": "c", "axes": []},
        ],
    )

    plan = compile_workflow(definition, catalogue=CATALOGUE)

    assert axes_of(plan.inputs["b"].layout) == ["cams", "tiles"]
    assert plan.inputs["b"].layout.axes[1].stationary is True
    assert plan.inputs["c"].required is True
    inconsistent = copy.deepcopy(definition)
    inconsistent["inputs"][1]["axes"][0]["stationary"] = True
    with pytest.raises(WorkflowCompileError, match="axis 'cams' differs"):
        compile_workflow(inconsistent, catalogue=CATALOGUE)


@pytest.mark.parametrize(
    "definition, expected",
    [
        pytest.param(
            {**workflow([]), "version": "1.0"},
            "version must be '2.0'",
            id="v1-version",
        ),
        pytest.param(
            {**workflow([]), "parameters": []},
            "unsupported keys ['parameters']",
            id="unknown-section",
        ),
        pytest.param(
            {**workflow([]), "steps": {}},
            "steps must be a list",
            id="steps-not-list",
        ),
        pytest.param(
            workflow([], inputs=[parameter("x"), parameter("x")]),
            "duplicate workflow input 'x'",
            id="duplicate-input",
        ),
        pytest.param(
            workflow([step("sink", "a"), step("sink", "a")]),
            "duplicate step name 'a'",
            id="duplicate-step",
        ),
        pytest.param(
            workflow([], inputs=[{"type": "WorkflowCamera", "name": "x"}]),
            "unknown input type 'WorkflowCamera'",
            id="unknown-input-type",
        ),
        pytest.param(
            workflow([], inputs=[batch_input("x", kind=["pixels"])]),
            "unknown kinds ['pixels']",
            id="unknown-kind",
        ),
        pytest.param(
            workflow([], inputs=[{**batch_input("x"), "dimensionality": 0}]),
            "dimensionality must be an integer of at least 1",
            id="batch-depth-zero",
        ),
        pytest.param(
            workflow(
                [],
                inputs=[
                    {
                        "name": "x",
                        "axes": [
                            {"id": "n", "kind": "sample"},
                            {"id": "t", "kind": "time"},
                        ],
                    }
                ],
            ),
            "must be stationary",
            id="nonstationary-parent-before-time",
        ),
        pytest.param(
            workflow(
                [
                    nested(
                        "child",
                        workflow_definition=workflow([step("sink", "s")]),
                        execution_mode="remote_dispatch",
                    )
                ]
            ),
            "only embeds child workflows",
            id="remote-dispatch",
        ),
        pytest.param(
            workflow([nested("child")]),
            "needs exactly one of a non-empty workflow_definition",
            id="nested-without-source",
        ),
        pytest.param(
            workflow(
                [
                    nested(
                        "child",
                        workflow_definition=workflow([step("sink", "s")]),
                        workflow_workspace_id="local",
                        workflow_id="saved",
                    )
                ]
            ),
            "needs exactly one of a non-empty workflow_definition",
            id="nested-with-both-sources",
        ),
        pytest.param(
            workflow([{"type": "test/sink@v1", "name": "s", "outputs": 1}], {}),
            "has invalid parameters",
            id="unknown-step-parameter",
        ),
    ],
)
def test_malformed_definitions_fail_with_the_offending_part(
    definition, expected
) -> None:
    with pytest.raises(WorkflowCompileError) as info:
        compile_workflow(definition, catalogue=CATALOGUE)

    assert expected in str(info.value)


def test_unknown_block_type_names_the_step_and_known_types() -> None:
    with pytest.raises(UnknownBlockError) as info:
        compile_workflow(workflow([step("missing", "m")]), catalogue=CATALOGUE)

    assert info.value.step_path == ("m",)
    assert "'test/missing@v1'" in str(info.value)
    assert "test/scale@v1" in str(info.value)


def test_aliases_resolve_to_the_canonical_block() -> None:
    definition = workflow([{"type": "Scale", "name": "s", "value": "$inputs.values"}])

    plan = compile_workflow(definition, catalogue=CATALOGUE)

    assert plan.step(("s",)).block_type == "test/scale@v1"
    assert plan.step(("s",)).namespace == "test"


def test_numeric_and_hyphenated_names_are_selector_segments() -> None:
    definition = workflow(
        [
            step(
                "parse_fields",
                "2026",
                raw="$inputs.camera-1",
                expected_fields=["class-name", "7"],
            ),
            step("echo", "use-it", value="$steps.2026.class-name"),
            step("echo", "3", value="$steps.2026.7"),
        ],
        {"out-1": "$steps.use-it.value", "0": "$steps.3.value"},
        inputs=[batch_input("camera-1", kind=["string"])],
    )

    plan = compile_workflow(definition, catalogue=CATALOGUE)

    assert [step.path for step in plan.steps] == [("2026",), ("use-it",), ("3",)]
    assert set(plan.step(("2026",)).outputs) == {"class-name", "7", "error_status"}
    assert plan.step(("use-it",)).bindings[0].source.output == "class-name"
    assert [output.name for output in plan.outputs] == ["out-1", "0"]


@pytest.mark.parametrize(
    "definition, error, shown",
    [
        pytest.param(
            workflow([], inputs=[parameter("x\n")]),
            WorkflowCompileError,
            "'x\\n'",
            id="input-name-newline",
        ),
        pytest.param(
            workflow([step("sink", "a;b")]),
            WorkflowCompileError,
            "'a;b'",
            id="step-name-junk",
        ),
        pytest.param(
            workflow([step("sink", "s")], {"out\n": "$inputs.values"}),
            WorkflowCompileError,
            "'out\\n'",
            id="output-name-newline",
        ),
        pytest.param(
            workflow([step("sink", "s")], {"out": "$inputs.values\n"}),
            SelectorError,
            "'$inputs.values\\n'",
            id="output-selector-newline",
        ),
        pytest.param(
            workflow([step("sink", "s")], {"out": "$steps.s.a.b"}),
            SelectorError,
            "'$steps.s.a.b'",
            id="output-selector-extra-segment",
        ),
        pytest.param(
            workflow([step("sink", "s")], {"out": "$steps.s"}),
            SelectorError,
            "'$steps.s'",
            id="output-selector-is-a-step",
        ),
        pytest.param(
            workflow([step("echo", "s", value="$inputs.values junk")]),
            SelectorError,
            "'$inputs.values junk'",
            id="param-selector-junk",
        ),
        pytest.param(
            workflow(
                [], inputs=[{"name": "x", "axes": [{"id": "a\n", "kind": "sample"}]}]
            ),
            WorkflowCompileError,
            "'a\\n'",
            id="axis-id-newline",
        ),
    ],
)
def test_names_and_selectors_must_match_completely(definition, error, shown) -> None:
    with pytest.raises(error) as info:
        compile_workflow(definition, catalogue=CATALOGUE)

    assert shown in str(info.value), "the offending text is shown verbatim"


def test_invalid_literal_names_the_step_and_field() -> None:
    definition = workflow([step("scale", "s", value="$inputs.values", factor=-1)])

    with pytest.raises(ParamsValidationError) as info:
        compile_workflow(definition, catalogue=CATALOGUE)

    assert info.value.step_path == ("s",)
    assert info.value.field_path[0] == "factor"


def test_compilation_neither_mutates_the_definition_nor_keeps_references() -> None:
    definition = workflow(
        [step("compound", "c", params={"a": "$inputs.values", "b": [1, 2]})],
        {"echo": "$steps.c.echo"},
    )
    snapshot = copy.deepcopy(definition)

    plan = compile_workflow(definition, catalogue=CATALOGUE)
    definition["steps"][0]["params"]["b"].append(3)

    assert definition != snapshot
    definition["steps"][0]["params"]["b"].pop()
    assert definition == snapshot
    assert plan.step(("c",)).params.params["b"] == [1, 2]
    assert plan.catalogue is CATALOGUE


def _with_image_kind():
    from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
    from roboflow_workflows.execution_engine.v2.kinds import Kind

    image_kind = Kind(name="image")
    catalogue = Catalogue.merge(CATALOGUE, Catalogue(kinds=[image_kind]))

    return catalogue


def test_compilation_constructs_no_block_and_calls_no_provider_or_run() -> None:
    from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
    from roboflow_workflows.execution_engine.v2.declaration import (
        Block,
        BlockParams,
        Output,
        Ref,
    )
    from roboflow_workflows.execution_engine.v2.resources import Factory

    calls = []

    class Explosive(Block):
        type = "test/explosive@v1"
        outputs = {"out": Output()}

        class Params(BlockParams):
            value: Ref()

        def __init__(self, *, model: object):
            calls.append("constructor")

        def run(self, *, value) -> dict:
            calls.append("run")
            return {"out": value}

    catalogue = Catalogue(
        [Explosive],
        namespace="boom",
        providers={"model": Factory(lambda: calls.append("provider"))},
    )
    definition = workflow(
        [step("explosive", "a", value="$inputs.values")],
        {"out": "$steps.a.out"},
        inputs=[batch_input("values")],
    )

    plan = compile_workflow(definition, catalogue=catalogue)
    plan.describe()

    assert calls == []
    assert plan.step(("a",)).spec.resources[0].name == "model"
