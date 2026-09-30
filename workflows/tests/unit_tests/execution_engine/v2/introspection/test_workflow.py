"""describe_workflow and discover_connections over a compiled nested plan."""

import json
from pathlib import Path

import pytest
from roboflow_workflows.execution_engine.v2.introspection import (
    describe_workflow,
    discover_connections,
)

from tests.unit_tests.execution_engine.v2.introspection.fixtures import (
    compile_deep_fixture,
    compile_fixture,
)


@pytest.fixture
def sentinel(tmp_path: Path) -> Path:
    return tmp_path / "executed.txt"


def steps_by_id(description: dict) -> dict:
    return {step["node_id"]: step for step in description["steps"]}


def test_inputs_report_required_and_defaults(sentinel: Path) -> None:
    # when
    inputs = describe_workflow(compile_fixture(sentinel))["inputs"]

    # then
    assert inputs["values"]["required"] is True
    assert inputs["values"]["default"] is None
    assert inputs["values"]["axes"] == ["inputs"]
    assert inputs["model"] == {
        "kinds": ["string"],
        "axes": [],
        "required": False,
        "default": {"value": "root-default-model"},
        "declared_type": "WorkflowParameter",
    }


def test_nested_steps_keep_their_scope_and_whole_child_gating(sentinel: Path) -> None:
    # when
    steps = steps_by_id(describe_workflow(compile_fixture(sentinel)))

    # then
    child = steps["$steps.child/detect"]
    assert child["path"] == ["child", "detect"] and child["scope"] == ["child"]
    assert child["type"] == "demo/detect@v1"
    assert [gate["controller"] for gate in child["gates"]] == ["$steps.gate"]
    assert steps["$steps.gate"]["control_targets"] == {
        "$steps.child": ["$steps.child/detect"]
    }
    gate_targets = steps["$steps.gate"]["parameters"]["next_steps"]["selectors"]
    assert gate_targets == [
        {
            "position": [0],
            "selector": "$steps.child",
            "governs": ["$steps.child/detect"],
        }
    ]


def test_parameters_separate_literals_explicit_values_and_resolved_selectors(
    sentinel: Path,
) -> None:
    # when
    steps = steps_by_id(describe_workflow(compile_fixture(sentinel)))

    # then
    root = steps["$steps.detect"]["parameters"]
    assert root["model_id"]["value"] == "$inputs.model"
    [root_model] = root["model_id"]["selectors"]
    assert root_model["source"] == "$inputs.model" and root_model["constant"] is None
    assert root_model["origin"] == "$inputs.model"
    [confidence] = root["confidence"]["selectors"]
    assert confidence["source"] == "$steps.collect.total"
    assert confidence["mode"] == "ancestor"
    assert root["note"] == {"value": None, "explicit": True, "selectors": []}
    assert root["classes"] == {"value": [], "explicit": False, "selectors": []}

    child = steps["$steps.child/detect"]["parameters"]
    [child_model] = child["model_id"]["selectors"]
    assert child_model["selector"] == "$inputs.model"
    assert child_model["source"] == "$steps.child: $inputs.model"
    assert child_model["origin"] is None
    assert child_model["constant"] == {"value": "child-model"}, "literal, not decoded"
    [child_image] = child["image"]["selectors"]
    assert child_image["source"] == "$steps.child: $inputs.image"
    assert child_image["origin"] == "$steps.split.part"
    assert child_image["source_axes"] == ["inputs", "split:parts"]
    assert child["note"]["explicit"] is False


def test_steps_report_their_actual_configured_outputs(sentinel: Path) -> None:
    # when
    steps = steps_by_id(describe_workflow(compile_fixture(sentinel)))

    # then: the class declares none statically; the step's keys name them
    pick = steps["$steps.pick"]
    assert list(pick["outputs"]) == ["left", "right"]
    assert pick["parameters"]["keys"] == {
        "value": ["left", "right"],
        "explicit": True,
        "selectors": [],
    }


def test_lineage_and_layout_transforms_are_explained_by_axis_origins(
    sentinel: Path,
) -> None:
    # when
    description = describe_workflow(compile_fixture(sentinel))

    # then
    steps = steps_by_id(description)
    assert description["axes"] == {
        "inputs": {"kind": "sample", "stationary": False, "origin": "$inputs.values"},
        "split:parts": {
            "kind": "dynamic_nesting",
            "stationary": False,
            "origin": "$steps.split.part",
        },
    }
    assert steps["$steps.split"]["outputs"]["part"]["transform"] == "expand"
    assert steps["$steps.split"]["outputs"]["part"]["axes"] == ["inputs", "split:parts"]
    [parts] = steps["$steps.collect"]["parameters"]["parts"]["selectors"]
    assert parts["mode"] == "group"
    assert steps["$steps.collect"]["invocation_axes"] == ["inputs"]
    assert steps["$steps.detect"]["invocation_axes"] == ["inputs", "split:parts"]
    json.dumps(description)


def test_connections_are_the_plan_edges_without_constants(sentinel: Path) -> None:
    # when
    connections = discover_connections(compile_fixture(sentinel))

    # then
    into_child = [c for c in connections if c.target == "$steps.child/detect"]
    assert [(c.kind, c.source, c.output, c.field_path) for c in into_child] == [
        ("data", "$steps.child: $inputs.image", None, ("image",)),
        ("data", "$steps.child: $inputs.model", None, ("model_id",)),
        ("control", "$steps.gate", None, ()),
    ]
    assert into_child[0].produced_kinds == ("float",), "the child's declared kinds"
    assert into_child[2].selector == "$steps.child"

    boundaries = [c for c in connections if c.kind == "child_input"]
    assert [(c.source, c.output, c.target, c.selector) for c in boundaries] == [
        ("$steps.split", "part", "$steps.child: $inputs.image", "$steps.split.part")
    ], "the literal-bound model input has no source node, so no edge"

    group = next(c for c in connections if c.target == "$steps.collect")
    assert (group.mode, group.produced_kinds, group.accepted_kinds) == (
        "group",
        ("float",),
        ("float",),
    )

    outputs = [c.describe() for c in connections if c.kind == "output"]
    assert outputs == [
        {
            "kind": "output",
            "source": "$steps.collect",
            "output": "total",
            "target": "$outputs.total",
            "field_path": [],
            "selector": "$steps.collect.total",
            "mode": None,
            "produced_kinds": ["float"],
            "accepted_kinds": [],
        },
        {
            "kind": "output",
            "source": "$steps.child/detect",
            "output": "predictions",
            "target": "$outputs.child",
            "field_path": [],
            "selector": "$steps.child.predictions",
            "mode": None,
            "produced_kinds": ["prediction"],
            "accepted_kinds": [],
        },
    ]


def test_inspection_never_constructs_blocks_or_runs_custom_code(
    sentinel: Path,
) -> None:
    # given
    plan = compile_fixture(sentinel)

    # when
    describe_workflow(plan)
    discover_connections(plan)

    # then: every fixture block raises when constructed
    assert not sentinel.exists()
    with pytest.raises(Exception, match="was constructed|runs submitted Python"):
        plan.create_session()


def test_child_inputs_are_described_with_kinds_layouts_and_origins() -> None:
    # when
    child_inputs = {
        item["port"]: item
        for item in describe_workflow(compile_deep_fixture())["child_inputs"]
    }

    # then: an alias chain keeps each hop and names its origin
    assert child_inputs["$steps.selected/inner: $inputs.image"] == {
        "port": "$steps.selected/inner: $inputs.image",
        "scope": ["selected", "inner"],
        "name": "image",
        "kinds": ["float"],
        "axes": ["inputs"],
        "source": "$steps.selected: $inputs.image",
        "origin": "$inputs.values",
        "constant": None,
    }
    literal = child_inputs["$steps.literal/inner: $inputs.model"]
    assert literal["source"] == "$steps.literal: $inputs.model"
    assert (
        literal["constant"] == {"value": "deep-literal"} and literal["origin"] is None
    )
    defaulted = child_inputs["$steps.defaulted: $inputs.model"]
    assert defaulted["source"] is None
    assert defaulted["constant"] == {"value": "middle-default"}
    assert child_inputs["$steps.selected: $inputs.model"]["origin"] == "$inputs.model"


def test_connections_follow_child_input_chains_and_direct_outputs() -> None:
    # when
    connections = discover_connections(compile_deep_fixture())

    # then
    image_path = [
        "$steps.selected: $inputs.image",
        "$steps.selected/inner: $inputs.image",
        "$steps.selected/inner/detect",
    ]
    selected_image = [
        (c.kind, c.source, c.target)
        for c in connections
        if c.target in image_path and c.field_path in ((), ("image",))
    ]
    assert selected_image == [
        ("child_input", "$inputs.values", "$steps.selected: $inputs.image"),
        (
            "child_input",
            "$steps.selected: $inputs.image",
            "$steps.selected/inner: $inputs.image",
        ),
        (
            "data",
            "$steps.selected/inner: $inputs.image",
            "$steps.selected/inner/detect",
        ),
    ]
    literal_sources = {c.source for c in connections if c.kind == "child_input"}
    assert "$steps.literal: $inputs.model" in literal_sources
    assert all(c.source is not None for c in connections), "constants create no edge"
    [echo] = [c for c in connections if c.kind == "output"]
    assert (echo.source, echo.output, echo.produced_kinds) == (
        "$steps.literal.image",
        None,
        ("float",),
    ), "a forwarded child input leaves through its child output port"
    [forwarded] = [c for c in connections if c.kind == "child_output"]
    assert (forwarded.source, forwarded.target, forwarded.produced_kinds) == (
        "$steps.literal: $inputs.image",
        "$steps.literal.image",
        ("float",),
    )
