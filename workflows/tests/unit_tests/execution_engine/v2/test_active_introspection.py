"""Structural introspection of active plans: sources, groups, domains, edges.

Every source and block here refuses construction, so a passing test proves
that inspection reads declarations and the plan only.
"""

import json

import pytest
from roboflow_workflows.execution_engine.v2 import (
    Catalogue,
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
from roboflow_workflows.execution_engine.v2.introspection import (
    describe_catalogue,
    describe_workflow,
    discover_connections,
    discover_workload,
)
from roboflow_workflows.execution_engine.v2.kinds import FLOAT_KIND, STRING_KIND

from tests.unit_tests.execution_engine.v2.compilation.fixture_blocks import (
    Gate,
    Sink,
    gate,
    nested,
    parameter,
)


class Camera(Source):
    """Scalar image port and a grouped tiles port; never constructed here."""

    type = "probe/camera@v1"
    aliases = ("probe/camera",)
    outputs = {
        "image": SourceOutput(FLOAT_KIND, description="one frame per pulse"),
        "tiles": SourceOutput(
            FLOAT_KIND, layout=EntryLayout((Axis("tile", "sample"),))
        ),
    }

    class Params(SourceParams):
        device: str | Ref(STRING_KIND) = "cam0"
        fps: float = 25.0

    def __init__(self, *, decoder, clock="wall"):
        raise AssertionError("introspection must not construct a source")

    def open(self, *, device, fps):
        raise AssertionError("introspection must not open a source")

    def read(self):
        raise AssertionError("introspection must not read a source")


class Thermometer(Source):
    type = "probe/thermometer@v1"
    outputs = {"temperature": SourceOutput(FLOAT_KIND)}

    def __init__(self):
        raise AssertionError("introspection must not construct a source")

    def open(self):
        raise AssertionError("introspection must not open a source")

    def read(self):
        raise AssertionError("introspection must not read a source")


class Scale(Block):
    type = "probe/scale@v1"
    outputs = {"scaled": Output(FLOAT_KIND, source="value")}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)
        factor: float | Ref(FLOAT_KIND) = 2.0

    def __init__(self):
        raise AssertionError("introspection must not construct a block")

    def run(self, *, value, factor):
        return {"scaled": value * factor}


class Label(Block):
    type = "probe/label@v1"
    outputs = {"text": Output(STRING_KIND)}

    class Params(BlockParams):
        text: str | Ref(STRING_KIND)

    def __init__(self):
        raise AssertionError("introspection must not construct a block")

    def run(self, *, text):
        return {"text": text}


CATALOGUE = Catalogue(
    [Scale, Label, Gate, Sink],
    sources=[Camera, Thermometer],
    namespace="probe",
    providers={"decoder": "software"},
)


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


CHILD = {
    "version": "2.0",
    "inputs": [{"type": "WorkflowParameter", "name": "x"}],
    "steps": [{"type": "probe/scale@v1", "name": "double", "value": "$inputs.x"}],
    "outputs": [
        {"type": "JsonField", "name": "doubled", "selector": "$steps.double.scaled"},
        {"type": "JsonField", "name": "fwd", "selector": "$inputs.x"},
    ],
}

DEFINITION = {
    "version": "2.0",
    "inputs": [parameter("device", default="cam7", kind=["string"])],
    "sources": [
        {"type": "probe/camera@v1", "name": "cam", "device": "$inputs.device"},
        {"type": "probe/thermometer@v1", "name": "temp"},
    ],
    "steps": [
        {
            "type": "probe/scale@v1",
            "name": "celsius",
            "value": "$sources.temp.temperature",
        },
        {"type": "probe/label@v1", "name": "note", "text": "static"},
        gate("g", "$sources.cam.image", ["child"]),
        nested(
            "child",
            workflow_definition=CHILD,
            parameter_bindings={"x": "$sources.cam.image"},
        ),
        {"type": "test/sink@v1", "name": "audit", "payload": "$steps.child.fwd"},
    ],
    "outputs": [
        group(
            "temps",
            "$sources.temp.temperature",
            c="$steps.celsius.scaled",
            n="$steps.note.text",
        ),
        group(
            "frames",
            "$sources.cam.image",
            f="$sources.cam.image",
            d="$steps.child.doubled",
        ),
        group("tiles", "$sources.cam.tiles", t="$sources.cam.tiles"),
    ],
}


@pytest.fixture(scope="module")
def plan():
    compiled = compile_workflow(DEFINITION, catalogue=CATALOGUE)

    return compiled


def test_catalogue_description_lists_sources_apart_from_blocks() -> None:
    description = describe_catalogue(CATALOGUE)

    [camera, thermometer] = description["sources"]
    assert camera["type"] == "probe/camera@v1" and camera["namespace"] == "probe"
    assert camera["identities"] == ["probe/camera@v1", "probe/camera"]
    assert camera["outputs"]["tiles"] == {
        "kinds": ["float"],
        "axes": ["tile"],
        "description": "",
    }
    assert camera["fields"]["device"]["default"] == {"value": "cam0"}
    assert [resource["name"] for resource in camera["resources"]] == [
        "decoder",
        "clock",
    ]
    assert thermometer["resources"] == []
    assert [block["type"] for block in description["blocks"]] == [
        "probe/scale@v1",
        "probe/label@v1",
        "test/gate@v1",
        "test/sink@v1",
    ]
    assert description["connections"]["probe/scale@v1"]["value"] == [
        "probe/scale@v1.scaled"
    ]
    assert description["source_connections"]["probe/scale@v1"]["value"] == [
        "probe/camera@v1.image",
        "probe/camera@v1.tiles",
        "probe/thermometer@v1.temperature",
    ]
    assert description["source_connections"]["probe/label@v1"]["text"] == []
    json.dumps(description)


def test_workflow_description_shows_sources_groups_and_step_domains(plan) -> None:
    description = describe_workflow(plan)

    cam = description["sources"]["cam"]
    assert cam["node_id"] == "$sources.cam" and cam["type"] == "probe/camera@v1"
    assert cam["parameters"]["device"]["value"] == "$inputs.device"
    assert cam["parameters"]["device"]["selectors"][0]["origin"] == "$inputs.device"
    assert cam["parameters"]["fps"] == {
        "value": 25.0,
        "explicit": False,
        "selectors": [],
    }
    assert cam["outputs"]["tiles"] == {
        "kinds": ["float"],
        "axes": ["sources.cam:tile"],
        "selector": "$sources.cam.tiles",
    }
    assert [resource["name"] for resource in cam["resources"]] == ["decoder", "clock"]
    assert cam["route"] == [
        "$steps.note",
        "$steps.g",
        "$steps.child/double",
        "$steps.audit",
    ]
    assert cam["groups"] == ["frames", "tiles"]
    assert description["sources"]["temp"]["route"] == ["$steps.celsius", "$steps.note"]

    domains = {item["node_id"]: item["domain"] for item in description["steps"]}
    assert domains == {
        "$steps.celsius": "$sources.temp",
        "$steps.note": None,
        "$steps.g": "$sources.cam",
        "$steps.child/double": "$sources.cam",
        "$steps.audit": "$sources.cam",
    }

    frames = description["output_groups"]["frames"]
    assert (
        frames["anchor"] == "$sources.cam.image" and frames["source"] == "$sources.cam"
    )
    assert frames["dependencies"] == ["$steps.child/double"]
    assert frames["outputs"]["f"]["origin"] == "$sources.cam.image"
    assert frames["outputs"]["d"]["source"] == "$steps.child/double.scaled"
    assert description["output_groups"]["tiles"]["outputs"]["t"]["axes"] == [
        "sources.cam:tile"
    ]
    assert description["output_groups"]["temps"]["dependencies"] == [
        "$steps.celsius",
        "$steps.note",
    ]
    assert description["outputs"] == {}
    assert description["axes"]["sources.cam:tile"] == {
        "kind": "sample",
        "stationary": False,
        "origin": "$sources.cam",
    }
    json.dumps(description)


def test_connections_include_source_data_edges_and_group_edges(plan) -> None:
    connections = discover_connections(plan)

    into_celsius = [c for c in connections if c.target == "$steps.celsius"]
    assert [
        (c.kind, c.source, c.output, c.produced_kinds, c.accepted_kinds)
        for c in into_celsius
    ] == [("data", "$sources.temp", "temperature", ("float",), ("float",))]
    into_child = [c for c in connections if c.kind == "child_input"]
    assert [(c.source, c.output, c.target, c.selector) for c in into_child] == [
        ("$sources.cam", "image", "$steps.child: $inputs.x", "$sources.cam.image")
    ]
    assert [
        c.kind for c in connections if c.target.startswith("$output_groups.frames")
    ] == [
        "anchor",
        "group_output",
        "group_output",
    ]
    frames = {
        c.target: c for c in connections if c.target.startswith("$output_groups.frames")
    }
    assert frames["$output_groups.frames"].source == "$sources.cam"
    assert frames["$output_groups.frames"].output == "image"
    assert frames["$output_groups.frames.f"].produced_kinds == ("float",)
    assert frames["$output_groups.frames.d"].source == "$steps.child/double"
    assert frames["$output_groups.frames.d"].selector == "$steps.child.doubled"
    tiles = next(c for c in connections if c.target == "$output_groups.tiles.t")
    assert (tiles.source, tiles.output, tiles.produced_kinds) == (
        "$sources.cam",
        "tiles",
        ("float",),
    )
    assert not any(c.kind == "output" for c in connections)
    json.dumps([c.describe() for c in connections])


def test_workload_lists_source_constructor_resources(plan) -> None:
    report = discover_workload(plan)

    assert [(s.node_id, s.source_type) for s in report.sources] == [
        ("$sources.cam", "probe/camera@v1"),
        ("$sources.temp", "probe/thermometer@v1"),
    ]
    assert [(r.name, r.required) for r in report.sources[0].constructor_resources] == [
        ("decoder", True),
        ("clock", False),
    ]
    assert (
        report.describe()["sources"]["$sources.cam"]["constructor_resources"][0]["name"]
        == "decoder"
    )
    assert [s.node_id for s in report.steps] == [
        "$steps.celsius",
        "$steps.note",
        "$steps.g",
        "$steps.child/double",
        "$steps.audit",
    ]


def test_passive_plans_keep_their_shape() -> None:
    passive = compile_workflow(
        {
            "version": "2.0",
            "inputs": [
                {"type": "WorkflowBatchInput", "name": "values", "kind": ["float"]}
            ],
            "steps": [
                {"type": "probe/scale@v1", "name": "s", "value": "$inputs.values"}
            ],
            "outputs": [
                {"type": "JsonField", "name": "out", "selector": "$steps.s.scaled"}
            ],
        },
        catalogue=CATALOGUE,
    )

    description = describe_workflow(passive)
    assert description["sources"] == {} and description["output_groups"] == {}
    assert description["steps"][0]["domain"] is None
    assert [c.kind for c in discover_connections(passive)] == ["data", "output"]
    assert discover_workload(passive).sources == ()
    assert describe_catalogue(Catalogue([Scale]))["sources"] == []
