"""Contract tests for the workload introspection response schemas
(``roboflow_workflows.execution_engine.introspection.workload_entities``).

Covers: JSON schema export + roundtrip for every entity, discriminators at
every nesting level, the cross-field rules of ``WorkflowIntrospection`` (each
rejection case plus valid hand-built examples, including a no-step
input->output workflow), ``steps_by_dimensionality`` string-key roundtrip and
the absence of every rejected field name from the schema.
"""

import json
import subprocess
import sys
from typing import Any, Dict, List, Type

import pytest
from pydantic import BaseModel, ValidationError
from roboflow_workflows.execution_engine.entities.workload import (
    ModelMetadata,
    RestrictionMetadata,
    Severity,
    WorkOperation,
    complete_discovery,
    incomplete_discovery,
)
from roboflow_workflows.execution_engine.introspection.workload_entities import (
    WORKLOAD_INTROSPECTION_SCHEMA_VERSION,
    GraphEdge,
    GraphNode,
    ModelSummary,
    StepMetadata,
    WorkflowIntrospection,
    WorkflowSummary,
)
from roboflow_workflows.prototypes.block import (
    STATEFUL_VIDEO_HTTP_SOFT_PORTABLE_RESTRICTION,
    ModelRequiredAction,
    roboflow_platform_model,
    roboflow_platform_project,
    third_party_model,
)

REJECTED_FIELD_NAMES = {
    "work",
    "work_expression",
    "expression",
    "lineage",
    "lineages",
    "inputs",
    "input_bindings",
    "origin_path",
    "compute",
    "io",
    "score",
    "scores",
    "weight",
    "execution_group",
    "execution_groups",
    "workers",
    "backend",
    "note",
}

EXPECTED_DISCRIMINATORS = {
    GraphNode: "graph_node",
    GraphEdge: "graph_edge",
    StepMetadata: "step_metadata",
    ModelSummary: "model_summary",
    WorkflowSummary: "workflow_summary",
    WorkflowIntrospection: "workflow_introspection",
}


# ---------------------------------------------------------------------------
# Builders for hand-made examples
# ---------------------------------------------------------------------------


def _step(
    node_id: str,
    input_dimensionality: int = 1,
    output_dimensionality: int = 1,
    **overrides: Any,
) -> StepMetadata:
    payload: Dict[str, Any] = dict(
        node_id=node_id,
        block_type="roboflow_core/roboflow_object_detection_model@v3",
        input_dimensionality=input_dimensionality,
        output_dimensionality=output_dimensionality,
        accepts_batch_input=True,
        resources=complete_discovery(
            [roboflow_platform_model(model_id="my_project/3")]
        ),
        restrictions=complete_discovery([]),
        operations=complete_discovery([WorkOperation.MODEL_INFERENCE]),
    )
    payload.update(overrides)
    return StepMetadata(**payload)


def _model(
    model_id: str = "my_project/3", used_by_steps: List[str] = ("$steps.detector",)
) -> ModelSummary:
    return ModelSummary(
        provider="roboflow",
        model_id=model_id,
        used_by_steps=list(used_by_steps),
        metadata=None,
        metadata_status="unavailable",
    )


def _introspection(**overrides: Any) -> WorkflowIntrospection:
    """A valid two-step workflow: image -> detector -> crop -> output, with a
    conditional (control) edge between the steps and a model inventory."""
    crop = _step(
        "$steps.crop",
        input_dimensionality=1,
        output_dimensionality=2,
        block_type="roboflow_core/dynamic_crop@v1",
        resources=complete_discovery([]),
        operations=complete_discovery([WorkOperation.IMAGE_CROP]),
        restrictions=incomplete_discovery(
            [STATEFUL_VIDEO_HTTP_SOFT_PORTABLE_RESTRICTION],
            ["step_declaration_missing:$steps.crop"],
        ),
    )
    payload: Dict[str, Any] = dict(
        execution_engine_version="1.7.0",
        nodes=[
            GraphNode(id="$inputs.image", kind="input"),
            GraphNode(id="$steps.detector", kind="step"),
            GraphNode(id="$steps.crop", kind="step"),
            GraphNode(id="$outputs.crops", kind="output"),
        ],
        edges=[
            GraphEdge(source="$inputs.image", target="$steps.detector", kind="data"),
            GraphEdge(source="$inputs.image", target="$steps.crop", kind="data"),
            GraphEdge(source="$steps.detector", target="$steps.crop", kind="data"),
            GraphEdge(source="$steps.detector", target="$steps.crop", kind="control"),
            GraphEdge(source="$steps.crop", target="$outputs.crops", kind="data"),
        ],
        steps=[_step("$steps.detector"), crop],
        summary=WorkflowSummary(
            models=complete_discovery([_model()]),
            max_dimensionality=2,
            steps_by_dimensionality={1: 2},
        ),
    )
    payload.update(overrides)
    return WorkflowIntrospection(**payload)


def _all_property_names(schema: Dict[str, Any]) -> List[str]:
    names: List[str] = []

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            properties = node.get("properties")
            if isinstance(properties, dict):
                names.extend(properties.keys())
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for value in node:
                _walk(value)

    _walk(schema)
    return names


def _collect_type_values(payload: Any) -> List[str]:
    found: List[str] = []

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            if "type" in node and isinstance(node["type"], str):
                found.append(node["type"])
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for value in node:
                _walk(value)

    _walk(payload)
    return found


# ---------------------------------------------------------------------------
# Module placement
# ---------------------------------------------------------------------------


def test_workload_entities_module_adds_no_v1_or_core_steps_imports() -> None:
    # `prototypes/block.py` already pulls `execution_engine.v1.entities` and
    # (through `entities/base.py`) an action-recognition entity module; the
    # response schemas must add nothing from `v1` or `core_steps` on top of
    # what importing prototypes alone loads.
    probe = (
        "import sys\n"
        "import roboflow_workflows.prototypes.block\n"
        "before = set(sys.modules)\n"
        "import roboflow_workflows.execution_engine.introspection.workload_entities\n"
        "added = sorted(set(sys.modules) - before)\n"
        "print('\\n'.join(added))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    added = set(result.stdout.split())
    assert (
        "roboflow_workflows.execution_engine.introspection.workload_entities" in added
    )
    forbidden = {
        module
        for module in added
        if module.startswith(
            (
                "roboflow_workflows.core_steps",
                "roboflow_workflows.execution_engine.v1",
            )
        )
    }
    assert forbidden == set(), forbidden


# ---------------------------------------------------------------------------
# Serialization: discriminators at every level, schema, roundtrip
# ---------------------------------------------------------------------------


def test_valid_example_roundtrips_and_carries_discriminators_everywhere() -> None:
    introspection = _introspection()

    python_payload = introspection.model_dump()
    json_payload = introspection.model_dump(mode="json")

    assert python_payload["type"] == "workflow_introspection"
    assert python_payload["summary"]["type"] == "workflow_summary"
    assert python_payload["summary"]["models"]["type"] == "discovery"
    assert python_payload["summary"]["models"]["items"][0]["type"] == "model_summary"
    assert python_payload["steps"][0]["type"] == "step_metadata"
    assert python_payload["steps"][0]["resources"]["type"] == "discovery"
    resource = python_payload["steps"][0]["resources"]["items"][0]
    assert resource["type"] == "dependent_resource"
    assert resource["metadata"]["type"] == "roboflow_platform_model"
    restriction = python_payload["steps"][1]["restrictions"]["items"][0]
    assert restriction["type"] == "restriction"
    assert restriction["when"]["type"] == "restriction_condition"
    assert python_payload["nodes"][0]["type"] == "graph_node"
    assert python_payload["edges"][0]["type"] == "graph_edge"

    assert set(_collect_type_values(json_payload)) == {
        "workflow_introspection",
        "workflow_summary",
        "discovery",
        "model_summary",
        "step_metadata",
        "dependent_resource",
        "roboflow_platform_model",
        "restriction",
        "restriction_condition",
        "graph_node",
        "graph_edge",
    }
    assert json_payload["schema_version"] == WORKLOAD_INTROSPECTION_SCHEMA_VERSION
    assert json_payload["steps"][1]["operations"]["items"] == ["image_crop"]

    parsed = WorkflowIntrospection.model_validate(json_payload)
    assert parsed == introspection
    assert WorkflowIntrospection.model_validate_json(json.dumps(json_payload)) == (
        introspection
    )


def test_steps_by_dimensionality_keys_are_strings_on_the_wire() -> None:
    introspection = _introspection()

    wire = json.loads(introspection.model_dump_json())

    assert wire["summary"]["steps_by_dimensionality"] == {"1": 2}
    parsed = WorkflowIntrospection.model_validate(wire)
    assert parsed.summary.steps_by_dimensionality == {1: 2}
    assert all(isinstance(key, int) for key in parsed.summary.steps_by_dimensionality)


def test_model_summary_with_metadata_roundtrips() -> None:
    summary = ModelSummary(
        provider="roboflow",
        model_id="my_project/3",
        used_by_steps=["$steps.b", "$steps.a"],
        metadata=ModelMetadata(model_type="rfdetr", task_type="object-detection"),
        metadata_status="available",
    )

    assert summary.used_by_steps == ["$steps.a", "$steps.b"]
    payload = summary.model_dump(mode="json")
    assert payload["metadata"]["type"] == "model_metadata"
    assert ModelSummary.model_validate(payload) == summary


@pytest.mark.parametrize(
    "entity_type", list(EXPECTED_DISCRIMINATORS), ids=lambda t: t.__name__
)
def test_schema_exports_type_const_and_default(entity_type: Type[BaseModel]) -> None:
    schema = entity_type.model_json_schema()

    json.dumps(schema)  # no python callables leak into the schema
    expected = EXPECTED_DISCRIMINATORS[entity_type]
    assert schema["properties"]["type"] == {
        "const": expected,
        "default": expected,
        "title": "Type",
        "type": "string",
    }


def test_workflow_introspection_schema_shows_type_on_every_entity() -> None:
    schema = WorkflowIntrospection.model_json_schema()

    for name, definition in schema["$defs"].items():
        if "properties" not in definition:
            continue  # enums
        assert "type" in definition["properties"], name
        type_property = definition["properties"]["type"]
        assert "const" in type_property and "default" in type_property, name
        assert type_property["const"] == type_property["default"], name
    assert schema["properties"]["schema_version"]["const"] == "1"


def test_schema_has_no_rejected_field_names() -> None:
    schema = WorkflowIntrospection.model_json_schema()

    offending = REJECTED_FIELD_NAMES & set(_all_property_names(schema))

    assert offending == set(), offending


def test_schema_excludes_resolver_and_registration_kwargs() -> None:
    schema = WorkflowIntrospection.model_json_schema()

    names = set(_all_property_names(schema))
    assert "model_id_resolver" not in names
    assert "model_registration_kwargs" not in names


@pytest.mark.parametrize(
    "entity",
    [
        GraphNode(id="$inputs.image", kind="input"),
        GraphEdge(source="$inputs.image", target="$steps.a", kind="data"),
        _step("$steps.a"),
        _model(),
        WorkflowSummary(
            models=complete_discovery([]),
            max_dimensionality=0,
            steps_by_dimensionality={},
        ),
    ],
    ids=lambda e: type(e).__name__,
)
def test_each_entity_roundtrips_through_json(entity: BaseModel) -> None:
    payload = json.loads(entity.model_dump_json())

    assert payload["type"] == EXPECTED_DISCRIMINATORS[type(entity)]
    assert entity.model_dump()["type"] == payload["type"]
    assert type(entity).model_validate(payload) == entity


@pytest.mark.parametrize(
    "entity_type", list(EXPECTED_DISCRIMINATORS), ids=lambda t: t.__name__
)
def test_wrong_type_discriminator_is_rejected(entity_type: Type[BaseModel]) -> None:
    example = {
        GraphNode: GraphNode(id="$inputs.image", kind="input"),
        GraphEdge: GraphEdge(source="$inputs.image", target="$steps.a", kind="data"),
        StepMetadata: _step("$steps.a"),
        ModelSummary: _model(),
        WorkflowSummary: WorkflowSummary(
            models=complete_discovery([]),
            max_dimensionality=0,
            steps_by_dimensionality={},
        ),
        WorkflowIntrospection: _introspection(),
    }[entity_type]
    payload = {**example.model_dump(mode="json"), "type": "something_else"}

    with pytest.raises(ValidationError):
        entity_type.model_validate(payload)


def test_entities_are_frozen() -> None:
    node = GraphNode(id="$inputs.image", kind="input")
    with pytest.raises(ValidationError):
        node.id = "$inputs.other"  # type: ignore[misc]
    introspection = _introspection()
    with pytest.raises(ValidationError):
        introspection.nodes = []  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Field-level rules
# ---------------------------------------------------------------------------


def test_graph_node_and_edge_reject_unknown_kinds_and_empty_ids() -> None:
    with pytest.raises(ValidationError):
        GraphNode(id="$steps.a", kind="super_input")
    with pytest.raises(ValidationError):
        GraphNode(id="", kind="step")
    with pytest.raises(ValidationError):
        GraphEdge(source="$steps.a", target="$steps.b", kind="schedule")
    with pytest.raises(ValidationError):
        GraphEdge(source="", target="$steps.b", kind="data")


def test_step_metadata_rejects_negative_dimensionality_and_blank_ids() -> None:
    with pytest.raises(ValidationError):
        _step("$steps.a", input_dimensionality=-1)
    with pytest.raises(ValidationError):
        _step("$steps.a", output_dimensionality=-1)
    with pytest.raises(ValidationError):
        _step("")
    with pytest.raises(ValidationError):
        _step("$steps.a", block_type="")


def test_step_metadata_discoveries_are_typed() -> None:
    with pytest.raises(ValidationError):
        _step("$steps.a", operations=complete_discovery(["not_an_operation"]))
    with pytest.raises(ValidationError):
        _step("$steps.a", restrictions=complete_discovery([WorkOperation.TRACKING]))
    with pytest.raises(ValidationError):
        _step("$steps.a", resources=complete_discovery([{"resource_type": "bogus"}]))


def test_step_metadata_keeps_access_vs_execution_and_third_party_resources() -> None:
    step = _step(
        "$steps.a",
        resources=complete_discovery(
            [
                third_party_model(provider="openai", model_id="gpt-4o"),
                roboflow_platform_model(
                    model_id="$inputs.model", required_action=ModelRequiredAction.ACCESS
                ),
                roboflow_platform_project(project_url="my_dataset"),
            ]
        ),
    )

    payload = step.model_dump(mode="json")["resources"]["items"]

    # Items sort by their sorted-keys JSON dump: `"metadata"` is the first
    # key, so `model_id` values ("$inputs.model" < "gpt-4o") come before the
    # project entry whose first metadata key is `project_url`.
    assert [item["metadata"]["type"] for item in payload] == [
        "roboflow_platform_model",
        "third_party_model",
        "roboflow_platform_project",
    ]
    assert payload[0]["metadata"]["required_action"] == "access"
    assert payload[0]["metadata"]["execution_location"] is None
    assert StepMetadata.model_validate(step.model_dump(mode="json")) == step


def test_model_summary_rules() -> None:
    with pytest.raises(ValidationError):
        _model(used_by_steps=[])
    with pytest.raises(ValidationError):
        _model(used_by_steps=["$steps.a", "$steps.a"])
    with pytest.raises(ValidationError):
        ModelSummary(
            provider="",
            model_id="x",
            used_by_steps=["$steps.a"],
            metadata_status="unavailable",
        )
    with pytest.raises(ValidationError):
        ModelSummary(
            provider="roboflow",
            model_id="",
            used_by_steps=["$steps.a"],
            metadata_status="unavailable",
        )
    with pytest.raises(ValidationError):
        ModelSummary(
            provider="roboflow",
            model_id="x",
            used_by_steps=["$steps.a"],
            metadata=None,
            metadata_status="available",
        )
    with pytest.raises(ValidationError):
        ModelSummary(
            provider="roboflow",
            model_id="x",
            used_by_steps=["$steps.a"],
            metadata=ModelMetadata(),
            metadata_status="available",
        )
    with pytest.raises(ValidationError):
        ModelSummary(
            provider="roboflow",
            model_id="x",
            used_by_steps=["$steps.a"],
            metadata=ModelMetadata(model_type="rfdetr"),
            metadata_status="disabled",
        )


def test_workflow_summary_histogram_rules() -> None:
    with pytest.raises(ValidationError):
        WorkflowSummary(
            models=complete_discovery([]),
            max_dimensionality=1,
            steps_by_dimensionality={-1: 1},
        )
    with pytest.raises(ValidationError):
        WorkflowSummary(
            models=complete_discovery([]),
            max_dimensionality=1,
            steps_by_dimensionality={1: 0},
        )
    with pytest.raises(ValidationError):
        WorkflowSummary(
            models=complete_discovery([]),
            max_dimensionality=-1,
            steps_by_dimensionality={},
        )
    summary = WorkflowSummary(
        models=complete_discovery([]),
        max_dimensionality=2,
        steps_by_dimensionality={"2": 1, "1": 3},
    )
    assert summary.steps_by_dimensionality == {1: 3, 2: 1}


# ---------------------------------------------------------------------------
# WorkflowIntrospection cross-field rules
# ---------------------------------------------------------------------------


def test_no_step_input_to_output_workflow_is_valid() -> None:
    introspection = WorkflowIntrospection(
        execution_engine_version="1.7.0",
        nodes=[
            GraphNode(id="$inputs.image", kind="input"),
            GraphNode(id="$outputs.image", kind="output"),
        ],
        edges=[GraphEdge(source="$inputs.image", target="$outputs.image", kind="data")],
        steps=[],
        summary=WorkflowSummary(
            models=complete_discovery([]),
            max_dimensionality=1,
            steps_by_dimensionality={},
        ),
    )

    assert introspection.summary.steps_by_dimensionality == {}
    assert introspection.summary.max_dimensionality == 1
    assert (
        WorkflowIntrospection.model_validate(introspection.model_dump(mode="json"))
        == introspection
    )


def test_incomplete_model_inventory_with_unresolved_selector_is_valid() -> None:
    introspection = _introspection(
        summary=WorkflowSummary(
            models=incomplete_discovery(
                [_model()], ["unresolved_model_selector:$steps.crop"]
            ),
            max_dimensionality=2,
            steps_by_dimensionality={1: 2},
        )
    )

    assert introspection.summary.models.complete is False


def test_rejects_duplicated_node_ids() -> None:
    with pytest.raises(ValidationError, match="Duplicated graph node id"):
        _introspection(
            nodes=[
                GraphNode(id="$inputs.image", kind="input"),
                GraphNode(id="$inputs.image", kind="input"),
                GraphNode(id="$steps.detector", kind="step"),
                GraphNode(id="$steps.crop", kind="step"),
                GraphNode(id="$outputs.crops", kind="output"),
            ]
        )


def test_rejects_duplicated_edge_triples() -> None:
    base = _introspection()
    with pytest.raises(ValidationError, match="Duplicated graph edge"):
        _introspection(edges=base.edges + [base.edges[0]])


def test_allows_data_and_control_edges_on_the_same_pair() -> None:
    introspection = _introspection()
    pairs = [(edge.source, edge.target, edge.kind) for edge in introspection.edges]
    assert ("$steps.detector", "$steps.crop", "data") in pairs
    assert ("$steps.detector", "$steps.crop", "control") in pairs


def test_rejects_edge_with_unknown_endpoint() -> None:
    base = _introspection()
    with pytest.raises(ValidationError, match="unknown node"):
        _introspection(
            edges=base.edges
            + [GraphEdge(source="$steps.ghost", target="$steps.crop", kind="data")]
        )
    with pytest.raises(ValidationError, match="unknown node"):
        _introspection(
            edges=base.edges
            + [GraphEdge(source="$steps.crop", target="$outputs.ghost", kind="data")]
        )


def test_rejects_data_edge_starting_at_output_node() -> None:
    base = _introspection()
    with pytest.raises(ValidationError, match="must not start at an output node"):
        _introspection(
            edges=base.edges
            + [GraphEdge(source="$outputs.crops", target="$steps.crop", kind="data")]
        )


@pytest.mark.parametrize(
    "source,target",
    [
        ("$inputs.image", "$steps.crop"),
        ("$steps.crop", "$outputs.crops"),
        ("$inputs.image", "$outputs.crops"),
    ],
)
def test_rejects_control_edge_not_between_steps(source: str, target: str) -> None:
    base = _introspection()
    with pytest.raises(ValidationError, match="must connect two step nodes"):
        _introspection(
            edges=base.edges + [GraphEdge(source=source, target=target, kind="control")]
        )


def test_rejects_step_metadata_for_unknown_step_node() -> None:
    base = _introspection()
    with pytest.raises(ValidationError, match="does not match any step node"):
        _introspection(
            steps=base.steps + [_step("$steps.ghost")],
            summary=WorkflowSummary(
                models=complete_discovery([_model()]),
                max_dimensionality=2,
                steps_by_dimensionality={1: 3},
            ),
        )


def test_rejects_step_metadata_for_input_node() -> None:
    base = _introspection()
    with pytest.raises(ValidationError, match="does not match any step node"):
        _introspection(
            steps=base.steps + [_step("$inputs.image")],
            summary=WorkflowSummary(
                models=complete_discovery([_model()]),
                max_dimensionality=2,
                steps_by_dimensionality={1: 3},
            ),
        )


def test_rejects_duplicated_step_metadata() -> None:
    base = _introspection()
    with pytest.raises(ValidationError, match="more than one StepMetadata"):
        _introspection(
            steps=base.steps + [_step("$steps.detector")],
            summary=WorkflowSummary(
                models=complete_discovery([_model()]),
                max_dimensionality=2,
                steps_by_dimensionality={1: 3},
            ),
        )


def test_rejects_step_node_without_step_metadata() -> None:
    base = _introspection()
    with pytest.raises(ValidationError, match="Step nodes without StepMetadata"):
        _introspection(
            steps=base.steps[:1],
            summary=WorkflowSummary(
                models=complete_discovery([_model()]),
                max_dimensionality=2,
                steps_by_dimensionality={1: 1},
            ),
        )


def test_rejects_model_used_by_unknown_step() -> None:
    with pytest.raises(ValidationError, match="used by unknown steps"):
        _introspection(
            summary=WorkflowSummary(
                models=complete_discovery([_model(used_by_steps=["$steps.ghost"])]),
                max_dimensionality=2,
                steps_by_dimensionality={1: 2},
            )
        )


def test_rejects_model_used_by_non_step_node() -> None:
    with pytest.raises(ValidationError, match="used by unknown steps"):
        _introspection(
            summary=WorkflowSummary(
                models=complete_discovery([_model(used_by_steps=["$inputs.image"])]),
                max_dimensionality=2,
                steps_by_dimensionality={1: 2},
            )
        )


def test_rejects_duplicated_model_inventory_entry() -> None:
    duplicate = ModelSummary(
        provider="roboflow",
        model_id="my_project/3",
        used_by_steps=["$steps.crop"],
        metadata_status="unavailable",
    )
    # Same (provider, model_id) with different used_by_steps: Discovery dedup
    # keeps both (different JSON), so the cross-field rule must catch it.
    with pytest.raises(ValidationError, match="Duplicated model inventory entry"):
        _introspection(
            summary=WorkflowSummary(
                models=complete_discovery([_model(), duplicate]),
                max_dimensionality=2,
                steps_by_dimensionality={1: 2},
            )
        )


def test_rejects_histogram_sum_mismatch() -> None:
    with pytest.raises(ValidationError, match="must sum to the number of steps"):
        _introspection(
            summary=WorkflowSummary(
                models=complete_discovery([_model()]),
                max_dimensionality=2,
                steps_by_dimensionality={1: 3},
            )
        )


def test_rejects_histogram_not_matching_step_input_dimensionalities() -> None:
    # Sum matches (2 steps) but the depths do not: both steps have input
    # dimensionality 1, the histogram claims one at depth 2.
    with pytest.raises(ValidationError, match="must equal the histogram"):
        _introspection(
            summary=WorkflowSummary(
                models=complete_discovery([_model()]),
                max_dimensionality=2,
                steps_by_dimensionality={1: 1, 2: 1},
            )
        )


def test_rejects_max_dimensionality_below_step_dimensionality() -> None:
    with pytest.raises(ValidationError, match="`max_dimensionality`"):
        _introspection(
            summary=WorkflowSummary(
                models=complete_discovery([_model()]),
                max_dimensionality=1,
                steps_by_dimensionality={1: 2},
            )
        )


def test_max_dimensionality_may_exceed_every_step() -> None:
    # The compiled graph's input/output depths can be deeper than any step
    # (e.g. a batch-of-batches input wired straight to an output).
    introspection = _introspection(
        summary=WorkflowSummary(
            models=complete_discovery([_model()]),
            max_dimensionality=3,
            steps_by_dimensionality={1: 2},
        )
    )
    assert introspection.summary.max_dimensionality == 3


def test_rejects_wrong_schema_version_and_blank_engine_version() -> None:
    base = _introspection()
    with pytest.raises(ValidationError):
        WorkflowIntrospection.model_validate(
            {**base.model_dump(mode="json"), "schema_version": "2"}
        )
    with pytest.raises(ValidationError):
        _introspection(execution_engine_version="")


def test_restriction_items_inside_steps_sort_deterministically() -> None:
    first = RestrictionMetadata(code="b_code", severity=Severity.SOFT)
    second = RestrictionMetadata(code="a_code", severity=Severity.HARD)
    step = _step("$steps.a", restrictions=complete_discovery([first, second, first]))

    assert [item.code for item in step.restrictions.items] == ["a_code", "b_code"]
