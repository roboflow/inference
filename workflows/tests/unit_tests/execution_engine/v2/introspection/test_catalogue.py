"""describe_catalogue: identities, schemas, defaults, grammar, connections and
the step keys each connectable field is written under."""

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pytest
from pydantic import AliasChoices, AliasPath, Field
from roboflow_workflows.execution_engine.v2 import (
    Block,
    BlockParams,
    Catalogue,
    Output,
    Ref,
    compile_workflow,
)
from roboflow_workflows.execution_engine.v2.dynamic_blocks import (
    build_dynamic_catalogue,
)
from roboflow_workflows.execution_engine.v2.errors import ParamsValidationError
from roboflow_workflows.execution_engine.v2.introspection import describe_catalogue
from roboflow_workflows.execution_engine.v2.kinds import (
    BOOLEAN_KIND,
    BUILTIN_KINDS,
    FLOAT_KIND,
)

from tests.unit_tests.execution_engine.v2.introspection.fixtures import (
    CATALOGUE,
    sentinel_definition,
)


class Measure(Block):
    """Pass a number on as a score."""

    type = "alias/measure@v1"
    outputs = {"score": Output(FLOAT_KIND)}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)

    def run(self, *, value: float) -> dict:
        return {"score": value}


class Audit(Block):
    """Compare a score with a limit; every data field is written under an alias."""

    type = "alias/audit@v1"
    outputs = {"passed": Output(BOOLEAN_KIND), "limit": Output(FLOAT_KIND)}

    class Params(BlockParams):
        threshold: float | Ref(FLOAT_KIND) = Field(
            default=0.5, alias="cutoff", ge=0, le=1
        )
        score: Ref(FLOAT_KIND) = Field(validation_alias=AliasPath("reading", "score"))
        margin: float | Ref(FLOAT_KIND) = Field(
            default=0.0,
            validation_alias=AliasChoices(AliasPath("margins", 0), "slack"),
        )

    def run(self, *, threshold: float, score: float, margin: float) -> dict:
        return {"passed": score + margin >= threshold, "limit": threshold}


ALIAS_CATALOGUE = Catalogue([Measure, Audit], kinds=BUILTIN_KINDS, namespace="alias")


def blocks_by_type(description: dict) -> dict:
    return {block["type"]: block for block in description["blocks"]}


def write_at(step: Dict[str, Any], path: List[str], value: Any) -> None:
    """Write like a JSON-only client: one nested object per path key."""
    target = step
    for key in path[:-1]:
        target = target.setdefault(key, {})
    target[path[-1]] = value


def test_blocks_expose_identities_namespace_and_class_owned_schema() -> None:
    # when
    blocks = blocks_by_type(describe_catalogue(CATALOGUE))

    # then
    detect = blocks["demo/detect@v1"]
    assert detect["identities"] == ["demo/detect@v1", "demo/detect"]
    assert detect["namespace"] == "demo"
    assert (
        detect["params_schema"]["properties"]["model_id"]["default"] == "default-model"
    )
    assert detect["fields"]["image"]["whole_selector"] == {
        "role": "item",
        "kinds": ["float"],
        "batch": "never",
    }
    assert detect["fields"]["image"]["literal_allowed"] is False
    assert detect["outputs"]["predictions"]["kinds"] == ["prediction"]
    assert (
        blocks["demo/collect@v1"]["fields"]["parts"]["whole_selector"]["role"]
        == "group"
    )
    assert blocks["demo/split@v1"]["outputs"]["part"]["transform"] == "expand"
    assert blocks["v2/continue_if"]["is_control"] is True
    pick = blocks["demo/pick@v1"]
    assert pick["configured_outputs"] is True and pick["outputs"] == {}
    assert pick["output_fields"] == ["keys"]


def test_defaults_keep_none_distinct_from_required_and_from_factories() -> None:
    # when
    fields = blocks_by_type(describe_catalogue(CATALOGUE))["demo/detect@v1"]["fields"]

    # then
    assert fields["image"]["default"] is None and fields["image"]["required"] is True
    assert fields["note"]["default"] == {"value": None}
    assert fields["model_id"]["default"] == {"value": "default-model"}
    assert fields["classes"]["default"] == {"factory": True}


def test_selector_grammar_is_published_and_matches_real_selectors() -> None:
    # when
    grammar = describe_catalogue(CATALOGUE)["selector_grammar"]

    # then
    data = re.compile(grammar["data_selector"])
    step = re.compile(grammar["step_selector"])
    assert data.fullmatch("$steps.parse.2026") and data.fullmatch("$inputs.camera-1")
    assert data.fullmatch("$steps.parse.*")
    assert not data.fullmatch("$steps.parse")
    assert step.fullmatch("$steps.parse") and not step.fullmatch("$steps.parse.out")
    assert grammar["batch_modes"] == ["never", "always", "if_varying"]


def test_connections_list_kind_compatible_producers_per_consumer_field() -> None:
    # when
    connections = describe_catalogue(CATALOGUE)["connections"]

    # then
    assert "demo/split@v1.part" in connections["demo/collect@v1"]["parts"]
    assert "demo/detect@v1.predictions" not in connections["demo/collect@v1"]["parts"]
    conditions = connections["v2/continue_if"]["condition"]
    assert "demo/positive@v1.ok" in conditions
    assert "demo/split@v1.part" not in conditions
    assert "demo/undeclared@v1.value" in conditions, "wildcard outputs fit any kind"
    assert "next_steps" not in connections["v2/continue_if"]
    wildcard_consumer = connections["demo/undeclared@v1"]["value"]
    assert "demo/detect@v1.predictions" in wildcard_consumer
    assert "demo/positive@v1.ok" in wildcard_consumer


def test_dynamic_blocks_are_described_without_running_their_code(
    tmp_path: Path,
) -> None:
    # given
    sentinel = tmp_path / "executed.txt"
    dynamic = build_dynamic_catalogue(
        [sentinel_definition(sentinel)], catalogue=CATALOGUE
    )

    # when
    description = describe_catalogue(Catalogue.merge(CATALOGUE, dynamic))

    # then
    block = blocks_by_type(description)["Sentinel"]
    assert block["namespace"] == "dynamic_workflows_blocks"
    assert (
        block["description"] == "Custom Python that must never run during inspection."
    )
    assert block["fields"]["value"]["default"] is None
    assert block["engine_compatibility"] == ">=2.0.0,<3.0.0"
    execution = block["metadata"]["dynamic"]
    assert execution["execution"] == "local_python"
    assert execution["sandboxed"] is False
    assert execution["local_code_allowed"] is False
    assert execution["declared_imports"] == ["numpy", "json"]
    assert execution["manifest"]["block_type"] == "Sentinel"
    assert [resource["name"] for resource in block["resources"]] == [
        "shared_state",
        "representation_policy",
    ]
    assert execution["tensor_compatibility"] == "legacy_compatibility"
    assert description["providers"]["dynamic_workflows_blocks"] == {
        "shared_state": {"factory": "dict", "scope": "session"}
    }
    json.dumps(description)
    assert not sentinel.exists()


def test_connection_inputs_name_the_keys_a_step_writes_for_each_field() -> None:
    # when
    description = describe_catalogue(ALIAS_CATALOGUE)

    # then
    inputs = description["connection_inputs"]["alias/audit@v1"]
    assert set(inputs) == set(description["connections"]["alias/audit@v1"])
    assert inputs["threshold"] == {
        "input_path": ["cutoff"],
        "input_paths": [["cutoff"]],
        "schema_property": "cutoff",
    }
    assert inputs["score"] == {
        "input_path": ["reading", "score"],
        "input_paths": [["reading", "score"]],
        "schema_property": "score",
    }
    assert inputs["margin"] == {
        "input_path": ["slack"],
        "input_paths": [["margins", 0], ["slack"]],
        "schema_property": "slack",
    }
    audit = blocks_by_type(description)["alias/audit@v1"]
    assert audit["fields"]["threshold"]["input_path"] == ["cutoff"]
    assert set(audit["params_schema"]["properties"]) == {"cutoff", "score", "slack"}
    plain = describe_catalogue(CATALOGUE)["connection_inputs"]["demo/detect@v1"]
    assert plain["image"] == {
        "input_path": ["image"],
        "input_paths": [["image"]],
        "schema_property": "image",
    }


def test_json_only_client_wires_aliased_fields_from_the_description() -> None:
    # given: only the serialized description is available to the client
    description = json.loads(json.dumps(describe_catalogue(ALIAS_CATALOGUE)))
    audit = blocks_by_type(description)["alias/audit@v1"]
    connections = description["connections"]["alias/audit@v1"]
    inputs = description["connection_inputs"]["alias/audit@v1"]
    properties = audit["params_schema"]["properties"]
    step = {"type": "alias/audit@v1", "name": "audit"}

    # when: every data field is bound to the measured score
    for field in ("threshold", "score", "margin"):
        assert "alias/measure@v1.score" in connections[field]
        assert inputs[field]["schema_property"] in properties
        write_at(step, inputs[field]["input_path"], "$steps.measure.score")
    definition = {
        "version": "2.0",
        "inputs": [
            {"type": "WorkflowBatchInput", "name": "number", "kind": ["float"]},
        ],
        "steps": [
            {"type": "alias/measure@v1", "name": "measure", "value": "$inputs.number"},
            step,
        ],
        "outputs": [
            {"type": "JsonField", "name": "passed", "selector": "$steps.audit.passed"},
            {"type": "JsonField", "name": "limit", "selector": "$steps.audit.limit"},
        ],
    }
    session = compile_workflow(definition, catalogue=ALIAS_CATALOGUE).create_session()
    rows = session.run({"number": [0.25, 0.75]}).rows()

    # then
    assert step == {
        "type": "alias/audit@v1",
        "name": "audit",
        "cutoff": "$steps.measure.score",
        "reading": {"score": "$steps.measure.score"},
        "slack": "$steps.measure.score",
    }
    assert properties[inputs["threshold"]["schema_property"]]["default"] == 0.5
    assert rows == [
        {"passed": True, "limit": 0.25},
        {"passed": True, "limit": 0.75},
    ]


def test_canonical_field_names_stay_rejected_where_the_schema_uses_aliases() -> None:
    # given
    inputs = describe_catalogue(ALIAS_CATALOGUE)["connection_inputs"]["alias/audit@v1"]
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowParameter", "name": "number", "kind": ["float"]}],
        "steps": [
            {
                "type": "alias/audit@v1",
                "name": "audit",
                "threshold": 0.4,
                "score": "$inputs.number",
            }
        ],
        "outputs": [
            {"type": "JsonField", "name": "passed", "selector": "$steps.audit.passed"}
        ],
    }

    # then
    assert ["threshold"] not in inputs["threshold"]["input_paths"]
    assert ["score"] not in inputs["score"]["input_paths"]
    with pytest.raises(
        ParamsValidationError, match="threshold: Extra inputs are not permitted"
    ):
        compile_workflow(definition, catalogue=ALIAS_CATALOGUE)
