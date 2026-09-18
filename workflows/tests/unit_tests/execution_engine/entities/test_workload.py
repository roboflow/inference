"""Contract tests for ``roboflow_workflows.execution_engine.entities.workload``.

Covers the dependency-light declaration primitives: enum relocation
(identity through ``prototypes.block``), the closed ``WorkOperation`` set,
``Discovery`` completeness/dedup/sort rules, ``RestrictionCondition`` /
``RestrictionMetadata`` validation, model-metadata status rules,
``normalize_declaration`` and JSON-schema/roundtrip for every entity.
"""

import json
import subprocess
import sys
from typing import Any, Dict, List, Type

import pytest
from pydantic import BaseModel, ValidationError
from roboflow_workflows.execution_engine.entities.workload import (
    Discovery,
    ModelMetadata,
    ModelMetadataLookup,
    ModelMetadataProvider,
    RestrictionCondition,
    RestrictionMetadata,
    Runtime,
    RuntimeInputMode,
    Severity,
    StepExecutionMode,
    WorkOperation,
    complete_discovery,
    ensure_model_metadata_status_consistent,
    incomplete_discovery,
    normalize_declaration,
)

EXPECTED_WORK_OPERATIONS = {
    "MODEL_INFERENCE": "model_inference",
    "CUSTOM_PYTHON": "custom_python",
    "IMAGE_RESIZE": "image_resize",
    "IMAGE_CROP": "image_crop",
    "IMAGE_TRANSFORM": "image_transform",
    "IMAGE_COMPOSITION": "image_composition",
    "IMAGE_FILTERING": "image_filtering",
    "IMAGE_ANALYSIS": "image_analysis",
    "IMAGE_ENCODING": "image_encoding",
    "VISUALIZATION": "visualization",
    "DETECTION_PROCESSING": "detection_processing",
    "DETECTION_MATCHING": "detection_matching",
    "TRACKING": "tracking",
    "NUMERICAL_COMPUTATION": "numerical_computation",
    "EXPRESSION_EVALUATION": "expression_evaluation",
    "DATA_TRANSFORMATION": "data_transformation",
    "DATA_AGGREGATION": "data_aggregation",
    "TEMPORAL_BUFFERING": "temporal_buffering",
    "FLOW_CONTROL": "flow_control",
    "EXTERNAL_REQUEST": "external_request",
    "STORAGE_READ": "storage_read",
    "STORAGE_WRITE": "storage_write",
    "CACHE_READ": "cache_read",
    "CACHE_WRITE": "cache_write",
    "ENVIRONMENT_READ": "environment_read",
}

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


# ---------------------------------------------------------------------------
# Module placement and enum relocation
# ---------------------------------------------------------------------------


def test_workload_module_imports_without_prototypes_core_steps_or_v1() -> None:
    # Fresh interpreter: the module must be importable on its own, without
    # pulling the block framework, the block library or the v1 engine.
    probe = (
        "import sys\n"
        "import roboflow_workflows.execution_engine.entities.workload\n"
        "loaded = sorted(m for m in sys.modules if m.startswith('roboflow_workflows'))\n"
        "print('\\n'.join(loaded))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
    )
    loaded = set(result.stdout.split())
    assert "roboflow_workflows.execution_engine.entities.workload" in loaded
    forbidden = {
        module
        for module in loaded
        if module.startswith(
            (
                "roboflow_workflows.prototypes",
                "roboflow_workflows.core_steps",
                "roboflow_workflows.execution_engine.v1",
                "roboflow_workflows.execution_engine.introspection",
            )
        )
    }
    assert forbidden == set(), forbidden


def test_enums_are_re_exported_from_prototypes_with_identity() -> None:
    from roboflow_workflows.prototypes import block

    assert block.Severity is Severity
    assert block.Runtime is Runtime
    assert block.RuntimeInputMode is RuntimeInputMode
    assert block.StepExecutionMode is StepExecutionMode


def test_enum_values_are_unchanged() -> None:
    assert {member.value for member in Severity} == {"soft", "hard"}
    assert {member.value for member in Runtime} == {
        "hosted_serverless",
        "dedicated_deployment",
        "self_hosted_cpu",
        "self_hosted_gpu",
        "inference_pipeline",
    }
    assert {member.value for member in RuntimeInputMode} == {"image", "video"}
    assert {member.value for member in StepExecutionMode} == {"local", "remote"}


def test_step_execution_mode_stays_a_plain_enum() -> None:
    assert not issubclass(StepExecutionMode, str)
    assert issubclass(Severity, str)
    assert issubclass(Runtime, str)
    assert issubclass(RuntimeInputMode, str)
    assert issubclass(WorkOperation, str)


def test_work_operation_has_exactly_the_contract_members() -> None:
    assert {member.name: member.value for member in WorkOperation} == (
        EXPECTED_WORK_OPERATIONS
    )
    assert len(WorkOperation) == 25
    assert not any(name in {"UNKNOWN", "OTHER"} for name in WorkOperation.__members__)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_complete_empty_discovery_is_known_absence() -> None:
    discovery = complete_discovery([])

    assert discovery.items == []
    assert discovery.complete is True
    assert discovery.unknown_reasons == []
    assert discovery.model_dump() == {
        "type": "discovery",
        "items": [],
        "complete": True,
        "unknown_reasons": [],
    }


def test_incomplete_discovery_keeps_known_items_and_reasons() -> None:
    discovery = incomplete_discovery(
        [WorkOperation.CUSTOM_PYTHON],
        ["custom_python_internal_operations_unknown:$steps.custom"],
    )

    assert discovery.items == [WorkOperation.CUSTOM_PYTHON]
    assert discovery.complete is False
    assert discovery.unknown_reasons == [
        "custom_python_internal_operations_unknown:$steps.custom"
    ]


def test_incomplete_discovery_without_reasons_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Discovery(items=[], complete=False, unknown_reasons=[])
    with pytest.raises(ValidationError):
        incomplete_discovery([WorkOperation.TRACKING], [])


def test_complete_discovery_with_reasons_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Discovery(items=[], complete=True, unknown_reasons=["x:$steps.a"])


def test_discovery_rejects_blank_reasons() -> None:
    with pytest.raises(ValidationError):
        Discovery(items=[], complete=False, unknown_reasons=["   "])


def test_discovery_deduplicates_and_sorts_enum_items_and_reasons() -> None:
    discovery = Discovery[WorkOperation](
        items=[
            WorkOperation.TRACKING,
            WorkOperation.MODEL_INFERENCE,
            WorkOperation.TRACKING,
            "image_crop",
        ],
        complete=False,
        unknown_reasons=["b:$steps.x", "a:$steps.x", "b:$steps.x"],
    )

    assert discovery.items == [
        WorkOperation.IMAGE_CROP,
        WorkOperation.MODEL_INFERENCE,
        WorkOperation.TRACKING,
    ]
    assert discovery.unknown_reasons == ["a:$steps.x", "b:$steps.x"]


def test_discovery_sorts_restriction_metadata_by_code_severity_condition() -> None:
    hard = RestrictionMetadata(code="b_code", severity=Severity.HARD)
    soft = RestrictionMetadata(code="b_code", severity=Severity.SOFT)
    video_only = RestrictionMetadata(
        code="a_code",
        severity=Severity.SOFT,
        when=RestrictionCondition(input_modes=[RuntimeInputMode.VIDEO]),
    )
    unconditional = RestrictionMetadata(code="a_code", severity=Severity.SOFT)

    discovery = complete_discovery([soft, hard, video_only, unconditional, soft])

    # Key = (code, severity, condition JSON with sorted keys): `a_code` before
    # `b_code`; within `a_code` the JSON `"input_modes": [` sorts before
    # `"input_modes": null`; within `b_code` "hard" sorts before "soft".
    assert discovery.items == [video_only, unconditional, hard, soft]


def test_discovery_deduplicates_models_by_canonical_json() -> None:
    first = ModelMetadata(model_type="object-detection")
    second = ModelMetadata(model_type="object-detection")
    other = ModelMetadata(model_type="classification")

    discovery = complete_discovery([first, second, other])

    assert discovery.items == [other, first]


def test_parametrised_discovery_validates_item_type_and_keeps_type_default() -> None:
    with pytest.raises(ValidationError):
        Discovery[WorkOperation](
            items=["not_an_operation"], complete=True, unknown_reasons=[]
        )
    discovery = Discovery[WorkOperation].model_validate(
        {"items": ["tracking"], "complete": True, "unknown_reasons": []}
    )
    assert discovery.type == "discovery"
    assert discovery.model_dump()["type"] == "discovery"
    assert discovery.model_dump(mode="json") == {
        "type": "discovery",
        "items": ["tracking"],
        "complete": True,
        "unknown_reasons": [],
    }


def test_discovery_rejects_wrong_type_discriminator() -> None:
    with pytest.raises(ValidationError):
        Discovery.model_validate(
            {
                "type": "not_discovery",
                "items": [],
                "complete": True,
                "unknown_reasons": [],
            }
        )


def test_discovery_is_frozen() -> None:
    discovery = complete_discovery([WorkOperation.TRACKING])
    with pytest.raises(ValidationError):
        discovery.complete = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# normalize_declaration
# ---------------------------------------------------------------------------


def test_normalize_declaration_maps_none_to_unknown() -> None:
    result = normalize_declaration(None, "step_declaration_missing:$steps.crop")

    assert result == Discovery(
        items=[],
        complete=False,
        unknown_reasons=["step_declaration_missing:$steps.crop"],
    )


def test_normalize_declaration_maps_list_to_complete() -> None:
    assert normalize_declaration([], "unused:$steps.a") == complete_discovery([])
    assert normalize_declaration(
        [WorkOperation.TRACKING, WorkOperation.IMAGE_CROP], "unused:$steps.a"
    ) == complete_discovery([WorkOperation.IMAGE_CROP, WorkOperation.TRACKING])


def test_normalize_declaration_revalidates_discovery_preserving_items() -> None:
    declared = incomplete_discovery(
        [WorkOperation.CUSTOM_PYTHON],
        ["custom_python_internal_operations_unknown:$steps.c"],
    )

    result = normalize_declaration(declared, "unused:$steps.c")

    assert result == declared
    assert result is not declared
    assert type(result) is type(declared)
    assert result.items[0] is declared.items[0]


# ---------------------------------------------------------------------------
# RestrictionCondition / RestrictionMetadata
# ---------------------------------------------------------------------------


def test_restriction_condition_default_is_unrestricted() -> None:
    condition = RestrictionCondition()

    assert condition.runtimes is None
    assert condition.step_execution_modes is None
    assert condition.input_modes is None
    assert condition.configuration_equals == {}
    assert condition.model_dump(mode="json") == {
        "type": "restriction_condition",
        "runtimes": None,
        "step_execution_modes": None,
        "input_modes": None,
        "configuration_equals": {},
    }


@pytest.mark.parametrize("axis", ["runtimes", "step_execution_modes", "input_modes"])
def test_restriction_condition_rejects_empty_axis_list(axis: str) -> None:
    with pytest.raises(ValidationError):
        RestrictionCondition(**{axis: []})


def test_restriction_condition_rejects_duplicated_axis_entries() -> None:
    with pytest.raises(ValidationError):
        RestrictionCondition(
            runtimes=[Runtime.SELF_HOSTED_CPU, Runtime.SELF_HOSTED_CPU]
        )


def test_restriction_condition_normalises_axis_order_by_value() -> None:
    condition = RestrictionCondition(
        runtimes=[
            Runtime.SELF_HOSTED_GPU,
            Runtime.DEDICATED_DEPLOYMENT,
            Runtime.HOSTED_SERVERLESS,
        ],
        step_execution_modes=[StepExecutionMode.REMOTE, StepExecutionMode.LOCAL],
        input_modes=[RuntimeInputMode.VIDEO, RuntimeInputMode.IMAGE],
    )

    assert condition.runtimes == [
        Runtime.DEDICATED_DEPLOYMENT,
        Runtime.HOSTED_SERVERLESS,
        Runtime.SELF_HOSTED_GPU,
    ]
    assert condition.step_execution_modes == [
        StepExecutionMode.LOCAL,
        StepExecutionMode.REMOTE,
    ]
    assert condition.input_modes == [RuntimeInputMode.IMAGE, RuntimeInputMode.VIDEO]


def test_restriction_condition_accepts_json_configuration_values() -> None:
    condition = RestrictionCondition(
        configuration_equals={"allow_local_files": False, "mode": "append", "n": 3}
    )

    assert condition.configuration_equals == {
        "allow_local_files": False,
        "mode": "append",
        "n": 3,
    }
    with pytest.raises(ValidationError):
        RestrictionCondition(configuration_equals={"": True})
    with pytest.raises(ValidationError):
        RestrictionCondition(configuration_equals={"x": object()})


def test_restriction_condition_is_frozen_and_hashable() -> None:
    first = RestrictionCondition(
        runtimes=[Runtime.HOSTED_SERVERLESS], configuration_equals={"a": [1, 2]}
    )
    second = RestrictionCondition(
        runtimes=[Runtime.HOSTED_SERVERLESS], configuration_equals={"a": [1, 2]}
    )

    assert first == second
    assert hash(first) == hash(second)
    assert len({first, second}) == 1
    with pytest.raises(ValidationError):
        first.runtimes = None  # type: ignore[misc]


def test_restriction_condition_roundtrips_json_with_plain_enum_axis() -> None:
    condition = RestrictionCondition(
        step_execution_modes=[StepExecutionMode.REMOTE],
        runtimes=[Runtime.HOSTED_SERVERLESS],
    )

    payload = json.loads(condition.model_dump_json())

    assert payload["step_execution_modes"] == ["remote"]
    assert RestrictionCondition.model_validate(payload) == condition


@pytest.mark.parametrize(
    "code",
    ["", "Upper_case", "1starts_with_digit", "has-dash", "has space", "_leading"],
)
def test_restriction_metadata_rejects_invalid_codes(code: str) -> None:
    with pytest.raises(ValidationError):
        RestrictionMetadata(code=code, severity=Severity.SOFT)


def test_restriction_metadata_accepts_valid_code_and_defaults_condition() -> None:
    restriction = RestrictionMetadata(
        code="writes_to_ephemeral_disk", severity=Severity.SOFT
    )

    assert restriction.when == RestrictionCondition()
    assert restriction.model_dump(mode="json") == {
        "type": "restriction",
        "code": "writes_to_ephemeral_disk",
        "severity": "soft",
        "when": {
            "type": "restriction_condition",
            "runtimes": None,
            "step_execution_modes": None,
            "input_modes": None,
            "configuration_equals": {},
        },
    }
    assert hash(restriction) == hash(
        RestrictionMetadata(code="writes_to_ephemeral_disk", severity=Severity.SOFT)
    )


def test_restriction_metadata_has_no_note_field_and_rejects_extras() -> None:
    assert "note" not in RestrictionMetadata.model_fields
    with pytest.raises(ValidationError):
        RestrictionMetadata(code="x", severity=Severity.SOFT, note="human text")
    with pytest.raises(ValidationError):
        RestrictionCondition(runtimes=None, score=1.0)
    with pytest.raises(ValidationError):
        Discovery(items=[], complete=True, unknown_reasons=[], weight=2)


# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------


def test_model_metadata_lookup_available_requires_known_field() -> None:
    lookup = ModelMetadataLookup(
        status="available", metadata=ModelMetadata(task_type="object-detection")
    )
    assert lookup.metadata is not None
    assert lookup.metadata.has_known_fields()

    with pytest.raises(ValidationError):
        ModelMetadataLookup(status="available", metadata=None)
    with pytest.raises(ValidationError):
        ModelMetadataLookup(status="available", metadata=ModelMetadata())


@pytest.mark.parametrize("status", ["disabled", "unavailable"])
def test_model_metadata_lookup_without_metadata_statuses(status: str) -> None:
    lookup = ModelMetadataLookup(status=status)
    assert lookup.metadata is None
    with pytest.raises(ValidationError):
        ModelMetadataLookup(status=status, metadata=ModelMetadata(model_type="x"))


def test_model_metadata_lookup_rejects_unknown_status() -> None:
    with pytest.raises(ValidationError):
        ModelMetadataLookup(status="partial")


def test_ensure_model_metadata_status_consistent_is_the_shared_rule() -> None:
    ensure_model_metadata_status_consistent("unavailable", None)
    ensure_model_metadata_status_consistent(
        "available", ModelMetadata(model_variant="v")
    )
    with pytest.raises(ValueError):
        ensure_model_metadata_status_consistent("available", None)
    with pytest.raises(ValueError):
        ensure_model_metadata_status_consistent(
            "disabled", ModelMetadata(model_type="x")
        )


def test_model_metadata_provider_protocol_is_structural() -> None:
    class Provider:
        def resolve_model_metadata(
            self, provider: str, model_id: str
        ) -> ModelMetadataLookup:
            return ModelMetadataLookup(status="unavailable")

    class NotAProvider:
        pass

    assert isinstance(Provider(), ModelMetadataProvider)
    assert not isinstance(NotAProvider(), ModelMetadataProvider)


# ---------------------------------------------------------------------------
# JSON schema and roundtrip for every entity
# ---------------------------------------------------------------------------

ENTITY_EXAMPLES: List[BaseModel] = [
    complete_discovery([WorkOperation.TRACKING, WorkOperation.MODEL_INFERENCE]),
    incomplete_discovery([], ["step_declaration_missing:$steps.a"]),
    RestrictionCondition(
        runtimes=[Runtime.HOSTED_SERVERLESS],
        step_execution_modes=[StepExecutionMode.REMOTE],
        input_modes=[RuntimeInputMode.VIDEO],
        configuration_equals={"disable_sink": False},
    ),
    RestrictionMetadata(
        code="stateful_video_state_resets_on_stateless_http", severity=Severity.SOFT
    ),
    ModelMetadata(
        model_type="rfdetr", model_variant="base", task_type="object-detection"
    ),
    ModelMetadataLookup(status="available", metadata=ModelMetadata(task_type="ocr")),
    ModelMetadataLookup(status="disabled"),
]


@pytest.mark.parametrize("entity", ENTITY_EXAMPLES, ids=lambda e: type(e).__name__)
def test_entity_roundtrips_through_json_with_type_discriminator(
    entity: BaseModel,
) -> None:
    payload = json.loads(entity.model_dump_json())

    assert payload["type"] == entity.type
    assert entity.model_dump()["type"] == entity.type
    assert type(entity).model_validate(payload) == entity
    assert type(entity).model_validate_json(json.dumps(payload)) == entity


@pytest.mark.parametrize(
    "entity_type",
    [
        Discovery[WorkOperation],
        Discovery[RestrictionMetadata],
        RestrictionCondition,
        RestrictionMetadata,
        ModelMetadata,
        ModelMetadataLookup,
    ],
    ids=lambda t: t.__name__,
)
def test_entity_schema_exports_type_const_and_default(
    entity_type: Type[BaseModel],
) -> None:
    schema = entity_type.model_json_schema()

    json.dumps(schema)  # exportable, no python callables
    type_property = schema["properties"]["type"]
    expected = entity_type.model_fields["type"].default
    assert type_property["const"] == expected
    assert type_property["default"] == expected
    assert not (REJECTED_FIELD_NAMES & set(_all_property_names(schema)))


def test_restriction_metadata_schema_pins_code_pattern() -> None:
    schema = RestrictionMetadata.model_json_schema()

    assert schema["properties"]["code"]["pattern"] == "^[a-z][a-z0-9_]*$"
    assert schema["properties"]["code"]["minLength"] == 1
