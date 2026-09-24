"""Contract tests for ``roboflow_workflows.execution_engine.entities.workload``.

Covers the dependency-light declaration primitives: enum relocation
(identity through ``prototypes.block``), the closed ``WorkOperation`` set, the
``DiscoveryProblem`` contract (closed codes, open JSON details, identity-based
deduplication), ``Discovery`` completeness/dedup/sort rules,
``RestrictionCondition`` / ``RestrictionMetadata`` validation, model-metadata
status rules, ``normalize_declaration`` and JSON-schema/roundtrip for every
entity.
"""

import json
import subprocess
import sys
from typing import Any, Dict, List, Tuple, Type, Union

import pytest
from pydantic import BaseModel, ValidationError
from roboflow_workflows.execution_engine.entities.workload import (
    Discovery,
    DiscoveryProblem,
    DiscoveryProblemCode,
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
    custom_python_internals_unknown_problem,
    declaration_failed_problem,
    declaration_unavailable_problem,
    ensure_model_metadata_status_consistent,
    incomplete_discovery,
    invalid_resource_identifier_problem,
    normalize_declaration,
    opaque_remote_workflow_problem,
    unresolved_selector_problem,
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
# DiscoveryProblem
# ---------------------------------------------------------------------------


def _problem(
    code: DiscoveryProblemCode = DiscoveryProblemCode.DECLARATION_UNAVAILABLE,
    description: str = "Step `$steps.a` does not declare its operations.",
    **details: Any,
) -> DiscoveryProblem:
    return DiscoveryProblem(code=code, description=description, details=details)


def test_discovery_problem_code_has_exactly_the_contract_members() -> None:
    assert {member.name: member.value for member in DiscoveryProblemCode} == {
        "DECLARATION_UNAVAILABLE": "declaration_unavailable",
        "DECLARATION_FAILED": "declaration_failed",
        "UNRESOLVED_SELECTOR": "unresolved_selector",
        "INVALID_RESOURCE_IDENTIFIER": "invalid_resource_identifier",
        "OPAQUE_REMOTE_WORKFLOW": "opaque_remote_workflow",
        "CUSTOM_PYTHON_INTERNALS_UNKNOWN": "custom_python_internals_unknown",
    }
    assert issubclass(DiscoveryProblemCode, str)


def test_discovery_problem_serializes_code_description_and_details() -> None:
    problem = _problem(
        code=DiscoveryProblemCode.UNRESOLVED_SELECTOR,
        description="Field `lmm_type` of step `$steps.a` is set by a selector.",
        node_id="$steps.a",
        declaration="operations",
        field="lmm_type",
        selector="$inputs.model",
    )

    payload = json.loads(problem.model_dump_json())

    assert payload == {
        "type": "discovery_problem_v1",
        "code": "unresolved_selector",
        "description": "Field `lmm_type` of step `$steps.a` is set by a selector.",
        "details": {
            "node_id": "$steps.a",
            "declaration": "operations",
            "field": "lmm_type",
            "selector": "$inputs.model",
        },
    }
    assert DiscoveryProblem.model_validate(payload) == problem
    assert DiscoveryProblem.model_validate_json(json.dumps(payload)) == problem


def test_discovery_problem_details_default_to_an_empty_map() -> None:
    problem = DiscoveryProblem(
        code=DiscoveryProblemCode.DECLARATION_FAILED, description="Hook failed."
    )

    assert problem.details == {}
    assert problem.type == "discovery_problem_v1"


def test_discovery_problem_details_keep_nested_json_data() -> None:
    # `details` is a loose-end contract: an open JSON map, understood per code,
    # with no per-code model constraining its shape.
    details = {
        "node_id": "$steps.a",
        "declaration": "resources",
        "offending_fields": ["model_id", "provider"],
        "context": {"nested": {"depth": 2}, "flag": False, "missing": None},
    }

    problem = DiscoveryProblem(
        code=DiscoveryProblemCode.INVALID_RESOURCE_IDENTIFIER,
        description="Blank identifiers.",
        details=details,
    )

    assert problem.details == details
    assert json.loads(problem.model_dump_json())["details"] == details
    assert DiscoveryProblem.model_validate(json.loads(problem.model_dump_json())) == (
        problem
    )


@pytest.mark.parametrize("description", ["", "   ", "\n\t"])
def test_discovery_problem_rejects_blank_description(description: str) -> None:
    with pytest.raises(ValidationError):
        DiscoveryProblem(
            code=DiscoveryProblemCode.DECLARATION_FAILED, description=description
        )


def test_discovery_problem_rejects_invalid_values() -> None:
    with pytest.raises(ValidationError):
        DiscoveryProblem(code="made_up_code", description="x")
    with pytest.raises(ValidationError):
        # details must be JSON data - a python object is not
        DiscoveryProblem(
            code=DiscoveryProblemCode.DECLARATION_FAILED,
            description="x",
            details={"callback": object()},
        )
    with pytest.raises(ValidationError):
        DiscoveryProblem(
            code=DiscoveryProblemCode.DECLARATION_FAILED,
            description="x",
            details={"": "blank key"},
        )
    with pytest.raises(ValidationError):
        DiscoveryProblem(
            code=DiscoveryProblemCode.DECLARATION_FAILED,
            description="x",
            severity="hard",
        )
    with pytest.raises(ValidationError):
        DiscoveryProblem.model_validate(
            {
                "type": "not_a_problem",
                "code": "declaration_failed",
                "description": "x",
                "details": {},
            }
        )


def test_discovery_problem_is_frozen() -> None:
    problem = _problem()
    with pytest.raises(ValidationError):
        problem.description = "other"  # type: ignore[misc]


def test_discovery_problem_identity_ignores_description_and_key_order() -> None:
    first = _problem(description="A wording.", node_id="$steps.a", field="model_id")
    reordered = _problem(
        description="Another wording.", field="model_id", node_id="$steps.a"
    )
    other_details = _problem(description="A wording.", node_id="$steps.b")

    assert first.identity() == reordered.identity()
    assert first.identity() != other_details.identity()


def test_discovery_problem_factories_carry_the_documented_details() -> None:
    unavailable = declaration_unavailable_problem(
        node_id="$steps.a", declaration="operations", block_type="plugin/block@v1"
    )
    failed = declaration_failed_problem(node_id="$steps.a", declaration="resources")
    selector = unresolved_selector_problem(
        node_id="$steps.a",
        declaration="restrictions",
        field="fire_and_forget",
        selector="$inputs.wait",
    )
    resource_selector = unresolved_selector_problem(
        node_id="$steps.a",
        declaration="resources",
        field="model_id",
        selector="$inputs.model",
        resource_type="roboflow_platform_model",
    )
    invalid = invalid_resource_identifier_problem(
        node_id="$steps.a",
        declaration="resources",
        field="model_id",
        resource_type="third_party_model",
    )
    opaque = opaque_remote_workflow_problem(
        node_id="$steps.a", declaration="operations"
    )
    custom_python = custom_python_internals_unknown_problem(
        node_id="$steps.a", declaration="operations"
    )

    assert unavailable.code is DiscoveryProblemCode.DECLARATION_UNAVAILABLE
    assert unavailable.details == {
        "node_id": "$steps.a",
        "declaration": "operations",
        "block_type": "plugin/block@v1",
    }
    assert failed.code is DiscoveryProblemCode.DECLARATION_FAILED
    assert failed.details == {"node_id": "$steps.a", "declaration": "resources"}
    assert selector.code is DiscoveryProblemCode.UNRESOLVED_SELECTOR
    assert selector.details == {
        "node_id": "$steps.a",
        "declaration": "restrictions",
        "field": "fire_and_forget",
        "selector": "$inputs.wait",
    }
    assert resource_selector.details["resource_type"] == "roboflow_platform_model"
    assert invalid.code is DiscoveryProblemCode.INVALID_RESOURCE_IDENTIFIER
    assert invalid.details == {
        "node_id": "$steps.a",
        "declaration": "resources",
        "field": "model_id",
        "resource_type": "third_party_model",
    }
    assert opaque.code is DiscoveryProblemCode.OPAQUE_REMOTE_WORKFLOW
    assert opaque.details == {"node_id": "$steps.a", "declaration": "operations"}
    assert custom_python.code is DiscoveryProblemCode.CUSTOM_PYTHON_INTERNALS_UNKNOWN
    assert custom_python.details == {
        "node_id": "$steps.a",
        "declaration": "operations",
    }
    for problem in (
        unavailable,
        failed,
        selector,
        resource_selector,
        invalid,
        opaque,
        custom_python,
    ):
        assert problem.description.strip() == problem.description
        assert len(problem.description) > 20, problem.code


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_complete_empty_discovery_is_known_absence() -> None:
    discovery = complete_discovery([])

    assert discovery.items == []
    assert discovery.complete is True
    assert discovery.unknown_reasons == []
    assert discovery.model_dump() == {
        "type": "discovery_v1",
        "items": [],
        "complete": True,
        "unknown_reasons": [],
    }


def test_incomplete_discovery_keeps_known_items_and_reasons() -> None:
    problem = custom_python_internals_unknown_problem(
        node_id="$steps.custom", declaration="operations"
    )

    discovery = incomplete_discovery([WorkOperation.CUSTOM_PYTHON], [problem])

    assert discovery.items == [WorkOperation.CUSTOM_PYTHON]
    assert discovery.complete is False
    assert discovery.unknown_reasons == [problem]


def test_incomplete_discovery_without_reasons_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Discovery(items=[], complete=False, unknown_reasons=[])
    with pytest.raises(ValidationError):
        incomplete_discovery([WorkOperation.TRACKING], [])


def test_complete_discovery_with_reasons_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Discovery(items=[], complete=True, unknown_reasons=[_problem()])


def test_discovery_rejects_reasons_that_are_not_problems() -> None:
    with pytest.raises(ValidationError):
        # the version "1" wire format (a bare string) is no longer accepted
        Discovery(
            items=[],
            complete=False,
            unknown_reasons=["step_resources_unknown:$steps.a"],
        )
    with pytest.raises(ValidationError):
        Discovery(
            items=[],
            complete=False,
            unknown_reasons=[{"code": "declaration_failed", "description": "  "}],
        )


def test_discovery_accepts_problems_from_their_wire_form() -> None:
    discovery = Discovery[WorkOperation].model_validate(
        {
            "items": ["tracking"],
            "complete": False,
            "unknown_reasons": [
                {
                    "code": "declaration_failed",
                    "description": "The operations declaration could not be read.",
                    "details": {"node_id": "$steps.a", "declaration": "operations"},
                }
            ],
        }
    )

    reason = discovery.unknown_reasons[0]
    assert isinstance(reason, DiscoveryProblem)
    assert reason.code is DiscoveryProblemCode.DECLARATION_FAILED
    assert reason.details == {"node_id": "$steps.a", "declaration": "operations"}


def test_discovery_deduplicates_and_sorts_enum_items_and_reasons() -> None:
    first = _problem(node_id="$steps.x", declaration="operations")
    second = _problem(
        code=DiscoveryProblemCode.OPAQUE_REMOTE_WORKFLOW,
        description="Child is remote.",
        node_id="$steps.x",
        declaration="operations",
    )

    discovery = Discovery[WorkOperation](
        items=[
            WorkOperation.TRACKING,
            WorkOperation.MODEL_INFERENCE,
            WorkOperation.TRACKING,
            "image_crop",
        ],
        complete=False,
        unknown_reasons=[second, first, second],
    )

    assert discovery.items == [
        WorkOperation.IMAGE_CROP,
        WorkOperation.MODEL_INFERENCE,
        WorkOperation.TRACKING,
    ]
    # sorted by (code, canonical details): `declaration_unavailable` first
    assert discovery.unknown_reasons == [first, second]


def test_discovery_reason_identity_ignores_details_insertion_order() -> None:
    one_order = _problem(
        code=DiscoveryProblemCode.UNRESOLVED_SELECTOR,
        description="Selector unresolved.",
        node_id="$steps.a",
        field="model_id",
        selector="$inputs.model",
    )
    other_order = _problem(
        code=DiscoveryProblemCode.UNRESOLVED_SELECTOR,
        description="Selector unresolved.",
        selector="$inputs.model",
        field="model_id",
        node_id="$steps.a",
    )

    discovery = incomplete_discovery([], [one_order, other_order])

    assert discovery.unknown_reasons == [one_order]


def test_discovery_reasons_with_equal_identity_pick_one_description() -> None:
    louder = _problem(description="Zulu wording.", node_id="$steps.a")
    quieter = _problem(description="Alpha wording.", node_id="$steps.a")

    forwards = incomplete_discovery([], [louder, quieter])
    backwards = incomplete_discovery([], [quieter, louder])

    assert forwards.unknown_reasons == backwards.unknown_reasons
    assert [reason.description for reason in forwards.unknown_reasons] == (
        ["Alpha wording."]
    )


def test_discovery_keeps_distinct_contexts_apart() -> None:
    first_step = unresolved_selector_problem(
        node_id="$steps.a",
        declaration="resources",
        field="model_id",
        selector="$inputs.model",
    )
    second_step = unresolved_selector_problem(
        node_id="$steps.b",
        declaration="resources",
        field="model_id",
        selector="$inputs.model",
    )
    other_field = unresolved_selector_problem(
        node_id="$steps.a",
        declaration="resources",
        field="provider",
        selector="$inputs.model",
    )
    other_selector = unresolved_selector_problem(
        node_id="$steps.a",
        declaration="resources",
        field="model_id",
        selector="$inputs.other_model",
    )

    discovery = incomplete_discovery(
        [], [other_selector, second_step, other_field, first_step]
    )

    # four distinct problems, ordered by (code, canonical details JSON): the
    # details keys compare in sorted order, so `field` decides before `node_id`
    assert discovery.unknown_reasons == [
        first_step,
        other_selector,
        second_step,
        other_field,
    ]


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
    assert discovery.type == "discovery_v1"
    assert discovery.model_dump()["type"] == "discovery_v1"
    assert discovery.model_dump(mode="json") == {
        "type": "discovery_v1",
        "items": ["tracking"],
        "complete": True,
        "unknown_reasons": [],
    }


def test_discovery_is_frozen() -> None:
    discovery = complete_discovery([WorkOperation.TRACKING])
    with pytest.raises(ValidationError):
        discovery.complete = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# normalize_declaration
# ---------------------------------------------------------------------------


def test_normalize_declaration_maps_none_to_unknown() -> None:
    unavailable = declaration_unavailable_problem(
        node_id="$steps.crop", declaration="operations"
    )

    result = normalize_declaration(None, unavailable)

    assert result == Discovery(items=[], complete=False, unknown_reasons=[unavailable])


def test_normalize_declaration_maps_list_to_complete() -> None:
    unused = declaration_unavailable_problem(
        node_id="$steps.a", declaration="operations"
    )
    assert normalize_declaration([], unused) == complete_discovery([])
    assert normalize_declaration(
        [WorkOperation.TRACKING, WorkOperation.IMAGE_CROP], unused
    ) == complete_discovery([WorkOperation.IMAGE_CROP, WorkOperation.TRACKING])


def test_normalize_declaration_revalidates_discovery_preserving_items() -> None:
    declared = incomplete_discovery(
        [WorkOperation.CUSTOM_PYTHON],
        [
            custom_python_internals_unknown_problem(
                node_id="$steps.c", declaration="operations"
            )
        ],
    )

    result = normalize_declaration(
        declared,
        declaration_unavailable_problem(node_id="$steps.c", declaration="operations"),
    )

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
        "type": "restriction_condition_v1",
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
        "type": "restriction_v1",
        "code": "writes_to_ephemeral_disk",
        "severity": "soft",
        "when": {
            "type": "restriction_condition_v1",
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
    with pytest.raises(ValidationError):
        DiscoveryProblem(
            code=DiscoveryProblemCode.DECLARATION_FAILED, description="x", score=1.0
        )


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
    incomplete_discovery(
        [],
        [declaration_unavailable_problem(node_id="$steps.a", declaration="resources")],
    ),
    declaration_failed_problem(
        node_id="$steps.a", declaration="operations", block_type="plugin/block@v1"
    ),
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


# (entity, path to the tagged object inside its JSON dump); an empty path is
# the entity itself.
TYPE_TAG_LOCATIONS: List[Tuple[BaseModel, Tuple[Union[str, int], ...]]] = [
    *((entity, ()) for entity in ENTITY_EXAMPLES),
    (
        incomplete_discovery(
            [],
            [declaration_failed_problem(node_id="$steps.a", declaration="resources")],
        ),
        ("unknown_reasons", 0),
    ),
    (
        RestrictionMetadata(code="writes_to_ephemeral_disk", severity=Severity.HARD),
        ("when",),
    ),
    (
        ModelMetadataLookup(
            status="available", metadata=ModelMetadata(task_type="ocr")
        ),
        ("metadata",),
    ),
]


@pytest.mark.parametrize(
    "wrong_type_template", ["something_else", "{unversioned}", "{unversioned}_v2"]
)
@pytest.mark.parametrize(
    "entity, tag_path",
    TYPE_TAG_LOCATIONS,
    ids=[
        f"{type(entity).__name__}:{'.'.join(map(str, path)) or 'self'}"
        for entity, path in TYPE_TAG_LOCATIONS
    ],
)
def test_entity_rejects_unknown_unversioned_and_other_version_types(
    entity: BaseModel,
    tag_path: Tuple[Union[str, int], ...],
    wrong_type_template: str,
) -> None:
    payload = entity.model_dump(mode="json")
    tagged = payload
    for key in tag_path:
        tagged = tagged[key]
    unversioned = tagged["type"].removesuffix("_v1")
    tagged["type"] = wrong_type_template.format(unversioned=unversioned)

    with pytest.raises(ValidationError):
        type(entity).model_validate(payload)


@pytest.mark.parametrize(
    "entity_type, expected_type",
    [
        (Discovery[WorkOperation], "discovery_v1"),
        (Discovery[RestrictionMetadata], "discovery_v1"),
        (DiscoveryProblem, "discovery_problem_v1"),
        (RestrictionCondition, "restriction_condition_v1"),
        (RestrictionMetadata, "restriction_v1"),
        (ModelMetadata, "model_metadata_v1"),
        (ModelMetadataLookup, "model_metadata_lookup_v1"),
    ],
    ids=lambda value: value.__name__ if isinstance(value, type) else value,
)
def test_entity_schema_exports_type_const_and_default(
    entity_type: Type[BaseModel], expected_type: str
) -> None:
    schema = entity_type.model_json_schema()

    json.dumps(schema)  # exportable, no python callables
    type_property = schema["properties"]["type"]
    assert type_property["const"] == expected_type
    assert type_property["default"] == expected_type
    assert entity_type.model_fields["type"].default == expected_type
    assert not (REJECTED_FIELD_NAMES & set(_all_property_names(schema)))


def test_restriction_metadata_schema_pins_code_pattern() -> None:
    schema = RestrictionMetadata.model_json_schema()

    assert schema["properties"]["code"]["pattern"] == "^[a-z][a-z0-9_]*$"
    assert schema["properties"]["code"]["minLength"] == 1
