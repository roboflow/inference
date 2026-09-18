"""Tests for the workload hooks and compatibility rules added to
``roboflow_workflows.prototypes.block``:

* ``discover_work_operations()`` / ``discover_portable_restrictions()``
  default to ``None`` (unknown) on a concrete manifest and follow the
  None / list / Discovery convention through ``normalize_declaration``;
* the three portable restriction presets;
* resource-entity envelope compatibility: old dicts without ``type`` parse,
  an explicit wrong ``type`` is rejected, a ``resource_type`` / metadata
  contradiction is rejected, ``to_dict()`` keeps the legacy shape while
  ``model_dump(mode="json")`` carries discriminators recursively, eq/hash
  are unchanged and resolver/kwargs stay excluded.
"""

import json
from typing import List, Literal

import pytest
from pydantic import ValidationError
from roboflow_workflows.errors import BlockInterfaceError
from roboflow_workflows.execution_engine.entities.base import OutputDefinition
from roboflow_workflows.execution_engine.entities.workload import (
    Discovery,
    RestrictionCondition,
    RestrictionMetadata,
    WorkOperation,
    complete_discovery,
    incomplete_discovery,
    normalize_declaration,
)
from roboflow_workflows.prototypes.block import (
    COOLDOWN_HTTP_SOFT_PORTABLE_RESTRICTION,
    COOLDOWN_HTTP_SOFT_RESTRICTION,
    REGISTERED_RESOURCE_METADATA_TYPES,
    STATEFUL_VIDEO_HTTP_SOFT_PORTABLE_RESTRICTION,
    STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION,
    STILL_IMAGE_INPUT_SOFT_PORTABLE_RESTRICTION,
    STILL_IMAGE_INPUT_SOFT_RESTRICTION,
    DependentResource,
    DependentResourceType,
    ModelExecutionLocation,
    ModelRequiredAction,
    RoboflowPlatformModelMetadata,
    RoboflowPlatformProjectMetadata,
    Runtime,
    RuntimeInputMode,
    RuntimeRestriction,
    Severity,
    StepExecutionMode,
    ThirdPartyModelMetadata,
    WorkflowBlockManifest,
    is_workflow_selector,
    roboflow_platform_model,
    roboflow_platform_project,
    third_party_model,
)


class _PlainManifest(WorkflowBlockManifest):
    type: Literal["test/plain@v1"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="output")]


class _AnnotatedManifest(WorkflowBlockManifest):
    type: Literal["test/annotated@v1"]
    resize: bool = True

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="output")]

    def discover_work_operations(self) -> List[WorkOperation]:
        if self.resize:
            return [WorkOperation.IMAGE_RESIZE]
        return [WorkOperation.IMAGE_TRANSFORM]

    def discover_portable_restrictions(self) -> List[RestrictionMetadata]:
        return [STILL_IMAGE_INPUT_SOFT_PORTABLE_RESTRICTION]


class _CustomPythonLikeManifest(WorkflowBlockManifest):
    type: Literal["test/custom@v1"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="output")]

    def discover_work_operations(self) -> Discovery[WorkOperation]:
        return incomplete_discovery(
            [WorkOperation.CUSTOM_PYTHON],
            [f"custom_python_internal_operations_unknown:$steps.{self.name}"],
        )

    def discover_portable_restrictions(self) -> List[RestrictionMetadata]:
        return []


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------


def test_hooks_default_to_none_on_a_concrete_manifest() -> None:
    manifest = _PlainManifest.model_validate({"type": "test/plain@v1", "name": "step"})

    assert manifest.discover_work_operations() is None
    assert manifest.discover_portable_restrictions() is None
    assert manifest.discover_dependent_resources() is None


def test_hooks_are_instance_methods_not_classmethods() -> None:
    assert not isinstance(
        WorkflowBlockManifest.__dict__["discover_work_operations"], classmethod
    )
    assert not isinstance(
        WorkflowBlockManifest.__dict__["discover_portable_restrictions"], classmethod
    )


def test_unannotated_manifest_normalises_to_unknown() -> None:
    manifest = _PlainManifest.model_validate({"type": "test/plain@v1", "name": "step"})

    operations = normalize_declaration(
        manifest.discover_work_operations(), "step_declaration_missing:$steps.step"
    )
    restrictions = normalize_declaration(
        manifest.discover_portable_restrictions(),
        "step_declaration_missing:$steps.step",
    )

    assert operations == Discovery(
        items=[],
        complete=False,
        unknown_reasons=["step_declaration_missing:$steps.step"],
    )
    assert restrictions == Discovery(
        items=[],
        complete=False,
        unknown_reasons=["step_declaration_missing:$steps.step"],
    )


def test_literal_manifest_settings_select_operations() -> None:
    resize = _AnnotatedManifest.model_validate(
        {"type": "test/annotated@v1", "name": "a", "resize": True}
    )
    transform = _AnnotatedManifest.model_validate(
        {"type": "test/annotated@v1", "name": "a", "resize": False}
    )

    assert normalize_declaration(
        resize.discover_work_operations(), "unused:$steps.a"
    ) == complete_discovery([WorkOperation.IMAGE_RESIZE])
    assert normalize_declaration(
        transform.discover_work_operations(), "unused:$steps.a"
    ) == complete_discovery([WorkOperation.IMAGE_TRANSFORM])
    assert normalize_declaration(
        resize.discover_portable_restrictions(), "unused:$steps.a"
    ) == complete_discovery([STILL_IMAGE_INPUT_SOFT_PORTABLE_RESTRICTION])


def test_discovery_return_value_carries_explicit_completeness() -> None:
    manifest = _CustomPythonLikeManifest.model_validate(
        {"type": "test/custom@v1", "name": "custom"}
    )

    operations = normalize_declaration(
        manifest.discover_work_operations(), "unused:$steps.custom"
    )
    restrictions = normalize_declaration(
        manifest.discover_portable_restrictions(), "unused:$steps.custom"
    )

    assert operations.items == [WorkOperation.CUSTOM_PYTHON]
    assert operations.complete is False
    assert operations.unknown_reasons == [
        "custom_python_internal_operations_unknown:$steps.custom"
    ]
    assert restrictions == complete_discovery([])


def test_legacy_restriction_api_is_untouched() -> None:
    assert WorkflowBlockManifest.get_restrictions() == []
    assert isinstance(WorkflowBlockManifest.__dict__["get_restrictions"], classmethod)
    assert RuntimeRestriction.__dataclass_fields__.keys() == {
        "severity",
        "note",
        "applies_to_runtimes",
        "applies_to_step_execution_modes",
        "applies_to_input_modes",
    }
    assert STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION.to_dict() == {
        "severity": "soft",
        "note": STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION.note,
        "applies_to_runtimes": ["hosted_serverless", "dedicated_deployment"],
        "applies_to_step_execution_modes": ["remote"],
        "applies_to_input_modes": ["video"],
    }


def test_manifest_model_config_keeps_validate_assignment() -> None:
    assert WorkflowBlockManifest.model_config.get("validate_assignment") is True


# ---------------------------------------------------------------------------
# Portable presets
# ---------------------------------------------------------------------------


def test_stateful_video_portable_preset_mirrors_legacy_axes() -> None:
    preset = STATEFUL_VIDEO_HTTP_SOFT_PORTABLE_RESTRICTION

    assert preset.code == "stateful_video_state_resets_on_stateless_http"
    assert preset.severity is Severity.SOFT
    assert preset.when == RestrictionCondition(
        runtimes=[Runtime.HOSTED_SERVERLESS, Runtime.DEDICATED_DEPLOYMENT],
        step_execution_modes=[StepExecutionMode.REMOTE],
        input_modes=[RuntimeInputMode.VIDEO],
    )
    assert set(preset.when.runtimes) == set(
        STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION.applies_to_runtimes
    )
    assert preset.when.step_execution_modes == (
        STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION.applies_to_step_execution_modes
    )
    assert preset.when.input_modes == (
        STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION.applies_to_input_modes
    )


def test_cooldown_portable_preset_mirrors_legacy_axes() -> None:
    preset = COOLDOWN_HTTP_SOFT_PORTABLE_RESTRICTION

    assert preset.code == "cooldown_timer_resets_on_stateless_http"
    assert preset.severity is Severity.SOFT
    assert preset.when == RestrictionCondition(
        runtimes=[Runtime.HOSTED_SERVERLESS, Runtime.DEDICATED_DEPLOYMENT],
        step_execution_modes=[StepExecutionMode.REMOTE],
    )
    assert preset.when.input_modes is None
    assert COOLDOWN_HTTP_SOFT_RESTRICTION.applies_to_input_modes is None


def test_still_image_portable_preset_mirrors_legacy_axes() -> None:
    preset = STILL_IMAGE_INPUT_SOFT_PORTABLE_RESTRICTION

    assert preset.code == "temporal_block_no_benefit_on_still_image"
    assert preset.severity is Severity.SOFT
    assert preset.when == RestrictionCondition(input_modes=[RuntimeInputMode.IMAGE])
    assert preset.when.runtimes is None
    assert preset.when.step_execution_modes is None
    assert STILL_IMAGE_INPUT_SOFT_RESTRICTION.applies_to_input_modes == [
        RuntimeInputMode.IMAGE
    ]


def test_portable_presets_have_no_note_and_serialise_with_discriminators() -> None:
    for preset in (
        STATEFUL_VIDEO_HTTP_SOFT_PORTABLE_RESTRICTION,
        COOLDOWN_HTTP_SOFT_PORTABLE_RESTRICTION,
        STILL_IMAGE_INPUT_SOFT_PORTABLE_RESTRICTION,
    ):
        payload = preset.model_dump(mode="json")
        assert "note" not in payload
        assert payload["type"] == "restriction"
        assert payload["when"]["type"] == "restriction_condition"
        assert RestrictionMetadata.model_validate(payload) == preset


# ---------------------------------------------------------------------------
# Resource entities: discriminators and envelope compatibility
# ---------------------------------------------------------------------------


def test_resource_entities_carry_defaulted_discriminators() -> None:
    assert DependentResource.model_fields["type"].default == "dependent_resource"
    assert (
        RoboflowPlatformModelMetadata.model_fields["type"].default
        == "roboflow_platform_model"
    )
    assert (
        RoboflowPlatformProjectMetadata.model_fields["type"].default
        == "roboflow_platform_project"
    )
    assert ThirdPartyModelMetadata.model_fields["type"].default == "third_party_model"
    # The metadata discriminators coincide with the `resource_type` values.
    for resource_type, metadata_type in REGISTERED_RESOURCE_METADATA_TYPES.items():
        assert metadata_type.model_fields["type"].default == resource_type.value


def test_model_dump_json_includes_discriminators_recursively() -> None:
    assert roboflow_platform_model(model_id="my_project/3").model_dump(mode="json") == {
        "type": "dependent_resource",
        "resource_type": "roboflow_platform_model",
        "metadata": {
            "type": "roboflow_platform_model",
            "model_id": "my_project/3",
            "required_action": "execution",
            "execution_location": "environment_defined",
        },
    }
    assert roboflow_platform_project(project_url="my_dataset").model_dump(
        mode="json"
    ) == {
        "type": "dependent_resource",
        "resource_type": "roboflow_platform_project",
        "metadata": {"type": "roboflow_platform_project", "project_url": "my_dataset"},
    }
    assert third_party_model(provider="openai", model_id="gpt-4o").model_dump(
        mode="json"
    ) == {
        "type": "dependent_resource",
        "resource_type": "third_party_model",
        "metadata": {
            "type": "third_party_model",
            "provider": "openai",
            "model_id": "gpt-4o",
        },
    }


def test_python_mode_dump_also_includes_discriminators() -> None:
    payload = roboflow_platform_model(model_id="my_project/3").model_dump()

    assert payload["type"] == "dependent_resource"
    assert payload["metadata"]["type"] == "roboflow_platform_model"
    assert payload["resource_type"] is DependentResourceType.ROBOFLOW_PLATFORM_MODEL


def test_to_dict_keeps_legacy_shape_without_type_keys() -> None:
    assert roboflow_platform_model(model_id="my_project/3").to_dict() == {
        "resource_type": "roboflow_platform_model",
        "metadata": {
            "model_id": "my_project/3",
            "required_action": "execution",
            "execution_location": "environment_defined",
        },
    }
    assert roboflow_platform_model(
        model_id="my_project/3", required_action=ModelRequiredAction.ACCESS
    ).to_dict() == {
        "resource_type": "roboflow_platform_model",
        "metadata": {"model_id": "my_project/3", "required_action": "access"},
    }
    assert roboflow_platform_project(project_url="my_dataset").to_dict() == {
        "resource_type": "roboflow_platform_project",
        "metadata": {"project_url": "my_dataset"},
    }
    assert third_party_model(provider="anthropic", model_id="claude").to_dict() == {
        "resource_type": "third_party_model",
        "metadata": {"provider": "anthropic", "model_id": "claude"},
    }


@pytest.mark.parametrize(
    "resource",
    [
        roboflow_platform_model(model_id="my_project/3"),
        roboflow_platform_model(
            model_id="my_project/3", required_action=ModelRequiredAction.ACCESS
        ),
        roboflow_platform_model(
            model_id="clip/ViT-B-16", execution_location=ModelExecutionLocation.LOCAL
        ),
        roboflow_platform_project(project_url="my_dataset"),
        third_party_model(provider="openai", model_id="gpt-4o"),
    ],
    ids=lambda r: f"{r.resource_type.value}:{type(r.metadata).__name__}",
)
def test_old_envelope_without_type_and_new_envelope_both_parse(
    resource: DependentResource,
) -> None:
    legacy = resource.to_dict()
    discriminated = resource.model_dump(mode="json")

    assert "type" not in legacy and "type" not in legacy["metadata"]
    for payload in (legacy, discriminated, json.loads(json.dumps(discriminated))):
        parsed = DependentResource.model_validate(payload)
        assert parsed == resource
        assert hash(parsed) == hash(resource)
        assert type(parsed.metadata) is type(resource.metadata)
        assert parsed.model_dump(mode="json") == discriminated


def test_explicit_wrong_envelope_type_is_rejected() -> None:
    with pytest.raises(ValidationError):
        DependentResource.model_validate(
            {
                "type": "resource",
                "resource_type": "third_party_model",
                "metadata": {"provider": "openai", "model_id": "gpt-4o"},
            }
        )


def test_explicit_wrong_metadata_type_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ThirdPartyModelMetadata(
            type="roboflow_platform_model", provider="a", model_id="b"
        )
    with pytest.raises(ValidationError):
        RoboflowPlatformModelMetadata.model_validate(
            {"type": "third_party_model", "model_id": "my_project/3"}
        )


def test_metadata_type_contradicting_resource_type_is_rejected() -> None:
    with pytest.raises(BlockInterfaceError):
        DependentResource.model_validate(
            {
                "resource_type": "roboflow_platform_model",
                "metadata": {
                    "type": "third_party_model",
                    "provider": "openai",
                    "model_id": "gpt-4o",
                },
            }
        )
    with pytest.raises(BlockInterfaceError):
        DependentResource.model_validate(
            {
                "resource_type": "third_party_model",
                "metadata": {"type": "roboflow_platform_project", "project_url": "x"},
            }
        )


def test_metadata_instance_contradicting_resource_type_is_rejected() -> None:
    with pytest.raises(BlockInterfaceError):
        DependentResource(
            resource_type=DependentResourceType.ROBOFLOW_PLATFORM_PROJECT,
            metadata=ThirdPartyModelMetadata(provider="openai", model_id="gpt-4o"),
        )


def test_equality_and_hash_are_unchanged() -> None:
    plain = roboflow_platform_model(model_id="$inputs.variant")
    with_resolver = roboflow_platform_model(
        model_id="$inputs.variant",
        model_id_resolver=lambda version: f"clip/{version}",
        model_registration_kwargs={"endpoint_type": "core-model"},
    )
    other = roboflow_platform_model(model_id="$inputs.other")

    assert with_resolver == plain
    assert hash(with_resolver) == hash(plain)
    assert len({plain, with_resolver}) == 1
    assert plain != other
    assert third_party_model(provider="a", model_id="b") == third_party_model(
        provider="a", model_id="b", model_id_resolver=lambda x: x
    )
    assert roboflow_platform_project(project_url="p") == roboflow_platform_project(
        project_url="p"
    )


def test_resolver_and_registration_kwargs_stay_excluded_everywhere() -> None:
    resource = roboflow_platform_model(
        model_id="clip/ViT-B-32",
        model_id_resolver=lambda version: f"clip/{version}",
        model_registration_kwargs={"endpoint_type": "core-model"},
    )
    third_party = third_party_model(
        provider="openrouter", model_id="$inputs.label", model_id_resolver=lambda x: x
    )

    for payload in (
        resource.to_dict()["metadata"],
        resource.model_dump()["metadata"],
        resource.model_dump(mode="json")["metadata"],
        json.loads(resource.model_dump_json())["metadata"],
        third_party.model_dump(mode="json")["metadata"],
    ):
        assert "model_id_resolver" not in payload
        assert "model_registration_kwargs" not in payload
    schema = json.dumps(DependentResource.model_json_schema())
    assert "model_id_resolver" not in schema
    assert "model_registration_kwargs" not in schema
    assert resource.metadata.model_id_resolver("ViT-B-32") == "clip/ViT-B-32"
    assert resource.metadata.model_registration_kwargs == {
        "endpoint_type": "core-model"
    }


def test_helpers_and_registry_are_unchanged() -> None:
    assert REGISTERED_RESOURCE_METADATA_TYPES == {
        DependentResourceType.ROBOFLOW_PLATFORM_MODEL: RoboflowPlatformModelMetadata,
        DependentResourceType.ROBOFLOW_PLATFORM_PROJECT: RoboflowPlatformProjectMetadata,
        DependentResourceType.THIRD_PARTY_MODEL: ThirdPartyModelMetadata,
    }
    assert is_workflow_selector("$inputs.model")
    assert is_workflow_selector("$steps.a.model_id")
    assert not is_workflow_selector("my_project/3")
    assert not is_workflow_selector(None)
    assert (
        roboflow_platform_model(model_id="m/1").metadata.requires_runtime_resolution()
        is False
    )
    assert (
        roboflow_platform_model(
            model_id="$inputs.m"
        ).metadata.requires_runtime_resolution()
        is True
    )


def test_resource_schema_shows_type_on_envelope_and_every_metadata_variant() -> None:
    schema = DependentResource.model_json_schema()

    assert schema["properties"]["type"] == {
        "const": "dependent_resource",
        "default": "dependent_resource",
        "title": "Type",
        "type": "string",
    }
    for name in (
        "RoboflowPlatformModelMetadata",
        "RoboflowPlatformProjectMetadata",
        "ThirdPartyModelMetadata",
    ):
        type_property = schema["$defs"][name]["properties"]["type"]
        assert type_property["const"] == type_property["default"]
    variants = {
        ref["$ref"].rsplit("/", 1)[-1]
        for ref in schema["properties"]["metadata"]["anyOf"]
    }
    assert variants == {
        "RoboflowPlatformModelMetadata",
        "RoboflowPlatformProjectMetadata",
        "ThirdPartyModelMetadata",
    }
