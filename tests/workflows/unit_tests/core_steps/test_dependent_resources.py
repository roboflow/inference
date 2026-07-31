"""
Tests for the dependent-resources discovery contract on block manifests
(``WorkflowBlockManifest.discover_dependent_resources()``).

Covers:
- the ``DependentResource`` envelope: metadata types are regulated by the
  Execution Engine (mismatches raise), serialization shape,
- one representative manifest per implementation pattern (direct model id,
  model + active-learning project, synthesized core-model id, multi-field
  composition, third-party model, backend routing, Roboflow project sink),
- a repo-wide coverage guard: every core block whose manifest declares a
  field accepting ``roboflow_model_id`` / ``roboflow_project`` selector
  kinds must override ``discover_dependent_resources()`` — with an explicit
  allowlist for blocks that only *carry* such values as generic payload.
"""

from typing import List, Literal, Optional, get_args

import pytest

from inference.core.roboflow_api import ModelEndpointType
from inference.core.workflows.core_steps.models.foundation.clip.v1 import (
    BlockManifest as ClipV1Manifest,
)
from inference.core.workflows.core_steps.models.foundation.openai.v2 import (
    BlockManifest as OpenAIV2Manifest,
)
from inference.core.workflows.core_steps.models.foundation.pp_ocr.v1 import (
    BlockManifest as PPOCRV1Manifest,
)
from inference.core.workflows.core_steps.models.foundation.qwen_vlm.v1 import (
    FINE_TUNED_NATIVE_LABEL,
)
from inference.core.workflows.core_steps.models.foundation.qwen_vlm.v1 import (
    BlockManifest as QwenVlmV1Manifest,
)
from inference.core.workflows.core_steps.models.roboflow.object_detection.v3 import (
    BlockManifest as ObjectDetectionV3Manifest,
)
from inference.core.workflows.core_steps.sinks.roboflow.dataset_upload.v2 import (
    BlockManifest as DatasetUploadV2Manifest,
)
from inference.core.workflows.core_steps.sinks.roboflow.model_monitoring_inference_aggregator.v1 import (
    BlockManifest as ModelMonitoringV1Manifest,
)
from inference.core.workflows.errors import BlockInterfaceError
from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.introspection.blocks_loader import (
    load_core_workflow_blocks,
)
from inference.core.workflows.execution_engine.introspection.schema_parser import (
    parse_block_manifest,
)
from inference.core.workflows.prototypes.block import (
    DependentResource,
    DependentResourceType,
    ModelExecutionLocation,
    ModelRequiredAction,
    RoboflowPlatformModelMetadata,
    RoboflowPlatformProjectMetadata,
    ThirdPartyModelMetadata,
    WorkflowBlockManifest,
    roboflow_platform_model,
    roboflow_platform_project,
    third_party_model,
)

# ---------------------------------------------------------------------------
# Envelope
# ---------------------------------------------------------------------------


def test_dependent_resource_rejects_metadata_not_matching_resource_type() -> None:
    with pytest.raises(BlockInterfaceError):
        DependentResource(
            resource_type=DependentResourceType.ROBOFLOW_PLATFORM_MODEL,
            metadata=ThirdPartyModelMetadata(provider="openai", model_id="gpt-4o"),
        )


def test_dependent_resource_factories_produce_registered_metadata_types() -> None:
    model = roboflow_platform_model(model_id="my_project/3")
    project = roboflow_platform_project(project_url="my_dataset")
    third_party = third_party_model(provider="openai", model_id="gpt-4o")

    assert model.resource_type is DependentResourceType.ROBOFLOW_PLATFORM_MODEL
    assert model.metadata == RoboflowPlatformModelMetadata(model_id="my_project/3")
    assert project.resource_type is DependentResourceType.ROBOFLOW_PLATFORM_PROJECT
    assert project.metadata == RoboflowPlatformProjectMetadata(project_url="my_dataset")
    assert third_party.resource_type is DependentResourceType.THIRD_PARTY_MODEL
    assert third_party.metadata == ThirdPartyModelMetadata(
        provider="openai", model_id="gpt-4o"
    )


def test_dependent_resource_serializes_to_plain_dict() -> None:
    resource = third_party_model(provider="anthropic", model_id="claude-sonnet-5")

    assert resource.to_dict() == {
        "resource_type": "third_party_model",
        "metadata": {"provider": "anthropic", "model_id": "claude-sonnet-5"},
    }


def test_roboflow_platform_model_defaults_to_environment_defined_execution() -> None:
    resource = roboflow_platform_model(model_id="my_project/3")

    assert resource.metadata.required_action is ModelRequiredAction.EXECUTION
    assert (
        resource.metadata.execution_location
        is ModelExecutionLocation.ENVIRONMENT_DEFINED
    )
    assert resource.to_dict() == {
        "resource_type": "roboflow_platform_model",
        "metadata": {
            "model_id": "my_project/3",
            "required_action": "execution",
            "execution_location": "environment_defined",
        },
    }


def test_roboflow_platform_model_access_only_omits_execution_location() -> None:
    resource = roboflow_platform_model(
        model_id="my_project/3", required_action=ModelRequiredAction.ACCESS
    )

    assert resource.metadata.execution_location is None
    assert resource.to_dict() == {
        "resource_type": "roboflow_platform_model",
        "metadata": {"model_id": "my_project/3", "required_action": "access"},
    }


def test_dependent_resource_round_trips_through_serialization() -> None:
    resources = [
        roboflow_platform_model(model_id="my_project/3"),
        roboflow_platform_model(
            model_id="my_project/3", required_action=ModelRequiredAction.ACCESS
        ),
        roboflow_platform_project(project_url="my_dataset"),
        third_party_model(provider="openai", model_id="gpt-4o"),
    ]

    for resource in resources:
        parsed = DependentResource.model_validate(resource.to_dict())
        assert parsed == resource
        assert type(parsed.metadata) is type(resource.metadata)


def test_model_id_resolver_is_excluded_from_serialization_and_equality() -> None:
    plain = roboflow_platform_model(model_id="$inputs.variant")
    with_resolver = roboflow_platform_model(
        model_id="$inputs.variant",
        model_id_resolver=lambda version: f"clip/{version}",
    )

    assert with_resolver == plain
    assert with_resolver.to_dict() == plain.to_dict()
    assert "model_id_resolver" not in with_resolver.to_dict()["metadata"]
    assert with_resolver.metadata.model_id_resolver("ViT-B-16") == "clip/ViT-B-16"


def test_model_registration_kwargs_are_excluded_from_serialization_and_equality() -> (
    None
):
    plain = roboflow_platform_model(model_id="clip/ViT-B-32")
    with_kwargs = roboflow_platform_model(
        model_id="clip/ViT-B-32",
        model_registration_kwargs={"endpoint_type": "core-model"},
    )

    assert with_kwargs == plain
    assert with_kwargs.to_dict() == plain.to_dict()
    assert "model_registration_kwargs" not in with_kwargs.to_dict()["metadata"]


def test_third_party_model_id_resolver_follows_the_same_contract() -> None:
    plain = third_party_model(provider="openrouter", model_id="$inputs.label")
    with_resolver = third_party_model(
        provider="openrouter",
        model_id="$inputs.label",
        model_id_resolver=lambda label: f"provider/{label}",
    )

    assert with_resolver == plain
    assert "model_id_resolver" not in with_resolver.to_dict()["metadata"]
    assert with_resolver.metadata.model_id_resolver("x") == "provider/x"


def test_roboflow_platform_model_metadata_enforces_action_location_consistency() -> (
    None
):
    with pytest.raises(BlockInterfaceError):
        RoboflowPlatformModelMetadata(model_id="m/1", execution_location=None)
    with pytest.raises(BlockInterfaceError):
        RoboflowPlatformModelMetadata(
            model_id="m/1",
            required_action=ModelRequiredAction.ACCESS,
            execution_location=ModelExecutionLocation.LOCAL,
        )


def test_roboflow_platform_model_metadata_reports_runtime_resolution() -> None:
    assert not RoboflowPlatformModelMetadata(
        model_id="my_project/3"
    ).requires_runtime_resolution()
    assert RoboflowPlatformModelMetadata(
        model_id="$inputs.model"
    ).requires_runtime_resolution()
    assert RoboflowPlatformModelMetadata(
        model_id="$steps.parser.model_id"
    ).requires_runtime_resolution()


def test_roboflow_platform_project_metadata_reports_runtime_resolution() -> None:
    assert not RoboflowPlatformProjectMetadata(
        project_url="my_dataset"
    ).requires_runtime_resolution()
    assert RoboflowPlatformProjectMetadata(
        project_url="$inputs.project"
    ).requires_runtime_resolution()


def test_third_party_model_metadata_reports_runtime_resolution() -> None:
    assert not ThirdPartyModelMetadata(
        provider="openai", model_id="gpt-4o"
    ).requires_runtime_resolution()
    assert ThirdPartyModelMetadata(
        provider="openai", model_id="$inputs.model_version"
    ).requires_runtime_resolution()


# ---------------------------------------------------------------------------
# Base contract
# ---------------------------------------------------------------------------


class _PlainManifest(WorkflowBlockManifest):
    type: Literal["test/plain@v1"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="output")]


def test_manifest_without_override_does_not_declare_dependencies() -> None:
    manifest = _PlainManifest.model_validate({"type": "test/plain@v1", "name": "step"})

    assert manifest.discover_dependent_resources() is None


# ---------------------------------------------------------------------------
# Representative manifests, one per implementation pattern
# ---------------------------------------------------------------------------


def test_object_detection_v3_declares_model_and_optional_project() -> None:
    manifest = ObjectDetectionV3Manifest.model_validate(
        {
            "type": "roboflow_core/roboflow_object_detection_model@v3",
            "name": "detector",
            "images": "$inputs.image",
            "model_id": "my_project/3",
            "disable_active_learning": False,
            "active_learning_target_dataset": "my_dataset",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="my_project/3"),
        roboflow_platform_project(project_url="my_dataset"),
    ]


def test_object_detection_v3_ignores_target_project_when_active_learning_disabled() -> (
    None
):
    # `disable_active_learning` defaults to True — a configured target project
    # is dead configuration then, not a dependency.
    manifest = ObjectDetectionV3Manifest.model_validate(
        {
            "type": "roboflow_core/roboflow_object_detection_model@v3",
            "name": "detector",
            "images": "$inputs.image",
            "model_id": "my_project/3",
            "active_learning_target_dataset": "my_dataset",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="my_project/3"),
    ]


def test_object_detection_v3_returns_selector_fed_model_id_verbatim() -> None:
    manifest = ObjectDetectionV3Manifest.model_validate(
        {
            "type": "roboflow_core/roboflow_object_detection_model@v3",
            "name": "detector",
            "images": "$inputs.image",
            "model_id": "$inputs.model",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="$inputs.model"),
    ]


def test_clip_v1_synthesizes_core_model_id_from_version() -> None:
    manifest = ClipV1Manifest.model_validate(
        {
            "type": "roboflow_core/clip@v1",
            "name": "embedder",
            "data": "$inputs.image",
            "version": "ViT-B-16",
        }
    )

    resources = manifest.discover_dependent_resources()

    assert resources == [
        roboflow_platform_model(model_id="clip/ViT-B-16"),
    ]
    # Core models must register in the model manager the same way
    # load_core_model() does.
    assert resources[0].metadata.model_registration_kwargs == {
        "endpoint_type": ModelEndpointType.CORE_MODEL
    }


def test_clip_v1_returns_selector_fed_version_verbatim() -> None:
    manifest = ClipV1Manifest.model_validate(
        {
            "type": "roboflow_core/clip@v1",
            "name": "embedder",
            "data": "$inputs.image",
            "version": "$inputs.variant",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="$inputs.variant"),
    ]


def test_pp_ocr_v1_composes_model_id_from_both_stage_fields() -> None:
    manifest = PPOCRV1Manifest.model_validate(
        {
            "type": "roboflow_core/pp_ocr@v1",
            "name": "ocr",
            "images": "$inputs.image",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="pp_ocr/small-small"),
    ]


def test_openai_v2_declares_third_party_model() -> None:
    manifest = OpenAIV2Manifest.model_validate(
        {
            "type": "roboflow_core/open_ai@v2",
            "name": "vlm",
            "images": "$inputs.image",
            "prompt": "What is in the image?",
            "api_key": "$inputs.openai_api_key",
        }
    )

    assert manifest.discover_dependent_resources() == [
        third_party_model(provider="openai", model_id="gpt-4o"),
    ]


def test_qwen_vlm_v1_native_backend_resolves_catalog_label() -> None:
    manifest = QwenVlmV1Manifest.model_validate(
        {
            "type": "roboflow_core/qwen_vlm@v1",
            "name": "vlm",
            "images": "$inputs.image",
            "prompt": "Describe the image.",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="qwen3_5-2b"),
    ]


def test_qwen_vlm_v1_native_backend_uses_fine_tuned_model_id() -> None:
    manifest = QwenVlmV1Manifest.model_validate(
        {
            "type": "roboflow_core/qwen_vlm@v1",
            "name": "vlm",
            "images": "$inputs.image",
            "prompt": "Describe the image.",
            "model_version": FINE_TUNED_NATIVE_LABEL,
            "fine_tuned_model_id": "my_workspace/3",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="my_workspace/3"),
    ]


def test_qwen_vlm_v1_openrouter_backend_declares_third_party_model() -> None:
    manifest = QwenVlmV1Manifest.model_validate(
        {
            "type": "roboflow_core/qwen_vlm@v1",
            "name": "vlm",
            "images": "$inputs.image",
            "prompt": "Describe the image.",
            "backend": "openrouter",
            "openrouter_model_version": "Qwen 3.6 27B",
        }
    )

    assert manifest.discover_dependent_resources() == [
        third_party_model(provider="openrouter", model_id="qwen/qwen3.6-27b"),
    ]


def test_model_monitoring_declares_access_only_dependency() -> None:
    manifest = ModelMonitoringV1Manifest.model_validate(
        {
            "type": "roboflow_core/model_monitoring_inference_aggregator@v1",
            "name": "monitor",
            "predictions": "$steps.model.predictions",
            "model_id": "$inputs.model",
            "unique_aggregator_key": "aggregator-1",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(
            model_id="$inputs.model",
            required_action=ModelRequiredAction.ACCESS,
        ),
    ]


def test_dataset_upload_v2_declares_target_project() -> None:
    manifest = DatasetUploadV2Manifest.model_validate(
        {
            "type": "roboflow_core/roboflow_dataset_upload@v2",
            "name": "sink",
            "images": "$inputs.image",
            "target_project": "my_dataset",
            "usage_quota_name": "quota-1",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_project(project_url="my_dataset"),
    ]


def test_dataset_upload_v2_declares_nothing_when_sink_literally_disabled() -> None:
    manifest = DatasetUploadV2Manifest.model_validate(
        {
            "type": "roboflow_core/roboflow_dataset_upload@v2",
            "name": "sink",
            "images": "$inputs.image",
            "target_project": "my_dataset",
            "usage_quota_name": "quota-1",
            "disable_sink": True,
        }
    )

    assert manifest.discover_dependent_resources() == []


# ---------------------------------------------------------------------------
# Repo-wide coverage guard
# ---------------------------------------------------------------------------

RESOURCE_KIND_NAMES = {"roboflow_model_id", "roboflow_project"}

# Blocks whose manifests accept resource-kind values as generic *payload*
# (carried to an external system, not consumed as a dependency) — they
# intentionally do not declare dependent resources.
CARRY_ONLY_ALLOWLIST = {
    "roboflow_core/webhook_sink@v1",
}

# Blocks loading their model weights outside the model manager (e.g. via
# AutoModel.from_pretrained) — per current policy they do not implement
# discover_dependent_resources() at all, so dependencies stay undeclared.
NON_MODEL_MANAGER_LOADERS_ALLOWLIST = {
    "roboflow_core/segment_anything_2_video@v1",
    "roboflow_core/sam3_video@v1",
}

UNDECLARED_BLOCKS_ALLOWLIST = CARRY_ONLY_ALLOWLIST | NON_MODEL_MANAGER_LOADERS_ALLOWLIST


def _canonical_block_type(manifest_class) -> str:
    return get_args(manifest_class.model_fields["type"].annotation)[0]


def _declares_resource_kind_field(manifest_class) -> bool:
    parsed = parse_block_manifest(manifest_class)
    declared_kinds = {
        kind.name
        for selector in parsed.selectors.values()
        for reference in selector.allowed_references
        for kind in reference.kind
    }
    return bool(declared_kinds & RESOURCE_KIND_NAMES)


def test_every_block_with_resource_kind_fields_declares_dependencies() -> None:
    flagged_types, missing_declarations = [], []
    for block in load_core_workflow_blocks():
        if not _declares_resource_kind_field(block.manifest_class):
            continue
        block_type = _canonical_block_type(block.manifest_class)
        flagged_types.append(block_type)
        overridden = (
            block.manifest_class.discover_dependent_resources
            is not WorkflowBlockManifest.discover_dependent_resources
        )
        if not overridden and block_type not in UNDECLARED_BLOCKS_ALLOWLIST:
            missing_declarations.append(block_type)

    # Guard against the allowlist going stale.
    assert UNDECLARED_BLOCKS_ALLOWLIST.issubset(set(flagged_types))
    assert not missing_declarations, (
        "Blocks declaring roboflow_model_id / roboflow_project fields without "
        f"discover_dependent_resources() override: {sorted(missing_declarations)}"
    )


# ---------------------------------------------------------------------------
# Dynamic (custom python) blocks
# ---------------------------------------------------------------------------


def test_dynamic_block_manifest_reports_unknown_dependencies() -> None:
    # Dynamic manifests are synthesized with `create_model` and do NOT
    # subclass WorkflowBlockManifest — the dependent-resources contract is
    # patched on in `assembly_manifest_class_methods`. The python body is
    # opaque, so the declared answer must be None (unknown), not [].
    from inference.core.workflows.execution_engine.v1.dynamic_blocks.block_assembler import (
        compile_dynamic_blocks,
    )

    dynamic_blocks = compile_dynamic_blocks(
        dynamic_blocks_definitions=[
            {
                "type": "DynamicBlockDefinition",
                "manifest": {
                    "type": "ManifestDescription",
                    "block_type": "CustomBlock",
                    "inputs": {
                        "predictions": {
                            "type": "DynamicInputDefinition",
                            "selector_types": ["step_output"],
                        },
                    },
                    "outputs": {
                        "result": {"type": "DynamicOutputDefinition", "kind": []}
                    },
                },
                "code": {
                    "type": "PythonCode",
                    "run_function_code": (
                        "def run(self, predictions):\n" '    return {"result": None}\n'
                    ),
                },
            }
        ],
        skip_class_eval=True,
    )

    manifest = dynamic_blocks[0].manifest_class.model_validate(
        {
            "type": "CustomBlock",
            "name": "custom",
            "predictions": "$steps.model.predictions",
        }
    )

    assert manifest.discover_dependent_resources() is None
