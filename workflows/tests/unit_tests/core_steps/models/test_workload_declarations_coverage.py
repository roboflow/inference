"""Registry census for the model blocks.

Every model block that the loader registers in THIS process must declare both
workload hooks explicitly. Inheriting the base `None` (which means "unknown")
is a silent gap, so it fails here rather than passing as a complete answer
downstream.

The cross-representation census (tensor mode on/off, enterprise and host
plugins) is owned elsewhere; this file only covers
`roboflow_workflows.core_steps.models.**` in the current process.
"""

from typing import Any, Dict, List, Set, Type

import pytest
from roboflow_workflows.execution_engine.entities.workload import (
    Discovery,
    RestrictionMetadata,
    WorkOperation,
)
from roboflow_workflows.execution_engine.introspection.blocks_loader import (
    load_workflow_blocks,
)
from roboflow_workflows.prototypes.block import WorkflowBlockManifest

MODELS_PACKAGE = "roboflow_workflows.core_steps.models"

# Codes minted for the model blocks. A new meaning gets a new code and is added
# to the coordinator's registry first; reusing a code for a different meaning
# is what this set is here to prevent.
ALLOWED_RESTRICTION_CODES = {
    "deprecated_block_always_raises",
    "hosted_endpoint_disabled_by_flag",
    "requires_gpu_for_local_execution",
    "roboflow_internal_endpoint_only",
    "stateful_video_state_resets_on_stateless_http",
    "temporal_block_no_benefit_on_still_image",
    "unsupported_in_tensor_representation",
}

# The server flags that gate a hosted model endpoint. Each one is declared for
# its False branch: the endpoint is not registered, so remote execution 404s.
ENDPOINT_FLAGS = {
    "CORE_MODEL_GAZE_ENABLED",
    "CORE_MODEL_PE_ENABLED",
    "CORE_MODEL_SAM2_ENABLED",
    "CORE_MODEL_SAM3_ENABLED",
    "COSMOS3_ENABLED",
    "DEPTH_ESTIMATION_ENABLED",
    "FLORENCE2_ENABLED",
    "GLM_OCR_ENABLED",
    "LMM_ENABLED",
    "MOONDREAM2_ENABLED",
    "QWEN_2_5_ENABLED",
    "QWEN_3_5_ENABLED",
    "QWEN_3_ENABLED",
    "SAM3_3D_OBJECTS_ENABLED",
    "SMOLVLM2_ENABLED",
}

# The representation switch is the one flag declared for its True branch: with
# tensor data representation ON, the block's tensor sibling raises.
TENSOR_REPRESENTATION_FLAG = "ENABLE_TENSOR_DATA_REPRESENTATION"

# Blocks whose run() raises FeatureDeprecatedError unconditionally. They
# truthfully perform no operation at all.
BLOCKS_PERFORMING_NO_WORK = {
    "roboflow_workflows.core_steps.models.foundation.cog_vlm.v1",
    "roboflow_workflows.core_steps.models.foundation.gaze.v1",
}

# Blocks that carry a portable caveat with no legacy get_restrictions()
# counterpart, because the legacy API has no way to express it.
PORTABLE_ONLY_CAVEAT_MODULES = {
    "roboflow_workflows.core_steps.models.foundation.yolo_world.v1",
    "roboflow_workflows.core_steps.models.foundation.yolo_world.v1_tensor",
}

# Model-scope blocks whose dependent-resource answer was audited as a proven
# absence: they fetch nothing at all.
AUDITED_NO_DEPENDENT_RESOURCE = {
    "roboflow_workflows.core_steps.models.foundation.cog_vlm.v1",
    "roboflow_workflows.core_steps.models.foundation.gaze.v1",
    "roboflow_workflows.core_steps.models.third_party.barcode_detection.v1",
    "roboflow_workflows.core_steps.models.third_party.barcode_detection.v1_tensor",
    "roboflow_workflows.core_steps.models.third_party.qr_code_detection.v1",
    "roboflow_workflows.core_steps.models.third_party.qr_code_detection.v1_tensor",
}

# Model-scope blocks whose dependent resource exists but has no declarable
# identity, so the audited answer is an explicit None (unknown). Guessing an
# identifier here would be worse than admitting it is unknown.
AUDITED_UNKNOWN_DEPENDENT_RESOURCE = {
    "roboflow_workflows.core_steps.models.foundation.google_vision_ocr.v1",
    "roboflow_workflows.core_steps.models.foundation.google_vision_ocr.v1_tensor",
    "roboflow_workflows.core_steps.models.foundation.seg_preview.v1",
    "roboflow_workflows.core_steps.models.foundation.seg_preview.v1_tensor",
    "roboflow_workflows.core_steps.models.foundation.segment_anything2_video.v1",
    "roboflow_workflows.core_steps.models.foundation.segment_anything2_video.v1_tensor",
    "roboflow_workflows.core_steps.models.foundation.segment_anything3_video.v1",
    "roboflow_workflows.core_steps.models.foundation.segment_anything3_video.v1_tensor",
    "roboflow_workflows.core_steps.models.foundation.stability_ai.inpainting.v1",
    "roboflow_workflows.core_steps.models.foundation.stability_ai.inpainting.v1_tensor",
    "roboflow_workflows.core_steps.models.foundation.stability_ai.outpainting.v1",
    "roboflow_workflows.core_steps.models.roboflow.action_recognition.v1",
}

# Blocks whose pre-existing declaration returns None only when the field that
# carries the model identity is selector-fed. The census builds instances with
# selector placeholders, so they land in the unknown bucket here; with a
# literal value they declare a real resource.
CONDITIONALLY_UNKNOWN_DEPENDENT_RESOURCE = {
    "roboflow_workflows.core_steps.models.foundation.lmm.v1",
    "roboflow_workflows.core_steps.models.foundation.lmm_classifier.v1",
}


def _instance(manifest_class: Type[WorkflowBlockManifest]) -> WorkflowBlockManifest:
    """Build an unvalidated instance with every required field populated.

    `model_construct()` fills defaults but leaves required fields unset, and an
    instance hook that reads a literal setting would raise AttributeError on
    such an object. Required fields get a selector placeholder, which is a
    legitimate compile-time state: a hook that cannot resolve it must say so
    with an incomplete Discovery rather than guess.
    """
    values: Dict[str, Any] = {}
    for field_name, field in manifest_class.model_fields.items():
        if not field.is_required():
            continue
        values[field_name] = "step" if field_name == "name" else "$inputs.placeholder"
    return manifest_class.model_construct(**values)


def _model_block_manifests() -> List[Type[WorkflowBlockManifest]]:
    registered = []
    for block in load_workflow_blocks():
        manifest_class = block.manifest_class
        if manifest_class.__module__.startswith(MODELS_PACKAGE):
            registered.append(manifest_class)
    return registered


@pytest.fixture(scope="module")
def model_block_manifests() -> List[Type[WorkflowBlockManifest]]:
    manifests = _model_block_manifests()
    assert manifests, "no model blocks were registered - the census is meaningless"
    return manifests


def test_every_registered_model_block_overrides_both_hooks(
    model_block_manifests: List[Type[WorkflowBlockManifest]],
) -> None:
    # given
    base_operations = WorkflowBlockManifest.discover_work_operations
    base_restrictions = WorkflowBlockManifest.discover_portable_restrictions

    # when
    not_declaring = [
        f"{manifest.__module__}.{manifest.__name__}"
        for manifest in model_block_manifests
        if manifest.discover_work_operations is base_operations
        or manifest.discover_portable_restrictions is base_restrictions
    ]

    # then
    assert not_declaring == []


def test_every_model_block_declares_its_operations(
    model_block_manifests: List[Type[WorkflowBlockManifest]],
) -> None:
    for manifest_class in model_block_manifests:
        manifest = _instance(manifest_class)

        operations = manifest.discover_work_operations()

        assert operations is not None, f"{manifest_class.__module__} returned unknown"
        if isinstance(operations, Discovery):
            # explicit incompleteness is allowed, but it must carry a reason
            assert operations.complete is False
            assert operations.unknown_reasons
            items = list(operations.items)
        else:
            assert isinstance(operations, list)
            items = operations
            if manifest_class.__module__ not in BLOCKS_PERFORMING_NO_WORK:
                assert items, f"{manifest_class.__module__} declares no operation"
        assert all(isinstance(entry, WorkOperation) for entry in items)
        assert len(set(items)) == len(items)


def test_only_the_deprecated_blocks_declare_an_empty_operations_list(
    model_block_manifests: List[Type[WorkflowBlockManifest]],
) -> None:
    empty = {
        manifest_class.__module__
        for manifest_class in model_block_manifests
        if _instance(manifest_class).discover_work_operations() == []
    }

    assert empty <= BLOCKS_PERFORMING_NO_WORK, empty - BLOCKS_PERFORMING_NO_WORK


def test_every_model_block_returns_known_restriction_codes(
    model_block_manifests: List[Type[WorkflowBlockManifest]],
) -> None:
    seen_codes: Set[str] = set()
    seen_flags: Set[str] = set()
    for manifest_class in model_block_manifests:
        manifest = _instance(manifest_class)

        restrictions = manifest.discover_portable_restrictions()

        assert isinstance(restrictions, list)
        for restriction in restrictions:
            assert isinstance(restriction, RestrictionMetadata)
            seen_codes.add(restriction.code)
            seen_flags.update(restriction.when.configuration_equals)
            for flag, value in restriction.when.configuration_equals.items():
                expected = (
                    value is True
                    if flag == TENSOR_REPRESENTATION_FLAG
                    else (value is False)
                )
                assert expected, f"{flag}={value} is not the declared branch"
    assert seen_codes <= ALLOWED_RESTRICTION_CODES, (
        seen_codes - ALLOWED_RESTRICTION_CODES
    )
    assert seen_flags <= ENDPOINT_FLAGS | {TENSOR_REPRESENTATION_FLAG}, seen_flags


def test_no_model_block_declares_more_than_one_endpoint_flag(
    model_block_manifests: List[Type[WorkflowBlockManifest]],
) -> None:
    # Every model block is gated by at most one endpoint flag today. A block
    # that starts combining flags needs its own reviewed declaration, so this
    # is a tripwire rather than a style rule.
    for manifest_class in model_block_manifests:
        manifest = _instance(manifest_class)

        flags = [
            flag
            for restriction in manifest.discover_portable_restrictions()
            for flag in restriction.when.configuration_equals
            if flag in ENDPOINT_FLAGS
        ]

        assert len(flags) <= 1, f"{manifest_class.__module__} declares {flags}"


def test_a_legacy_restriction_declaration_is_mirrored_by_a_portable_one(
    model_block_manifests: List[Type[WorkflowBlockManifest]],
) -> None:
    # A model manifest that overrides get_restrictions() inside the models
    # package carries a real caveat, so its portable list cannot be empty.
    for manifest_class in model_block_manifests:
        declaring_module = getattr(manifest_class.get_restrictions, "__module__", "")
        if not declaring_module.startswith(MODELS_PACKAGE):
            continue

        restrictions = _instance(manifest_class).discover_portable_restrictions()

        assert restrictions, (
            f"{manifest_class.__module__} overrides get_restrictions() but declares "
            f"no portable restriction"
        )


def test_model_blocks_without_a_legacy_declaration_declare_no_caveat(
    model_block_manifests: List[Type[WorkflowBlockManifest]],
) -> None:
    # The mirror of the previous test: nothing was invented on the portable
    # side that the legacy API does not already say, except the explicitly
    # reviewed portable-only caveats.
    for manifest_class in model_block_manifests:
        if manifest_class.get_restrictions.__func__ is not (
            WorkflowBlockManifest.get_restrictions.__func__
        ):
            continue
        if manifest_class.__module__ in PORTABLE_ONLY_CAVEAT_MODULES:
            continue

        restrictions = _instance(manifest_class).discover_portable_restrictions()

        assert (
            restrictions == []
        ), f"{manifest_class.__module__} invented {restrictions}"


def test_the_deprecated_blocks_declare_the_always_raises_caveat(
    model_block_manifests: List[Type[WorkflowBlockManifest]],
) -> None:
    by_module = {
        manifest_class.__module__: manifest_class
        for manifest_class in model_block_manifests
    }
    for module in BLOCKS_PERFORMING_NO_WORK:
        assert module in by_module, f"{module} is not registered any more"

        codes = [
            restriction.code
            for restriction in _instance(
                by_module[module]
            ).discover_portable_restrictions()
        ]

        assert "deprecated_block_always_raises" in codes


# ---------------------------------------------------------------------------
# dependent resources: an audited declaration on every model block
# ---------------------------------------------------------------------------


def test_every_registered_model_block_audits_its_dependent_resources(
    model_block_manifests: List[Type[WorkflowBlockManifest]],
) -> None:
    # Inheriting the base None is indistinguishable from "nobody looked". The
    # declaration has to be written in the class body, even when the audited
    # answer is None.
    not_audited = [
        f"{manifest.__module__}.{manifest.__name__}"
        for manifest in model_block_manifests
        if "discover_dependent_resources" not in vars(manifest)
    ]

    assert not_audited == []


def test_the_audited_none_blocks_return_none(
    model_block_manifests: List[Type[WorkflowBlockManifest]],
) -> None:
    by_module = {
        manifest_class.__module__: manifest_class
        for manifest_class in model_block_manifests
    }
    for module in AUDITED_UNKNOWN_DEPENDENT_RESOURCE & set(by_module):

        resources = _instance(by_module[module]).discover_dependent_resources()

        assert resources is None, f"{module} claims {resources}"


def test_the_audited_empty_blocks_return_an_empty_list(
    model_block_manifests: List[Type[WorkflowBlockManifest]],
) -> None:
    by_module = {
        manifest_class.__module__: manifest_class
        for manifest_class in model_block_manifests
    }
    for module in AUDITED_NO_DEPENDENT_RESOURCE & set(by_module):

        resources = _instance(by_module[module]).discover_dependent_resources()

        assert resources == [], f"{module} claims {resources}"


def test_no_model_block_outside_the_audited_sets_returns_none(
    model_block_manifests: List[Type[WorkflowBlockManifest]],
) -> None:
    # A block that starts returning None without being audited is a silent
    # regression, so the unknown set is closed.
    unknown = {
        manifest_class.__module__
        for manifest_class in model_block_manifests
        if _instance(manifest_class).discover_dependent_resources() is None
    }

    allowed = (
        AUDITED_UNKNOWN_DEPENDENT_RESOURCE | CONDITIONALLY_UNKNOWN_DEPENDENT_RESOURCE
    )
    assert unknown <= allowed, unknown - allowed
