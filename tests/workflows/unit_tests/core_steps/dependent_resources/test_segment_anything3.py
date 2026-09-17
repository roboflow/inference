"""
Dependent-resources discovery tests for the SAM 3 image blocks
(``roboflow_core/sam3@v1|v2|v3``) — the model id is held directly in the
Optional ``model_id`` field (default ``sam3/sam3_final``); an explicit
``None`` declares no dependencies (empty list).
"""

import pytest

from inference.core.workflows.core_steps.models.foundation.segment_anything3.v1 import (
    BlockManifest as SegmentAnything3V1Manifest,
)
from inference.core.workflows.core_steps.models.foundation.segment_anything3.v2 import (
    BlockManifest as SegmentAnything3V2Manifest,
)
from inference.core.workflows.core_steps.models.foundation.segment_anything3.v3 import (
    BlockManifest as SegmentAnything3V3Manifest,
)
from inference.core.workflows.prototypes.block import roboflow_platform_model

MANIFEST_VERSIONS = [
    (SegmentAnything3V1Manifest, "roboflow_core/sam3@v1"),
    (SegmentAnything3V2Manifest, "roboflow_core/sam3@v2"),
    (SegmentAnything3V3Manifest, "roboflow_core/sam3@v3"),
]
VERSION_IDS = ["v1", "v2", "v3"]


@pytest.mark.parametrize(
    "manifest_class,block_type", MANIFEST_VERSIONS, ids=VERSION_IDS
)
def test_sam3_declares_default_model_id(manifest_class, block_type) -> None:
    manifest = manifest_class.model_validate(
        {
            "type": block_type,
            "name": "model",
            "images": "$inputs.image",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="sam3/sam3_final"),
    ]


@pytest.mark.parametrize(
    "manifest_class,block_type", MANIFEST_VERSIONS, ids=VERSION_IDS
)
def test_sam3_declares_explicit_model_id(manifest_class, block_type) -> None:
    manifest = manifest_class.model_validate(
        {
            "type": block_type,
            "name": "model",
            "images": "$inputs.image",
            "model_id": "my_workspace/3",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="my_workspace/3"),
    ]


@pytest.mark.parametrize(
    "manifest_class,block_type", MANIFEST_VERSIONS, ids=VERSION_IDS
)
def test_sam3_with_null_model_id_declares_no_dependencies(
    manifest_class, block_type
) -> None:
    manifest = manifest_class.model_validate(
        {
            "type": block_type,
            "name": "model",
            "images": "$inputs.image",
            "model_id": None,
        }
    )

    assert manifest.discover_dependent_resources() == []


@pytest.mark.parametrize(
    "manifest_class,block_type", MANIFEST_VERSIONS, ids=VERSION_IDS
)
def test_sam3_returns_selector_fed_model_id_verbatim(
    manifest_class, block_type
) -> None:
    manifest = manifest_class.model_validate(
        {
            "type": block_type,
            "name": "model",
            "images": "$inputs.image",
            "model_id": "$inputs.model_variant",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="$inputs.model_variant"),
    ]


# ---------------------------------------------------------------------------
# SAM3_EXEC_MODE governs declared execution locality
# ---------------------------------------------------------------------------

import inference.core.workflows.core_steps.models.foundation.segment_anything3.v1 as sam3_v1_module
import inference.core.workflows.core_steps.models.foundation.segment_anything3.v2 as sam3_v2_module
import inference.core.workflows.core_steps.models.foundation.segment_anything3.v3 as sam3_v3_module
from inference.core.workflows.prototypes.block import ModelExecutionLocation

MANIFEST_VERSIONS_WITH_MODULES = [
    (SegmentAnything3V1Manifest, "roboflow_core/sam3@v1", sam3_v1_module),
    (SegmentAnything3V2Manifest, "roboflow_core/sam3@v2", sam3_v2_module),
    (SegmentAnything3V3Manifest, "roboflow_core/sam3@v3", sam3_v3_module),
]


@pytest.mark.parametrize(
    "manifest_class,block_type,module", MANIFEST_VERSIONS_WITH_MODULES, ids=VERSION_IDS
)
def test_sam3_declares_environment_defined_execution_when_exec_mode_local(
    manifest_class, block_type, module, monkeypatch
) -> None:
    monkeypatch.setattr(module, "SAM3_EXEC_MODE", "local")
    manifest = manifest_class.model_validate(
        {
            "type": block_type,
            "name": "model",
            "images": "$inputs.image",
        }
    )

    (resource,) = manifest.discover_dependent_resources()

    assert (
        resource.metadata.execution_location
        is ModelExecutionLocation.ENVIRONMENT_DEFINED
    )


@pytest.mark.parametrize(
    "manifest_class,block_type,module", MANIFEST_VERSIONS_WITH_MODULES, ids=VERSION_IDS
)
def test_sam3_declares_nothing_under_proxy_execution_mode(
    manifest_class, block_type, module, monkeypatch
) -> None:
    monkeypatch.setattr(module, "SAM3_EXEC_MODE", "remote")
    manifest = manifest_class.model_validate(
        {
            "type": block_type,
            "name": "model",
            "images": "$inputs.image",
        }
    )

    # The proxy ignores the configured model id and runs its own fixed SAM3
    # server-side — nothing to declare.
    assert manifest.discover_dependent_resources() == []
