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
