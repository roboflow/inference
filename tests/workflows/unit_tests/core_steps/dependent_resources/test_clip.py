"""
Manifest creation + ``discover_dependent_resources()`` for the CLIP embedding
block (``roboflow_core/clip@v1``).

The declared model id mirrors ``run()``: ``load_core_model(...,
core_model="clip")`` loads ``clip/<version>`` with ``version`` taken from the
manifest field (default ``"ViT-B-32"``). Selector-fed versions are returned
verbatim — callers substitute and apply the family prefix.
"""

from inference.core.workflows.core_steps.models.foundation.clip.v1 import (
    BlockManifest as ClipV1Manifest,
)
from inference.core.workflows.prototypes.block import roboflow_platform_model


def test_clip_v1_default_version_synthesizes_model_id() -> None:
    manifest = ClipV1Manifest.model_validate(
        {
            "type": "roboflow_core/clip@v1",
            "name": "embedder",
            "data": "$inputs.image",
        }
    )

    assert manifest.version == "ViT-B-32"
    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="clip/ViT-B-32"),
    ]


def test_clip_v1_explicit_version_synthesizes_model_id() -> None:
    manifest = ClipV1Manifest.model_validate(
        {
            "type": "roboflow_core/clip@v1",
            "name": "embedder",
            "data": "$inputs.image",
            "version": "ViT-B-16",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="clip/ViT-B-16"),
    ]


def test_clip_v1_selector_fed_version_is_returned_verbatim() -> None:
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
