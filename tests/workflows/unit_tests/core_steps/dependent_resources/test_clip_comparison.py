"""
Manifest creation + ``discover_dependent_resources()`` for the Clip Comparison
blocks.

- v1 (``roboflow_core/clip_comparison@v1``) has NO version field: ``run()``
  builds a ``ClipCompareRequest`` without ``clip_version_id``, so the request
  default (``CLIP_VERSION_ID`` from ``inference.core.env``) is what
  ``load_core_model(..., core_model="clip")`` loads — the declared id is
  ``clip/<CLIP_VERSION_ID>``.
- v2 (``roboflow_core/clip_comparison@v2``) has a ``version`` field (default
  ``"ViT-B-16"``) passed straight into the request, so the declared id is
  ``clip/<version>``; selector-fed versions are returned verbatim.
"""

from inference.core.env import CLIP_VERSION_ID
from inference.core.workflows.core_steps.models.foundation.clip_comparison.v1 import (
    BlockManifest as ClipComparisonV1Manifest,
)
from inference.core.workflows.core_steps.models.foundation.clip_comparison.v2 import (
    BlockManifest as ClipComparisonV2Manifest,
)
from inference.core.workflows.prototypes.block import roboflow_platform_model


def test_clip_comparison_v1_declares_server_default_clip_variant() -> None:
    manifest = ClipComparisonV1Manifest.model_validate(
        {
            "type": "roboflow_core/clip_comparison@v1",
            "name": "comparison",
            "images": "$inputs.image",
            "texts": ["a", "b"],
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id=f"clip/{CLIP_VERSION_ID}"),
    ]


def test_clip_comparison_v2_default_version_synthesizes_model_id() -> None:
    manifest = ClipComparisonV2Manifest.model_validate(
        {
            "type": "roboflow_core/clip_comparison@v2",
            "name": "comparison",
            "images": "$inputs.image",
            "classes": ["a", "b"],
        }
    )

    assert manifest.version == "ViT-B-16"
    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="clip/ViT-B-16"),
    ]


def test_clip_comparison_v2_explicit_version_synthesizes_model_id() -> None:
    manifest = ClipComparisonV2Manifest.model_validate(
        {
            "type": "roboflow_core/clip_comparison@v2",
            "name": "comparison",
            "images": "$inputs.image",
            "classes": ["a", "b"],
            "version": "ViT-B-32",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="clip/ViT-B-32"),
    ]


def test_clip_comparison_v2_selector_fed_version_is_returned_verbatim() -> None:
    manifest = ClipComparisonV2Manifest.model_validate(
        {
            "type": "roboflow_core/clip_comparison@v2",
            "name": "comparison",
            "images": "$inputs.image",
            "classes": ["a", "b"],
            "version": "$inputs.variant",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="$inputs.variant"),
    ]
