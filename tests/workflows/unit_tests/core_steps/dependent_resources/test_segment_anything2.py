"""
Manifest creation + ``discover_dependent_resources()`` for the Segment
Anything 2 block (``roboflow_core/segment_anything@v1``).

The declared model id mirrors ``run()``: ``load_core_model(...,
core_model="sam2")`` loads ``sam2/<version>`` with ``version`` taken from the
manifest field (default ``"hiera_tiny"``) — note the family prefix is ``sam2``
even though the block package is named ``segment_anything2``. Selector-fed
versions are returned verbatim.
"""

from inference.core.workflows.core_steps.models.foundation.segment_anything2.v1 import (
    BlockManifest as SegmentAnything2V1Manifest,
)
from inference.core.workflows.prototypes.block import roboflow_platform_model


def test_segment_anything2_v1_default_version_synthesizes_model_id() -> None:
    manifest = SegmentAnything2V1Manifest.model_validate(
        {
            "type": "roboflow_core/segment_anything@v1",
            "name": "segmenter",
            "images": "$inputs.image",
        }
    )

    assert manifest.version == "hiera_tiny"
    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="sam2/hiera_tiny"),
    ]


def test_segment_anything2_v1_explicit_version_synthesizes_model_id() -> None:
    manifest = SegmentAnything2V1Manifest.model_validate(
        {
            "type": "roboflow_core/segment_anything@v1",
            "name": "segmenter",
            "images": "$inputs.image",
            "version": "hiera_large",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="sam2/hiera_large"),
    ]


def test_segment_anything2_v1_selector_fed_version_is_returned_verbatim() -> None:
    manifest = SegmentAnything2V1Manifest.model_validate(
        {
            "type": "roboflow_core/segment_anything@v1",
            "name": "segmenter",
            "images": "$inputs.image",
            "version": "$inputs.variant",
        }
    )

    resources = manifest.discover_dependent_resources()

    assert resources == [
        roboflow_platform_model(model_id="$inputs.variant"),
    ]
    resolver = resources[0].metadata.model_id_resolver
    assert resolver is not None
    assert resolver("hiera_small") == "sam2/hiera_small"
