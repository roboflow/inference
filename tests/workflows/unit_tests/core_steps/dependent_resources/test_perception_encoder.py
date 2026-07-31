"""
Manifest creation + ``discover_dependent_resources()`` for the Perception
Encoder block (``roboflow_core/perception_encoder@v1``).

The declared model id mirrors ``run()``: ``load_core_model(...,
core_model="perception_encoder")`` loads ``perception_encoder/<version>`` with
``version`` taken from the manifest field (default ``"PE-Core-L14-336"``).
Selector-fed versions are returned verbatim.
"""

from inference.core.workflows.core_steps.models.foundation.perception_encoder.v1 import (
    BlockManifest as PerceptionEncoderV1Manifest,
)
from inference.core.workflows.prototypes.block import roboflow_platform_model


def test_perception_encoder_v1_default_version_synthesizes_model_id() -> None:
    manifest = PerceptionEncoderV1Manifest.model_validate(
        {
            "type": "roboflow_core/perception_encoder@v1",
            "name": "embedder",
            "data": "$inputs.image",
        }
    )

    assert manifest.version == "PE-Core-L14-336"
    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="perception_encoder/PE-Core-L14-336"),
    ]


def test_perception_encoder_v1_explicit_version_synthesizes_model_id() -> None:
    manifest = PerceptionEncoderV1Manifest.model_validate(
        {
            "type": "roboflow_core/perception_encoder@v1",
            "name": "embedder",
            "data": "$inputs.image",
            "version": "PE-Core-B16-224",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="perception_encoder/PE-Core-B16-224"),
    ]


def test_perception_encoder_v1_selector_fed_version_is_returned_verbatim() -> None:
    manifest = PerceptionEncoderV1Manifest.model_validate(
        {
            "type": "roboflow_core/perception_encoder@v1",
            "name": "embedder",
            "data": "$inputs.image",
            "version": "$inputs.variant",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="$inputs.variant"),
    ]
