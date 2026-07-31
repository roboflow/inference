"""
Dependent-resources discovery tests for the SAM2 Video Tracker block
(``roboflow_core/segment_anything_2_video@v1``) — the model id is held
directly in the ``model_id`` field (default ``sam2video/small``).
"""

from inference.core.workflows.core_steps.models.foundation.segment_anything2_video.v1 import (
    BlockManifest as SegmentAnything2VideoV1Manifest,
)
from inference.core.workflows.prototypes.block import roboflow_platform_model


def test_sam2_video_v1_declares_default_model_id() -> None:
    manifest = SegmentAnything2VideoV1Manifest.model_validate(
        {
            "type": "roboflow_core/segment_anything_2_video@v1",
            "name": "tracker",
            "images": "$inputs.image",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="sam2video/small"),
    ]


def test_sam2_video_v1_declares_explicit_model_id() -> None:
    manifest = SegmentAnything2VideoV1Manifest.model_validate(
        {
            "type": "roboflow_core/segment_anything_2_video@v1",
            "name": "tracker",
            "images": "$inputs.image",
            "model_id": "sam3trackervideo",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="sam3trackervideo"),
    ]


def test_sam2_video_v1_returns_selector_fed_model_id_verbatim() -> None:
    manifest = SegmentAnything2VideoV1Manifest.model_validate(
        {
            "type": "roboflow_core/segment_anything_2_video@v1",
            "name": "tracker",
            "images": "$inputs.image",
            "model_id": "$inputs.model",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="$inputs.model"),
    ]
