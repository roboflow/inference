"""
Dependent-resources discovery tests for the SAM3 Video Tracker block
(``roboflow_core/sam3_video@v1``) — the model id is held directly in the
``model_id`` field (default ``sam3video``); ``class_names`` is required.
"""

from inference.core.workflows.core_steps.models.foundation.segment_anything3_video.v1 import (
    BlockManifest as SegmentAnything3VideoV1Manifest,
)
from inference.core.workflows.prototypes.block import roboflow_platform_model


def test_sam3_video_v1_declares_default_model_id() -> None:
    manifest = SegmentAnything3VideoV1Manifest.model_validate(
        {
            "type": "roboflow_core/sam3_video@v1",
            "name": "tracker",
            "images": "$inputs.image",
            "class_names": ["person", "forklift"],
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="sam3video"),
    ]


def test_sam3_video_v1_declares_explicit_model_id() -> None:
    manifest = SegmentAnything3VideoV1Manifest.model_validate(
        {
            "type": "roboflow_core/sam3_video@v1",
            "name": "tracker",
            "images": "$inputs.image",
            "class_names": ["person"],
            "model_id": "my_workspace/3",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="my_workspace/3"),
    ]


def test_sam3_video_v1_returns_selector_fed_model_id_verbatim() -> None:
    manifest = SegmentAnything3VideoV1Manifest.model_validate(
        {
            "type": "roboflow_core/sam3_video@v1",
            "name": "tracker",
            "images": "$inputs.image",
            "class_names": ["person"],
            "model_id": "$inputs.model",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="$inputs.model"),
    ]
