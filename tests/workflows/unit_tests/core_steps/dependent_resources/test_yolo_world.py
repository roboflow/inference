"""
Manifest creation + ``discover_dependent_resources()`` for the YOLO-World
block (``roboflow_core/yolo_world_model@v1``).

The declared model id mirrors ``run()``: ``load_core_model(...,
core_model="yolo_world")`` loads ``yolo_world/<version>`` with ``version``
taken from the manifest field (default ``"v2-s"``). Selector-fed versions are
returned verbatim.
"""

from inference.core.workflows.core_steps.models.foundation.yolo_world.v1 import (
    BlockManifest as YoloWorldV1Manifest,
)
from inference.core.workflows.prototypes.block import roboflow_platform_model


def test_yolo_world_v1_default_version_synthesizes_model_id() -> None:
    manifest = YoloWorldV1Manifest.model_validate(
        {
            "type": "roboflow_core/yolo_world_model@v1",
            "name": "detector",
            "images": "$inputs.image",
            "class_names": ["person", "car"],
        }
    )

    assert manifest.version == "v2-s"
    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="yolo_world/v2-s"),
    ]


def test_yolo_world_v1_explicit_version_synthesizes_model_id() -> None:
    manifest = YoloWorldV1Manifest.model_validate(
        {
            "type": "roboflow_core/yolo_world_model@v1",
            "name": "detector",
            "images": "$inputs.image",
            "class_names": ["person", "car"],
            "version": "v2-l",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="yolo_world/v2-l"),
    ]


def test_yolo_world_v1_selector_fed_version_is_returned_verbatim() -> None:
    manifest = YoloWorldV1Manifest.model_validate(
        {
            "type": "roboflow_core/yolo_world_model@v1",
            "name": "detector",
            "images": "$inputs.image",
            "class_names": ["person", "car"],
            "version": "$inputs.variant",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="$inputs.variant"),
    ]
