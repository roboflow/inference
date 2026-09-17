"""
Dependent-resources discovery tests for the SAM2 Video Tracker block
(``roboflow_core/segment_anything_2_video@v1``).

The block loads its weights via ``AutoModel.from_pretrained``, not the model
manager — ``discover_dependent_resources()`` is deliberately not implemented,
so dependencies stay undeclared (``None``) for now.
"""

from inference.core.workflows.core_steps.models.foundation.segment_anything2_video.v1 import (
    BlockManifest as SegmentAnything2VideoV1Manifest,
)


def test_sam2_video_v1_does_not_declare_dependencies() -> None:
    manifest = SegmentAnything2VideoV1Manifest.model_validate(
        {
            "type": "roboflow_core/segment_anything_2_video@v1",
            "name": "tracker",
            "images": "$inputs.image",
        }
    )

    assert manifest.discover_dependent_resources() is None
