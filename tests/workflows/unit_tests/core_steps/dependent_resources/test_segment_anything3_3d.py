"""
Dependent-resources discovery tests for the SAM3D block
(``roboflow_core/segment_anything3_3d_objects@v1``) — the manifest exposes
NO model field; the block always runs the literal ``sam3-3d-objects`` id.
"""

from inference.core.workflows.core_steps.models.foundation.segment_anything3_3d.v1 import (
    BlockManifest as SegmentAnything33DV1Manifest,
)
from inference.core.workflows.prototypes.block import roboflow_platform_model


def test_sam3_3d_v1_declares_literal_model_id() -> None:
    manifest = SegmentAnything33DV1Manifest.model_validate(
        {
            "type": "roboflow_core/segment_anything3_3d_objects@v1",
            "name": "mesh",
            "images": "$inputs.image",
            "mask_input": "$steps.sam2.predictions",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="sam3-3d-objects"),
    ]
