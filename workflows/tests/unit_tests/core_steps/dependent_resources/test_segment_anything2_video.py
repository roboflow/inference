"""
Dependent-resources discovery tests for the SAM2 Video Tracker block
(``roboflow_core/segment_anything_2_video@v1``), numpy and tensor variants.

The block loads its streaming model in-process itself, whatever the generic
step execution mode is, so it declares that model as LOCAL EXECUTION and keeps
it away from the generic model-manager preloader (``preloadable=False``).
"""

from typing import Type

import pytest
from roboflow_workflows.core_steps.models.foundation.segment_anything2_video.v1 import (
    BlockManifest as SegmentAnything2VideoV1Manifest,
)
from roboflow_workflows.core_steps.models.foundation.segment_anything2_video.v1_tensor import (
    BlockManifest as SegmentAnything2VideoV1TensorManifest,
)
from roboflow_workflows.prototypes.block import (
    DependentResourceType,
    ModelExecutionLocation,
    ModelRequiredAction,
    WorkflowBlockManifest,
)

MANIFESTS = [SegmentAnything2VideoV1Manifest, SegmentAnything2VideoV1TensorManifest]


def _manifest(manifest_class: Type[WorkflowBlockManifest], **fields):
    return manifest_class.model_validate(
        {
            "type": "roboflow_core/segment_anything_2_video@v1",
            "name": "tracker",
            "images": "$inputs.image",
            **fields,
        }
    )


@pytest.mark.parametrize("manifest_class", MANIFESTS)
@pytest.mark.parametrize(
    "fields, expected_model_id",
    [
        ({}, "sam2video/small"),
        ({"model_id": "sam2video/large"}, "sam2video/large"),
        ({"model_id": "$inputs.tracker_model"}, "$inputs.tracker_model"),
    ],
)
def test_sam2_video_v1_declares_its_model_as_local_non_preloadable(
    manifest_class: Type[WorkflowBlockManifest],
    fields: dict,
    expected_model_id: str,
) -> None:
    # when
    result = _manifest(manifest_class, **fields).discover_dependent_resources()

    # then
    assert len(result) == 1
    resource = result[0]
    assert resource.resource_type is DependentResourceType.ROBOFLOW_PLATFORM_MODEL
    assert resource.metadata.model_id == expected_model_id
    assert resource.metadata.required_action is ModelRequiredAction.EXECUTION
    assert resource.metadata.execution_location is ModelExecutionLocation.LOCAL
    assert resource.metadata.preloadable is False
    assert resource.to_dict() == {
        "resource_type": "roboflow_platform_model",
        "metadata": {
            "model_id": expected_model_id,
            "required_action": "execution",
            "execution_location": "local",
        },
    }
