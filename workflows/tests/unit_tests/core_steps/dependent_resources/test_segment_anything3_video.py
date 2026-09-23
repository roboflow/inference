"""
Dependent-resources discovery tests for the SAM3 Video Tracker block
(``roboflow_core/sam3_video@v1``), numpy and tensor variants.

The block loads its streaming model in-process itself, whatever the generic
step execution mode is, so it declares that model as LOCAL EXECUTION and keeps
it away from the generic model-manager preloader (``preloadable=False``). The
literal ``tracking_mode`` picks the one model the block loads: ``model_id``
for ``concept``, ``visual_model_id`` for ``visual`` - never both.
"""

from typing import Type

import pytest
from roboflow_workflows.core_steps.models.foundation.segment_anything3_video.v1 import (
    BlockManifest as SegmentAnything3VideoV1Manifest,
)
from roboflow_workflows.core_steps.models.foundation.segment_anything3_video.v1_tensor import (
    BlockManifest as SegmentAnything3VideoV1TensorManifest,
)
from roboflow_workflows.core_steps.models.foundation.segment_anything_common.streaming_video import (
    SAM3_CONCEPT_VIDEO_MODEL_ID,
    SAM3_VISUAL_VIDEO_MODEL_ID,
)
from roboflow_workflows.prototypes.block import (
    DependentResourceType,
    ModelExecutionLocation,
    ModelRequiredAction,
    WorkflowBlockManifest,
)

MANIFESTS = [SegmentAnything3VideoV1Manifest, SegmentAnything3VideoV1TensorManifest]
CONCEPT_FIELDS = {"tracking_mode": "concept", "class_names": ["person"]}
VISUAL_FIELDS = {
    "tracking_mode": "visual",
    "boxes": "$steps.detector.predictions",
}


def _declared_model_ids(manifest_class: Type[WorkflowBlockManifest], **fields) -> list:
    manifest = manifest_class.model_validate(
        {
            "type": "roboflow_core/sam3_video@v1",
            "name": "tracker",
            "images": "$inputs.image",
            **fields,
        }
    )
    resources = manifest.discover_dependent_resources()
    for resource in resources:
        assert resource.resource_type is DependentResourceType.ROBOFLOW_PLATFORM_MODEL
        assert resource.metadata.required_action is ModelRequiredAction.EXECUTION
        assert resource.metadata.execution_location is ModelExecutionLocation.LOCAL
        assert resource.metadata.preloadable is False
    return [resource.metadata.model_id for resource in resources]


@pytest.mark.parametrize("manifest_class", MANIFESTS)
def test_sam3_video_concept_mode_declares_default_concept_model_only(
    manifest_class: Type[WorkflowBlockManifest],
) -> None:
    # when
    result = _declared_model_ids(manifest_class, **CONCEPT_FIELDS)

    # then
    assert result == [SAM3_CONCEPT_VIDEO_MODEL_ID]


@pytest.mark.parametrize("manifest_class", MANIFESTS)
def test_sam3_video_default_tracking_mode_is_concept(
    manifest_class: Type[WorkflowBlockManifest],
) -> None:
    # when
    result = _declared_model_ids(manifest_class, class_names=["person"])

    # then
    assert result == [SAM3_CONCEPT_VIDEO_MODEL_ID]


@pytest.mark.parametrize("manifest_class", MANIFESTS)
def test_sam3_video_visual_mode_declares_default_visual_model_only(
    manifest_class: Type[WorkflowBlockManifest],
) -> None:
    # when
    result = _declared_model_ids(manifest_class, **VISUAL_FIELDS)

    # then
    assert result == [SAM3_VISUAL_VIDEO_MODEL_ID]


@pytest.mark.parametrize("manifest_class", MANIFESTS)
@pytest.mark.parametrize(
    "fields, expected_model_ids",
    [
        (
            {
                **CONCEPT_FIELDS,
                "model_id": "custom-concept",
                "visual_model_id": "custom-visual",
            },
            ["custom-concept"],
        ),
        (
            {
                **VISUAL_FIELDS,
                "model_id": "custom-concept",
                "visual_model_id": "custom-visual",
            },
            ["custom-visual"],
        ),
        (
            {**CONCEPT_FIELDS, "model_id": "$inputs.concept_model"},
            ["$inputs.concept_model"],
        ),
        (
            {**VISUAL_FIELDS, "visual_model_id": "$inputs.visual_model"},
            ["$inputs.visual_model"],
        ),
    ],
)
def test_sam3_video_declares_configured_id_of_selected_mode_verbatim(
    manifest_class: Type[WorkflowBlockManifest],
    fields: dict,
    expected_model_ids: list,
) -> None:
    # when
    result = _declared_model_ids(manifest_class, **fields)

    # then
    assert result == expected_model_ids


@pytest.mark.parametrize("manifest_class", MANIFESTS)
def test_sam3_video_tracking_mode_rejects_selectors(
    manifest_class: Type[WorkflowBlockManifest],
) -> None:
    # when
    with pytest.raises(ValueError):
        manifest_class.model_validate(
            {
                "type": "roboflow_core/sam3_video@v1",
                "name": "tracker",
                "images": "$inputs.image",
                "class_names": ["person"],
                "tracking_mode": "$inputs.mode",
            }
        )
