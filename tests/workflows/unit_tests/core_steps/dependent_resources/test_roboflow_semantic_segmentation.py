"""
Dependent-resources discovery tests for the Roboflow semantic segmentation
model blocks (``roboflow_core/roboflow_semantic_segmentation_model@v1..v2``).

Unlike the other Roboflow model families, semantic segmentation manifests
have no ``active_learning_target_dataset`` field — both versions declare
the model dependency only.
"""

from inference.core.workflows.core_steps.models.roboflow.semantic_segmentation.v1 import (
    BlockManifest as SemanticSegmentationV1Manifest,
)
from inference.core.workflows.core_steps.models.roboflow.semantic_segmentation.v2 import (
    BlockManifest as SemanticSegmentationV2Manifest,
)
from inference.core.workflows.prototypes.block import roboflow_platform_model

# ---------------------------------------------------------------------------
# v1
# ---------------------------------------------------------------------------


def test_semantic_segmentation_v1_declares_model_for_literal_model_id() -> None:
    manifest = SemanticSegmentationV1Manifest.model_validate(
        {
            "type": "roboflow_core/roboflow_semantic_segmentation_model@v1",
            "name": "segmenter",
            "images": "$inputs.image",
            "model_id": "my_project/3",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="my_project/3"),
    ]


def test_semantic_segmentation_v1_returns_selector_fed_model_id_verbatim() -> None:
    manifest = SemanticSegmentationV1Manifest.model_validate(
        {
            "type": "roboflow_core/roboflow_semantic_segmentation_model@v1",
            "name": "segmenter",
            "images": "$inputs.image",
            "model_id": "$inputs.model",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="$inputs.model"),
    ]


# ---------------------------------------------------------------------------
# v2
# ---------------------------------------------------------------------------


def test_semantic_segmentation_v2_declares_model_for_literal_model_id() -> None:
    manifest = SemanticSegmentationV2Manifest.model_validate(
        {
            "type": "roboflow_core/roboflow_semantic_segmentation_model@v2",
            "name": "segmenter",
            "images": "$inputs.image",
            "model_id": "my_project/3",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="my_project/3"),
    ]


def test_semantic_segmentation_v2_returns_selector_fed_model_id_verbatim() -> None:
    manifest = SemanticSegmentationV2Manifest.model_validate(
        {
            "type": "roboflow_core/roboflow_semantic_segmentation_model@v2",
            "name": "segmenter",
            "images": "$inputs.image",
            "model_id": "$inputs.model",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="$inputs.model"),
    ]


# ---------------------------------------------------------------------------
# Family invariant: no active-learning field in any version
# ---------------------------------------------------------------------------


def test_semantic_segmentation_manifests_declare_no_active_learning_field() -> None:
    for manifest_class in (
        SemanticSegmentationV1Manifest,
        SemanticSegmentationV2Manifest,
    ):
        assert (
            "active_learning_target_dataset" not in manifest_class.model_fields
        ), f"{manifest_class.__module__} unexpectedly grew an active-learning field"
