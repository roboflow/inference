"""
Dependent-resources discovery tests for the Depth Estimation block
(``roboflow_core/depth_estimation@v1``) — the model id is held directly in
the ``model_version`` field as full ids like ``depth-anything-v2/small``.
"""

from inference.core.workflows.core_steps.models.foundation.depth_estimation.v1 import (
    BlockManifest as DepthEstimationV1Manifest,
)
from inference.core.workflows.prototypes.block import roboflow_platform_model


def test_depth_estimation_v1_declares_default_model_version() -> None:
    manifest = DepthEstimationV1Manifest.model_validate(
        {
            "type": "roboflow_core/depth_estimation@v1",
            "name": "depth",
            "images": "$inputs.image",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="depth-anything-v3/small"),
    ]


def test_depth_estimation_v1_declares_explicit_model_version() -> None:
    manifest = DepthEstimationV1Manifest.model_validate(
        {
            "type": "roboflow_core/depth_estimation@v1",
            "name": "depth",
            "images": "$inputs.image",
            "model_version": "depth-anything-v2/small",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="depth-anything-v2/small"),
    ]


def test_depth_estimation_v1_returns_selector_fed_model_version_verbatim() -> None:
    manifest = DepthEstimationV1Manifest.model_validate(
        {
            "type": "roboflow_core/depth_estimation@v1",
            "name": "depth",
            "images": "$inputs.image",
            "model_version": "$inputs.variant",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="$inputs.variant"),
    ]
