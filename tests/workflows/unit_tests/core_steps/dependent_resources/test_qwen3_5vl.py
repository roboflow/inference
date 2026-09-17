"""
Dependent-resources discovery tests for the Qwen3.5-VL blocks
(``roboflow_core/qwen3_5vl@v1`` and ``roboflow_core/qwen3_5vl@v2``) — the
model id is held directly in the ``model_version`` field of both versions.
"""

from inference.core.workflows.core_steps.models.foundation.qwen3_5vl.v1 import (
    BlockManifest as Qwen35VLV1Manifest,
)
from inference.core.workflows.core_steps.models.foundation.qwen3_5vl.v2 import (
    BlockManifest as Qwen35VLV2Manifest,
)
from inference.core.workflows.prototypes.block import roboflow_platform_model


def test_qwen3_5vl_v1_declares_default_model_version() -> None:
    manifest = Qwen35VLV1Manifest.model_validate(
        {
            "type": "roboflow_core/qwen3_5vl@v1",
            "name": "model",
            "images": "$inputs.image",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="qwen3_5-0.8b"),
    ]


def test_qwen3_5vl_v1_declares_explicit_model_version() -> None:
    manifest = Qwen35VLV1Manifest.model_validate(
        {
            "type": "roboflow_core/qwen3_5vl@v1",
            "name": "model",
            "images": "$inputs.image",
            "model_version": "qwen3_5-2b",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="qwen3_5-2b"),
    ]


def test_qwen3_5vl_v1_returns_selector_fed_model_version_verbatim() -> None:
    manifest = Qwen35VLV1Manifest.model_validate(
        {
            "type": "roboflow_core/qwen3_5vl@v1",
            "name": "model",
            "images": "$inputs.image",
            "model_version": "$inputs.model",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="$inputs.model"),
    ]


def test_qwen3_5vl_v2_declares_default_model_version() -> None:
    manifest = Qwen35VLV2Manifest.model_validate(
        {
            "type": "roboflow_core/qwen3_5vl@v2",
            "name": "model",
            "images": "$inputs.image",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="qwen3_5-0.8b"),
    ]


def test_qwen3_5vl_v2_declares_explicit_model_version() -> None:
    manifest = Qwen35VLV2Manifest.model_validate(
        {
            "type": "roboflow_core/qwen3_5vl@v2",
            "name": "model",
            "images": "$inputs.image",
            "model_version": "qwen3_5-4b",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="qwen3_5-4b"),
    ]


def test_qwen3_5vl_v2_returns_selector_fed_model_version_verbatim() -> None:
    manifest = Qwen35VLV2Manifest.model_validate(
        {
            "type": "roboflow_core/qwen3_5vl@v2",
            "name": "model",
            "images": "$inputs.image",
            "model_version": "$inputs.model",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="$inputs.model"),
    ]
