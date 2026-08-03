"""
Dependent-resources discovery tests for the Qwen2.5-VL block
(``roboflow_core/qwen25vl@v1``, module ``qwen.v1``) — the model id is held
directly in the ``model_version`` field.
"""

from inference.core.workflows.core_steps.models.foundation.qwen.v1 import (
    BlockManifest as Qwen25VLV1Manifest,
)
from inference.core.workflows.prototypes.block import roboflow_platform_model


def test_qwen25vl_v1_declares_default_model_version() -> None:
    manifest = Qwen25VLV1Manifest.model_validate(
        {
            "type": "roboflow_core/qwen25vl@v1",
            "name": "model",
            "images": "$inputs.image",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="qwen25-vl-7b"),
    ]


def test_qwen25vl_v1_declares_explicit_model_version() -> None:
    manifest = Qwen25VLV1Manifest.model_validate(
        {
            "type": "roboflow_core/qwen25vl@v1",
            "name": "model",
            "images": "$inputs.image",
            "model_version": "my_workspace/3",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="my_workspace/3"),
    ]


def test_qwen25vl_v1_returns_selector_fed_model_version_verbatim() -> None:
    manifest = Qwen25VLV1Manifest.model_validate(
        {
            "type": "roboflow_core/qwen25vl@v1",
            "name": "model",
            "images": "$inputs.image",
            "model_version": "$inputs.model",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="$inputs.model"),
    ]
