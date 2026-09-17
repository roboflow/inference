"""
Dependent-resources discovery tests for the GLM-OCR block
(``roboflow_core/glm_ocr@v1``) — the model id is held directly in the
``model_version`` field.
"""

from inference.core.workflows.core_steps.models.foundation.glm_ocr.v1 import (
    BlockManifest as GLMOCRV1Manifest,
)
from inference.core.workflows.prototypes.block import roboflow_platform_model


def test_glm_ocr_v1_declares_default_model_version() -> None:
    manifest = GLMOCRV1Manifest.model_validate(
        {
            "type": "roboflow_core/glm_ocr@v1",
            "name": "ocr",
            "images": "$inputs.image",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="glm-ocr"),
    ]


def test_glm_ocr_v1_declares_explicit_model_version() -> None:
    manifest = GLMOCRV1Manifest.model_validate(
        {
            "type": "roboflow_core/glm_ocr@v1",
            "name": "ocr",
            "images": "$inputs.image",
            "model_version": "my_workspace/3",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="my_workspace/3"),
    ]


def test_glm_ocr_v1_returns_selector_fed_model_version_verbatim() -> None:
    manifest = GLMOCRV1Manifest.model_validate(
        {
            "type": "roboflow_core/glm_ocr@v1",
            "name": "ocr",
            "images": "$inputs.image",
            "model_version": "$inputs.model",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="$inputs.model"),
    ]
