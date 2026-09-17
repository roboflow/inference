"""
Manifest creation + ``discover_dependent_resources()`` for the DocTR OCR block
(``roboflow_core/ocr_model@v1``).

The manifest has NO version field: ``run()`` builds a
``DoctrOCRInferenceRequest`` without ``doctr_version_id``, so the request
default (the inline literal ``"default"``) is what ``load_core_model(...,
core_model="doctr")`` loads — the declared id is the constant
``doctr/default``.
"""

from inference.core.workflows.core_steps.models.foundation.ocr.v1 import (
    BlockManifest as OCRModelV1Manifest,
)
from inference.core.workflows.prototypes.block import roboflow_platform_model


def test_ocr_v1_declares_constant_doctr_model_id() -> None:
    manifest = OCRModelV1Manifest.model_validate(
        {
            "type": "roboflow_core/ocr_model@v1",
            "name": "ocr",
            "images": "$inputs.image",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="doctr/default"),
    ]
