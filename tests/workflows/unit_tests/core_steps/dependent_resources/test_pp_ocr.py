"""
Manifest creation + ``discover_dependent_resources()`` for the PP-OCR block
(``roboflow_core/pp_ocr@v1``).

The model id composes BOTH stage fields as
``pp_ocr/<text_detection>-<text_recognition>`` — mirroring ``run()``, where
``PPOCRInferenceRequest`` derives ``pp_ocr_version_id = f"{det}-{rec}"`` and
``load_core_model(..., core_model="pp_ocr")`` loads it. Both stage fields are
pure Literals (no selector support), so there is no verbatim-selector case.
A manifest-level validator rejects disabling both stages at once.
"""

import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.models.foundation.pp_ocr.v1 import (
    BlockManifest as PPOCRV1Manifest,
)
from inference.core.workflows.prototypes.block import roboflow_platform_model


def test_pp_ocr_v1_default_stages_synthesize_model_id() -> None:
    manifest = PPOCRV1Manifest.model_validate(
        {
            "type": "roboflow_core/pp_ocr@v1",
            "name": "ocr",
            "images": "$inputs.image",
        }
    )

    assert manifest.text_detection == "small"
    assert manifest.text_recognition == "small"
    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="pp_ocr/small-small"),
    ]


def test_pp_ocr_v1_disabled_detection_stage_synthesizes_model_id() -> None:
    manifest = PPOCRV1Manifest.model_validate(
        {
            "type": "roboflow_core/pp_ocr@v1",
            "name": "ocr",
            "images": "$inputs.image",
            "text_detection": "none",
            "text_recognition": "small",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="pp_ocr/none-small"),
    ]


def test_pp_ocr_v1_mixed_stage_sizes_synthesize_model_id() -> None:
    manifest = PPOCRV1Manifest.model_validate(
        {
            "type": "roboflow_core/pp_ocr@v1",
            "name": "ocr",
            "images": "$inputs.image",
            "text_detection": "tiny",
            "text_recognition": "medium",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="pp_ocr/tiny-medium"),
    ]


def test_pp_ocr_v1_rejects_disabling_both_stages() -> None:
    with pytest.raises(ValidationError):
        PPOCRV1Manifest.model_validate(
            {
                "type": "roboflow_core/pp_ocr@v1",
                "name": "ocr",
                "images": "$inputs.image",
                "text_detection": "none",
                "text_recognition": "none",
            }
        )
