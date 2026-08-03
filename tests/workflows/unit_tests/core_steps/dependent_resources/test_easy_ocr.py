"""
Manifest creation + ``discover_dependent_resources()`` for the EasyOCR block
(``roboflow_core/easy_ocr@v1``).

The ``language`` field is a pure Literal (no selector support) mapped through
the block's ``MODELS`` dict to a version token — ``run()`` passes
``easy_ocr_version_id=MODELS[language][0]`` and ``load_core_model(...,
core_model="easy_ocr")`` loads ``easy_ocr/<token>``. Default ``"English"``
maps to ``english_g2``. ``quantize`` does not participate in the model id.
"""

from inference.core.workflows.core_steps.models.foundation.easy_ocr.v1 import (
    BlockManifest as EasyOCRV1Manifest,
)
from inference.core.workflows.prototypes.block import roboflow_platform_model


def test_easy_ocr_v1_default_language_synthesizes_model_id() -> None:
    manifest = EasyOCRV1Manifest.model_validate(
        {
            "type": "roboflow_core/easy_ocr@v1",
            "name": "ocr",
            "images": "$inputs.image",
        }
    )

    assert manifest.language == "English"
    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="easy_ocr/english_g2"),
    ]


def test_easy_ocr_v1_non_default_language_maps_to_model_token() -> None:
    manifest = EasyOCRV1Manifest.model_validate(
        {
            "type": "roboflow_core/easy_ocr@v1",
            "name": "ocr",
            "images": "$inputs.image",
            "language": "Simplified Chinese",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="easy_ocr/zh_sim_g2"),
    ]


def test_easy_ocr_v1_quantize_does_not_change_model_id() -> None:
    manifest = EasyOCRV1Manifest.model_validate(
        {
            "type": "roboflow_core/easy_ocr@v1",
            "name": "ocr",
            "images": "$inputs.image",
            "quantize": True,
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="easy_ocr/english_g2"),
    ]
