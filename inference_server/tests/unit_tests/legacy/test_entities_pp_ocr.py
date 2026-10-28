import dataclasses

import pytest

from inference_model_manager import pipelines
from inference_model_manager.pipelines import default_stage_tokens
from inference_server.legacy.entities import PPOCRInferenceRequest

_IMAGE = {"type": "base64", "value": "AA=="}


def test_omitted_stages_take_the_manager_defaults():
    request = PPOCRInferenceRequest(image=_IMAGE)
    assert (request.text_detection, request.text_recognition) == default_stage_tokens(
        "pp_ocr"
    )
    assert request.pp_ocr_version_id == "small-small"
    assert request.model_id == "pp_ocr/small-small"


def test_entity_reads_tokens_from_the_manager_at_validation_time(monkeypatch):
    family = pipelines.PIPELINE_FAMILIES["pp_ocr"]
    monkeypatch.setitem(
        pipelines.PIPELINE_FAMILIES,
        "pp_ocr",
        dataclasses.replace(family, stage_tokens=family.stage_tokens | {"huge"}),
    )
    request = PPOCRInferenceRequest(image=_IMAGE, text_detection="huge")
    assert request.model_id == "pp_ocr/huge-small"


def test_entity_reads_defaults_from_the_manager_at_validation_time(monkeypatch):
    from inference_models.model_pipelines.auto_loaders import pipelines_registry

    monkeypatch.setitem(
        pipelines_registry.DEFAULT_PIPELINES_PARAMETERS,
        "pp-ocrv6",
        ["pp-ocrv6-det/tiny", "pp-ocrv6-rec/medium"],
    )
    assert PPOCRInferenceRequest(image=_IMAGE).model_id == "pp_ocr/tiny-medium"


@pytest.mark.parametrize(
    "payload,message",
    [
        ({"text_detection": "huge"}, "Invalid PP-OCR text_detection value: huge"),
        ({"text_recognition": "huge"}, "Invalid PP-OCR text_recognition value: huge"),
        (
            {"text_detection": None, "text_recognition": None},
            "PP-OCR requires at least one of detection or recognition",
        ),
        (
            {"pp_ocr_version_id": "a-b-c"},
            "Invalid PP-OCR pp_ocr_version_id value: a-b-c",
        ),
    ],
)
def test_error_messages_are_unchanged(payload, message):
    with pytest.raises(ValueError, match=message):
        PPOCRInferenceRequest(image=_IMAGE, **payload)


@pytest.mark.parametrize(
    "payload,model_id",
    [
        ({"pp_ocr_version_id": "tiny"}, "pp_ocr/tiny-tiny"),
        ({"pp_ocr_version_id": "medium-small"}, "pp_ocr/medium-small"),
        ({"text_recognition": None}, "pp_ocr/small-none"),
        (
            {"text_detection": "None", "text_recognition": "Medium"},
            "pp_ocr/none-medium",
        ),
    ],
)
def test_version_id_and_stage_forms(payload, model_id):
    assert PPOCRInferenceRequest(image=_IMAGE, **payload).model_id == model_id
