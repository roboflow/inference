import pytest

from inference.core.entities.requests.pp_ocr import PPOCRInferenceRequest


def test_pp_ocr_request_defaults_small_small() -> None:
    request = PPOCRInferenceRequest(
        image={"type": "url", "value": "https://some/image.jpg"}
    )

    assert request.pp_ocr_version_id == "small-small"
    assert request.model_id == "pp_ocr/small-small"


def test_pp_ocr_request_detect_only() -> None:
    request = PPOCRInferenceRequest(
        image={"type": "url", "value": "https://some/image.jpg"},
        text_recognition="none",
    )

    assert request.pp_ocr_version_id == "small-none"
    assert request.model_id == "pp_ocr/small-none"


def test_pp_ocr_request_recognize_only() -> None:
    request = PPOCRInferenceRequest(
        image={"type": "url", "value": "https://some/image.jpg"},
        text_detection="none",
    )

    assert request.pp_ocr_version_id == "none-small"
    assert request.model_id == "pp_ocr/none-small"


def test_pp_ocr_request_none_python_value_treated_as_none() -> None:
    request = PPOCRInferenceRequest(
        image={"type": "url", "value": "https://some/image.jpg"},
        text_detection=None,
        text_recognition="medium",
    )

    assert request.pp_ocr_version_id == "none-medium"
    assert request.model_id == "pp_ocr/none-medium"


def test_pp_ocr_request_both_none_raises() -> None:
    with pytest.raises(ValueError):
        PPOCRInferenceRequest(
            image={"type": "url", "value": "https://some/image.jpg"},
            text_detection="none",
            text_recognition="none",
        )


def test_pp_ocr_request_invalid_value_raises() -> None:
    with pytest.raises(ValueError):
        PPOCRInferenceRequest(
            image={"type": "url", "value": "https://some/image.jpg"},
            text_detection="huge",
        )
