import numpy as np
import pytest
from inference_exp import AutoModel, Detections


@pytest.mark.e2e_model_inference
def test_easyocr_english(
    ocr_test_image_numpy: np.ndarray, roboflow_api_key: str
) -> None:
    # given
    model = AutoModel.from_pretrained("easy-ocr-english")

    # when
    result = model(ocr_test_image_numpy)

    # then
    assert len(result) == 2
    assert result[0][0].startswith("This is a test image for OCR")
    assert isinstance(result[1][0], Detections)


@pytest.mark.e2e_model_inference
def test_easyocr_latin(ocr_test_image_numpy: np.ndarray, roboflow_api_key: str) -> None:
    # given
    model = AutoModel.from_pretrained("easy-ocr-latin")

    # when
    result = model(ocr_test_image_numpy)

    # then
    assert len(result) == 2
    assert result[0][0].startswith("This is a test image for OCR")
    assert isinstance(result[1][0], Detections)


@pytest.mark.e2e_model_inference
def test_easyocr_japanese(
    ocr_test_image_numpy: np.ndarray, roboflow_api_key: str
) -> None:
    # given
    model = AutoModel.from_pretrained("easy-ocr-japanese")

    # when
    result = model(ocr_test_image_numpy)

    # then
    assert len(result) == 2
    assert isinstance(result[1][0], Detections)
