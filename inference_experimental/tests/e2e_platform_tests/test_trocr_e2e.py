import numpy as np
import pytest
from inference_exp import AutoModel, Detections
from inference_exp.configuration import DEFAULT_DEVICE


@pytest.mark.e2e_model_inference
def test_small_ocr_model(
    ocr_test_image_numpy: np.ndarray, roboflow_api_key: str
) -> None:
    # given
    model = AutoModel.from_pretrained(
        "microsoft/trocr-small-printed",
        api_key=roboflow_api_key,
        device=DEFAULT_DEVICE,
    )

    # when
    result = model(ocr_test_image_numpy)

    # then
    assert len(result) == 1
    assert result[0] == "THIS IS A TEST IMAGE FOR OCR."


@pytest.mark.e2e_model_inference
def test_base_ocr_model(
    ocr_test_image_numpy: np.ndarray, roboflow_api_key: str
) -> None:
    # given
    model = AutoModel.from_pretrained(
        "microsoft/trocr-base-printed",
        api_key=roboflow_api_key,
        device=DEFAULT_DEVICE,
    )

    # when
    result = model(ocr_test_image_numpy)

    # then
    assert len(result) == 1
    assert result[0] == "THIS IS A TEST IMAGE FOR OCR."
