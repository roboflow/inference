import numpy as np
import pytest
from inference_exp import AutoModel, Detections
from inference_exp.configuration import DEFAULT_DEVICE


@pytest.mark.e2e_model_inference
def test_default_ocr_package(
    ocr_test_image_numpy: np.ndarray, roboflow_api_key: str
) -> None:
    # given
    model = AutoModel.from_pretrained(
        "doctr-dbnet-rn50/crnn-vgg16",
        api_key=roboflow_api_key,
        device=DEFAULT_DEVICE,
    )

    # when
    result = model(ocr_test_image_numpy)

    # then
    assert len(result) == 2
    assert result[0][0] == "This is a test image for OCR."
    assert isinstance(result[1][0], Detections)


@pytest.mark.e2e_model_inference
def test_non_default_ocr_package(
    ocr_test_image_numpy: np.ndarray, roboflow_api_key: str
) -> None:
    # given
    model = AutoModel.from_pretrained(
        "doctr-linknet-rn18/crnn-vgg16",
        api_key=roboflow_api_key,
        device=DEFAULT_DEVICE,
    )

    # when
    result = model(ocr_test_image_numpy)

    # then
    assert len(result) == 2
    assert result[0][0] == "This is a test image for OCR."
    assert isinstance(result[1][0], Detections)
