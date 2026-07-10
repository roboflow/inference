import numpy as np
import pytest

from inference_models import AutoModel, AutoModelPipeline


@pytest.mark.e2e_model_inference
def test_pp_ocrv6_detection(
    ocr_test_image_numpy: np.ndarray, roboflow_api_key: str
) -> None:
    # given
    model = AutoModel.from_pretrained("pp-ocrv6-det/small")

    # when
    detections = model(ocr_test_image_numpy)[0]

    # then
    assert len(detections.xyxy) > 0, "Expected text lines to be detected"
    assert all("polygon" in meta for meta in detections.bboxes_metadata)


@pytest.mark.e2e_model_inference
def test_pp_ocrv6_recognition(
    ocr_test_image_numpy: np.ndarray, roboflow_api_key: str
) -> None:
    # given
    model = AutoModel.from_pretrained("pp-ocrv6-rec/small")
    # recognition expects a single text-line crop - cut the first line out
    text_line_crop = ocr_test_image_numpy[: ocr_test_image_numpy.shape[0] // 4]

    # when
    texts = model(text_line_crop)

    # then
    assert len(texts) == 1
    assert isinstance(texts[0], str)
    assert len(texts[0].strip()) > 0, "Expected text to be recognized"


@pytest.mark.e2e_model_inference
def test_pp_ocrv6_pipeline(
    ocr_test_image_numpy: np.ndarray, roboflow_api_key: str
) -> None:
    # given
    pipeline = AutoModelPipeline.from_pretrained("pp-ocrv6")

    # when
    result = pipeline(ocr_test_image_numpy)[0]

    # then
    assert "This is a test image for OCR" in result.text
    assert len(result.line_texts) == len(result.detections.xyxy)
