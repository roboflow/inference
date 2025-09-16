import numpy as np
import pytest
import torch

from inference_exp.models.florence2.florence2_hf import Florence2HF


@pytest.fixture(scope="module")
def florence2_model(florence2_base_ft_path: str) -> Florence2HF:
    return Florence2HF.from_pretrained(florence2_base_ft_path)


@pytest.mark.slow
def test_classify_image_region(
    florence2_model: Florence2HF, dog_image_numpy: np.ndarray
):
    # given
    xyxy = [100, 100, 300, 300]
    # when
    result = florence2_model.classify_image_region(images=dog_image_numpy, xyxy=xyxy)
    # then
    assert result == ["human face"]


@pytest.mark.slow
def test_caption_image_region(
    florence2_model: Florence2HF, dog_image_numpy: np.ndarray
):
    # given
    xyxy = [100, 100, 300, 300]
    # when
    result = florence2_model.caption_image_region(images=dog_image_numpy, xyxy=xyxy)
    # then
    assert result == ["human face"]


@pytest.mark.slow
def test_ocr_image_region(
    florence2_model: Florence2HF, ocr_test_image_numpy: np.ndarray
):
    # TODO: figure out if this is imlementation error? doesnt really seem to work, like just returns text from the whole image
    # given
    xyxy = [0, 0, 100, 150]
    # when
    result = florence2_model.ocr_image_region(images=ocr_test_image_numpy, xyxy=xyxy)
    # then
    assert result == ["This is a test image for OCR."]


@pytest.mark.slow
def test_segment_region(florence2_model: Florence2HF, dog_image_numpy: np.ndarray):
    # given
    xyxy = [100, 100, 300, 300]
    # when
    result = florence2_model.segment_region(images=dog_image_numpy, xyxy=xyxy)
    # then
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].xyxy.shape == (1, 4)
    assert torch.allclose(
        result[0].xyxy, torch.tensor([[100, 100, 302, 303]], dtype=torch.int32), atol=2
    )
    assert result[0].mask.shape == (1, 1280, 720)


@pytest.mark.slow
def test_segment_phrase(florence2_model: Florence2HF, dog_image_numpy: np.ndarray):
    # when
    result = florence2_model.segment_phrase(images=dog_image_numpy, phrase="dog")
    # then
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].xyxy.shape == (1, 4)
    assert torch.allclose(
        result[0].xyxy, torch.tensor([[71, 249, 649, 926]], dtype=torch.int32), atol=5
    )
    assert result[0].mask.shape == (1, 1280, 720)


@pytest.mark.slow
def test_detect_objects(florence2_model: Florence2HF, dog_image_numpy: np.ndarray):
    # when
    result = florence2_model.detect_objects(images=dog_image_numpy)
    # then
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].xyxy.shape == (4, 4)
    expected_bboxes_metadata = [
        {"class_name": "backpack"},
        {"class_name": "dog"},
        {"class_name": "hat"},
        {"class_name": "person"},
    ]
    assert result[0].bboxes_metadata == expected_bboxes_metadata


@pytest.mark.slow
def test_caption_image(florence2_model: Florence2HF, dog_image_numpy: np.ndarray):
    # when
    result = florence2_model.caption_image(images=dog_image_numpy)
    # then
    assert result == ["A man carrying a blue dog on his back."]


@pytest.mark.slow
def test_parse_document(florence2_model: Florence2HF, ocr_test_image_numpy: np.ndarray):
    # when
    result = florence2_model.parse_document(images=ocr_test_image_numpy)
    # then
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].xyxy.shape[0] >= 1
    assert result[0].xyxy.shape[1] == 4
    full_text = "".join(
        meta["class_name"] for meta in result[0].bboxes_metadata
    ).lstrip("</s>")
    assert full_text == "This is a test image for OCR."


@pytest.mark.slow
def test_ocr_image(florence2_model: Florence2HF, ocr_test_image_numpy: np.ndarray):
    # when
    result = florence2_model.ocr_image(images=ocr_test_image_numpy)
    # then
    assert result == ["This is a test image for OCR."]
