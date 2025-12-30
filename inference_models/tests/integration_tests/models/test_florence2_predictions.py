import numpy as np
import pytest
import torch

from inference_models.models.florence2.florence2_hf import Florence2HF


@pytest.fixture(scope="module")
def florence2_model(florence2_base_ft_path: str) -> Florence2HF:
    return Florence2HF.from_pretrained(florence2_base_ft_path)


@pytest.mark.slow
@pytest.mark.hf_vlm_models
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
@pytest.mark.hf_vlm_models
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
@pytest.mark.hf_vlm_models
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
@pytest.mark.hf_vlm_models
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
        result[0].xyxy.cpu(),
        torch.tensor([[100, 100, 302, 303]], dtype=torch.int32),
        atol=2,
    )
    assert result[0].mask.shape == (1, 1280, 720)


@pytest.mark.slow
@pytest.mark.hf_vlm_models
@pytest.mark.skip(
    "This test may indicate broken package or implementation broken - TODO - fix"
)
def test_segment_phrase(florence2_model: Florence2HF, dog_image_numpy: np.ndarray):
    # when
    result = florence2_model.segment_phrase(images=dog_image_numpy, phrase="dog")
    # then
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].xyxy.shape == (1, 4)
    assert torch.allclose(
        result[0].xyxy.cpu(),
        torch.tensor([[73, 251, 628, 928]], dtype=torch.int32),
        atol=5,
    )
    assert result[0].mask.shape == (1, 1280, 720)


@pytest.mark.slow
@pytest.mark.hf_vlm_models
def test_detect_objects(florence2_model: Florence2HF, dog_image_numpy: np.ndarray):
    # when
    result = florence2_model.detect_objects(images=dog_image_numpy)
    # then
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].xyxy.shape[1] == 4


@pytest.mark.slow
@pytest.mark.hf_vlm_models
def test_caption_image(florence2_model: Florence2HF, dog_image_numpy: np.ndarray):
    # when
    result = florence2_model.caption_image(images=dog_image_numpy)
    # then
    assert isinstance(result[0], str)


@pytest.mark.slow
@pytest.mark.hf_vlm_models
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
@pytest.mark.hf_vlm_models
def test_ocr_image(florence2_model: Florence2HF, ocr_test_image_numpy: np.ndarray):
    # when
    result = florence2_model.ocr_image(images=ocr_test_image_numpy)
    # then
    assert result == ["This is a test image for OCR."]


@pytest.mark.slow
@pytest.mark.hf_vlm_models
def test_caption_image_input_formats(
    florence2_model: Florence2HF,
    dog_image_numpy: np.ndarray,
    dog_image_torch: torch.Tensor,
):
    # when
    result_numpy = florence2_model.caption_image(images=dog_image_numpy)
    result_tensor = florence2_model.caption_image(images=dog_image_torch)
    # then
    assert result_numpy[0] == result_tensor[0]
