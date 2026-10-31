import numpy as np
import pytest

from inference_models.configuration import DEFAULT_DEVICE
from inference_models.models.glm_ocr.glm_ocr_hf import GlmOcrHF


@pytest.fixture(scope="module")
def glm_ocr_model(glm_ocr_path: str) -> GlmOcrHF:
    return GlmOcrHF.from_pretrained(glm_ocr_path, device=DEFAULT_DEVICE)


@pytest.mark.slow
@pytest.mark.hf_vlm_models
def test_glm_ocr_prompt_with_numpy_image(
    glm_ocr_model: GlmOcrHF, ocr_test_image_numpy: np.ndarray
) -> None:
    # when
    result = glm_ocr_model.prompt(images=ocr_test_image_numpy)

    # then
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], str)
    assert len(result[0]) > 0


@pytest.mark.slow
@pytest.mark.hf_vlm_models
def test_glm_ocr_prompt_with_custom_prompt(
    glm_ocr_model: GlmOcrHF, ocr_test_image_numpy: np.ndarray
) -> None:
    # when
    result = glm_ocr_model.prompt(
        images=ocr_test_image_numpy, prompt="What text is in this image?"
    )

    # then
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], str)
    assert len(result[0]) > 0
