import numpy as np
import pytest
from inference_exp.models.paligemma.paligemma_hf import PaliGemmaHF


@pytest.fixture(scope="module")
def paligemma_model(paligemma_3b_224_path: str) -> PaliGemmaHF:
    return PaliGemmaHF.from_pretrained(paligemma_3b_224_path)


@pytest.mark.slow
@pytest.mark.hf_vlm_models
def test_prompt(paligemma_model: PaliGemmaHF, dog_image_numpy: np.ndarray):
    # when
    result = paligemma_model.prompt(
        images=dog_image_numpy, prompt="What is in the image?"
    )
    # then
    assert isinstance(result[0], str)
    assert len(result[0]) > 0


@pytest.mark.slow
@pytest.mark.hf_vlm_models
def test_prompt_dog_type(paligemma_model: PaliGemmaHF, dog_image_numpy: np.ndarray):
    # when
    result = paligemma_model.prompt(
        images=dog_image_numpy, prompt="What type of dog is this?"
    )
    # then
    assert isinstance(result[0], str)
    assert len(result[0]) > 0
