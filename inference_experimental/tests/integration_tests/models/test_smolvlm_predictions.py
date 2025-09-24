import numpy as np
import pytest

from inference_exp.models.smolvlm.smolvlm_hf import SmolVLMHF


@pytest.fixture(scope="module")
def smolvlm_model(smolvlm_256m_path: str) -> SmolVLMHF:
    return SmolVLMHF.from_pretrained(smolvlm_256m_path)


@pytest.mark.slow
def test_prompt(smolvlm_model: SmolVLMHF, dog_image_numpy: np.ndarray):
    # when
    result = smolvlm_model.prompt(
        images=dog_image_numpy, prompt="What is in the image?"
    )
    # then
    assert result == ["There is a person and a dog in the image."]
