import numpy as np
import pytest

from inference_models.models.smolvlm.smolvlm_hf import SmolVLMHF


@pytest.fixture(scope="module")
def smolvlm_model(smolvlm_256m_path: str) -> SmolVLMHF:
    return SmolVLMHF.from_pretrained(smolvlm_256m_path)


@pytest.mark.slow
@pytest.mark.hf_vlm_models
def test_prompt(smolvlm_model: SmolVLMHF, dog_image_numpy: np.ndarray):
    # when
    result = smolvlm_model.prompt(
        images=dog_image_numpy, prompt="What is in the image?"
    )
    # then
    assert isinstance(result[0], str)
    assert len(result[0]) > 0
