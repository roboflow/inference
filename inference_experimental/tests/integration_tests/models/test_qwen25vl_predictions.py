import numpy as np
import pytest
from inference_exp.models.qwen25vl.qwen25vl_hf import Qwen25VLHF


@pytest.fixture(scope="module")
def qwen_model(qwen25vl_3b_path: str) -> Qwen25VLHF:
    return Qwen25VLHF.from_pretrained(qwen25vl_3b_path)


@pytest.mark.slow
@pytest.mark.hf_vlm_models
def test_prompt(qwen_model: Qwen25VLHF, dog_image_numpy: np.ndarray):
    # when
    result = qwen_model.prompt(images=dog_image_numpy, prompt="What is in the image?")
    # then
    assert isinstance(result[0], str)
    assert len(result[0]) > 0
