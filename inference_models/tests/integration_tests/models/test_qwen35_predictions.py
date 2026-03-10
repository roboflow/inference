import numpy as np
import pytest

from inference_models.models.qwen3_5.qwen3_5_hf import Qwen35HF


@pytest.fixture(scope="module")
def qwen_model(qwen35_08b_path: str) -> Qwen35HF:
    return Qwen35HF.from_pretrained(qwen35_08b_path)


@pytest.mark.slow
@pytest.mark.hf_vlm_models
def test_prompt(qwen_model: Qwen35HF, dog_image_numpy: np.ndarray):
    # when
    result = qwen_model.prompt(images=dog_image_numpy, prompt="What is in the image?")
    # then
    assert isinstance(result[0], str)
    assert len(result[0]) > 0
