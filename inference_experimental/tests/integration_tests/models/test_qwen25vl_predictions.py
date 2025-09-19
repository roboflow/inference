import numpy as np
import pytest
from inference_exp.models.qwen25vl.qwen25vl_hf import Qwen25VLHF


@pytest.fixture(scope="module")
def qwen_model(qwen25vl_3b_path: str) -> Qwen25VLHF:
    return Qwen25VLHF.from_pretrained(qwen25vl_3b_path)


@pytest.mark.slow
def test_prompt(qwen_model: Qwen25VLHF, dog_image_numpy: np.ndarray):
    # when
    result = qwen_model.prompt(images=dog_image_numpy, prompt="What is in the image?")
    # then
    assert (
        result[0]
        == "The image shows a person carrying a dog on their back. The dog appears to be a Beagle, with its tongue out and ears floppy. The person is wearing a white shirt and a black cap. They have a backpack on, which has a logo on it. The background includes a street scene with buildings and a clear blue sky.<|im_end|>"
    )
