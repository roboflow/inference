import numpy as np
import pytest
from inference_exp import AutoModel


@pytest.mark.e2e_model_inference
@pytest.mark.slow
def test_qwen25vl_base_model(dog_image_numpy: np.ndarray):
    # GIVEN
    model = AutoModel.from_pretrained("qwen25vl-7b")

    # WHEN
    captions = model.prompt(images=dog_image_numpy, prompt="What is in the image?")

    # THEN
    assert isinstance(captions, list)
    assert len(captions) == 1
    assert isinstance(captions[0], str)
    assert (
        captions[0]
        == "The image shows a person carrying a dog on their back. The dog appears to be painted blue, and its tongue is sticking out, suggesting it might be panting or excited. The person is wearing a black cap and a backpack with a visible logo on it. The background includes an urban setting with buildings and a street.<|im_end|>"
    )


@pytest.mark.e2e_model_inference
@pytest.mark.slow
def test_qwen25vl_lora_model(dog_image_numpy: np.ndarray):
    # GIVEN
    model = AutoModel.from_pretrained("qwen-lora-test")

    # WHEN
    captions = model.prompt(images=dog_image_numpy, prompt="What is in the image?")

    # THEN
    assert isinstance(captions, list)
    assert len(captions) == 1
    assert isinstance(captions[0], str)
    assert (
        captions[0]
        == "The image shows a person carrying a Beagle dog on their shoulders. The dog appears to be happy, with its tongue out and looking upwards. The person is wearing a white shirt, a black cap, and a backpack with a visible logo on the front. The background includes a street scene with buildings, a clear blue sky, and some vehicles parked in the distance. The overall atmosphere suggests a casual, sunny day.<|im_end|>"
    )
