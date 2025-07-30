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
        == "The image shows a person carrying a Beagle dog on their shoulders. The dog appears to be happy, with its tongue out and looking upwards. The person is wearing a white shirt, a black cap, and a backpack. The background includes a street scene with buildings and a clear sky.<|im_end|>"
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
        == "The image shows a person carrying a Beagle dog on their shoulders. The dog appears to be happy, with its tongue out and looking upwards. The person is wearing a white shirt, a black cap, and a backpack. The background includes a street scene with buildings and a clear sky.<|im_end|>"
    )
