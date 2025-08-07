import numpy as np
import pytest
from inference_exp import AutoModel


@pytest.mark.e2e_model_inference
def test_smolvlm_base_model(dog_image_numpy: np.ndarray):
    # GIVEN
    model = AutoModel.from_pretrained("smolvlm-256m")

    # WHEN
    captions = model.prompt(images=dog_image_numpy, prompt="What is in the image?")

    # THEN
    assert isinstance(captions, list)
    assert len(captions) == 1
    assert isinstance(captions[0], str)
    assert captions[0] == "There is a person and a dog in the image."


@pytest.mark.e2e_model_inference
def test_smolvlm_lora_model(dog_image_numpy: np.ndarray):
    # GIVEN
    model = AutoModel.from_pretrained("smolvlm-lora-test")

    # WHEN
    captions = model.prompt(images=dog_image_numpy, prompt="What is in the image?")

    # THEN
    assert isinstance(captions, list)
    assert len(captions) == 1
    assert isinstance(captions[0], str)
    assert captions[0] == "There is a man in the image."
