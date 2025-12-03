import numpy as np
import pytest
from inference_exp import AutoModel


@pytest.mark.e2e_model_inference
def test_paligemma_base_model(dog_image_numpy: np.ndarray):
    # GIVEN
    model = AutoModel.from_pretrained("paligemma2-3b-pt-224")

    # WHEN
    captions = model.prompt(images=dog_image_numpy, prompt="What is in the image?")

    # THEN
    assert isinstance(captions, list)
    assert len(captions) == 1
    assert isinstance(captions[0], str)
    assert len(captions[0]) > 0


@pytest.mark.e2e_model_inference
@pytest.mark.gpu_only
def test_paligemma_lora_model(dog_image_numpy: np.ndarray):
    # GIVEN
    model = AutoModel.from_pretrained("paligemma-lora-test")

    # WHEN
    captions = model.prompt(images=dog_image_numpy, prompt="What is in the image?")

    # THEN
    assert isinstance(captions, list)
    assert len(captions) == 1
    assert isinstance(captions[0], str)
    assert len(captions[0]) > 0
