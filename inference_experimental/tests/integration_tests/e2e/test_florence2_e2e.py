import numpy as np
import pytest
import torch
from inference_exp import AutoModel


@pytest.mark.e2e_model_inference
def test_florence2_base_model(dog_image_numpy: np.ndarray):
    # GIVEN
    model = AutoModel.from_pretrained("florence-2-base")

    # WHEN
    captions = model.caption_image(dog_image_numpy)

    # THEN
    assert isinstance(captions, list)
    assert len(captions) == 1
    assert isinstance(captions[0], str)
    assert captions[0] == "A man carrying a dog on his back."


@pytest.mark.e2e_model_inference
def test_florence2_lora_model(
    dog_image_numpy: np.ndarray, dog_image_torch: torch.Tensor
):
    # GIVEN
    model = AutoModel.from_pretrained("florence-2-lora-test")

    # WHEN
    captions = model.caption_image(dog_image_numpy)

    # THEN
    assert isinstance(captions, list)
    assert len(captions) == 1
    assert isinstance(captions[0], str)
    assert captions[0] == "Disease"
