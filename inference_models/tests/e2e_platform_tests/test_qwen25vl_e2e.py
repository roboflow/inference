import numpy as np
import pytest

from inference_models import AutoModel


@pytest.mark.e2e_model_inference
@pytest.mark.slow
@pytest.mark.gpu_only
def test_qwen25vl_base_model(dog_image_numpy: np.ndarray):
    # GIVEN
    model = AutoModel.from_pretrained("qwen25vl-7b")

    # WHEN
    captions = model.prompt(images=dog_image_numpy, prompt="What is in the image?")

    # THEN
    assert isinstance(captions, list)
    assert len(captions) == 1
    assert isinstance(captions[0], str)
    assert len(captions[0]) > 0
