import numpy as np
import pytest

from inference_models import AutoModel


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
    assert len(captions[0]) > 0


@pytest.mark.e2e_model_inference
@pytest.mark.gpu_only
def test_florence2_base_model(dog_image_numpy: np.ndarray):
    # GIVEN
    model = AutoModel.from_pretrained("florence-2-large")

    # WHEN
    captions = model.caption_image(dog_image_numpy)

    # THEN
    assert isinstance(captions, list)
    assert len(captions) == 1
    assert isinstance(captions[0], str)
    assert len(captions[0]) > 0
