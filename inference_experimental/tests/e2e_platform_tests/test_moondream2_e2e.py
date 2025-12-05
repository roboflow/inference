import numpy as np
import pytest
from inference_exp import AutoModel


@pytest.mark.e2e_model_inference
@pytest.mark.slow
@pytest.mark.gpu_only
def test_moondream2_model(dog_image_numpy: np.ndarray):
    # GIVEN
    model = AutoModel.from_pretrained("moondream2")

    # WHEN
    answer = model.query(images=dog_image_numpy, question="What is in the image?")

    # THEN
    assert isinstance(answer, list)
    assert len(answer) == 1
    assert isinstance(answer[0], str)
    assert len(answer[0]) > 0
