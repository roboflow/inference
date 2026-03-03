import numpy as np
import pytest

from inference_models import AutoModel


@pytest.mark.e2e_model_inference
def test_grounding_dino(dog_image_numpy: np.ndarray, roboflow_api_key: str) -> None:
    # given
    model = AutoModel.from_pretrained("grounding-dino", api_key=roboflow_api_key)

    # when
    predictions = model(dog_image_numpy, ["dog", "person", "bagpack"], conf_thresh=0.33)

    # then
    assert len(predictions[0].xyxy) >= 1
