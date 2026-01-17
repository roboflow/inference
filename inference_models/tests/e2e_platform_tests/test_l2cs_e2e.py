import numpy as np
import pytest

from inference_models import AutoModel
from inference_models.models.l2cs.l2cs_onnx import L2CSGazeDetection


@pytest.mark.e2e_model_inference
def test_l2cs_e2e(man_image_numpy: np.ndarray, roboflow_api_key: str) -> None:
    # given
    model = AutoModel.from_pretrained("l2cs-net/rn50", api_key=roboflow_api_key)

    # when
    results = model(man_image_numpy)

    # then
    assert isinstance(results, L2CSGazeDetection)
