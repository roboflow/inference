import numpy as np
import pytest

from inference_models import AutoModel
from inference_models.configuration import DEFAULT_DEVICE


@pytest.mark.e2e_model_inference
@pytest.mark.gpu_only
def test_glm_ocr_model(ocr_test_image_numpy: np.ndarray, roboflow_api_key: str) -> None:
    # given
    model = AutoModel.from_pretrained(
        "glm-ocr",
        api_key=roboflow_api_key,
        device=DEFAULT_DEVICE,
    )

    # when
    result = model.prompt(images=ocr_test_image_numpy)

    # then
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], str)
    assert len(result[0]) > 0
