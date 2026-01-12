import numpy as np
import pytest

from inference_models import AutoModel


@pytest.mark.e2e_model_inference
@pytest.mark.gpu_only
def test_depth_anything_v3_small(
    roboflow_api_key: str, dog_image_numpy: np.ndarray
) -> None:
    # given
    model = AutoModel.from_pretrained(
        "depth-anything-v3/small", api_key=roboflow_api_key
    )

    # when
    results = model(dog_image_numpy)

    # then
    assert results[0].cpu().shape == (1280, 720)


@pytest.mark.e2e_model_inference
@pytest.mark.gpu_only
def test_depth_anything_v3_base(
    roboflow_api_key: str, dog_image_numpy: np.ndarray
) -> None:
    # given
    model = AutoModel.from_pretrained(
        "depth-anything-v3/base", api_key=roboflow_api_key
    )

    # when
    results = model(dog_image_numpy)

    # then
    assert results[0].cpu().shape == (1280, 720)
