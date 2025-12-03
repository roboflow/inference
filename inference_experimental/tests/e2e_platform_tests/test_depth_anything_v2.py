import numpy as np
import pytest

from inference_exp import AutoModel


@pytest.mark.e2e_model_inference
@pytest.mark.gpu_only
def test_depth_anything_v2_small(roboflow_api_key: str, dog_image_numpy: np.ndarray) -> None:
    # given
    model = AutoModel.from_pretrained("depth-anything-v2/small", api_key=roboflow_api_key)

    # when
    results = model(dog_image_numpy)

    # then
    assert results[0].cpu().shape == (1280, 720)


@pytest.mark.e2e_model_inference
@pytest.mark.gpu_only
def test_depth_anything_v2_small_via_alias(roboflow_api_key: str, dog_image_numpy: np.ndarray) -> None:
    # given
    model = AutoModel.from_pretrained("depth-anything-v2", api_key=roboflow_api_key)

    # when
    results = model(dog_image_numpy)

    # then
    assert results[0].cpu().shape == (1280, 720)


@pytest.mark.e2e_model_inference
@pytest.mark.gpu_only
def test_depth_anything_v2_base(roboflow_api_key: str, dog_image_numpy: np.ndarray) -> None:
    # given
    model = AutoModel.from_pretrained("depth-anything-v2/base", api_key=roboflow_api_key)

    # when
    results = model(dog_image_numpy)

    # then
    assert results[0].cpu().shape == (1280, 720)


@pytest.mark.e2e_model_inference
@pytest.mark.gpu_only
def test_depth_anything_v2_large(roboflow_api_key: str, dog_image_numpy: np.ndarray) -> None:
    # given
    model = AutoModel.from_pretrained("depth-anything-v2/large", api_key=roboflow_api_key)

    # when
    results = model(dog_image_numpy)

    # then
    assert results[0].cpu().shape == (1280, 720)
