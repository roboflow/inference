import numpy as np
import pytest
from inference_exp import AutoModel, Detections
from inference_exp.configuration import DEFAULT_DEVICE


@pytest.mark.e2e_model_inference
@pytest.mark.slow
def test_owlv2_model(dog_image_numpy: np.ndarray, roboflow_api_key: str) -> None:
    # given
    model = AutoModel.from_pretrained(
        "google/owlv2-large-patch14-ensemble",
        api_key=roboflow_api_key,
        device=DEFAULT_DEVICE,
    )

    # when
    results = model(dog_image_numpy, classes=["dog", "person"])

    # then
    assert isinstance(results[0], Detections)
    assert len(results[0].xyxy) == 2


@pytest.mark.e2e_model_inference
@pytest.mark.slow
def test_roboflow_instant_model(
    coins_counting_image_numpy: np.ndarray, roboflow_api_key: str
) -> None:
    # given
    model = AutoModel.from_pretrained(
        "paul-guerrie-tang1/coin-counting-instant-2",
        api_key=roboflow_api_key,
        device=DEFAULT_DEVICE,
    )

    # when
    results = model(
        coins_counting_image_numpy,
        confidence_threshold=0.95,
    )

    # then
    assert isinstance(results[0], Detections)
    assert len(results[0].xyxy) == 9
