import numpy as np
import pytest

from inference_models import AutoModel
from inference_models.configuration import DEFAULT_DEVICE
from inference_models.models.sam2.entities import SAM2Prediction


@pytest.mark.e2e_model_inference
@pytest.mark.slow
def test_sam_model(truck_image_numpy: np.ndarray, roboflow_api_key: str) -> None:
    # given
    model = AutoModel.from_pretrained(
        "sam2/hiera_tiny",
        api_key=roboflow_api_key,
        device=DEFAULT_DEVICE,
    )

    # when
    results = model.segment_images(truck_image_numpy)

    # then
    assert isinstance(results[0], SAM2Prediction)
