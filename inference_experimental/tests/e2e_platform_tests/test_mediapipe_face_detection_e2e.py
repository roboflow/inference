import numpy as np
import pytest
from inference_exp import AutoModel, Detections, KeyPoints


@pytest.mark.e2e_model_inference
def test_mediapipe_face_detection(
    man_image_numpy: np.ndarray, roboflow_api_key: str
) -> None:
    # given
    model = AutoModel.from_pretrained(
        "mediapipe/face-detector", api_key=roboflow_api_key
    )

    # when
    results = model(man_image_numpy)

    # then
    assert len(results) == 2
    assert isinstance(results[0][0], KeyPoints)
    assert isinstance(results[1][0], Detections)
