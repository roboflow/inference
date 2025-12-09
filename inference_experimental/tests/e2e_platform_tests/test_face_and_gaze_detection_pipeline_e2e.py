import numpy as np
import pytest
from inference_exp import AutoModelPipeline, Detections, KeyPoints
from inference_exp.models.l2cs.l2cs_onnx import L2CSGazeDetection


@pytest.mark.e2e_model_inference
def test_pipeline(man_image_numpy: np.ndarray, roboflow_api_key: str) -> None:
    # given
    pipeline = AutoModelPipeline.from_pretrained("face-and-gaze-detection")

    # when
    result = pipeline(man_image_numpy)

    # then
    assert len(result) == 3
    assert isinstance(result[0][0], KeyPoints)
    assert isinstance(result[1][0], Detections)
    assert isinstance(result[2][0], L2CSGazeDetection)
