import numpy as np
import pytest
import torch
from inference_exp import Detections
from inference_exp.models.moondream2.moondream2_hf import MoonDream2HF, Points


@pytest.fixture(scope="module")
def moondream2_model(moondream2_path: str) -> MoonDream2HF:
    return MoonDream2HF.from_pretrained(moondream2_path)


@pytest.mark.slow
@pytest.mark.hf_vlm_models
def test_detect(moondream2_model: MoonDream2HF, dog_image_numpy: np.ndarray):
    # when
    detections = moondream2_model.detect(
        images=dog_image_numpy, classes=["dog", "person"]
    )

    # then
    assert isinstance(detections, list)
    assert len(detections) == 1
    assert isinstance(detections[0], Detections)
    assert len(detections[0].xyxy) == 2


@pytest.mark.slow
@pytest.mark.hf_vlm_models
def test_caption(moondream2_model: MoonDream2HF, dog_image_numpy: np.ndarray):
    # when
    caption = moondream2_model.caption(images=dog_image_numpy)

    # then
    assert isinstance(caption, list)
    assert len(caption) == 1
    assert isinstance(caption[0], str)


@pytest.mark.slow
@pytest.mark.hf_vlm_models
def test_query(moondream2_model: MoonDream2HF, dog_image_numpy: np.ndarray):
    # when
    answer = moondream2_model.query(
        images=dog_image_numpy, question="What is in the image?"
    )

    # then
    assert isinstance(answer, list)
    assert len(answer) == 1
    assert isinstance(answer[0], str)


@pytest.mark.slow
@pytest.mark.hf_vlm_models
def test_point(moondream2_model: MoonDream2HF, dog_image_numpy: np.ndarray):
    # when
    points = moondream2_model.point(images=dog_image_numpy, classes=["dog", "person"])

    # then
    assert isinstance(points, list)
    assert len(points) == 1
    assert isinstance(points[0], Points)
    assert len(points[0].xy) == 2
