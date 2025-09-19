import numpy as np
import pytest
import torch
from inference_exp import Detections
from inference_exp.models.moondream2.moondream2_hf import MoonDream2HF, Points


@pytest.fixture(scope="module")
def moondream2_model(moondream2_path: str) -> MoonDream2HF:
    return MoonDream2HF.from_pretrained(moondream2_path)


@pytest.mark.slow
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
    assert torch.allclose(
        detections[0].xyxy,
        torch.tensor([[64, 253, 628, 925], [0, 358, 646, 1277]], dtype=torch.int32),
    )
    assert torch.allclose(
        detections[0].class_id,
        torch.tensor([0, 1], dtype=torch.int32),
    )


@pytest.mark.slow
def test_caption(moondream2_model: MoonDream2HF, dog_image_numpy: np.ndarray):
    # when
    caption = moondream2_model.caption(images=dog_image_numpy)

    # then
    assert isinstance(caption, list)
    assert len(caption) == 1
    assert isinstance(caption[0], str)
    assert (
        caption[0]
        == "A person wearing a black baseball cap and a white t-shirt is carrying a beagle on their back. The beagle, with its light brown and white fur, is sitting comfortably on the person's shoulder, its tongue hanging out in a playful manner. The person is also wearing a black backpack with a white logo. The background features a cityscape with a tall building and a street, with a red car visible in the distance. The sky is a clear blue with a few clouds."
    )


@pytest.mark.slow
def test_query(moondream2_model: MoonDream2HF, dog_image_numpy: np.ndarray):
    # when
    answer = moondream2_model.query(
        images=dog_image_numpy, question="What is in the image?"
    )

    # then
    assert isinstance(answer, list)
    assert len(answer) == 1
    assert isinstance(answer[0], str)
    assert (
        answer[0]
        == "The image features a man carrying a beagle on his back, with the dog sitting on his shoulder."
    )


@pytest.mark.slow
def test_point(moondream2_model: MoonDream2HF, dog_image_numpy: np.ndarray):
    # when
    points = moondream2_model.point(images=dog_image_numpy, classes=["dog", "person"])

    # then
    assert isinstance(points, list)
    assert len(points) == 1
    assert isinstance(points[0], Points)
    assert len(points[0].xy) == 2
    assert torch.allclose(
        points[0].xy,
        torch.tensor([[367, 355], [323, 872]], dtype=torch.int32),
    )
    assert torch.allclose(
        points[0].class_id,
        torch.tensor([0, 1], dtype=torch.int32),
    )
