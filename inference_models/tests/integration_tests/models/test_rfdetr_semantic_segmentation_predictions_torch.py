import numpy as np
import pytest
import torch

from inference_models.configuration import DEFAULT_DEVICE
from inference_models.models.rfdetr.rfdetr_semantic_segmentation_pytorch import (
    RFDetrForSemanticSegmentationTorch,
)


@pytest.mark.slow
@pytest.mark.torch_models
def test_torch_package_with_stretch_numpy(
    balloons_rfdetr_sem_nano_torch_stretch_package: str,
    balloons_image_numpy: np.ndarray,
) -> None:
    # given
    model = RFDetrForSemanticSegmentationTorch.from_pretrained(
        model_name_or_path=balloons_rfdetr_sem_nano_torch_stretch_package,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(balloons_image_numpy, confidence=0.5)

    # then
    _assert_balloon_prediction(predictions[0])


@pytest.mark.slow
@pytest.mark.torch_models
def test_torch_package_with_stretch_batch_numpy(
    balloons_rfdetr_sem_nano_torch_stretch_package: str,
    balloons_image_numpy: np.ndarray,
) -> None:
    # given
    model = RFDetrForSemanticSegmentationTorch.from_pretrained(
        model_name_or_path=balloons_rfdetr_sem_nano_torch_stretch_package,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model([balloons_image_numpy, balloons_image_numpy], confidence=0.5)

    # then
    _assert_balloon_prediction(predictions[0])
    _assert_balloon_prediction(predictions[1])


@pytest.mark.slow
@pytest.mark.torch_models
def test_torch_package_with_stretch_torch_tensor(
    balloons_rfdetr_sem_nano_torch_stretch_package: str,
    balloons_image_torch: torch.Tensor,
) -> None:
    # given
    model = RFDetrForSemanticSegmentationTorch.from_pretrained(
        model_name_or_path=balloons_rfdetr_sem_nano_torch_stretch_package,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(balloons_image_torch, confidence=0.5)

    # then
    _assert_balloon_prediction(predictions[0])


def _assert_balloon_prediction(prediction) -> None:
    assert sorted(torch.unique(prediction.segmentation_map).cpu().tolist()) == [0, 1]
    assert torch.allclose(
        torch.mean(prediction.confidence).cpu(),
        torch.tensor(0.9985),
        atol=0.001,
    )
    assert 247000 <= torch.sum(prediction.segmentation_map.cpu() == 0).item() <= 247450
    assert 14700 <= torch.sum(prediction.segmentation_map.cpu() == 1).item() <= 15150
