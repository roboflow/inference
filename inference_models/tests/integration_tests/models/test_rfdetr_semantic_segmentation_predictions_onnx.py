import numpy as np
import pytest
import torch


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_stretch_numpy(
    balloons_rfdetr_sem_nano_onnx_static_bs_stretch_package: str,
    balloons_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_semantic_segmentation_onnx import (
        RFDetrForSemanticSegmentationOnnx,
    )

    model = RFDetrForSemanticSegmentationOnnx.from_pretrained(
        model_name_or_path=balloons_rfdetr_sem_nano_onnx_static_bs_stretch_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(balloons_image_numpy, confidence=0.5)

    # then
    _assert_balloon_prediction(predictions[0])


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_stretch_batch_numpy(
    balloons_rfdetr_sem_nano_onnx_static_bs_stretch_package: str,
    balloons_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_semantic_segmentation_onnx import (
        RFDetrForSemanticSegmentationOnnx,
    )

    model = RFDetrForSemanticSegmentationOnnx.from_pretrained(
        model_name_or_path=balloons_rfdetr_sem_nano_onnx_static_bs_stretch_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([balloons_image_numpy, balloons_image_numpy], confidence=0.5)

    # then
    _assert_balloon_prediction(predictions[0])
    _assert_balloon_prediction(predictions[1])


def _assert_balloon_prediction(prediction) -> None:
    assert sorted(torch.unique(prediction.segmentation_map).cpu().tolist()) == [0, 1]
    assert torch.allclose(
        torch.mean(prediction.confidence).cpu(),
        torch.tensor(0.9985),
        atol=0.001,
    )
    assert 247000 <= torch.sum(prediction.segmentation_map.cpu() == 0).item() <= 247450
    assert 14700 <= torch.sum(prediction.segmentation_map.cpu() == 1).item() <= 15150
