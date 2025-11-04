import numpy as np
import pytest
import torch


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_batch_size_and_stretch_numpy(
    yolov8_cls_static_bs_onnx_package: str,
    dog_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_classification_onnx import (
        YOLOv8ForClassificationOnnx,
    )

    model = YOLOv8ForClassificationOnnx.from_pretrained(
        model_name_or_path=yolov8_cls_static_bs_onnx_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(dog_image_numpy)

    # then
    np.allclose(predictions.class_id.cpu().numpy(), 162)
    assert np.allclose(
        np.max(predictions.confidence.cpu().numpy()), [0.0013962699], atol=0.0001
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_batch_size_and_stretch_numpy_list(
    yolov8_cls_static_bs_onnx_package: str,
    dog_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_classification_onnx import (
        YOLOv8ForClassificationOnnx,
    )

    model = YOLOv8ForClassificationOnnx.from_pretrained(
        model_name_or_path=yolov8_cls_static_bs_onnx_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([dog_image_numpy, dog_image_numpy])

    # then
    np.allclose(predictions.class_id.cpu().numpy(), [162, 162])
    assert np.allclose(
        np.max(predictions.confidence.cpu().numpy(), axis=1),
        [0.0013962699],
        atol=0.0001,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_batch_size_and_stretch_torch(
    yolov8_cls_static_bs_onnx_package: str,
    dog_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_classification_onnx import (
        YOLOv8ForClassificationOnnx,
    )

    model = YOLOv8ForClassificationOnnx.from_pretrained(
        model_name_or_path=yolov8_cls_static_bs_onnx_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(dog_image_torch)

    # then
    np.allclose(predictions.class_id.cpu().numpy(), 162)
    assert np.allclose(
        np.max(predictions.confidence.cpu().numpy()), [0.0013962699], atol=0.0001
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_batch_size_and_stretch_torch_batch(
    yolov8_cls_static_bs_onnx_package: str,
    dog_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_classification_onnx import (
        YOLOv8ForClassificationOnnx,
    )

    model = YOLOv8ForClassificationOnnx.from_pretrained(
        model_name_or_path=yolov8_cls_static_bs_onnx_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(torch.stack([dog_image_torch, dog_image_torch], dim=0))

    # then
    np.allclose(predictions.class_id.cpu().numpy(), [162, 162])
    assert np.allclose(
        np.max(predictions.confidence.cpu().numpy(), axis=1),
        [0.0013962699],
        atol=0.0001,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_batch_size_and_stretch_torch_list(
    yolov8_cls_static_bs_onnx_package: str,
    dog_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_classification_onnx import (
        YOLOv8ForClassificationOnnx,
    )

    model = YOLOv8ForClassificationOnnx.from_pretrained(
        model_name_or_path=yolov8_cls_static_bs_onnx_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([dog_image_torch, dog_image_torch])

    # then
    np.allclose(predictions.class_id.cpu().numpy(), [162, 162])
    assert np.allclose(
        np.max(predictions.confidence.cpu().numpy(), axis=1),
        [0.0013962699],
        atol=0.0001,
    )
