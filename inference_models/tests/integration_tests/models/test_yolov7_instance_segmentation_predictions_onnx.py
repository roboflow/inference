import numpy as np
import pytest
import torch


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_seg_onnx_package_with_static_batch_size_and_letterbox_numpy(
    asl_yolov7_onnx_seg_static_bs_letterbox: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yolov7.yolov7_instance_segmentation_onnx import (
        YOLOv7ForInstanceSegmentationOnnx,
    )

    model = YOLOv7ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov7_onnx_seg_static_bs_letterbox,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(asl_image_numpy, conf_thresh=0.6)

    # then
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].xyxy[0].cpu().numpy(), [59, 162, 192, 387], atol=1
    )
    assert np.allclose(predictions[0].class_id[0].cpu().numpy(), [0], atol=1)
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), [0.7349], atol=0.005)
    assert 16600 <= predictions[0].mask[0].cpu().numpy().sum() <= 16900


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_seg_onnx_package_with_static_batch_size_and_letterbox_numpy_list(
    asl_yolov7_onnx_seg_static_bs_letterbox: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yolov7.yolov7_instance_segmentation_onnx import (
        YOLOv7ForInstanceSegmentationOnnx,
    )

    model = YOLOv7ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov7_onnx_seg_static_bs_letterbox,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([asl_image_numpy, asl_image_numpy], conf_thresh=0.6)

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy[0].cpu().numpy(), [59, 162, 192, 387], atol=1
    )
    assert np.allclose(predictions[0].class_id[0].cpu().numpy(), [0], atol=1)
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), [0.7349], atol=0.005)
    assert 16600 <= predictions[0].mask[0].cpu().numpy().sum() <= 16900
    assert np.allclose(
        predictions[1].xyxy[0].cpu().numpy(), [59, 162, 192, 387], atol=1
    )
    assert np.allclose(predictions[1].class_id[0].cpu().numpy(), [0], atol=1)
    assert np.allclose(predictions[1].confidence[0].cpu().numpy(), [0.7349], atol=0.005)
    assert 16600 <= predictions[1].mask[0].cpu().numpy().sum() <= 16900


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_seg_onnx_package_with_static_batch_size_and_letterbox_torch(
    asl_yolov7_onnx_seg_static_bs_letterbox: str, asl_image_torch: torch.Tensor
) -> None:
    # given
    from inference_models.models.yolov7.yolov7_instance_segmentation_onnx import (
        YOLOv7ForInstanceSegmentationOnnx,
    )

    model = YOLOv7ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov7_onnx_seg_static_bs_letterbox,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(asl_image_torch, conf_thresh=0.6)

    # then
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].xyxy[0].cpu().numpy(), [59, 162, 192, 387], atol=1
    )
    assert np.allclose(predictions[0].class_id[0].cpu().numpy(), [0], atol=1)
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), [0.7349], atol=0.005)
    assert 16600 <= predictions[0].mask[0].cpu().numpy().sum() <= 16900


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_seg_onnx_package_with_static_batch_size_and_letterbox_torch_list(
    asl_yolov7_onnx_seg_static_bs_letterbox: str, asl_image_torch: torch.Tensor
) -> None:
    # given
    from inference_models.models.yolov7.yolov7_instance_segmentation_onnx import (
        YOLOv7ForInstanceSegmentationOnnx,
    )

    model = YOLOv7ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov7_onnx_seg_static_bs_letterbox,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([asl_image_torch, asl_image_torch], conf_thresh=0.6)

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy[0].cpu().numpy(), [59, 162, 192, 387], atol=1
    )
    assert np.allclose(predictions[0].class_id[0].cpu().numpy(), [0], atol=1)
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), [0.7349], atol=0.005)
    assert 16600 <= predictions[0].mask[0].cpu().numpy().sum() <= 16900
    assert np.allclose(
        predictions[1].xyxy[0].cpu().numpy(), [59, 162, 192, 387], atol=1
    )
    assert np.allclose(predictions[1].class_id[0].cpu().numpy(), [0], atol=1)
    assert np.allclose(predictions[1].confidence[0].cpu().numpy(), [0.7349], atol=0.005)
    assert 16600 <= predictions[1].mask[0].cpu().numpy().sum() <= 16900


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_seg_onnx_package_with_static_batch_size_and_letterbox_torch_tensor(
    asl_yolov7_onnx_seg_static_bs_letterbox: str, asl_image_torch: torch.Tensor
) -> None:
    # given
    from inference_models.models.yolov7.yolov7_instance_segmentation_onnx import (
        YOLOv7ForInstanceSegmentationOnnx,
    )

    model = YOLOv7ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov7_onnx_seg_static_bs_letterbox,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(
        torch.stack([asl_image_torch, asl_image_torch], dim=0), conf_thresh=0.6
    )

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy[0].cpu().numpy(), [59, 162, 192, 387], atol=1
    )
    assert np.allclose(predictions[0].class_id[0].cpu().numpy(), [0], atol=1)
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), [0.7349], atol=0.005)
    assert 16600 <= predictions[0].mask[0].cpu().numpy().sum() <= 16900
    assert np.allclose(
        predictions[1].xyxy[0].cpu().numpy(), [59, 162, 192, 387], atol=1
    )
    assert np.allclose(predictions[1].class_id[0].cpu().numpy(), [0], atol=1)
    assert np.allclose(predictions[1].confidence[0].cpu().numpy(), [0.7349], atol=0.005)
    assert 16600 <= predictions[1].mask[0].cpu().numpy().sum() <= 16900
