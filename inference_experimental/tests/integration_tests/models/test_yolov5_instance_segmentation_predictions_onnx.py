import numpy as np
import pytest
import torch


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_seg_onnx_package_with_static_batch_size_and_letterbox_numpy(
    asl_yolov5_onnx_seg_static_bs_letterbox: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_exp.models.yolov5.yolov5_instance_segmentation_onnx import (
        YOLOv5ForInstanceSegmentationOnnx,
    )

    model = YOLOv5ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov5_onnx_seg_static_bs_letterbox,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(asl_image_numpy)

    # then
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].xyxy[0].cpu().numpy(), [61, 174, 188, 370], atol=1
    )
    assert np.allclose(predictions[0].class_id[0].cpu().numpy(), [21], atol=1)
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), [0.9928], atol=0.001)
    assert 16000 <= predictions[0].mask[0].cpu().numpy().sum() <= 16100


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_seg_onnx_package_with_static_batch_size_and_letterbox_numpy_list(
    asl_yolov5_onnx_seg_static_bs_letterbox: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_exp.models.yolov5.yolov5_instance_segmentation_onnx import (
        YOLOv5ForInstanceSegmentationOnnx,
    )

    model = YOLOv5ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov5_onnx_seg_static_bs_letterbox,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([asl_image_numpy, asl_image_numpy])

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy[0].cpu().numpy(), [61, 174, 188, 370], atol=1
    )
    assert np.allclose(predictions[0].class_id[0].cpu().numpy(), [21], atol=1)
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), [0.9928], atol=0.001)
    assert 16000 <= predictions[0].mask[0].cpu().numpy().sum() <= 16100
    assert np.allclose(
        predictions[1].xyxy[0].cpu().numpy(), [61, 174, 188, 370], atol=1
    )
    assert np.allclose(predictions[1].class_id[0].cpu().numpy(), [21], atol=1)
    assert np.allclose(predictions[1].confidence[0].cpu().numpy(), [0.9928], atol=0.001)
    assert 16000 <= predictions[1].mask[0].cpu().numpy().sum() <= 16100


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_seg_onnx_package_with_static_batch_size_and_letterbox_torch(
    asl_yolov5_onnx_seg_static_bs_letterbox: str, asl_image_torch: torch.Tensor
) -> None:
    # given
    from inference_exp.models.yolov5.yolov5_instance_segmentation_onnx import (
        YOLOv5ForInstanceSegmentationOnnx,
    )

    model = YOLOv5ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov5_onnx_seg_static_bs_letterbox,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(asl_image_torch)

    # then
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].xyxy[0].cpu().numpy(), [61, 174, 188, 370], atol=1
    )
    assert np.allclose(predictions[0].class_id[0].cpu().numpy(), [21], atol=1)
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), [0.9928], atol=0.001)
    assert 16000 <= predictions[0].mask[0].cpu().numpy().sum() <= 16100


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_seg_onnx_package_with_static_batch_size_and_letterbox_torch_list(
    asl_yolov5_onnx_seg_static_bs_letterbox: str, asl_image_torch: torch.Tensor
) -> None:
    # given
    from inference_exp.models.yolov5.yolov5_instance_segmentation_onnx import (
        YOLOv5ForInstanceSegmentationOnnx,
    )

    model = YOLOv5ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov5_onnx_seg_static_bs_letterbox,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([asl_image_torch, asl_image_torch])

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy[0].cpu().numpy(), [61, 174, 188, 370], atol=1
    )
    assert np.allclose(predictions[0].class_id[0].cpu().numpy(), [21], atol=1)
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), [0.9928], atol=0.001)
    assert 16000 <= predictions[0].mask[0].cpu().numpy().sum() <= 16100
    assert np.allclose(
        predictions[1].xyxy[0].cpu().numpy(), [61, 174, 188, 370], atol=1
    )
    assert np.allclose(predictions[1].class_id[0].cpu().numpy(), [21], atol=1)
    assert np.allclose(predictions[1].confidence[0].cpu().numpy(), [0.9928], atol=0.001)
    assert 16000 <= predictions[1].mask[0].cpu().numpy().sum() <= 16100


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_seg_onnx_package_with_static_batch_size_and_letterbox_torch_tensor(
    asl_yolov5_onnx_seg_static_bs_letterbox: str, asl_image_torch: torch.Tensor
) -> None:
    # given
    from inference_exp.models.yolov5.yolov5_instance_segmentation_onnx import (
        YOLOv5ForInstanceSegmentationOnnx,
    )

    model = YOLOv5ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov5_onnx_seg_static_bs_letterbox,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(torch.stack([asl_image_torch, asl_image_torch], dim=0))

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy[0].cpu().numpy(), [61, 174, 188, 370], atol=1
    )
    assert np.allclose(predictions[0].class_id[0].cpu().numpy(), [21], atol=1)
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), [0.9928], atol=0.001)
    assert 16000 <= predictions[0].mask[0].cpu().numpy().sum() <= 16100
    assert np.allclose(
        predictions[1].xyxy[0].cpu().numpy(), [61, 174, 188, 370], atol=1
    )
    assert np.allclose(predictions[1].class_id[0].cpu().numpy(), [21], atol=1)
    assert np.allclose(predictions[1].confidence[0].cpu().numpy(), [0.9928], atol=0.001)
    assert 16000 <= predictions[1].mask[0].cpu().numpy().sum() <= 16100
