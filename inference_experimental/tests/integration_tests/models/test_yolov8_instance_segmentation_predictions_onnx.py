import numpy as np
import pytest
import torch


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_dynamic_batch_size_and_stretch_numpy(
    asl_yolov8n_onnx_seg_dynamic_bs_stretch: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_dynamic_bs_stretch,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(asl_image_numpy)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.98464]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor([[63, 174, 187, 371]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        16000 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 16200
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_dynamic_batch_size_and_stretch_numpy_custom_size(
    asl_yolov8n_onnx_seg_dynamic_bs_stretch: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_dynamic_bs_stretch,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(asl_image_numpy, image_size=(200, 200))

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.8392724]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([17], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor([[60, 170, 185, 380]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        17000 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 17200
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_dynamic_batch_size_and_stretch_torch_custom_size(
    asl_yolov8n_onnx_seg_dynamic_bs_stretch: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_dynamic_bs_stretch,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(asl_image_torch, image_size=(200, 200))

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.8392724]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([17], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor([[60, 170, 185, 380]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        17000 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 17200
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_dynamic_batch_size_and_stretch_batch_numpy(
    asl_yolov8n_onnx_seg_dynamic_bs_stretch: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_dynamic_bs_stretch,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([asl_image_numpy, asl_image_numpy])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.98464]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.98464]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor([[63, 174, 187, 371]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        16000 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 16200
    )
    assert (
        16000 <= predictions[1].to_supervision().mask[0, 174:371, 63:187].sum() <= 16200
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_dynamic_batch_size_and_stretch_torch(
    asl_yolov8n_onnx_seg_dynamic_bs_stretch: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_dynamic_bs_stretch,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(asl_image_torch)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.98464]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor([[63, 174, 187, 371]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        16000 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 16200
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_dynamic_batch_size_and_stretch_batch_torch(
    asl_yolov8n_onnx_seg_dynamic_bs_stretch: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_dynamic_bs_stretch,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(torch.stack([asl_image_torch, asl_image_torch], dim=0))

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.98464]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.98464]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor([[63, 174, 187, 371]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        16000 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 16200
    )
    assert (
        16000 <= predictions[1].to_supervision().mask[0, 174:371, 63:187].sum() <= 16200
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_dynamic_batch_size_and_stretch_list_torch(
    asl_yolov8n_onnx_seg_dynamic_bs_stretch: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_dynamic_bs_stretch,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([asl_image_torch, asl_image_torch])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.98464]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.98464]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor([[63, 174, 187, 371]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        16000 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 16200
    )
    assert (
        16000 <= predictions[1].to_supervision().mask[0, 174:371, 63:187].sum() <= 16200
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_dynamic_batch_size_and_stretch_fused_nms_numpy(
    asl_yolov8n_onnx_seg_dynamic_bs_stretch_fused_nms: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_dynamic_bs_stretch_fused_nms,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(asl_image_numpy)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.98464]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor([[63, 174, 187, 371]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        16000 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 16200
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_dynamic_batch_size_and_stretch_fused_nms_batch_numpy(
    asl_yolov8n_onnx_seg_dynamic_bs_stretch_fused_nms: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_dynamic_bs_stretch_fused_nms,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([asl_image_numpy, asl_image_numpy])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.98464]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.98464]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor([[63, 174, 187, 371]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        16000 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 16200
    )
    assert (
        16000 <= predictions[1].to_supervision().mask[0, 174:371, 63:187].sum() <= 16200
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_dynamic_batch_size_and_stretch_fused_nms_torch(
    asl_yolov8n_onnx_seg_dynamic_bs_stretch_fused_nms: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_dynamic_bs_stretch_fused_nms,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(asl_image_torch)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.98464]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor([[63, 174, 187, 371]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        16000 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 16200
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_dynamic_batch_size_and_stretch_fused_nms_batch_torch(
    asl_yolov8n_onnx_seg_dynamic_bs_stretch_fused_nms: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_dynamic_bs_stretch_fused_nms,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(torch.stack([asl_image_torch, asl_image_torch], dim=0))

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.98464]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.98464]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor([[63, 174, 187, 371]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        16000 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 16200
    )
    assert (
        16000 <= predictions[1].to_supervision().mask[0, 174:371, 63:187].sum() <= 16200
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_dynamic_batch_size_and_stretch_fused_nms_list_torch(
    asl_yolov8n_onnx_seg_dynamic_bs_stretch_fused_nms: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_dynamic_bs_stretch_fused_nms,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([asl_image_torch, asl_image_torch])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.98464]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.98464]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor([[63, 174, 187, 371]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        16000 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 16200
    )
    assert (
        16000 <= predictions[1].to_supervision().mask[0, 174:371, 63:187].sum() <= 16200
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_batch_size_and_stretch_numpy(
    asl_yolov8n_onnx_seg_static_bs_stretch: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_static_bs_stretch,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(asl_image_numpy)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.98464]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor([[63, 174, 187, 371]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        16000 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 16200
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_batch_size_and_stretch_batch_numpy(
    asl_yolov8n_onnx_seg_static_bs_stretch: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_static_bs_stretch,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([asl_image_numpy, asl_image_numpy])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.98464]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.98464]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor([[63, 174, 187, 371]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        16000 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 16200
    )
    assert (
        16000 <= predictions[1].to_supervision().mask[0, 174:371, 63:187].sum() <= 16200
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_batch_size_and_stretch_torch(
    asl_yolov8n_onnx_seg_static_bs_stretch: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_static_bs_stretch,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(asl_image_torch)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.98464]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor([[63, 174, 187, 371]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        16000 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 16200
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_batch_size_and_stretch_batch_torch(
    asl_yolov8n_onnx_seg_static_bs_stretch: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_static_bs_stretch,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(torch.stack([asl_image_torch, asl_image_torch], dim=0))

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.98464]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.98464]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor([[63, 174, 187, 371]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        16000 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 16200
    )
    assert (
        16000 <= predictions[1].to_supervision().mask[0, 174:371, 63:187].sum() <= 16200
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_batch_size_and_stretch_list_torch(
    asl_yolov8n_onnx_seg_static_bs_stretch: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_static_bs_stretch,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([asl_image_torch, asl_image_torch])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.98464]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.98464]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor([[63, 174, 187, 371]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        16000 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 16200
    )
    assert (
        16000 <= predictions[1].to_supervision().mask[0, 174:371, 63:187].sum() <= 16200
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_dynamic_batch_size_and_static_crop_stretch_numpy(
    asl_yolov8n_onnx_seg_dynamic_bs_static_crop_stretch: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_dynamic_bs_static_crop_stretch,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(asl_image_numpy)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9735]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor([[63, 171, 186, 343]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        13800 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 14000
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_dynamic_batch_size_and_static_crop_stretch_numpy_custom_size(
    asl_yolov8n_onnx_seg_dynamic_bs_static_crop_stretch: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_dynamic_bs_static_crop_stretch,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(asl_image_numpy, image_size=(300, 300))

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.4323]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([21], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor([[61, 169, 186, 339]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        13800 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 14000
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_dynamic_batch_size_and_static_crop_stretch_torch_custom_size(
    asl_yolov8n_onnx_seg_dynamic_bs_static_crop_stretch: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_dynamic_bs_static_crop_stretch,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(asl_image_torch, image_size=(300, 300))

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.4323]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([21], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor([[61, 169, 186, 339]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        13800 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 14000
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_dynamic_batch_size_and_static_crop_stretch_batch_numpy(
    asl_yolov8n_onnx_seg_dynamic_bs_static_crop_stretch: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_dynamic_bs_static_crop_stretch,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([asl_image_numpy, asl_image_numpy])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9735]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.9735]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor([[63, 171, 186, 343]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        13800 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 14000
    )
    assert (
        13800 <= predictions[1].to_supervision().mask[0, 174:371, 63:187].sum() <= 14000
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_dynamic_batch_size_and_static_crop_stretch_torch(
    asl_yolov8n_onnx_seg_dynamic_bs_static_crop_stretch: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_dynamic_bs_static_crop_stretch,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(asl_image_torch)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9735]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor([[63, 171, 186, 343]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        13800 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 14000
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_dynamic_batch_size_and_static_crop_stretch_batch_torch(
    asl_yolov8n_onnx_seg_dynamic_bs_static_crop_stretch: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_dynamic_bs_static_crop_stretch,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(torch.stack([asl_image_torch, asl_image_torch], dim=0))

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9735]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.9735]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor([[63, 171, 186, 343]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        13800 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 14000
    )
    assert (
        13800 <= predictions[1].to_supervision().mask[0, 174:371, 63:187].sum() <= 14000
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_dynamic_batch_size_and_static_crop_stretch_list_torch(
    asl_yolov8n_onnx_seg_dynamic_bs_static_crop_stretch: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_dynamic_bs_static_crop_stretch,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([asl_image_torch, asl_image_torch])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9735]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.9735]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor([[63, 171, 186, 343]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        13800 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 14000
    )
    assert (
        13800 <= predictions[1].to_supervision().mask[0, 174:371, 63:187].sum() <= 14000
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_dynamic_batch_size_and_static_crop_stretch_fused_nms_numpy(
    asl_yolov8n_onnx_seg_dynamic_bs_static_crop_stretch_fused_nms: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_dynamic_bs_static_crop_stretch_fused_nms,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(asl_image_numpy)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9735]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor([[63, 171, 186, 343]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        13800 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 14000
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_dynamic_batch_size_and_static_crop_stretch_fused_nms_numpy_custom_size(
    asl_yolov8n_onnx_seg_dynamic_bs_static_crop_stretch_fused_nms: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_dynamic_bs_static_crop_stretch_fused_nms,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(asl_image_numpy, image_size=(300, 300))

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.4323]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([21], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor([[61, 169, 186, 339]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        13800 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 14000
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_dynamic_batch_size_and_static_crop_stretch_fused_nms_torch_custom_size(
    asl_yolov8n_onnx_seg_dynamic_bs_static_crop_stretch_fused_nms: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_dynamic_bs_static_crop_stretch_fused_nms,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(asl_image_torch, image_size=(300, 300))

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.4323]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([21], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor([[61, 169, 186, 339]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        13800 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 14000
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_dynamic_batch_size_and_static_crop_stretch_fused_nms_batch_numpy(
    asl_yolov8n_onnx_seg_dynamic_bs_static_crop_stretch_fused_nms: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_dynamic_bs_static_crop_stretch_fused_nms,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([asl_image_numpy, asl_image_numpy])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9735]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.9735]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor([[63, 171, 186, 343]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        13800 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 14000
    )
    assert (
        13800 <= predictions[1].to_supervision().mask[0, 174:371, 63:187].sum() <= 14000
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_dynamic_batch_size_and_static_crop_stretch_fused_nms_torch(
    asl_yolov8n_onnx_seg_dynamic_bs_static_crop_stretch_fused_nms: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_dynamic_bs_static_crop_stretch_fused_nms,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(asl_image_torch)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9735]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor([[63, 171, 186, 343]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        13800 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 14000
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_dynamic_batch_size_and_static_crop_stretch_fused_nms_batch_torch(
    asl_yolov8n_onnx_seg_dynamic_bs_static_crop_stretch_fused_nms: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_dynamic_bs_static_crop_stretch_fused_nms,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(torch.stack([asl_image_torch, asl_image_torch], dim=0))

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9735]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.9735]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor([[63, 171, 186, 343]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        13800 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 14000
    )
    assert (
        13800 <= predictions[1].to_supervision().mask[0, 174:371, 63:187].sum() <= 14000
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_dynamic_batch_size_and_static_crop_stretch_fused_nms_list_torch(
    asl_yolov8n_onnx_seg_dynamic_bs_static_crop_stretch_fused_nms: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_dynamic_bs_static_crop_stretch_fused_nms,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([asl_image_torch, asl_image_torch])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9735]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.9735]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor([[63, 171, 186, 343]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        13800 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 14000
    )
    assert (
        13800 <= predictions[1].to_supervision().mask[0, 174:371, 63:187].sum() <= 14000
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_batch_size_and_static_crop_stretch_numpy(
    asl_yolov8n_onnx_seg_static_bs_static_crop_stretch: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_static_bs_static_crop_stretch,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(asl_image_numpy)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9735]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor([[63, 171, 186, 343]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        13800 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 14000
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_batch_size_and_static_crop_stretch_batch_numpy(
    asl_yolov8n_onnx_seg_static_bs_static_crop_stretch: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_static_bs_static_crop_stretch,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([asl_image_numpy, asl_image_numpy])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9735]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.9735]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor([[63, 171, 186, 343]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        13800 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 14000
    )
    assert (
        13800 <= predictions[1].to_supervision().mask[0, 174:371, 63:187].sum() <= 14000
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_batch_size_and_static_crop_stretch_torch(
    asl_yolov8n_onnx_seg_static_bs_static_crop_stretch: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_static_bs_static_crop_stretch,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(asl_image_torch)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9735]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor([[63, 171, 186, 343]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        13800 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 14000
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_batch_size_and_static_crop_stretch_batch_torch(
    asl_yolov8n_onnx_seg_static_bs_static_crop_stretch: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_static_bs_static_crop_stretch,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(torch.stack([asl_image_torch, asl_image_torch], dim=0))

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9735]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.9735]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor([[63, 171, 186, 343]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        13800 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 14000
    )
    assert (
        13800 <= predictions[1].to_supervision().mask[0, 174:371, 63:187].sum() <= 14000
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_batch_size_and_static_crop_stretch_list_torch(
    asl_yolov8n_onnx_seg_static_bs_static_crop_stretch: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_static_bs_static_crop_stretch,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([asl_image_torch, asl_image_torch])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9735]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.9735]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor([[63, 171, 186, 343]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        13800 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 14000
    )
    assert (
        13800 <= predictions[1].to_supervision().mask[0, 174:371, 63:187].sum() <= 14000
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_dynamic_batch_size_and_static_crop_center_crop_numpy(
    asl_yolov8n_onnx_seg_dynamic_bs_static_crop_center_crop: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_dynamic_bs_static_crop_center_crop,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(asl_image_numpy)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.8159524]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor([[64, 175, 188, 341]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        13800 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 14000
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_dynamic_batch_size_and_static_crop_center_crop_numpy_custom_size(
    asl_yolov8n_onnx_seg_dynamic_bs_static_crop_center_crop: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_dynamic_bs_static_crop_center_crop,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(asl_image_numpy, image_size=(200, 200))

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.47219265]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor([[63, 173, 187, 329]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        12800 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 13000
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_dynamic_batch_size_and_static_crop_center_crop_torch_custom_size(
    asl_yolov8n_onnx_seg_dynamic_bs_static_crop_center_crop: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_dynamic_bs_static_crop_center_crop,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(asl_image_torch, image_size=(200, 200))

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.47219265]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor([[63, 173, 187, 329]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        12800 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 13000
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_dynamic_batch_size_and_tatic_crop_center_crop_batch_numpy(
    asl_yolov8n_onnx_seg_dynamic_bs_static_crop_center_crop: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_dynamic_bs_static_crop_center_crop,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([asl_image_numpy, asl_image_numpy])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.8159524]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.8159524]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor([[64, 175, 188, 341]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        13800 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 14000
    )
    assert (
        13800 <= predictions[1].to_supervision().mask[0, 174:371, 63:187].sum() <= 14000
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_dynamic_batch_size_and_static_crop_center_crop_torch(
    asl_yolov8n_onnx_seg_dynamic_bs_static_crop_center_crop: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_dynamic_bs_static_crop_center_crop,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(asl_image_torch)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.8159524]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor([[64, 175, 188, 341]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        13800 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 14000
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_dynamic_batch_size_and_static_crop_center_crop_batch_torch(
    asl_yolov8n_onnx_seg_dynamic_bs_static_crop_center_crop: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_dynamic_bs_static_crop_center_crop,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(torch.stack([asl_image_torch, asl_image_torch], dim=0))

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.8159524]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.8159524]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor([[64, 175, 188, 341]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        13800 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 14000
    )
    assert (
        13800 <= predictions[1].to_supervision().mask[0, 174:371, 63:187].sum() <= 14000
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_dynamic_batch_size_and_static_crop_center_crop_list_torch(
    asl_yolov8n_onnx_seg_dynamic_bs_static_crop_center_crop: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_dynamic_bs_static_crop_center_crop,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([asl_image_torch, asl_image_torch])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.8159524]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.8159524]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor([[64, 175, 188, 341]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        13800 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 14000
    )
    assert (
        13800 <= predictions[1].to_supervision().mask[0, 174:371, 63:187].sum() <= 14000
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_dynamic_batch_size_and_static_crop_center_crop_fused_nms_numpy(
    asl_yolov8n_onnx_seg_dynamic_bs_static_crop_center_crop_fused_nms: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_dynamic_bs_static_crop_center_crop_fused_nms,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(asl_image_numpy)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.8159524]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor([[64, 175, 188, 341]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        13800 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 14000
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_dynamic_batch_size_and_static_crop_center_crop_fused_nms_numpy_custom_size(
    asl_yolov8n_onnx_seg_dynamic_bs_static_crop_center_crop_fused_nms: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_dynamic_bs_static_crop_center_crop_fused_nms,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(asl_image_numpy, image_size=(200, 200))

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.47219265]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor([[63, 173, 187, 329]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        12800 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 13000
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_dynamic_batch_size_and_static_crop_center_crop_fused_nms_torch_custom_size(
    asl_yolov8n_onnx_seg_dynamic_bs_static_crop_center_crop_fused_nms: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_dynamic_bs_static_crop_center_crop_fused_nms,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(asl_image_torch, image_size=(200, 200))

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.47219265]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor([[63, 173, 187, 329]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        12800 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 13000
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_dynamic_batch_size_and_tatic_crop_center_crop_fused_nms_batch_numpy(
    asl_yolov8n_onnx_seg_dynamic_bs_static_crop_center_crop_fused_nms: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_dynamic_bs_static_crop_center_crop_fused_nms,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([asl_image_numpy, asl_image_numpy])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.8159524]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.8159524]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor([[64, 175, 188, 341]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        13800 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 14000
    )
    assert (
        13800 <= predictions[1].to_supervision().mask[0, 174:371, 63:187].sum() <= 14000
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_dynamic_batch_size_and_static_crop_center_crop_fused_nms_torch(
    asl_yolov8n_onnx_seg_dynamic_bs_static_crop_center_crop_fused_nms: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_dynamic_bs_static_crop_center_crop_fused_nms,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(asl_image_torch)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.8159524]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor([[64, 175, 188, 341]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        13800 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 14000
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_dynamic_batch_size_and_static_crop_center_crop_fused_nms_batch_torch(
    asl_yolov8n_onnx_seg_dynamic_bs_static_crop_center_crop_fused_nms: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_dynamic_bs_static_crop_center_crop_fused_nms,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(torch.stack([asl_image_torch, asl_image_torch], dim=0))

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.8159524]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.8159524]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor([[64, 175, 188, 341]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        13800 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 14000
    )
    assert (
        13800 <= predictions[1].to_supervision().mask[0, 174:371, 63:187].sum() <= 14000
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_dynamic_batch_size_and_static_crop_center_crop_fused_nms_list_torch(
    asl_yolov8n_onnx_seg_dynamic_bs_static_crop_center_crop_fused_nms: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_dynamic_bs_static_crop_center_crop_fused_nms,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([asl_image_torch, asl_image_torch])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.8159524]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.8159524]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor([[64, 175, 188, 341]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        13800 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 14000
    )
    assert (
        13800 <= predictions[1].to_supervision().mask[0, 174:371, 63:187].sum() <= 14000
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_batch_size_and_static_crop_center_crop_numpy(
    asl_yolov8n_onnx_seg_static_bs_static_crop_center_crop: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_static_bs_static_crop_center_crop,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(asl_image_numpy)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.8159524]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor([[64, 175, 188, 341]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        13800 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 14000
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_batch_size_and_static_crop_center_crop_batch_numpy(
    asl_yolov8n_onnx_seg_static_bs_static_crop_center_crop: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_static_bs_static_crop_center_crop,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([asl_image_numpy, asl_image_numpy])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.8159524]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.8159524]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor([[64, 175, 188, 341]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        13800 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 14000
    )
    assert (
        13800 <= predictions[1].to_supervision().mask[0, 174:371, 63:187].sum() <= 14000
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_batch_size_and_static_crop_center_crop_torch(
    asl_yolov8n_onnx_seg_static_bs_static_crop_center_crop: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_static_bs_static_crop_center_crop,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(asl_image_torch)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.8159524]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor([[64, 175, 188, 341]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        13800 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 14000
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_batch_size_and_static_crop_center_crop_batch_torch(
    asl_yolov8n_onnx_seg_static_bs_static_crop_center_crop: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_static_bs_static_crop_center_crop,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(torch.stack([asl_image_torch, asl_image_torch], dim=0))

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.8159524]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.8159524]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor([[64, 175, 188, 341]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        13800 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 14000
    )
    assert (
        13800 <= predictions[1].to_supervision().mask[0, 174:371, 63:187].sum() <= 14000
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_batch_size_and_static_crop_center_crop_list_torch(
    asl_yolov8n_onnx_seg_static_bs_static_crop_center_crop: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_static_bs_static_crop_center_crop,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([asl_image_torch, asl_image_torch])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.8159524]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.8159524]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor([[64, 175, 188, 341]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        13800 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 14000
    )
    assert (
        13800 <= predictions[1].to_supervision().mask[0, 174:371, 63:187].sum() <= 14000
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_dynamic_batch_size_and_center_crop_numpy(
    asl_yolov8n_onnx_seg_dynamic_bs_center_crop: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_dynamic_bs_center_crop,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(asl_image_numpy)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9711]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor([[61, 172, 187, 367]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        16200 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 16400
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_dynamic_batch_size_and_center_crop_batch_numpy(
    asl_yolov8n_onnx_seg_dynamic_bs_center_crop: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_dynamic_bs_center_crop,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([asl_image_numpy, asl_image_numpy])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9711]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.9711]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor([[61, 172, 187, 367]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        16200 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 16400
    )
    assert (
        16200 <= predictions[1].to_supervision().mask[0, 174:371, 63:187].sum() <= 16400
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_dynamic_batch_size_and_center_crop_torch(
    asl_yolov8n_onnx_seg_dynamic_bs_center_crop: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_dynamic_bs_center_crop,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(asl_image_torch)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9711]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor([[61, 172, 187, 367]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        16200 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 16400
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_dynamic_batch_size_and_center_crop_batch_torch(
    asl_yolov8n_onnx_seg_dynamic_bs_center_crop: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_dynamic_bs_center_crop,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(torch.stack([asl_image_torch, asl_image_torch], dim=0))

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9711]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.9711]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor([[61, 172, 187, 367]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        16200 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 16400
    )
    assert (
        16200 <= predictions[1].to_supervision().mask[0, 174:371, 63:187].sum() <= 16400
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_dynamic_batch_size_and_center_crop_list_torch(
    asl_yolov8n_onnx_seg_dynamic_bs_center_crop: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_dynamic_bs_center_crop,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([asl_image_torch, asl_image_torch])

    # then
    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9711]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.9711]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor([[61, 172, 187, 367]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        16200 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 16400
    )
    assert (
        16200 <= predictions[1].to_supervision().mask[0, 174:371, 63:187].sum() <= 16400
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_dynamic_batch_size_and_static_crop_letterbox_numpy(
    asl_yolov8n_onnx_seg_dynamic_bs_static_crop_letterbox: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_dynamic_bs_static_crop_letterbox,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(asl_image_numpy)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.6637, 0.5337]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20, 17], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[63, 173, 187, 341], [61, 175, 187, 342]], dtype=torch.int32
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        13900 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 14100
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_dynamic_batch_size_and_static_crop_letterbox_batch_numpy(
    asl_yolov8n_onnx_seg_dynamic_bs_static_crop_letterbox: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_dynamic_bs_static_crop_letterbox,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([asl_image_numpy, asl_image_numpy])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.6637, 0.5337]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.6637, 0.5337]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20, 17], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([20, 17], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[63, 173, 187, 341], [61, 175, 187, 342]], dtype=torch.int32
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        13900 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 14100
    )
    assert (
        13900 <= predictions[1].to_supervision().mask[0, 174:371, 63:187].sum() <= 14100
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_dynamic_batch_size_and_static_crop_letterbox_torch(
    asl_yolov8n_onnx_seg_dynamic_bs_static_crop_letterbox: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_dynamic_bs_static_crop_letterbox,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(asl_image_torch)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.6599, 0.5505]).cpu(),
        atol=0.02,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20, 17], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[63, 173, 187, 341], [61, 175, 187, 342]], dtype=torch.int32
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        13900 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 14100
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_dynamic_batch_size_and_static_crop_letterbox_batch_torch(
    asl_yolov8n_onnx_seg_dynamic_bs_static_crop_letterbox: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_dynamic_bs_static_crop_letterbox,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(torch.stack([asl_image_torch, asl_image_torch], dim=0))

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.65991, 0.55051]).cpu(),
        atol=0.02,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.65991, 0.55051]).cpu(),
        atol=0.02,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20, 17], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([20, 17], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[63, 173, 187, 341], [61, 175, 187, 342]], dtype=torch.int32
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        13900 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 14100
    )
    assert (
        13900 <= predictions[1].to_supervision().mask[0, 174:371, 63:187].sum() <= 14100
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_dynamic_batch_size_and_static_crop_letterbox_list_torch(
    asl_yolov8n_onnx_seg_dynamic_bs_static_crop_letterbox: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )

    model = YOLOv8ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov8n_onnx_seg_dynamic_bs_static_crop_letterbox,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([asl_image_torch, asl_image_torch])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.65991, 0.55051]).cpu(),
        atol=0.02,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.65991, 0.55051]).cpu(),
        atol=0.02,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20, 17], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([20, 17], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[63, 173, 187, 341], [61, 175, 187, 342]], dtype=torch.int32
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert (
        13900 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 14100
    )
    assert (
        13900 <= predictions[1].to_supervision().mask[0, 174:371, 63:187].sum() <= 14100
    )
