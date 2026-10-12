import numpy as np
import pytest
import supervision as sv
import torch

from inference_models.configuration import DEFAULT_DEVICE
from inference_models.models.common.rle_utils import coco_rle_masks_to_torch_mask
from inference_models.models.common.roboflow.model_packages import StaticCrop
from inference_models.models.yolov8.yolov8_instance_segmentation_torch_script import (
    YOLOv8ForInstanceSegmentationTorchScript,
)


@pytest.mark.slow
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_stretch_numpy(
    asl_yolov8n_torchscript_seg_static_bs_stretch: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    model = YOLOv8ForInstanceSegmentationTorchScript.from_pretrained(
        model_name_or_path=asl_yolov8n_torchscript_seg_static_bs_stretch,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(
        asl_image_numpy,
        confidence=0.25,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
    )

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
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_stretch_numpy_rle_variant(
    asl_yolov8n_torchscript_seg_static_bs_stretch: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    model = YOLOv8ForInstanceSegmentationTorchScript.from_pretrained(
        model_name_or_path=asl_yolov8n_torchscript_seg_static_bs_stretch,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(
        asl_image_numpy,
        confidence=0.25,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
        mask_format="rle",
    )
    predictions_ref = model(
        asl_image_numpy,
        confidence=0.25,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
    )
    decoded_mask = coco_rle_masks_to_torch_mask(
        instances_masks=predictions[0].mask, device=torch.device("cpu")
    )

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
    assert np.allclose(
        decoded_mask.cpu().numpy(), predictions_ref[0].mask.cpu().numpy()
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_stretch_batch_numpy(
    asl_yolov8n_torchscript_seg_static_bs_stretch: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    model = YOLOv8ForInstanceSegmentationTorchScript.from_pretrained(
        model_name_or_path=asl_yolov8n_torchscript_seg_static_bs_stretch,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(
        [asl_image_numpy, asl_image_numpy],
        confidence=0.25,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
    )

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
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_stretch_batch_numpy_rle_variant(
    asl_yolov8n_torchscript_seg_static_bs_stretch: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    model = YOLOv8ForInstanceSegmentationTorchScript.from_pretrained(
        model_name_or_path=asl_yolov8n_torchscript_seg_static_bs_stretch,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(
        [asl_image_numpy, asl_image_numpy],
        confidence=0.25,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
        mask_format="rle",
    )
    predictions_ref = model(
        [asl_image_numpy, asl_image_numpy],
        confidence=0.25,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
    )
    decoded_mask_1 = coco_rle_masks_to_torch_mask(
        instances_masks=predictions[0].mask, device=torch.device("cpu")
    )
    decoded_mask_2 = coco_rle_masks_to_torch_mask(
        instances_masks=predictions[1].mask, device=torch.device("cpu")
    )

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
    assert np.allclose(
        decoded_mask_1.cpu().numpy(), predictions_ref[0].mask.cpu().numpy()
    )
    assert np.allclose(
        decoded_mask_2.cpu().numpy(), predictions_ref[1].mask.cpu().numpy()
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_stretch_torch(
    asl_yolov8n_torchscript_seg_static_bs_stretch: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    model = YOLOv8ForInstanceSegmentationTorchScript.from_pretrained(
        model_name_or_path=asl_yolov8n_torchscript_seg_static_bs_stretch,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(
        asl_image_torch,
        confidence=0.25,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
    )

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
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_stretch_batch_torch(
    asl_yolov8n_torchscript_seg_static_bs_stretch: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    model = YOLOv8ForInstanceSegmentationTorchScript.from_pretrained(
        model_name_or_path=asl_yolov8n_torchscript_seg_static_bs_stretch,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(
        torch.stack([asl_image_torch, asl_image_torch], dim=0),
        confidence=0.25,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
    )

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
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_stretch_list_torch(
    asl_yolov8n_torchscript_seg_static_bs_stretch: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    model = YOLOv8ForInstanceSegmentationTorchScript.from_pretrained(
        model_name_or_path=asl_yolov8n_torchscript_seg_static_bs_stretch,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(
        [asl_image_torch, asl_image_torch],
        confidence=0.25,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
    )

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
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_stretch_fused_nms_numpy(
    asl_yolov8n_torchscript_seg_static_bs_stretch_fused_nms: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    model = YOLOv8ForInstanceSegmentationTorchScript.from_pretrained(
        model_name_or_path=asl_yolov8n_torchscript_seg_static_bs_stretch_fused_nms,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(
        asl_image_numpy,
        confidence=0.25,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
    )

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
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_stretch_fused_nms_batch_numpy(
    asl_yolov8n_torchscript_seg_static_bs_stretch_fused_nms: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    model = YOLOv8ForInstanceSegmentationTorchScript.from_pretrained(
        model_name_or_path=asl_yolov8n_torchscript_seg_static_bs_stretch_fused_nms,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(
        [asl_image_numpy, asl_image_numpy],
        confidence=0.25,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
    )

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
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_stretch_fused_nms_torch(
    asl_yolov8n_torchscript_seg_static_bs_stretch_fused_nms: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    model = YOLOv8ForInstanceSegmentationTorchScript.from_pretrained(
        model_name_or_path=asl_yolov8n_torchscript_seg_static_bs_stretch_fused_nms,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(
        asl_image_torch,
        confidence=0.25,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
    )

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
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_stretch_fused_nms_batch_torch(
    asl_yolov8n_torchscript_seg_static_bs_stretch_fused_nms: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    model = YOLOv8ForInstanceSegmentationTorchScript.from_pretrained(
        model_name_or_path=asl_yolov8n_torchscript_seg_static_bs_stretch_fused_nms,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(
        torch.stack([asl_image_torch, asl_image_torch], dim=0),
        confidence=0.25,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
    )

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
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_stretch_fused_nms_list_torch(
    asl_yolov8n_torchscript_seg_static_bs_stretch_fused_nms: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    model = YOLOv8ForInstanceSegmentationTorchScript.from_pretrained(
        model_name_or_path=asl_yolov8n_torchscript_seg_static_bs_stretch_fused_nms,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(
        [asl_image_torch, asl_image_torch],
        confidence=0.25,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
    )

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
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_static_crop_stretch_numpy(
    asl_yolov8n_torchscript_seg_static_bs_static_crop_stretch: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    model = YOLOv8ForInstanceSegmentationTorchScript.from_pretrained(
        model_name_or_path=asl_yolov8n_torchscript_seg_static_bs_static_crop_stretch,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(
        asl_image_numpy,
        confidence=0.25,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
    )

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
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_static_crop_stretch_batch_numpy(
    asl_yolov8n_torchscript_seg_static_bs_static_crop_stretch: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    model = YOLOv8ForInstanceSegmentationTorchScript.from_pretrained(
        model_name_or_path=asl_yolov8n_torchscript_seg_static_bs_static_crop_stretch,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(
        [asl_image_numpy, asl_image_numpy],
        confidence=0.25,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
    )

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
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_center_crop_stretch_torch(
    asl_yolov8n_torchscript_seg_static_bs_static_crop_stretch: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    model = YOLOv8ForInstanceSegmentationTorchScript.from_pretrained(
        model_name_or_path=asl_yolov8n_torchscript_seg_static_bs_static_crop_stretch,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(
        asl_image_torch,
        confidence=0.25,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
    )

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
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_center_crop_stretch_batch_torch(
    asl_yolov8n_torchscript_seg_static_bs_static_crop_stretch: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    model = YOLOv8ForInstanceSegmentationTorchScript.from_pretrained(
        model_name_or_path=asl_yolov8n_torchscript_seg_static_bs_static_crop_stretch,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(
        torch.stack([asl_image_torch, asl_image_torch], dim=0),
        confidence=0.25,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
    )

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
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_center_crop_stretch_list_torch(
    asl_yolov8n_torchscript_seg_static_bs_static_crop_stretch: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    model = YOLOv8ForInstanceSegmentationTorchScript.from_pretrained(
        model_name_or_path=asl_yolov8n_torchscript_seg_static_bs_static_crop_stretch,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(
        [asl_image_torch, asl_image_torch],
        confidence=0.25,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
    )

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
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_static_crop_stretch_fused_nms_numpy(
    asl_yolov8n_torchscript_seg_static_bs_static_crop_stretch_fused_nms: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    model = YOLOv8ForInstanceSegmentationTorchScript.from_pretrained(
        model_name_or_path=asl_yolov8n_torchscript_seg_static_bs_static_crop_stretch_fused_nms,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(
        asl_image_numpy,
        confidence=0.25,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
    )

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
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_static_crop_stretch_fused_nms_batch_numpy(
    asl_yolov8n_torchscript_seg_static_bs_static_crop_stretch_fused_nms: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    model = YOLOv8ForInstanceSegmentationTorchScript.from_pretrained(
        model_name_or_path=asl_yolov8n_torchscript_seg_static_bs_static_crop_stretch_fused_nms,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(
        [asl_image_numpy, asl_image_numpy],
        confidence=0.25,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
    )

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
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_center_crop_stretch_fused_nms_torch(
    asl_yolov8n_torchscript_seg_static_bs_static_crop_stretch_fused_nms: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    model = YOLOv8ForInstanceSegmentationTorchScript.from_pretrained(
        model_name_or_path=asl_yolov8n_torchscript_seg_static_bs_static_crop_stretch_fused_nms,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(
        asl_image_torch,
        confidence=0.25,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
    )

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
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_center_crop_stretch_fused_nms_batch_torch(
    asl_yolov8n_torchscript_seg_static_bs_static_crop_stretch_fused_nms: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    model = YOLOv8ForInstanceSegmentationTorchScript.from_pretrained(
        model_name_or_path=asl_yolov8n_torchscript_seg_static_bs_static_crop_stretch_fused_nms,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(
        torch.stack([asl_image_torch, asl_image_torch], dim=0),
        confidence=0.25,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
    )

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
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_center_crop_stretch_fused_nms_list_torch(
    asl_yolov8n_torchscript_seg_static_bs_static_crop_stretch_fused_nms: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    model = YOLOv8ForInstanceSegmentationTorchScript.from_pretrained(
        model_name_or_path=asl_yolov8n_torchscript_seg_static_bs_static_crop_stretch_fused_nms,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(
        [asl_image_torch, asl_image_torch],
        confidence=0.25,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
    )

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
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_static_crop_stretch_numpy(
    asl_yolov8n_torchscript_seg_static_bs_static_crop_stretch: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    model = YOLOv8ForInstanceSegmentationTorchScript.from_pretrained(
        model_name_or_path=asl_yolov8n_torchscript_seg_static_bs_static_crop_stretch,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(
        asl_image_numpy,
        confidence=0.25,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
    )

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
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_static_crop_stretch_batch_numpy(
    asl_yolov8n_torchscript_seg_static_bs_static_crop_stretch: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    model = YOLOv8ForInstanceSegmentationTorchScript.from_pretrained(
        model_name_or_path=asl_yolov8n_torchscript_seg_static_bs_static_crop_stretch,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(
        [asl_image_numpy, asl_image_numpy],
        confidence=0.25,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
    )

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
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_center_crop_stretch_torch(
    asl_yolov8n_torchscript_seg_static_bs_static_crop_stretch: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    model = YOLOv8ForInstanceSegmentationTorchScript.from_pretrained(
        model_name_or_path=asl_yolov8n_torchscript_seg_static_bs_static_crop_stretch,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(
        asl_image_torch,
        confidence=0.25,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
    )

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
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_center_crop_stretch_batch_torch(
    asl_yolov8n_torchscript_seg_static_bs_static_crop_stretch: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    model = YOLOv8ForInstanceSegmentationTorchScript.from_pretrained(
        model_name_or_path=asl_yolov8n_torchscript_seg_static_bs_static_crop_stretch,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(
        torch.stack([asl_image_torch, asl_image_torch], dim=0),
        confidence=0.25,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
    )

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
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_center_crop_stretch_list_torch(
    asl_yolov8n_torchscript_seg_static_bs_static_crop_stretch: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    model = YOLOv8ForInstanceSegmentationTorchScript.from_pretrained(
        model_name_or_path=asl_yolov8n_torchscript_seg_static_bs_static_crop_stretch,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(
        [asl_image_torch, asl_image_torch],
        confidence=0.25,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
    )

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
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_static_crop_stretch_fused_nms_numpy(
    asl_yolov8n_torchscript_seg_static_bs_static_crop_stretch_fused_nms: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    model = YOLOv8ForInstanceSegmentationTorchScript.from_pretrained(
        model_name_or_path=asl_yolov8n_torchscript_seg_static_bs_static_crop_stretch_fused_nms,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(
        asl_image_numpy,
        confidence=0.25,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
    )

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
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_static_crop_stretch_fused_nms_batch_numpy(
    asl_yolov8n_torchscript_seg_static_bs_static_crop_stretch_fused_nms: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    model = YOLOv8ForInstanceSegmentationTorchScript.from_pretrained(
        model_name_or_path=asl_yolov8n_torchscript_seg_static_bs_static_crop_stretch_fused_nms,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(
        [asl_image_numpy, asl_image_numpy],
        confidence=0.25,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
    )

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
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_center_crop_stretch_fused_nms_torch(
    asl_yolov8n_torchscript_seg_static_bs_static_crop_stretch_fused_nms: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    model = YOLOv8ForInstanceSegmentationTorchScript.from_pretrained(
        model_name_or_path=asl_yolov8n_torchscript_seg_static_bs_static_crop_stretch_fused_nms,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(
        asl_image_torch,
        confidence=0.25,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
    )

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
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_center_crop_stretch_fused_nms_batch_torch(
    asl_yolov8n_torchscript_seg_static_bs_static_crop_stretch_fused_nms: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    model = YOLOv8ForInstanceSegmentationTorchScript.from_pretrained(
        model_name_or_path=asl_yolov8n_torchscript_seg_static_bs_static_crop_stretch_fused_nms,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(
        torch.stack([asl_image_torch, asl_image_torch], dim=0),
        confidence=0.25,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
    )

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
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_center_crop_stretch_fused_nms_list_torch(
    asl_yolov8n_torchscript_seg_static_bs_static_crop_stretch_fused_nms: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    model = YOLOv8ForInstanceSegmentationTorchScript.from_pretrained(
        model_name_or_path=asl_yolov8n_torchscript_seg_static_bs_static_crop_stretch_fused_nms,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(
        [asl_image_torch, asl_image_torch],
        confidence=0.25,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
    )

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
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_static_crop_center_crop_numpy(
    asl_yolov8n_torchscript_seg_static_bs_static_crop_center_crop: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    model = YOLOv8ForInstanceSegmentationTorchScript.from_pretrained(
        model_name_or_path=asl_yolov8n_torchscript_seg_static_bs_static_crop_center_crop,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(
        asl_image_numpy,
        confidence=0.25,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
    )

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
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_static_crop_center_crop_batch_numpy(
    asl_yolov8n_torchscript_seg_static_bs_static_crop_center_crop: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    model = YOLOv8ForInstanceSegmentationTorchScript.from_pretrained(
        model_name_or_path=asl_yolov8n_torchscript_seg_static_bs_static_crop_center_crop,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(
        [asl_image_numpy, asl_image_numpy],
        confidence=0.25,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
    )

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
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_static_crop_center_crop_torch(
    asl_yolov8n_torchscript_seg_static_bs_static_crop_center_crop: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    model = YOLOv8ForInstanceSegmentationTorchScript.from_pretrained(
        model_name_or_path=asl_yolov8n_torchscript_seg_static_bs_static_crop_center_crop,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(
        asl_image_torch,
        confidence=0.25,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
    )

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
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_static_crop_center_crop_batch_torch(
    asl_yolov8n_torchscript_seg_static_bs_static_crop_center_crop: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    model = YOLOv8ForInstanceSegmentationTorchScript.from_pretrained(
        model_name_or_path=asl_yolov8n_torchscript_seg_static_bs_static_crop_center_crop,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(
        torch.stack([asl_image_torch, asl_image_torch], dim=0),
        confidence=0.25,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
    )

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
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_static_crop_center_crop_list_torch(
    asl_yolov8n_torchscript_seg_static_bs_static_crop_center_crop: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    model = YOLOv8ForInstanceSegmentationTorchScript.from_pretrained(
        model_name_or_path=asl_yolov8n_torchscript_seg_static_bs_static_crop_center_crop,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(
        [asl_image_torch, asl_image_torch],
        confidence=0.25,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
    )

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
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_center_crop_numpy(
    asl_yolov8n_torchscript_seg_static_bs_center_crop: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    model = YOLOv8ForInstanceSegmentationTorchScript.from_pretrained(
        model_name_or_path=asl_yolov8n_torchscript_seg_static_bs_center_crop,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(
        asl_image_numpy,
        confidence=0.25,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
    )

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
        16000 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 16400
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_center_crop_batch_numpy(
    asl_yolov8n_torchscript_seg_static_bs_center_crop: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    model = YOLOv8ForInstanceSegmentationTorchScript.from_pretrained(
        model_name_or_path=asl_yolov8n_torchscript_seg_static_bs_center_crop,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(
        [asl_image_numpy, asl_image_numpy],
        confidence=0.25,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
    )

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
        16000 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 16400
    )
    assert (
        16000 <= predictions[1].to_supervision().mask[0, 174:371, 63:187].sum() <= 16400
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_center_crop_torch(
    asl_yolov8n_torchscript_seg_static_bs_center_crop: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    model = YOLOv8ForInstanceSegmentationTorchScript.from_pretrained(
        model_name_or_path=asl_yolov8n_torchscript_seg_static_bs_center_crop,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(
        asl_image_torch,
        confidence=0.25,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
    )

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
        16000 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 16400
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_center_crop_batch_torch(
    asl_yolov8n_torchscript_seg_static_bs_center_crop: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    model = YOLOv8ForInstanceSegmentationTorchScript.from_pretrained(
        model_name_or_path=asl_yolov8n_torchscript_seg_static_bs_center_crop,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(
        torch.stack([asl_image_torch, asl_image_torch], dim=0),
        confidence=0.25,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
    )

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
        16000 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 16400
    )
    assert (
        16000 <= predictions[1].to_supervision().mask[0, 174:371, 63:187].sum() <= 16400
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_center_crop_list_torch(
    asl_yolov8n_torchscript_seg_static_bs_center_crop: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    model = YOLOv8ForInstanceSegmentationTorchScript.from_pretrained(
        model_name_or_path=asl_yolov8n_torchscript_seg_static_bs_center_crop,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(
        [asl_image_torch, asl_image_torch],
        confidence=0.25,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
    )

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
        16000 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 16400
    )
    assert (
        16000 <= predictions[1].to_supervision().mask[0, 174:371, 63:187].sum() <= 16400
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_static_crop_letterbox_crop_numpy(
    asl_yolov8n_torchscript_seg_static_bs_static_crop_letterbox: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    model = YOLOv8ForInstanceSegmentationTorchScript.from_pretrained(
        model_name_or_path=asl_yolov8n_torchscript_seg_static_bs_static_crop_letterbox,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(
        asl_image_numpy,
        confidence=0.25,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
    )

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
        13800 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 14100
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_static_crop_letterbox_batch_numpy(
    asl_yolov8n_torchscript_seg_static_bs_static_crop_letterbox: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    model = YOLOv8ForInstanceSegmentationTorchScript.from_pretrained(
        model_name_or_path=asl_yolov8n_torchscript_seg_static_bs_static_crop_letterbox,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(
        [asl_image_numpy, asl_image_numpy],
        confidence=0.25,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
    )

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
        13800 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 14100
    )
    assert (
        13800 <= predictions[1].to_supervision().mask[0, 174:371, 63:187].sum() <= 14100
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_static_crop_letterbox_crop_torch(
    asl_yolov8n_torchscript_seg_static_bs_static_crop_letterbox: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    model = YOLOv8ForInstanceSegmentationTorchScript.from_pretrained(
        model_name_or_path=asl_yolov8n_torchscript_seg_static_bs_static_crop_letterbox,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(
        asl_image_torch,
        confidence=0.25,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
    )

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
        13800 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 14100
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_static_crop_letterbox_batch_torch(
    asl_yolov8n_torchscript_seg_static_bs_static_crop_letterbox: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    model = YOLOv8ForInstanceSegmentationTorchScript.from_pretrained(
        model_name_or_path=asl_yolov8n_torchscript_seg_static_bs_static_crop_letterbox,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(
        torch.stack([asl_image_torch, asl_image_torch], dim=0),
        confidence=0.25,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
    )

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
        13800 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 14100
    )
    assert (
        13800 <= predictions[1].to_supervision().mask[0, 174:371, 63:187].sum() <= 14100
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_static_crop_letterbox_list_torch(
    asl_yolov8n_torchscript_seg_static_bs_static_crop_letterbox: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    model = YOLOv8ForInstanceSegmentationTorchScript.from_pretrained(
        model_name_or_path=asl_yolov8n_torchscript_seg_static_bs_static_crop_letterbox,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(
        [asl_image_torch, asl_image_torch],
        confidence=0.25,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
    )

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
        13800 <= predictions[0].to_supervision().mask[0, 174:371, 63:187].sum() <= 14100
    )
    assert (
        13800 <= predictions[1].to_supervision().mask[0, 174:371, 63:187].sum() <= 14100
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_torchscript_per_class_confidence_blocks_specific_class(
    asl_yolov8n_torchscript_seg_static_bs_stretch: str,
    asl_image_numpy: np.ndarray,
) -> None:
    """ASL dataset baseline returns class 20 detection. Setting a 1.01
    per-class threshold on class 20 leaves no detections."""
    from inference_models.weights_providers.entities import RecommendedParameters

    model = YOLOv8ForInstanceSegmentationTorchScript.from_pretrained(
        model_name_or_path=asl_yolov8n_torchscript_seg_static_bs_stretch,
        device=DEFAULT_DEVICE,
    )
    class_names = list(model.class_names)
    model.recommended_parameters = RecommendedParameters(
        confidence=0.25,
        per_class_confidence={class_names[20]: 1.01},
    )
    predictions = model(asl_image_numpy, confidence="best")
    assert predictions[0].class_id.numel() == 0


@pytest.mark.slow
@pytest.mark.torch_models
@pytest.mark.parametrize(
    "factor,expected_grid,expected_area",
    [(1.0, (416, 416), 16069), (0.5, (288, 288), 16040), (0.0, (160, 160), 16063)],
)
@pytest.mark.parametrize(
    "crop_start", [None, 0, 25], ids=["uncropped", "origin-crop", "offset-crop"]
)
def test_mask_resolution_and_crop_round_trip(
    asl_yolov8n_torchscript_seg_static_bs_stretch: str,
    asl_image_numpy: np.ndarray,
    monkeypatch: pytest.MonkeyPatch,
    factor: float,
    crop_start: int | None,
    expected_grid: tuple,
    expected_area: int,
) -> None:
    # given
    from inference_models.models.yolov8.yolov8_instance_segmentation_torch_script import (
        YOLOv8ForInstanceSegmentationTorchScript,
    )

    model = YOLOv8ForInstanceSegmentationTorchScript.from_pretrained(
        model_name_or_path=asl_yolov8n_torchscript_seg_static_bs_stretch,
    )
    image = np.pad(
        asl_image_numpy,
        ((0, asl_image_numpy.shape[0] % 2), (0, asl_image_numpy.shape[1] % 2), (0, 0)),
    )
    height, width = image.shape[:2]
    scene = image
    offset_x = offset_y = 0
    crop_config = None
    if crop_start is not None:
        offset_y = 0 if crop_start == 0 else height // 2
        offset_x = 0 if crop_start == 0 else width // 2
        scene = np.zeros((2 * height, 2 * width, 3), dtype=image.dtype)
        scene[offset_y : offset_y + height, offset_x : offset_x + width] = image
        crop_config = StaticCrop(
            enabled=True,
            x_min=crop_start,
            y_min=crop_start,
            x_max=crop_start + 50,
            y_max=crop_start + 50,
        )
    monkeypatch.setattr(
        model._inference_config.image_pre_processing, "static_crop", crop_config
    )

    # when
    dense = model(
        scene, confidence=0.25, mask_format="dense", masks_resolution_factor=factor
    )[0]
    rle = model(
        scene, confidence=0.25, mask_format="rle", masks_resolution_factor=factor
    )[0]
    decoded = coco_rle_masks_to_torch_mask(
        instances_masks=rle.mask, device=torch.device("cpu")
    )
    detections = dense.to_supervision()

    # then
    offset = np.array([offset_x, offset_y, offset_x, offset_y])
    expected_boxes = np.array([[63, 174, 187, 371]]) + offset
    np.testing.assert_allclose(detections.xyxy, expected_boxes, atol=2, rtol=0)
    np.testing.assert_array_equal(detections.class_id, [20])
    np.testing.assert_allclose(detections.confidence, [0.9846], atol=0.01, rtol=0)
    assert dense.image_size == scene.shape[:2]
    assert dense.mask_frame_size == scene.shape[:2]
    assert rle.image_size == scene.shape[:2]
    assert rle.mask_frame_size == scene.shape[:2]
    canvas_multiplier = 1 if crop_start is None else 2
    assert dense.mask_size == tuple(size * canvas_multiplier for size in expected_grid)
    assert rle.mask_size == dense.mask_size

    grid_scale = np.array(
        [dense.mask_size[1] / scene.shape[1], dense.mask_size[0] / scene.shape[0]] * 2
    )
    np.testing.assert_allclose(
        dense.xyxy.cpu().numpy(),
        expected_boxes * grid_scale,
        atol=2 * max(grid_scale),
        rtol=0,
    )
    if dense.mask_size != scene.shape[:2]:
        assert dense.xyxy.is_floating_point()
    assert detections.mask.shape == (1, *scene.shape[:2])
    # Areas were checked against overlays of this fixture at all three factors.
    assert detections.mask.sum() == pytest.approx(expected_area, rel=0.02)
    expected_mask_boxes = np.array([[63, 174, 186, 369]]) + offset
    pixel_tolerance = (
        np.ceil(
            max(
                scene.shape[0] / dense.mask_size[0], scene.shape[1] / dense.mask_size[1]
            )
        )
        + 2
    )
    np.testing.assert_allclose(
        sv.mask_to_xyxy(detections.mask),
        expected_mask_boxes,
        atol=pixel_tolerance,
        rtol=0,
    )
    crop_masks = detections.mask[
        :, offset_y : offset_y + height, offset_x : offset_x + width
    ]
    assert crop_masks.sum() == detections.mask.sum()
    torch.testing.assert_close(dense.xyxy.cpu(), rle.xyxy.cpu())
    torch.testing.assert_close(dense.class_id.cpu(), rle.class_id.cpu())
    torch.testing.assert_close(dense.confidence.cpu(), rle.confidence.cpu())
    np.testing.assert_array_equal(decoded.numpy(), dense.mask.cpu().numpy())
    np.testing.assert_array_equal(detections.mask, rle.to_supervision().mask)
    sv.MaskAnnotator().annotate(scene.copy(), detections)
