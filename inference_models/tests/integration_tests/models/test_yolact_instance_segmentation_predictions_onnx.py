import numpy as np
import pytest
import supervision as sv
import torch

from inference_models.models.common.rle_utils import coco_rle_masks_to_torch_mask
from inference_models.models.common.roboflow.model_packages import StaticCrop


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_seg_onnx_package_with_static_batch_size_and_letterbox_numpy(
    asl_yolact_onnx_seg_static_bs_letterbox: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yolact.yolact_instance_segmentation_onnx import (
        YOLOACTForInstanceSegmentationOnnx,
    )

    model = YOLOACTForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolact_onnx_seg_static_bs_letterbox,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
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
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].xyxy[0].cpu().numpy(), [66, 167, 188, 373], atol=1
    )
    assert np.allclose(predictions[0].class_id[0].cpu().numpy(), [18], atol=1)
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), [0.5068], atol=0.001)
    assert 14800 <= predictions[0].mask[0].cpu().numpy().sum() <= 16100


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_seg_onnx_package_with_static_batch_size_and_letterbox_numpy_rle_variant(
    asl_yolact_onnx_seg_static_bs_letterbox: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yolact.yolact_instance_segmentation_onnx import (
        YOLOACTForInstanceSegmentationOnnx,
    )

    model = YOLOACTForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolact_onnx_seg_static_bs_letterbox,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
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
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].xyxy[0].cpu().numpy(), [66, 167, 188, 373], atol=1
    )
    assert np.allclose(predictions[0].class_id[0].cpu().numpy(), [18], atol=1)
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), [0.5068], atol=0.001)
    assert 14800 <= decoded_mask.cpu().numpy().sum() <= 16100
    assert np.allclose(
        decoded_mask.cpu().numpy(), predictions_ref[0].mask.cpu().numpy()
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_seg_onnx_package_with_static_batch_size_and_letterbox_numpy_list(
    asl_yolact_onnx_seg_static_bs_letterbox: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yolact.yolact_instance_segmentation_onnx import (
        YOLOACTForInstanceSegmentationOnnx,
    )

    model = YOLOACTForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolact_onnx_seg_static_bs_letterbox,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
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
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy[0].cpu().numpy(), [66, 167, 188, 373], atol=1
    )
    assert np.allclose(predictions[0].class_id[0].cpu().numpy(), [18], atol=1)
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), [0.5068], atol=0.001)
    assert 14800 <= predictions[0].mask[0].cpu().numpy().sum() <= 16100
    assert np.allclose(
        predictions[1].xyxy[0].cpu().numpy(), [66, 167, 188, 373], atol=1
    )
    assert np.allclose(predictions[1].class_id[0].cpu().numpy(), [18], atol=1)
    assert np.allclose(predictions[1].confidence[0].cpu().numpy(), [0.5068], atol=0.001)
    assert 14800 <= predictions[1].mask[0].cpu().numpy().sum() <= 16100


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_seg_onnx_package_with_static_batch_size_and_letterbox_numpy_list_rle_variant(
    asl_yolact_onnx_seg_static_bs_letterbox: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yolact.yolact_instance_segmentation_onnx import (
        YOLOACTForInstanceSegmentationOnnx,
    )

    model = YOLOACTForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolact_onnx_seg_static_bs_letterbox,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
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
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy[0].cpu().numpy(), [66, 167, 188, 373], atol=1
    )
    assert np.allclose(predictions[0].class_id[0].cpu().numpy(), [18], atol=1)
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), [0.5068], atol=0.001)
    assert 14800 <= decoded_mask_1.cpu().numpy().sum() <= 16100
    assert np.allclose(
        predictions[1].xyxy[0].cpu().numpy(), [66, 167, 188, 373], atol=1
    )
    assert np.allclose(predictions[1].class_id[0].cpu().numpy(), [18], atol=1)
    assert np.allclose(predictions[1].confidence[0].cpu().numpy(), [0.5068], atol=0.001)
    assert 14800 <= decoded_mask_2.cpu().numpy().sum() <= 16100
    assert np.allclose(
        decoded_mask_1.cpu().numpy(), predictions_ref[0].mask.cpu().numpy()
    )
    assert np.allclose(
        decoded_mask_2.cpu().numpy(), predictions_ref[1].mask.cpu().numpy()
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_seg_onnx_package_with_static_batch_size_and_letterbox_torch(
    asl_yolact_onnx_seg_static_bs_letterbox: str, asl_image_torch: torch.Tensor
) -> None:
    # given
    from inference_models.models.yolact.yolact_instance_segmentation_onnx import (
        YOLOACTForInstanceSegmentationOnnx,
    )

    model = YOLOACTForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolact_onnx_seg_static_bs_letterbox,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
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
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].xyxy[0].cpu().numpy(), [66, 167, 188, 373], atol=1
    )
    assert np.allclose(predictions[0].class_id[0].cpu().numpy(), [18], atol=1)
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), [0.5068], atol=0.001)
    assert 14800 <= predictions[0].mask[0].cpu().numpy().sum() <= 16100


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_seg_onnx_package_with_static_batch_size_and_letterbox_torch_list(
    asl_yolact_onnx_seg_static_bs_letterbox: str, asl_image_torch: torch.Tensor
) -> None:
    # given
    from inference_models.models.yolact.yolact_instance_segmentation_onnx import (
        YOLOACTForInstanceSegmentationOnnx,
    )

    model = YOLOACTForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolact_onnx_seg_static_bs_letterbox,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
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
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy[0].cpu().numpy(), [66, 167, 188, 373], atol=1
    )
    assert np.allclose(predictions[0].class_id[0].cpu().numpy(), [18], atol=1)
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), [0.5068], atol=0.001)
    assert 14800 <= predictions[0].mask[0].cpu().numpy().sum() <= 16100
    assert np.allclose(
        predictions[1].xyxy[0].cpu().numpy(), [66, 167, 188, 373], atol=1
    )
    assert np.allclose(predictions[1].class_id[0].cpu().numpy(), [18], atol=1)
    assert np.allclose(predictions[1].confidence[0].cpu().numpy(), [0.5068], atol=0.001)
    assert 14800 <= predictions[1].mask[0].cpu().numpy().sum() <= 16100


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_seg_onnx_package_with_static_batch_size_and_letterbox_torch_tensor(
    asl_yolact_onnx_seg_static_bs_letterbox: str, asl_image_torch: torch.Tensor
) -> None:
    # given
    from inference_models.models.yolact.yolact_instance_segmentation_onnx import (
        YOLOACTForInstanceSegmentationOnnx,
    )

    model = YOLOACTForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolact_onnx_seg_static_bs_letterbox,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
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
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy[0].cpu().numpy(), [66, 167, 188, 373], atol=1
    )
    assert np.allclose(predictions[0].class_id[0].cpu().numpy(), [18], atol=1)
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), [0.5068], atol=0.001)
    assert 14800 <= predictions[0].mask[0].cpu().numpy().sum() <= 16100
    assert np.allclose(
        predictions[1].xyxy[0].cpu().numpy(), [66, 167, 188, 373], atol=1
    )
    assert np.allclose(predictions[1].class_id[0].cpu().numpy(), [18], atol=1)
    assert np.allclose(predictions[1].confidence[0].cpu().numpy(), [0.5068], atol=0.001)
    assert 14800 <= predictions[1].mask[0].cpu().numpy().sum() <= 16100


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_seg_onnx_package_with_static_batch_size_and_stretch_numpy(
    asl_yolact_onnx_seg_static_bs_stretch: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yolact.yolact_instance_segmentation_onnx import (
        YOLOACTForInstanceSegmentationOnnx,
    )

    model = YOLOACTForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolact_onnx_seg_static_bs_stretch,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(
        asl_image_numpy,
        confidence=0.8,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
    )

    # then
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].xyxy[0].cpu().numpy(), [63, 170, 188, 377], atol=1
    )
    assert np.allclose(predictions[0].class_id[0].cpu().numpy(), [21], atol=1)
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), [0.9783], atol=0.001)
    assert 15000 <= predictions[0].mask[0].cpu().numpy().sum() <= 16100


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_seg_onnx_package_with_static_batch_size_and_stretch_numpy_list(
    asl_yolact_onnx_seg_static_bs_stretch: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yolact.yolact_instance_segmentation_onnx import (
        YOLOACTForInstanceSegmentationOnnx,
    )

    model = YOLOACTForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolact_onnx_seg_static_bs_stretch,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(
        [asl_image_numpy, asl_image_numpy],
        confidence=0.8,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
    )

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy[0].cpu().numpy(), [63, 170, 188, 377], atol=1
    )
    assert np.allclose(predictions[0].class_id[0].cpu().numpy(), [21], atol=1)
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), [0.9783], atol=0.001)
    assert 15000 <= predictions[0].mask[0].cpu().numpy().sum() <= 16100
    assert np.allclose(
        predictions[1].xyxy[0].cpu().numpy(), [63, 170, 188, 377], atol=1
    )
    assert np.allclose(predictions[1].class_id[0].cpu().numpy(), [21], atol=1)
    assert np.allclose(predictions[1].confidence[0].cpu().numpy(), [0.9783], atol=0.001)
    assert 15000 <= predictions[1].mask[0].cpu().numpy().sum() <= 16100


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_seg_onnx_package_with_static_batch_size_and_stretch_torch(
    asl_yolact_onnx_seg_static_bs_stretch: str, asl_image_torch: torch.Tensor
) -> None:
    # given
    from inference_models.models.yolact.yolact_instance_segmentation_onnx import (
        YOLOACTForInstanceSegmentationOnnx,
    )

    model = YOLOACTForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolact_onnx_seg_static_bs_stretch,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(
        asl_image_torch,
        confidence=0.8,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
    )

    # then
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].xyxy[0].cpu().numpy(), [63, 170, 188, 377], atol=1
    )
    assert np.allclose(predictions[0].class_id[0].cpu().numpy(), [21], atol=1)
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), [0.9783], atol=0.001)
    assert 15000 <= predictions[0].mask[0].cpu().numpy().sum() <= 16100


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_seg_onnx_package_with_static_batch_size_and_stretch_torch_list(
    asl_yolact_onnx_seg_static_bs_stretch: str, asl_image_torch: torch.Tensor
) -> None:
    # given
    from inference_models.models.yolact.yolact_instance_segmentation_onnx import (
        YOLOACTForInstanceSegmentationOnnx,
    )

    model = YOLOACTForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolact_onnx_seg_static_bs_stretch,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(
        [asl_image_torch, asl_image_torch],
        confidence=0.8,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
    )

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy[0].cpu().numpy(), [63, 170, 188, 377], atol=1
    )
    assert np.allclose(predictions[0].class_id[0].cpu().numpy(), [21], atol=1)
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), [0.9783], atol=0.001)
    assert 15000 <= predictions[0].mask[0].cpu().numpy().sum() <= 16100
    assert np.allclose(
        predictions[1].xyxy[0].cpu().numpy(), [63, 170, 188, 377], atol=1
    )
    assert np.allclose(predictions[1].class_id[0].cpu().numpy(), [21], atol=1)
    assert np.allclose(predictions[1].confidence[0].cpu().numpy(), [0.9783], atol=0.001)
    assert 15000 <= predictions[1].mask[0].cpu().numpy().sum() <= 16100


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_seg_onnx_package_with_static_batch_size_and_stretch_torch_tensor(
    asl_yolact_onnx_seg_static_bs_stretch: str, asl_image_torch: torch.Tensor
) -> None:
    # given
    from inference_models.models.yolact.yolact_instance_segmentation_onnx import (
        YOLOACTForInstanceSegmentationOnnx,
    )

    model = YOLOACTForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolact_onnx_seg_static_bs_stretch,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(
        torch.stack([asl_image_torch, asl_image_torch], dim=0),
        confidence=0.8,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
    )

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy[0].cpu().numpy(), [63, 170, 188, 377], atol=1
    )
    assert np.allclose(predictions[0].class_id[0].cpu().numpy(), [21], atol=1)
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), [0.9783], atol=0.001)
    assert 15000 <= predictions[0].mask[0].cpu().numpy().sum() <= 16100
    assert np.allclose(
        predictions[1].xyxy[0].cpu().numpy(), [63, 170, 188, 377], atol=1
    )
    assert np.allclose(predictions[1].class_id[0].cpu().numpy(), [21], atol=1)
    assert np.allclose(predictions[1].confidence[0].cpu().numpy(), [0.9783], atol=0.001)
    assert 15000 <= predictions[1].mask[0].cpu().numpy().sum() <= 16100


####
@pytest.mark.slow
@pytest.mark.onnx_extras
def test_seg_onnx_package_with_static_batch_size_static_crop_and_stretch_numpy(
    asl_yolact_onnx_seg_static_bs_static_crop_stretch: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yolact.yolact_instance_segmentation_onnx import (
        YOLOACTForInstanceSegmentationOnnx,
    )

    model = YOLOACTForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolact_onnx_seg_static_bs_static_crop_stretch,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
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
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].xyxy[0].cpu().numpy(), [109, 169, 208, 311], atol=1
    )
    assert np.allclose(predictions[0].class_id[0].cpu().numpy(), [24], atol=1)
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), [0.2552], atol=0.003)
    assert 7000 <= predictions[0].mask[0].cpu().numpy().sum() <= 7500


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_seg_onnx_package_with_static_batch_size_static_crop_and_stretch_numpy_list(
    asl_yolact_onnx_seg_static_bs_static_crop_stretch: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yolact.yolact_instance_segmentation_onnx import (
        YOLOACTForInstanceSegmentationOnnx,
    )

    model = YOLOACTForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolact_onnx_seg_static_bs_static_crop_stretch,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
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
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy[0].cpu().numpy(), [109, 169, 208, 311], atol=1
    )
    assert np.allclose(predictions[0].class_id[0].cpu().numpy(), [24], atol=1)
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), [0.2552], atol=0.003)
    assert 7000 <= predictions[0].mask[0].cpu().numpy().sum() <= 7500
    assert np.allclose(
        predictions[1].xyxy[0].cpu().numpy(), [109, 169, 208, 311], atol=1
    )
    assert np.allclose(predictions[1].class_id[0].cpu().numpy(), [24], atol=1)
    assert np.allclose(predictions[1].confidence[0].cpu().numpy(), [0.2552], atol=0.003)
    assert 7000 <= predictions[1].mask[0].cpu().numpy().sum() <= 7500


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_seg_onnx_package_with_static_batch_size_static_crop_and_stretch_torch(
    asl_yolact_onnx_seg_static_bs_static_crop_stretch: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.yolact.yolact_instance_segmentation_onnx import (
        YOLOACTForInstanceSegmentationOnnx,
    )

    model = YOLOACTForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolact_onnx_seg_static_bs_static_crop_stretch,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
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
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].xyxy[0].cpu().numpy(), [109, 169, 208, 311], atol=1
    )
    assert np.allclose(predictions[0].class_id[0].cpu().numpy(), [24], atol=1)
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), [0.2552], atol=0.006)
    assert 7000 <= predictions[0].mask[0].cpu().numpy().sum() <= 7500


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_seg_onnx_package_with_static_batch_size_static_crop_and_stretch_torch_list(
    asl_yolact_onnx_seg_static_bs_static_crop_stretch: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.yolact.yolact_instance_segmentation_onnx import (
        YOLOACTForInstanceSegmentationOnnx,
    )

    model = YOLOACTForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolact_onnx_seg_static_bs_static_crop_stretch,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
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
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy[0].cpu().numpy(), [109, 169, 208, 311], atol=1
    )
    assert np.allclose(predictions[0].class_id[0].cpu().numpy(), [24], atol=1)
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), [0.2552], atol=0.006)
    assert 7000 <= predictions[0].mask[0].cpu().numpy().sum() <= 7500
    assert np.allclose(
        predictions[1].xyxy[0].cpu().numpy(), [109, 169, 208, 311], atol=1
    )
    assert np.allclose(predictions[1].class_id[0].cpu().numpy(), [24], atol=1)
    assert np.allclose(predictions[1].confidence[0].cpu().numpy(), [0.2552], atol=0.006)
    assert 7000 <= predictions[1].mask[0].cpu().numpy().sum() <= 7500


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_seg_onnx_package_with_static_batch_size_static_crop_and_stretch_torch_tensor(
    asl_yolact_onnx_seg_static_bs_static_crop_stretch: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.yolact.yolact_instance_segmentation_onnx import (
        YOLOACTForInstanceSegmentationOnnx,
    )

    model = YOLOACTForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolact_onnx_seg_static_bs_static_crop_stretch,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
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
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy[0].cpu().numpy(), [109, 169, 208, 311], atol=1
    )
    assert np.allclose(predictions[0].class_id[0].cpu().numpy(), [24], atol=1)
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), [0.2552], atol=0.006)
    assert 7000 <= predictions[0].mask[0].cpu().numpy().sum() <= 7500
    assert np.allclose(
        predictions[1].xyxy[0].cpu().numpy(), [109, 169, 208, 311], atol=1
    )
    assert np.allclose(predictions[1].class_id[0].cpu().numpy(), [24], atol=1)
    assert np.allclose(predictions[1].confidence[0].cpu().numpy(), [0.2552], atol=0.006)
    assert 7000 <= predictions[1].mask[0].cpu().numpy().sum() <= 7500


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_per_class_confidence_blocks_specific_class(
    asl_yolact_onnx_seg_static_bs_letterbox: str,
    asl_image_numpy: np.ndarray,
) -> None:
    """Baseline (see `test_seg_onnx_package_with_static_batch_size_and_letterbox_numpy`
    above) top detection is class 18 at conf 0.5068. Setting a 0.99 per-class
    threshold on class 18 drops the detection."""
    from inference_models.models.yolact.yolact_instance_segmentation_onnx import (
        YOLOACTForInstanceSegmentationOnnx,
    )
    from inference_models.weights_providers.entities import RecommendedParameters

    model = YOLOACTForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolact_onnx_seg_static_bs_letterbox,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    class_names = list(model.class_names)
    model.recommended_parameters = RecommendedParameters(
        confidence=0.25,
        per_class_confidence={class_names[18]: 0.99},
    )
    predictions = model(asl_image_numpy, confidence="best")
    assert 18 not in predictions[0].class_id.cpu().tolist()


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.parametrize(
    "factor,expected_grid,expected_area",
    [(1.0, (416, 416), 14979), (0.5, (277, 277), 14934), (0.0, (138, 138), 14895)],
)
@pytest.mark.parametrize(
    "crop_start", [None, 0, 25], ids=["uncropped", "origin-crop", "offset-crop"]
)
def test_mask_resolution_and_crop_round_trip(
    asl_yolact_onnx_seg_static_bs_letterbox: str,
    asl_image_numpy: np.ndarray,
    monkeypatch: pytest.MonkeyPatch,
    factor: float,
    crop_start: int | None,
    expected_grid: tuple,
    expected_area: int,
) -> None:
    # given
    from inference_models.models.yolact.yolact_instance_segmentation_onnx import (
        YOLOACTForInstanceSegmentationOnnx,
    )

    model = YOLOACTForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolact_onnx_seg_static_bs_letterbox,
        onnx_execution_providers=["CPUExecutionProvider"],
        device=torch.device("cpu"),
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
    expected_boxes = np.array([[66, 167, 188, 373]]) + offset
    np.testing.assert_allclose(detections.xyxy, expected_boxes, atol=2, rtol=0)
    np.testing.assert_array_equal(detections.class_id, [18])
    np.testing.assert_allclose(detections.confidence, [0.5063], atol=0.01, rtol=0)
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
    expected_mask_boxes = np.array([[66, 166, 181, 366]]) + offset
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
