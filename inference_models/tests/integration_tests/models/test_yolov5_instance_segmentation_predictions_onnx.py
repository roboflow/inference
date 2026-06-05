import numpy as np
import pytest
import torch

from inference_models.models.common.rle_utils import coco_rle_masks_to_torch_mask


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_seg_onnx_package_with_static_batch_size_and_letterbox_numpy(
    asl_yolov5_onnx_seg_static_bs_letterbox: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yolov5.yolov5_instance_segmentation_onnx import (
        YOLOv5ForInstanceSegmentationOnnx,
    )

    model = YOLOv5ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov5_onnx_seg_static_bs_letterbox,
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
        predictions[0].xyxy[0].cpu().numpy(), [61, 174, 188, 370], atol=1
    )
    assert np.allclose(predictions[0].class_id[0].cpu().numpy(), [21], atol=1)
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), [0.9928], atol=0.001)
    assert 16000 <= predictions[0].mask[0].cpu().numpy().sum() <= 16200


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_seg_onnx_package_with_static_batch_size_and_letterbox_numpy_rle_variant(
    asl_yolov5_onnx_seg_static_bs_letterbox: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yolov5.yolov5_instance_segmentation_onnx import (
        YOLOv5ForInstanceSegmentationOnnx,
    )

    model = YOLOv5ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov5_onnx_seg_static_bs_letterbox,
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
        predictions[0].xyxy[0].cpu().numpy(), [61, 174, 188, 370], atol=1
    )
    assert np.allclose(predictions[0].class_id[0].cpu().numpy(), [21], atol=1)
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), [0.9928], atol=0.001)
    assert 16000 <= decoded_mask[0].cpu().numpy().sum() <= 16200
    assert np.allclose(
        decoded_mask.cpu().numpy(), predictions_ref[0].mask.cpu().numpy()
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_seg_onnx_package_with_static_batch_size_and_letterbox_numpy_list(
    asl_yolov5_onnx_seg_static_bs_letterbox: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yolov5.yolov5_instance_segmentation_onnx import (
        YOLOv5ForInstanceSegmentationOnnx,
    )

    model = YOLOv5ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov5_onnx_seg_static_bs_letterbox,
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
        predictions[0].xyxy[0].cpu().numpy(), [61, 174, 188, 370], atol=1
    )
    assert np.allclose(predictions[0].class_id[0].cpu().numpy(), [21], atol=1)
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), [0.9928], atol=0.001)
    assert 16000 <= predictions[0].mask[0].cpu().numpy().sum() <= 16200
    assert np.allclose(
        predictions[1].xyxy[0].cpu().numpy(), [61, 174, 188, 370], atol=1
    )
    assert np.allclose(predictions[1].class_id[0].cpu().numpy(), [21], atol=1)
    assert np.allclose(predictions[1].confidence[0].cpu().numpy(), [0.9928], atol=0.001)
    assert 16000 <= predictions[1].mask[0].cpu().numpy().sum() <= 16200


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_seg_onnx_package_with_static_batch_size_and_letterbox_numpy_list_rle_variant(
    asl_yolov5_onnx_seg_static_bs_letterbox: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yolov5.yolov5_instance_segmentation_onnx import (
        YOLOv5ForInstanceSegmentationOnnx,
    )

    model = YOLOv5ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov5_onnx_seg_static_bs_letterbox,
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
        predictions[0].xyxy[0].cpu().numpy(), [61, 174, 188, 370], atol=1
    )
    assert np.allclose(predictions[0].class_id[0].cpu().numpy(), [21], atol=1)
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), [0.9928], atol=0.001)
    assert 16000 <= decoded_mask_1[0].cpu().numpy().sum() <= 16200
    assert np.allclose(
        predictions[1].xyxy[0].cpu().numpy(), [61, 174, 188, 370], atol=1
    )
    assert np.allclose(predictions[1].class_id[0].cpu().numpy(), [21], atol=1)
    assert np.allclose(predictions[1].confidence[0].cpu().numpy(), [0.9928], atol=0.001)
    assert 16000 <= decoded_mask_2[0].cpu().numpy().sum() <= 16200
    assert np.allclose(
        decoded_mask_1.cpu().numpy(), predictions_ref[0].mask.cpu().numpy()
    )
    assert np.allclose(
        decoded_mask_2.cpu().numpy(), predictions_ref[1].mask.cpu().numpy()
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_seg_onnx_package_with_static_batch_size_and_letterbox_torch(
    asl_yolov5_onnx_seg_static_bs_letterbox: str, asl_image_torch: torch.Tensor
) -> None:
    # given
    from inference_models.models.yolov5.yolov5_instance_segmentation_onnx import (
        YOLOv5ForInstanceSegmentationOnnx,
    )

    model = YOLOv5ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov5_onnx_seg_static_bs_letterbox,
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
        predictions[0].xyxy[0].cpu().numpy(), [61, 174, 188, 370], atol=1
    )
    assert np.allclose(predictions[0].class_id[0].cpu().numpy(), [21], atol=1)
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), [0.9928], atol=0.001)
    assert 16000 <= predictions[0].mask[0].cpu().numpy().sum() <= 16200


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_seg_onnx_package_with_static_batch_size_and_letterbox_torch_list(
    asl_yolov5_onnx_seg_static_bs_letterbox: str, asl_image_torch: torch.Tensor
) -> None:
    # given
    from inference_models.models.yolov5.yolov5_instance_segmentation_onnx import (
        YOLOv5ForInstanceSegmentationOnnx,
    )

    model = YOLOv5ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov5_onnx_seg_static_bs_letterbox,
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
        predictions[0].xyxy[0].cpu().numpy(), [61, 174, 188, 370], atol=1
    )
    assert np.allclose(predictions[0].class_id[0].cpu().numpy(), [21], atol=1)
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), [0.9928], atol=0.001)
    assert 16000 <= predictions[0].mask[0].cpu().numpy().sum() <= 16200
    assert np.allclose(
        predictions[1].xyxy[0].cpu().numpy(), [61, 174, 188, 370], atol=1
    )
    assert np.allclose(predictions[1].class_id[0].cpu().numpy(), [21], atol=1)
    assert np.allclose(predictions[1].confidence[0].cpu().numpy(), [0.9928], atol=0.001)
    assert 16000 <= predictions[1].mask[0].cpu().numpy().sum() <= 16200


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_seg_onnx_package_with_static_batch_size_and_letterbox_torch_tensor(
    asl_yolov5_onnx_seg_static_bs_letterbox: str, asl_image_torch: torch.Tensor
) -> None:
    # given
    from inference_models.models.yolov5.yolov5_instance_segmentation_onnx import (
        YOLOv5ForInstanceSegmentationOnnx,
    )

    model = YOLOv5ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov5_onnx_seg_static_bs_letterbox,
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
        predictions[0].xyxy[0].cpu().numpy(), [61, 174, 188, 370], atol=1
    )
    assert np.allclose(predictions[0].class_id[0].cpu().numpy(), [21], atol=1)
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), [0.9928], atol=0.001)
    assert 16000 <= predictions[0].mask[0].cpu().numpy().sum() <= 16200
    assert np.allclose(
        predictions[1].xyxy[0].cpu().numpy(), [61, 174, 188, 370], atol=1
    )
    assert np.allclose(predictions[1].class_id[0].cpu().numpy(), [21], atol=1)
    assert np.allclose(predictions[1].confidence[0].cpu().numpy(), [0.9928], atol=0.001)
    assert 16000 <= predictions[1].mask[0].cpu().numpy().sum() <= 16200


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_per_class_confidence_blocks_specific_class(
    asl_yolov5_onnx_seg_static_bs_letterbox: str,
    asl_image_numpy: np.ndarray,
) -> None:
    """YOLOv5 IS uses slice-space class_ids (0=obj, k>=1=class_{k-1}).
    Baseline (see `test_seg_onnx_package_with_static_batch_size_and_letterbox_numpy`
    above) returns 1 detection of slice_id 21 (real class 20) at conf 0.9928.
    Setting a 0.99 per-class threshold on class 20 drops the detection."""
    from inference_models.models.yolov5.yolov5_instance_segmentation_onnx import (
        YOLOv5ForInstanceSegmentationOnnx,
    )
    from inference_models.weights_providers.entities import RecommendedParameters

    model = YOLOv5ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov5_onnx_seg_static_bs_letterbox,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    class_names = list(model.class_names)
    model.recommended_parameters = RecommendedParameters(
        confidence=0.25,
        per_class_confidence={class_names[20]: 0.99},
    )
    predictions = model(asl_image_numpy, confidence="best")
    assert predictions[0].class_id.numel() == 0
