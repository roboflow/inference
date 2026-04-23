import numpy as np
import pytest
import torch

from inference_models.models.common.rle_utils import coco_rle_masks_to_torch_mask


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
    predictions = model(
        asl_image_numpy,
        confidence=0.6,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
    )

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
def test_seg_onnx_package_with_static_batch_size_and_letterbox_numpy_rle_variant(
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
    predictions = model(
        asl_image_numpy,
        confidence=0.6,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
        mask_format="rle",
    )
    predictions_ref = model(
        asl_image_numpy,
        confidence=0.6,
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
        predictions[0].xyxy[0].cpu().numpy(), [59, 162, 192, 387], atol=1
    )
    assert np.allclose(predictions[0].class_id[0].cpu().numpy(), [0], atol=1)
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), [0.7349], atol=0.005)
    assert 16600 <= decoded_mask[0].cpu().numpy().sum() <= 16900
    assert np.allclose(
        decoded_mask.cpu().numpy(), predictions_ref[0].mask.cpu().numpy()
    )


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
    predictions = model(
        [asl_image_numpy, asl_image_numpy],
        confidence=0.6,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
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


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_seg_onnx_package_with_static_batch_size_and_letterbox_numpy_list_rle_variant(
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
    predictions = model(
        [asl_image_numpy, asl_image_numpy],
        confidence=0.6,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
        mask_format="rle",
    )
    predictions_ref = model(
        [asl_image_numpy, asl_image_numpy],
        confidence=0.6,
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
        predictions[0].xyxy[0].cpu().numpy(), [59, 162, 192, 387], atol=1
    )
    assert np.allclose(predictions[0].class_id[0].cpu().numpy(), [0], atol=1)
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), [0.7349], atol=0.005)
    assert 16600 <= decoded_mask_1[0].cpu().numpy().sum() <= 16900
    assert np.allclose(
        predictions[1].xyxy[0].cpu().numpy(), [59, 162, 192, 387], atol=1
    )
    assert np.allclose(predictions[1].class_id[0].cpu().numpy(), [0], atol=1)
    assert np.allclose(predictions[1].confidence[0].cpu().numpy(), [0.7349], atol=0.005)
    assert 16600 <= decoded_mask_2[0].cpu().numpy().sum() <= 16900
    assert np.allclose(
        decoded_mask_1.cpu().numpy(), predictions_ref[0].mask.cpu().numpy()
    )
    assert np.allclose(
        decoded_mask_2.cpu().numpy(), predictions_ref[1].mask.cpu().numpy()
    )


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
    predictions = model(
        asl_image_torch,
        confidence=0.6,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
    )

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
    predictions = model(
        [asl_image_torch, asl_image_torch],
        confidence=0.6,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
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
        torch.stack([asl_image_torch, asl_image_torch], dim=0),
        confidence=0.6,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
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


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_per_class_confidence_preserves_objectness_row(
    asl_yolov7_onnx_seg_static_bs_letterbox: str,
    asl_image_numpy: np.ndarray,
) -> None:
    """YOLOv7 IS uses slice-space class_ids where class_id 0 is the objectness
    slot (no real class). The baseline (see
    `test_seg_onnx_package_with_static_batch_size_and_letterbox_numpy` above)
    returns one obj-dominated detection at conf 0.7349. Per-class thresholds
    must not block objectness-dominated rows, so the detection survives even
    with every real class set to 0.99."""
    from inference_models.models.yolov7.yolov7_instance_segmentation_onnx import (
        YOLOv7ForInstanceSegmentationOnnx,
    )
    from inference_models.weights_providers.entities import RecommendedParameters

    model = YOLOv7ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov7_onnx_seg_static_bs_letterbox,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    model.recommended_parameters = RecommendedParameters(
        confidence=0.6,
        per_class_confidence={name: 1.01 for name in model.class_names},
    )
    predictions = model(asl_image_numpy, confidence="best")
    assert np.allclose(predictions[0].class_id[0].cpu().numpy(), [0], atol=1)
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), [0.7349], atol=0.005)
