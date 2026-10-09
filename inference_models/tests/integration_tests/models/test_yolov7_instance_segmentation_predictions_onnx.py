import numpy as np
import pytest
import supervision as sv
import torch

from inference_models.models.common.rle_utils import coco_rle_masks_to_torch_mask
from inference_models.models.common.roboflow.model_packages import StaticCrop


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


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.parametrize(
    "factor,expected_grid,expected_area",
    [(1.0, (416, 416), 16757), (0.5, (288, 288), 16797), (0.0, (160, 160), 16693)],
)
@pytest.mark.parametrize(
    "crop_start", [None, 0, 25], ids=["uncropped", "origin-crop", "offset-crop"]
)
def test_mask_resolution_and_crop_round_trip(
    asl_yolov7_onnx_seg_static_bs_letterbox: str,
    asl_image_numpy: np.ndarray,
    monkeypatch: pytest.MonkeyPatch,
    factor: float,
    crop_start: int | None,
    expected_grid: tuple,
    expected_area: int,
) -> None:
    # given
    from inference_models.models.yolov7.yolov7_instance_segmentation_onnx import (
        YOLOv7ForInstanceSegmentationOnnx,
    )

    model = YOLOv7ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolov7_onnx_seg_static_bs_letterbox,
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
        scene, confidence=0.6, mask_format="dense", masks_resolution_factor=factor
    )[0]
    rle = model(
        scene, confidence=0.6, mask_format="rle", masks_resolution_factor=factor
    )[0]
    decoded = coco_rle_masks_to_torch_mask(
        instances_masks=rle.mask, device=torch.device("cpu")
    )
    detections = dense.to_supervision()

    # then
    offset = np.array([offset_x, offset_y, offset_x, offset_y])
    expected_boxes = np.array([[59, 162, 192, 387]]) + offset
    np.testing.assert_allclose(detections.xyxy, expected_boxes, atol=2, rtol=0)
    np.testing.assert_array_equal(detections.class_id, [0])
    np.testing.assert_allclose(detections.confidence, [0.7313], atol=0.01, rtol=0)
    assert dense.image_size == scene.shape[:2]
    assert dense.mask_frame_size == scene.shape[:2]
    assert rle.image_size == scene.shape[:2]
    assert rle.mask_frame_size == scene.shape[:2]
    canvas_multiplier = 1 if crop_start is None else 2
    assert dense.mask_size == tuple(size * canvas_multiplier for size in expected_grid)
    assert rle.mask_size == dense.mask_size
    assert detections.mask.shape == (1, *scene.shape[:2])
    # Areas were checked against overlays of this fixture at all three factors.
    assert detections.mask.sum() == pytest.approx(expected_area, rel=0.02)
    expected_mask_boxes = np.array([[59, 168, 187, 379]]) + offset
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
