import numpy as np
import pytest
import supervision as sv
import torch

from inference_models.models.common.rle_utils import coco_rle_masks_to_torch_mask
from inference_models.models.common.roboflow.model_packages import StaticCrop


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_numpy(
    yolov8_seg_asl_trt_package: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yolov8.yolov8_instance_segmentation_trt import (
        YOLOv8ForInstanceSegmentationTRT,
    )

    model = YOLOv8ForInstanceSegmentationTRT.from_pretrained(
        model_name_or_path=yolov8_seg_asl_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model(asl_image_numpy)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9795]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[63, 174, 187, 368]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert 16000 <= predictions[0].mask.cpu().sum().item() <= 16200


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_numpy_rle_variant(
    yolov8_seg_asl_trt_package: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yolov8.yolov8_instance_segmentation_trt import (
        YOLOv8ForInstanceSegmentationTRT,
    )

    model = YOLOv8ForInstanceSegmentationTRT.from_pretrained(
        model_name_or_path=yolov8_seg_asl_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model(asl_image_numpy, mask_format="rle")
    predictions_ref = model(asl_image_numpy)
    decoded_mask = coco_rle_masks_to_torch_mask(
        instances_masks=predictions[0].mask, device=torch.device("cpu")
    )

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9795]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[63, 174, 187, 368]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert 16000 <= decoded_mask.cpu().sum().item() <= 16200
    assert np.allclose(
        decoded_mask.cpu().numpy(), predictions_ref[0].mask.cpu().numpy()
    )


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_batch_numpy(
    yolov8_seg_asl_trt_package: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yolov8.yolov8_instance_segmentation_trt import (
        YOLOv8ForInstanceSegmentationTRT,
    )

    model = YOLOv8ForInstanceSegmentationTRT.from_pretrained(
        model_name_or_path=yolov8_seg_asl_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model([asl_image_numpy, asl_image_numpy])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9795]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[63, 174, 187, 368]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert 16050 <= predictions[0].mask.cpu().sum().item() <= 16200
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.9795]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[63, 174, 187, 368]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert 16000 <= predictions[1].mask.cpu().sum().item() <= 16200


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_batch_numpy_rle_variant(
    yolov8_seg_asl_trt_package: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yolov8.yolov8_instance_segmentation_trt import (
        YOLOv8ForInstanceSegmentationTRT,
    )

    model = YOLOv8ForInstanceSegmentationTRT.from_pretrained(
        model_name_or_path=yolov8_seg_asl_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model([asl_image_numpy, asl_image_numpy], mask_format="rle")
    predictions_ref = model([asl_image_numpy, asl_image_numpy])
    decoded_mask_1 = coco_rle_masks_to_torch_mask(
        instances_masks=predictions[0].mask, device=torch.device("cpu")
    )
    decoded_mask_2 = coco_rle_masks_to_torch_mask(
        instances_masks=predictions[1].mask, device=torch.device("cpu")
    )

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9795]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[63, 174, 187, 368]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert 16050 <= decoded_mask_1.cpu().sum().item() <= 16200
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.9795]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[63, 174, 187, 368]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert 16000 <= decoded_mask_2.cpu().sum().item() <= 16200
    assert np.allclose(
        decoded_mask_1.cpu().numpy(), predictions_ref[0].mask.cpu().numpy()
    )
    assert np.allclose(
        decoded_mask_2.cpu().numpy(), predictions_ref[1].mask.cpu().numpy()
    )


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_torch(
    yolov8_seg_asl_trt_package: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.yolov8.yolov8_instance_segmentation_trt import (
        YOLOv8ForInstanceSegmentationTRT,
    )

    model = YOLOv8ForInstanceSegmentationTRT.from_pretrained(
        model_name_or_path=yolov8_seg_asl_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model(asl_image_torch)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9795]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[63, 174, 187, 368]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert 16000 <= predictions[0].mask.cpu().sum().item() <= 16200


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_torch_multiple_predictions_in_row(
    yolov8_seg_asl_trt_package: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.yolov8.yolov8_instance_segmentation_trt import (
        YOLOv8ForInstanceSegmentationTRT,
    )

    model = YOLOv8ForInstanceSegmentationTRT.from_pretrained(
        model_name_or_path=yolov8_seg_asl_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    for _ in range(8):
        predictions = model(asl_image_torch)

        # then
        assert torch.allclose(
            predictions[0].confidence.cpu(),
            torch.tensor([0.9795]).cpu(),
            atol=0.01,
        )
        assert torch.allclose(
            predictions[0].class_id.cpu(),
            torch.tensor([20], dtype=torch.int32).cpu(),
        )
        expected_xyxy = torch.tensor(
            [[63, 174, 187, 368]],
            dtype=torch.int32,
        )
        assert torch.allclose(
            predictions[0].xyxy.cpu(),
            expected_xyxy.cpu(),
            atol=5,
        )
        assert 16000 <= predictions[0].mask.cpu().sum().item() <= 16200


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_torch_list(
    yolov8_seg_asl_trt_package: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.yolov8.yolov8_instance_segmentation_trt import (
        YOLOv8ForInstanceSegmentationTRT,
    )

    model = YOLOv8ForInstanceSegmentationTRT.from_pretrained(
        model_name_or_path=yolov8_seg_asl_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model([asl_image_torch, asl_image_torch])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9795]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[63, 174, 187, 368]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert 16050 <= predictions[0].mask.cpu().sum().item() <= 16200
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.9795]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[63, 174, 187, 368]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert 16000 <= predictions[1].mask.cpu().sum().item() <= 16200


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_torch_batch(
    yolov8_seg_asl_trt_package: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.yolov8.yolov8_instance_segmentation_trt import (
        YOLOv8ForInstanceSegmentationTRT,
    )

    model = YOLOv8ForInstanceSegmentationTRT.from_pretrained(
        model_name_or_path=yolov8_seg_asl_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model(torch.stack([asl_image_torch, asl_image_torch], dim=0))

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9795]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[63, 174, 187, 368]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert 16000 <= predictions[0].mask.cpu().sum().item() <= 16200
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.9795]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[63, 174, 187, 368]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert 16000 <= predictions[1].mask.cpu().sum().item() <= 16200


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_per_class_confidence_blocks_specific_class(
    yolov8_seg_asl_trt_package: str,
    asl_image_numpy: np.ndarray,
) -> None:
    """Baseline (see `test_trt_package_numpy` above) returns 1 detection of
    class 20. Setting a 1.01 per-class threshold on class 20 leaves no
    detections."""
    from inference_models.models.yolov8.yolov8_instance_segmentation_trt import (
        YOLOv8ForInstanceSegmentationTRT,
    )
    from inference_models.weights_providers.entities import RecommendedParameters

    model = YOLOv8ForInstanceSegmentationTRT.from_pretrained(
        model_name_or_path=yolov8_seg_asl_trt_package,
        engine_host_code_allowed=True,
    )
    class_names = list(model.class_names)
    model.recommended_parameters = RecommendedParameters(
        confidence=0.25,
        per_class_confidence={class_names[20]: 1.01},
    )
    predictions = model(asl_image_numpy, confidence="best")
    assert predictions[0].class_id.numel() == 0


@pytest.mark.slow
@pytest.mark.trt_extras
@pytest.mark.gpu_only
@pytest.mark.parametrize("factor", [1.0, 0.5, 0.0])
@pytest.mark.parametrize(
    "crop_start", [None, 0, 25], ids=["uncropped", "origin-crop", "offset-crop"]
)
def test_mask_resolution_and_crop_round_trip(
    yolov8_seg_asl_trt_package: str,
    asl_image_numpy: np.ndarray,
    monkeypatch: pytest.MonkeyPatch,
    factor: float,
    crop_start: int | None,
) -> None:
    # given
    from inference_models.models.yolov8.yolov8_instance_segmentation_trt import (
        YOLOv8ForInstanceSegmentationTRT,
    )

    model = YOLOv8ForInstanceSegmentationTRT.from_pretrained(
        model_name_or_path=yolov8_seg_asl_trt_package,
        engine_host_code_allowed=True,
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
    expected_boxes = np.array([[63, 174, 187, 368]]) + offset
    np.testing.assert_allclose(detections.xyxy, expected_boxes, atol=5, rtol=0)
    np.testing.assert_array_equal(detections.class_id, [20])
    np.testing.assert_allclose(detections.confidence, [0.9795], atol=0.01, rtol=0)
    assert dense.image_size == scene.shape[:2]
    assert dense.mask_frame_size == scene.shape[:2]
    assert rle.image_size == scene.shape[:2]
    assert rle.mask_frame_size == scene.shape[:2]
    if factor == 1.0:
        assert dense.mask_size == scene.shape[:2]
    else:
        assert dense.mask_size != scene.shape[:2]
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
    # Reuse the full-resolution area baseline in test_trt_package_numpy.
    expected_area = 16100
    assert detections.mask.sum() == pytest.approx(expected_area, rel=0.05)
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
