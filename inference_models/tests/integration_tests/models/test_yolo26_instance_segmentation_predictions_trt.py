import numpy as np
import pytest
import torch

from inference_models.models.common.rle_utils import coco_rle_masks_to_torch_mask


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_numpy(
    yolo26_seg_asl_trt_package: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yolo26.yolo26_instance_segmentation_trt import (
        YOLO26ForInstanceSegmentationTRT,
    )

    model = YOLO26ForInstanceSegmentationTRT.from_pretrained(
        model_name_or_path=yolo26_seg_asl_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model(asl_image_numpy)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9671]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[63, 174, 186, 368]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert 16100 <= predictions[0].mask.cpu().sum().item() <= 16700


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_numpy_rle_variant(
    yolo26_seg_asl_trt_package: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yolo26.yolo26_instance_segmentation_trt import (
        YOLO26ForInstanceSegmentationTRT,
    )

    model = YOLO26ForInstanceSegmentationTRT.from_pretrained(
        model_name_or_path=yolo26_seg_asl_trt_package,
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
        torch.tensor([0.9671]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[63, 174, 186, 368]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert 16100 <= decoded_mask.cpu().sum().item() <= 16700
    assert np.allclose(
        decoded_mask.cpu().numpy(), predictions_ref[0].mask.cpu().numpy()
    )


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_batch_numpy(
    yolo26_seg_asl_trt_package: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yolo26.yolo26_instance_segmentation_trt import (
        YOLO26ForInstanceSegmentationTRT,
    )

    model = YOLO26ForInstanceSegmentationTRT.from_pretrained(
        model_name_or_path=yolo26_seg_asl_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model([asl_image_numpy, asl_image_numpy])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9671]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[63, 174, 186, 368]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert 16100 <= predictions[0].mask.cpu().sum().item() <= 16700
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.9671]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[63, 174, 186, 368]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert 16100 <= predictions[1].mask.cpu().sum().item() <= 16700


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_batch_numpy_rle_variant(
    yolo26_seg_asl_trt_package: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yolo26.yolo26_instance_segmentation_trt import (
        YOLO26ForInstanceSegmentationTRT,
    )

    model = YOLO26ForInstanceSegmentationTRT.from_pretrained(
        model_name_or_path=yolo26_seg_asl_trt_package,
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
        torch.tensor([0.9671]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[63, 174, 186, 368]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert 16100 <= decoded_mask_1.cpu().sum().item() <= 16700
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.9671]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[63, 174, 186, 368]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert 16100 <= decoded_mask_2.cpu().sum().item() <= 16700
    assert np.allclose(
        decoded_mask_1.cpu().numpy(), predictions_ref[0].mask.cpu().numpy()
    )
    assert np.allclose(
        decoded_mask_2.cpu().numpy(), predictions_ref[1].mask.cpu().numpy()
    )


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_torch(
    yolo26_seg_asl_trt_package: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.yolo26.yolo26_instance_segmentation_trt import (
        YOLO26ForInstanceSegmentationTRT,
    )

    model = YOLO26ForInstanceSegmentationTRT.from_pretrained(
        model_name_or_path=yolo26_seg_asl_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model(asl_image_torch)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9671]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[63, 174, 186, 368]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert 16100 <= predictions[0].mask.cpu().sum().item() <= 16700


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_torch_multiple_predictions_in_row(
    yolo26_seg_asl_trt_package: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.yolo26.yolo26_instance_segmentation_trt import (
        YOLO26ForInstanceSegmentationTRT,
    )

    model = YOLO26ForInstanceSegmentationTRT.from_pretrained(
        model_name_or_path=yolo26_seg_asl_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    for _ in range(8):
        predictions = model(asl_image_torch)

        # then
        assert torch.allclose(
            predictions[0].confidence.cpu(),
            torch.tensor([0.9671]).cpu(),
            atol=0.01,
        )
        assert torch.allclose(
            predictions[0].class_id.cpu(),
            torch.tensor([20], dtype=torch.int32).cpu(),
        )
        expected_xyxy = torch.tensor(
            [[63, 174, 186, 368]],
            dtype=torch.int32,
        )
        assert torch.allclose(
            predictions[0].xyxy.cpu(),
            expected_xyxy.cpu(),
            atol=5,
        )
        assert 16100 <= predictions[0].mask.cpu().sum().item() <= 16700


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_torch_list(
    yolo26_seg_asl_trt_package: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.yolo26.yolo26_instance_segmentation_trt import (
        YOLO26ForInstanceSegmentationTRT,
    )

    model = YOLO26ForInstanceSegmentationTRT.from_pretrained(
        model_name_or_path=yolo26_seg_asl_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model([asl_image_torch, asl_image_torch])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9671]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[63, 174, 186, 368]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert 16100 <= predictions[0].mask.cpu().sum().item() <= 16700
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.9671]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[63, 174, 186, 368]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert 16100 <= predictions[1].mask.cpu().sum().item() <= 16700


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_torch_batch(
    yolo26_seg_asl_trt_package: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.yolo26.yolo26_instance_segmentation_trt import (
        YOLO26ForInstanceSegmentationTRT,
    )

    model = YOLO26ForInstanceSegmentationTRT.from_pretrained(
        model_name_or_path=yolo26_seg_asl_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model(torch.stack([asl_image_torch, asl_image_torch], dim=0))

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9671]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[63, 174, 186, 368]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert 16100 <= predictions[0].mask.cpu().sum().item() <= 16700
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.9671]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[63, 174, 186, 368]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert 16100 <= predictions[1].mask.cpu().sum().item() <= 16700


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_per_class_confidence_filters_detections(
    yolo26_seg_asl_trt_package: str,
    asl_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.yolo26.yolo26_instance_segmentation_trt import (
        YOLO26ForInstanceSegmentationTRT,
    )
    from inference_models.weights_providers.entities import RecommendedParameters

    model = YOLO26ForInstanceSegmentationTRT.from_pretrained(
        model_name_or_path=yolo26_seg_asl_trt_package,
        engine_host_code_allowed=True,
    )
    class_names = list(model.class_names)
    model.recommended_parameters = RecommendedParameters(
        confidence=0.25,
        per_class_confidence={class_names[20]: 1.01},
    )
    predictions = model(asl_image_numpy, confidence="best")
    assert predictions[0].class_id.numel() == 0
