import numpy as np
import pytest
import torch

from inference_models.configuration import DEFAULT_DEVICE
from inference_models.models.common.rle_utils import coco_rle_masks_to_torch_mask

CONFIDENCE_ATOL = 0.01
XYXY_ATOL = 2


@pytest.mark.slow
@pytest.mark.torch_models
def test_torchscript_package_stretch_numpy(
    yolo26n_seg_snakes_stretch_torch_script_package: str,
    snake_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.yolo26.yolo26_instance_segmentation_torch_script import (
        YOLO26ForInstanceSegmentationTorchScript,
    )

    model = YOLO26ForInstanceSegmentationTorchScript.from_pretrained(
        model_name_or_path=yolo26n_seg_snakes_stretch_torch_script_package,
        device=DEFAULT_DEVICE,
    )

    predictions = model(snake_image_numpy, confidence=0.25)

    xyxy = predictions[0].xyxy.cpu().tolist()[0]
    mask_region_sum = (
        predictions[0]
        .to_supervision()
        .mask[0, xyxy[1] : xyxy[3], xyxy[0] : xyxy[2]]
        .sum()
    )

    assert len(predictions) == 1
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9644]),
        atol=CONFIDENCE_ATOL,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([0], dtype=torch.int32),
    )
    expected_xyxy = torch.tensor([[128, 326, 1263, 558]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy,
        atol=XYXY_ATOL,
    )
    assert 209000 <= mask_region_sum <= 210200


@pytest.mark.slow
@pytest.mark.torch_models
def test_torchscript_package_stretch_numpy_rle_variant(
    yolo26n_seg_snakes_stretch_torch_script_package: str,
    snake_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.yolo26.yolo26_instance_segmentation_torch_script import (
        YOLO26ForInstanceSegmentationTorchScript,
    )

    model = YOLO26ForInstanceSegmentationTorchScript.from_pretrained(
        model_name_or_path=yolo26n_seg_snakes_stretch_torch_script_package,
        device=DEFAULT_DEVICE,
    )

    predictions = model(snake_image_numpy, confidence=0.25, mask_format="rle")
    predictions_ref = model(snake_image_numpy, confidence=0.25)
    decoded_mask = coco_rle_masks_to_torch_mask(
        instances_masks=predictions[0].mask, device=torch.device("cpu")
    )

    xyxy = predictions[0].xyxy.cpu().tolist()[0]
    mask_region_sum = (
        predictions[0]
        .to_supervision()
        .mask[0, xyxy[1] : xyxy[3], xyxy[0] : xyxy[2]]
        .sum()
    )

    assert len(predictions) == 1
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9644]),
        atol=CONFIDENCE_ATOL,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([0], dtype=torch.int32),
    )
    expected_xyxy = torch.tensor([[128, 326, 1263, 558]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy,
        atol=XYXY_ATOL,
    )
    assert 209000 <= mask_region_sum <= 210200
    assert np.allclose(
        decoded_mask.cpu().numpy(), predictions_ref[0].mask.cpu().numpy()
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_torchscript_package_stretch_batch_numpy(
    yolo26n_seg_snakes_stretch_torch_script_package: str,
    snake_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.yolo26.yolo26_instance_segmentation_torch_script import (
        YOLO26ForInstanceSegmentationTorchScript,
    )

    model = YOLO26ForInstanceSegmentationTorchScript.from_pretrained(
        model_name_or_path=yolo26n_seg_snakes_stretch_torch_script_package,
        device=DEFAULT_DEVICE,
    )

    predictions = model([snake_image_numpy, snake_image_numpy], confidence=0.25)

    xyxy_0 = predictions[0].xyxy.cpu().tolist()[0]
    xyxy_1 = predictions[1].xyxy.cpu().tolist()[0]
    mask_region_sum_0 = (
        predictions[0]
        .to_supervision()
        .mask[0, xyxy_0[1] : xyxy_0[3], xyxy_0[0] : xyxy_0[2]]
        .sum()
    )
    mask_region_sum_1 = (
        predictions[1]
        .to_supervision()
        .mask[0, xyxy_1[1] : xyxy_1[3], xyxy_1[0] : xyxy_1[2]]
        .sum()
    )

    assert len(predictions) == 2
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9644]),
        atol=CONFIDENCE_ATOL,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.9644]),
        atol=CONFIDENCE_ATOL,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([0], dtype=torch.int32),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([0], dtype=torch.int32),
    )
    expected_xyxy = torch.tensor([[128, 326, 1263, 558]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy,
        atol=XYXY_ATOL,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy,
        atol=XYXY_ATOL,
    )
    assert 209000 <= mask_region_sum_0 <= 210200
    assert 209000 <= mask_region_sum_1 <= 210200


@pytest.mark.slow
@pytest.mark.torch_models
def test_torchscript_package_stretch_batch_numpy_rle_variant(
    yolo26n_seg_snakes_stretch_torch_script_package: str,
    snake_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.yolo26.yolo26_instance_segmentation_torch_script import (
        YOLO26ForInstanceSegmentationTorchScript,
    )

    model = YOLO26ForInstanceSegmentationTorchScript.from_pretrained(
        model_name_or_path=yolo26n_seg_snakes_stretch_torch_script_package,
        device=DEFAULT_DEVICE,
    )

    predictions = model(
        [snake_image_numpy, snake_image_numpy], confidence=0.25, mask_format="rle"
    )
    predictions_ref = model([snake_image_numpy, snake_image_numpy], confidence=0.25)
    decoded_mask_1 = coco_rle_masks_to_torch_mask(
        instances_masks=predictions[0].mask, device=torch.device("cpu")
    )
    decoded_mask_2 = coco_rle_masks_to_torch_mask(
        instances_masks=predictions[1].mask, device=torch.device("cpu")
    )

    xyxy_0 = predictions[0].xyxy.cpu().tolist()[0]
    xyxy_1 = predictions[1].xyxy.cpu().tolist()[0]
    mask_region_sum_0 = (
        predictions[0]
        .to_supervision()
        .mask[0, xyxy_0[1] : xyxy_0[3], xyxy_0[0] : xyxy_0[2]]
        .sum()
    )
    mask_region_sum_1 = (
        predictions[1]
        .to_supervision()
        .mask[0, xyxy_1[1] : xyxy_1[3], xyxy_1[0] : xyxy_1[2]]
        .sum()
    )

    assert len(predictions) == 2
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9644]),
        atol=CONFIDENCE_ATOL,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.9644]),
        atol=CONFIDENCE_ATOL,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([0], dtype=torch.int32),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([0], dtype=torch.int32),
    )
    expected_xyxy = torch.tensor([[128, 326, 1263, 558]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy,
        atol=XYXY_ATOL,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy,
        atol=XYXY_ATOL,
    )
    assert 209000 <= mask_region_sum_0 <= 210200
    assert 209000 <= mask_region_sum_1 <= 210200
    assert np.allclose(
        decoded_mask_1.cpu().numpy(), predictions_ref[0].mask.cpu().numpy()
    )
    assert np.allclose(
        decoded_mask_2.cpu().numpy(), predictions_ref[1].mask.cpu().numpy()
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_torchscript_package_stretch_torch(
    yolo26n_seg_snakes_stretch_torch_script_package: str,
    snake_image_torch: torch.Tensor,
) -> None:
    from inference_models.models.yolo26.yolo26_instance_segmentation_torch_script import (
        YOLO26ForInstanceSegmentationTorchScript,
    )

    model = YOLO26ForInstanceSegmentationTorchScript.from_pretrained(
        model_name_or_path=yolo26n_seg_snakes_stretch_torch_script_package,
        device=DEFAULT_DEVICE,
    )

    predictions = model(snake_image_torch, confidence=0.25)

    xyxy = predictions[0].xyxy.cpu().tolist()[0]
    mask_region_sum = (
        predictions[0]
        .to_supervision()
        .mask[0, xyxy[1] : xyxy[3], xyxy[0] : xyxy[2]]
        .sum()
    )

    assert len(predictions) == 1
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9641]),
        atol=CONFIDENCE_ATOL,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([0], dtype=torch.int32),
    )
    expected_xyxy = torch.tensor([[128, 326, 1263, 558]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy,
        atol=XYXY_ATOL,
    )
    assert 209000 <= mask_region_sum <= 210200


@pytest.mark.slow
@pytest.mark.torch_models
def test_torchscript_package_letterbox_numpy(
    yolo26n_seg_snakes_letterbox_torch_script_package: str,
    snake_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.yolo26.yolo26_instance_segmentation_torch_script import (
        YOLO26ForInstanceSegmentationTorchScript,
    )

    model = YOLO26ForInstanceSegmentationTorchScript.from_pretrained(
        model_name_or_path=yolo26n_seg_snakes_letterbox_torch_script_package,
        device=DEFAULT_DEVICE,
    )

    predictions = model(snake_image_numpy, confidence=0.25)

    xyxy = predictions[0].xyxy.cpu().tolist()[0]
    mask_region_sum = (
        predictions[0]
        .to_supervision()
        .mask[0, xyxy[1] : xyxy[3], xyxy[0] : xyxy[2]]
        .sum()
    )

    assert len(predictions) == 1
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.274]),
        atol=CONFIDENCE_ATOL,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([0], dtype=torch.int32),
    )
    expected_xyxy = torch.tensor([[107, 325, 1293, 562]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy,
        atol=XYXY_ATOL,
    )
    assert 219500 <= mask_region_sum <= 221300


@pytest.mark.slow
@pytest.mark.torch_models
def test_torchscript_package_letterbox_batch_numpy(
    yolo26n_seg_snakes_letterbox_torch_script_package: str,
    snake_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.yolo26.yolo26_instance_segmentation_torch_script import (
        YOLO26ForInstanceSegmentationTorchScript,
    )

    model = YOLO26ForInstanceSegmentationTorchScript.from_pretrained(
        model_name_or_path=yolo26n_seg_snakes_letterbox_torch_script_package,
        device=DEFAULT_DEVICE,
    )

    predictions = model([snake_image_numpy, snake_image_numpy], confidence=0.25)

    xyxy_0 = predictions[0].xyxy.cpu().tolist()[0]
    xyxy_1 = predictions[1].xyxy.cpu().tolist()[0]
    mask_region_sum_0 = (
        predictions[0]
        .to_supervision()
        .mask[0, xyxy_0[1] : xyxy_0[3], xyxy_0[0] : xyxy_0[2]]
        .sum()
    )
    mask_region_sum_1 = (
        predictions[1]
        .to_supervision()
        .mask[0, xyxy_1[1] : xyxy_1[3], xyxy_1[0] : xyxy_1[2]]
        .sum()
    )

    assert len(predictions) == 2
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.274]),
        atol=CONFIDENCE_ATOL,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.274]),
        atol=CONFIDENCE_ATOL,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([0], dtype=torch.int32),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([0], dtype=torch.int32),
    )
    expected_xyxy = torch.tensor([[107, 325, 1293, 562]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy,
        atol=XYXY_ATOL,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy,
        atol=XYXY_ATOL,
    )
    assert 219500 <= mask_region_sum_0 <= 221300
    assert 219500 <= mask_region_sum_1 <= 221300


@pytest.mark.slow
@pytest.mark.torch_models
def test_torchscript_package_letterbox_torch(
    yolo26n_seg_snakes_letterbox_torch_script_package: str,
    snake_image_torch: torch.Tensor,
) -> None:
    from inference_models.models.yolo26.yolo26_instance_segmentation_torch_script import (
        YOLO26ForInstanceSegmentationTorchScript,
    )

    model = YOLO26ForInstanceSegmentationTorchScript.from_pretrained(
        model_name_or_path=yolo26n_seg_snakes_letterbox_torch_script_package,
        device=DEFAULT_DEVICE,
    )

    predictions = model(snake_image_torch, confidence=0.25)

    xyxy = predictions[0].xyxy.cpu().tolist()[0]
    mask_region_sum = (
        predictions[0]
        .to_supervision()
        .mask[0, xyxy[1] : xyxy[3], xyxy[0] : xyxy[2]]
        .sum()
    )

    assert len(predictions) == 1
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.270]),
        atol=CONFIDENCE_ATOL,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([0], dtype=torch.int32),
    )
    expected_xyxy = torch.tensor([[107, 325, 1293, 562]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy,
        atol=XYXY_ATOL,
    )
    assert 219500 <= mask_region_sum <= 221300


@pytest.mark.slow
@pytest.mark.torch_models
def test_torchscript_per_class_confidence_filters_detections(
    yolo26n_seg_snakes_stretch_torch_script_package: str,
    snake_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.yolo26.yolo26_instance_segmentation_torch_script import (
        YOLO26ForInstanceSegmentationTorchScript,
    )
    from inference_models.weights_providers.entities import RecommendedParameters

    model = YOLO26ForInstanceSegmentationTorchScript.from_pretrained(
        model_name_or_path=yolo26n_seg_snakes_stretch_torch_script_package,
        device=DEFAULT_DEVICE,
    )
    class_names = list(model.class_names)
    model.recommended_parameters = RecommendedParameters(
        confidence=0.25,
        per_class_confidence={class_names[0]: 1.01},
    )
    predictions = model(snake_image_numpy, confidence="best")
    assert predictions[0].class_id.numel() == 0
