import numpy as np
import pytest
import torch

from inference_models.configuration import DEFAULT_DEVICE

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

    predictions = model(snake_image_numpy)

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
    assert 210000 <= mask_region_sum <= 210200


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

    predictions = model([snake_image_numpy, snake_image_numpy])

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
    assert 210000 <= mask_region_sum_0 <= 210200
    assert 210000 <= mask_region_sum_1 <= 210200


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

    predictions = model(snake_image_torch)

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
    assert 210000 <= mask_region_sum <= 210200


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

    predictions = model(snake_image_numpy)

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
    assert 221000 <= mask_region_sum <= 221300


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

    predictions = model([snake_image_numpy, snake_image_numpy])

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
    assert 221000 <= mask_region_sum_0 <= 221300
    assert 221000 <= mask_region_sum_1 <= 221300


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

    predictions = model(snake_image_torch)

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
    assert 221000 <= mask_region_sum <= 221300
