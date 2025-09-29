import numpy as np
import pytest
import torch
from inference_exp.configuration import DEFAULT_DEVICE
from inference_exp.models.deep_lab_v3_plus.deep_lab_v3_plus_segmentation_torch import (
    DeepLabV3PlusForSemanticSegmentationTorch,
)


@pytest.mark.slow
@pytest.mark.torch_models
def test_torch_package_with_stretch_numpy(
    balloons_deep_lab_v3_torch_stretch_package: str,
    balloons_image_numpy: np.ndarray,
) -> None:
    # given
    model = DeepLabV3PlusForSemanticSegmentationTorch.from_pretrained(
        model_name_or_path=balloons_deep_lab_v3_torch_stretch_package,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(balloons_image_numpy)

    # then
    assert sorted(torch.unique(predictions[0].segmentation_map).cpu().tolist()) == [
        0,
        3,
    ]
    assert torch.allclose(
        torch.mean(predictions[0].confidence).cpu(),
        torch.tensor(0.9646).cpu(),
        atol=0.001,
    )
    assert (
        245000 <= torch.sum(predictions[0].segmentation_map.cpu() == 0).item() <= 246000
    )
    assert (
        16600 <= torch.sum(predictions[0].segmentation_map.cpu() == 3).item() <= 16700
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_torch_package_with_stretch_batch_numpy(
    balloons_deep_lab_v3_torch_stretch_package: str,
    balloons_image_numpy: np.ndarray,
) -> None:
    # given
    model = DeepLabV3PlusForSemanticSegmentationTorch.from_pretrained(
        model_name_or_path=balloons_deep_lab_v3_torch_stretch_package,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model([balloons_image_numpy, balloons_image_numpy])

    # then
    assert sorted(torch.unique(predictions[0].segmentation_map).cpu().tolist()) == [
        0,
        3,
    ]
    assert torch.allclose(
        torch.mean(predictions[0].confidence).cpu(),
        torch.tensor(0.9646).cpu(),
        atol=0.001,
    )
    assert (
        245000 <= torch.sum(predictions[0].segmentation_map.cpu() == 0).item() <= 246000
    )
    assert (
        16600 <= torch.sum(predictions[0].segmentation_map.cpu() == 3).item() <= 16700
    )
    assert sorted(torch.unique(predictions[1].segmentation_map).cpu().tolist()) == [
        0,
        3,
    ]
    assert torch.allclose(
        torch.mean(predictions[1].confidence).cpu(),
        torch.tensor(0.9646).cpu(),
        atol=0.001,
    )
    assert (
        245000 <= torch.sum(predictions[1].segmentation_map.cpu() == 0).item() <= 246000
    )
    assert (
        16600 <= torch.sum(predictions[1].segmentation_map.cpu() == 3).item() <= 16700
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_torch_package_with_stretch_torch(
    balloons_deep_lab_v3_torch_stretch_package: str,
    balloons_image_torch: torch.Tensor,
) -> None:
    # given
    model = DeepLabV3PlusForSemanticSegmentationTorch.from_pretrained(
        model_name_or_path=balloons_deep_lab_v3_torch_stretch_package,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(balloons_image_torch)

    # then
    assert sorted(torch.unique(predictions[0].segmentation_map).cpu().tolist()) == [
        0,
        3,
    ]
    assert torch.allclose(
        torch.mean(predictions[0].confidence).cpu(),
        torch.tensor(0.9646).cpu(),
        atol=0.001,
    )
    assert (
        245000 <= torch.sum(predictions[0].segmentation_map.cpu() == 0).item() <= 246000
    )
    assert (
        16600 <= torch.sum(predictions[0].segmentation_map.cpu() == 3).item() <= 16700
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_torch_package_with_stretch_batch_torch(
    balloons_deep_lab_v3_torch_stretch_package: str,
    balloons_image_torch: torch.Tensor,
) -> None:
    # given
    model = DeepLabV3PlusForSemanticSegmentationTorch.from_pretrained(
        model_name_or_path=balloons_deep_lab_v3_torch_stretch_package,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(
        torch.stack([balloons_image_torch, balloons_image_torch], dim=0)
    )

    # then
    assert sorted(torch.unique(predictions[0].segmentation_map).cpu().tolist()) == [
        0,
        3,
    ]
    assert torch.allclose(
        torch.mean(predictions[0].confidence).cpu(),
        torch.tensor(0.9646).cpu(),
        atol=0.001,
    )
    assert (
        245000 <= torch.sum(predictions[0].segmentation_map.cpu() == 0).item() <= 246000
    )
    assert (
        16600 <= torch.sum(predictions[0].segmentation_map.cpu() == 3).item() <= 16700
    )
    assert sorted(torch.unique(predictions[1].segmentation_map).cpu().tolist()) == [
        0,
        3,
    ]
    assert torch.allclose(
        torch.mean(predictions[1].confidence).cpu(),
        torch.tensor(0.9646).cpu(),
        atol=0.001,
    )
    assert (
        245000 <= torch.sum(predictions[1].segmentation_map.cpu() == 0).item() <= 246000
    )
    assert (
        16600 <= torch.sum(predictions[1].segmentation_map.cpu() == 3).item() <= 16700
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_torch_package_with_stretch_batch_torch_list(
    balloons_deep_lab_v3_torch_stretch_package: str,
    balloons_image_torch: torch.Tensor,
) -> None:
    # given
    model = DeepLabV3PlusForSemanticSegmentationTorch.from_pretrained(
        model_name_or_path=balloons_deep_lab_v3_torch_stretch_package,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model([balloons_image_torch, balloons_image_torch])

    # then
    assert sorted(torch.unique(predictions[0].segmentation_map).cpu().tolist()) == [
        0,
        3,
    ]
    assert torch.allclose(
        torch.mean(predictions[0].confidence).cpu(),
        torch.tensor(0.9646).cpu(),
        atol=0.001,
    )
    assert (
        245000 <= torch.sum(predictions[0].segmentation_map.cpu() == 0).item() <= 246000
    )
    assert (
        16600 <= torch.sum(predictions[0].segmentation_map.cpu() == 3).item() <= 16700
    )
    assert sorted(torch.unique(predictions[1].segmentation_map).cpu().tolist()) == [
        0,
        3,
    ]
    assert torch.allclose(
        torch.mean(predictions[1].confidence).cpu(),
        torch.tensor(0.9646).cpu(),
        atol=0.001,
    )
    assert (
        245000 <= torch.sum(predictions[1].segmentation_map.cpu() == 0).item() <= 246000
    )
    assert (
        16600 <= torch.sum(predictions[1].segmentation_map.cpu() == 3).item() <= 16700
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_torch_package_with_static_crop_letterbox_numpy(
    balloons_deep_lab_v3_torch_static_crop_letterbox_package: str,
    balloons_image_numpy: np.ndarray,
) -> None:
    # given
    model = DeepLabV3PlusForSemanticSegmentationTorch.from_pretrained(
        model_name_or_path=balloons_deep_lab_v3_torch_static_crop_letterbox_package,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(balloons_image_numpy)

    # then
    assert sorted(torch.unique(predictions[0].segmentation_map).cpu().tolist()) == [
        0,
        3,
    ]
    assert torch.allclose(
        torch.mean(predictions[0].confidence).cpu(),
        torch.tensor(0.2423).cpu(),
        atol=0.001,
    )
    assert (
        247000 <= torch.sum(predictions[0].segmentation_map.cpu() == 0).item() <= 248000
    )
    assert (
        14800 <= torch.sum(predictions[0].segmentation_map.cpu() == 3).item() <= 15000
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_torch_package_with_static_crop_letterbox_batch_numpy(
    balloons_deep_lab_v3_torch_static_crop_letterbox_package: str,
    balloons_image_numpy: np.ndarray,
) -> None:
    # given
    model = DeepLabV3PlusForSemanticSegmentationTorch.from_pretrained(
        model_name_or_path=balloons_deep_lab_v3_torch_static_crop_letterbox_package,
        device=DEFAULT_DEVICE,
    )
    # when
    predictions = model([balloons_image_numpy, balloons_image_numpy])

    # then
    assert sorted(torch.unique(predictions[0].segmentation_map).cpu().tolist()) == [
        0,
        3,
    ]
    assert torch.allclose(
        torch.mean(predictions[0].confidence).cpu(),
        torch.tensor(0.2423).cpu(),
        atol=0.001,
    )
    assert (
        247000 <= torch.sum(predictions[0].segmentation_map.cpu() == 0).item() <= 248000
    )
    assert (
        14800 <= torch.sum(predictions[0].segmentation_map.cpu() == 3).item() <= 15000
    )
    assert sorted(torch.unique(predictions[1].segmentation_map).cpu().tolist()) == [
        0,
        3,
    ]
    assert torch.allclose(
        torch.mean(predictions[1].confidence).cpu(),
        torch.tensor(0.2423).cpu(),
        atol=0.001,
    )
    assert (
        247000 <= torch.sum(predictions[1].segmentation_map.cpu() == 0).item() <= 248000
    )
    assert (
        14800 <= torch.sum(predictions[1].segmentation_map.cpu() == 3).item() <= 15000
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_torch_package_with_static_crop_letterbox_torch(
    balloons_deep_lab_v3_torch_static_crop_letterbox_package: str,
    balloons_image_torch: torch.Tensor,
) -> None:
    # given
    model = DeepLabV3PlusForSemanticSegmentationTorch.from_pretrained(
        model_name_or_path=balloons_deep_lab_v3_torch_static_crop_letterbox_package,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(balloons_image_torch)

    # then
    assert sorted(torch.unique(predictions[0].segmentation_map).cpu().tolist()) == [
        0,
        3,
    ]
    assert torch.allclose(
        torch.mean(predictions[0].confidence).cpu(),
        torch.tensor(0.2423).cpu(),
        atol=0.001,
    )
    assert (
        247000 <= torch.sum(predictions[0].segmentation_map.cpu() == 0).item() <= 248000
    )
    assert (
        14800 <= torch.sum(predictions[0].segmentation_map.cpu() == 3).item() <= 15000
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_torch_package_with_static_crop_letterbox_batch_torch(
    balloons_deep_lab_v3_torch_static_crop_letterbox_package: str,
    balloons_image_torch: torch.Tensor,
) -> None:
    # given
    model = DeepLabV3PlusForSemanticSegmentationTorch.from_pretrained(
        model_name_or_path=balloons_deep_lab_v3_torch_static_crop_letterbox_package,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(
        torch.stack([balloons_image_torch, balloons_image_torch], dim=0)
    )

    # then
    assert sorted(torch.unique(predictions[0].segmentation_map).cpu().tolist()) == [
        0,
        3,
    ]
    assert torch.allclose(
        torch.mean(predictions[0].confidence).cpu(),
        torch.tensor(0.2423).cpu(),
        atol=0.001,
    )
    assert (
        247000 <= torch.sum(predictions[0].segmentation_map.cpu() == 0).item() <= 248000
    )
    assert (
        14800 <= torch.sum(predictions[0].segmentation_map.cpu() == 3).item() <= 15000
    )
    assert sorted(torch.unique(predictions[1].segmentation_map).cpu().tolist()) == [
        0,
        3,
    ]
    assert torch.allclose(
        torch.mean(predictions[1].confidence).cpu(),
        torch.tensor(0.2423).cpu(),
        atol=0.001,
    )
    assert (
        247000 <= torch.sum(predictions[1].segmentation_map.cpu() == 0).item() <= 248000
    )
    assert (
        14800 <= torch.sum(predictions[1].segmentation_map.cpu() == 3).item() <= 15000
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_torch_package_with_static_crop_letterbox_batch_torch_list(
    balloons_deep_lab_v3_torch_static_crop_letterbox_package: str,
    balloons_image_torch: torch.Tensor,
) -> None:
    # given
    model = DeepLabV3PlusForSemanticSegmentationTorch.from_pretrained(
        model_name_or_path=balloons_deep_lab_v3_torch_static_crop_letterbox_package,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model([balloons_image_torch, balloons_image_torch])

    # then
    assert sorted(torch.unique(predictions[0].segmentation_map).cpu().tolist()) == [
        0,
        3,
    ]
    assert torch.allclose(
        torch.mean(predictions[0].confidence).cpu(),
        torch.tensor(0.2423).cpu(),
        atol=0.001,
    )
    assert (
        247000 <= torch.sum(predictions[0].segmentation_map.cpu() == 0).item() <= 248000
    )
    assert (
        14800 <= torch.sum(predictions[0].segmentation_map.cpu() == 3).item() <= 15000
    )
    assert sorted(torch.unique(predictions[1].segmentation_map).cpu().tolist()) == [
        0,
        3,
    ]
    assert torch.allclose(
        torch.mean(predictions[1].confidence).cpu(),
        torch.tensor(0.2423).cpu(),
        atol=0.001,
    )
    assert (
        247000 <= torch.sum(predictions[1].segmentation_map.cpu() == 0).item() <= 248000
    )
    assert (
        14800 <= torch.sum(predictions[1].segmentation_map.cpu() == 3).item() <= 15000
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_torch_package_with_static_crop_center_crop_numpy(
    balloons_deep_lab_v3_torch_static_crop_center_crop_package: str,
    balloons_image_numpy: np.ndarray,
) -> None:
    # given
    model = DeepLabV3PlusForSemanticSegmentationTorch.from_pretrained(
        model_name_or_path=balloons_deep_lab_v3_torch_static_crop_center_crop_package,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(balloons_image_numpy)

    # then
    assert sorted(torch.unique(predictions[0].segmentation_map).cpu().tolist()) == [
        0,
        1,
        3,
    ]
    assert torch.allclose(
        torch.mean(predictions[0].confidence).cpu(),
        torch.tensor(0.2461).cpu(),
        atol=0.001,
    )
    assert (
        248000 <= torch.sum(predictions[0].segmentation_map.cpu() == 0).item() <= 249000
    )
    assert (
        13700 <= torch.sum(predictions[0].segmentation_map.cpu() == 3).item() <= 13900
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_torch_package_with_static_crop_center_crop_batch_numpy(
    balloons_deep_lab_v3_torch_static_crop_center_crop_package: str,
    balloons_image_numpy: np.ndarray,
) -> None:
    # given
    model = DeepLabV3PlusForSemanticSegmentationTorch.from_pretrained(
        model_name_or_path=balloons_deep_lab_v3_torch_static_crop_center_crop_package,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model([balloons_image_numpy, balloons_image_numpy])

    # then
    assert sorted(torch.unique(predictions[0].segmentation_map).cpu().tolist()) == [
        0,
        1,
        3,
    ]
    assert torch.allclose(
        torch.mean(predictions[0].confidence).cpu(),
        torch.tensor(0.2461).cpu(),
        atol=0.001,
    )
    assert (
        248000 <= torch.sum(predictions[0].segmentation_map.cpu() == 0).item() <= 249000
    )
    assert (
        13700 <= torch.sum(predictions[0].segmentation_map.cpu() == 3).item() <= 13900
    )
    assert sorted(torch.unique(predictions[1].segmentation_map).cpu().tolist()) == [
        0,
        1,
        3,
    ]
    assert torch.allclose(
        torch.mean(predictions[1].confidence).cpu(),
        torch.tensor(0.2461).cpu(),
        atol=0.001,
    )
    assert (
        248000 <= torch.sum(predictions[1].segmentation_map.cpu() == 0).item() <= 249000
    )
    assert (
        13700 <= torch.sum(predictions[1].segmentation_map.cpu() == 3).item() <= 13900
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_torch_package_with_static_crop_center_crop_torch(
    balloons_deep_lab_v3_torch_static_crop_center_crop_package: str,
    balloons_image_torch: torch.Tensor,
) -> None:
    # given
    model = DeepLabV3PlusForSemanticSegmentationTorch.from_pretrained(
        model_name_or_path=balloons_deep_lab_v3_torch_static_crop_center_crop_package,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(balloons_image_torch)

    # then
    assert sorted(torch.unique(predictions[0].segmentation_map).cpu().tolist()) == [
        0,
        1,
        3,
    ]
    assert torch.allclose(
        torch.mean(predictions[0].confidence).cpu(),
        torch.tensor(0.2461).cpu(),
        atol=0.001,
    )
    assert (
        248000 <= torch.sum(predictions[0].segmentation_map.cpu() == 0).item() <= 249000
    )
    assert (
        13700 <= torch.sum(predictions[0].segmentation_map.cpu() == 3).item() <= 13900
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_torch_package_with_static_crop_center_crop_batch_torch(
    balloons_deep_lab_v3_torch_static_crop_center_crop_package: str,
    balloons_image_torch: torch.Tensor,
) -> None:
    # given
    model = DeepLabV3PlusForSemanticSegmentationTorch.from_pretrained(
        model_name_or_path=balloons_deep_lab_v3_torch_static_crop_center_crop_package,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(
        torch.stack([balloons_image_torch, balloons_image_torch], dim=0)
    )

    # then
    assert sorted(torch.unique(predictions[0].segmentation_map).cpu().tolist()) == [
        0,
        1,
        3,
    ]
    assert torch.allclose(
        torch.mean(predictions[0].confidence).cpu(),
        torch.tensor(0.2461).cpu(),
        atol=0.001,
    )
    assert (
        248000 <= torch.sum(predictions[0].segmentation_map.cpu() == 0).item() <= 249000
    )
    assert (
        13700 <= torch.sum(predictions[0].segmentation_map.cpu() == 3).item() <= 13900
    )
    assert sorted(torch.unique(predictions[1].segmentation_map).cpu().tolist()) == [
        0,
        1,
        3,
    ]
    assert torch.allclose(
        torch.mean(predictions[1].confidence).cpu(),
        torch.tensor(0.2461).cpu(),
        atol=0.001,
    )
    assert (
        248000 <= torch.sum(predictions[1].segmentation_map.cpu() == 0).item() <= 249000
    )
    assert (
        13700 <= torch.sum(predictions[1].segmentation_map.cpu() == 3).item() <= 13900
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_torch_package_with_static_crop_center_crop_batch_torch_list(
    balloons_deep_lab_v3_torch_static_crop_center_crop_package: str,
    balloons_image_torch: torch.Tensor,
) -> None:
    # given
    model = DeepLabV3PlusForSemanticSegmentationTorch.from_pretrained(
        model_name_or_path=balloons_deep_lab_v3_torch_static_crop_center_crop_package,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model([balloons_image_torch, balloons_image_torch])

    # then
    assert sorted(torch.unique(predictions[0].segmentation_map).cpu().tolist()) == [
        0,
        1,
        3,
    ]
    assert torch.allclose(
        torch.mean(predictions[0].confidence).cpu(),
        torch.tensor(0.2461).cpu(),
        atol=0.001,
    )
    assert (
        248000 <= torch.sum(predictions[0].segmentation_map.cpu() == 0).item() <= 249000
    )
    assert (
        13700 <= torch.sum(predictions[0].segmentation_map.cpu() == 3).item() <= 13900
    )
    assert sorted(torch.unique(predictions[1].segmentation_map).cpu().tolist()) == [
        0,
        1,
        3,
    ]
    assert torch.allclose(
        torch.mean(predictions[1].confidence).cpu(),
        torch.tensor(0.2461).cpu(),
        atol=0.001,
    )
    assert (
        248000 <= torch.sum(predictions[1].segmentation_map.cpu() == 0).item() <= 249000
    )
    assert (
        13700 <= torch.sum(predictions[1].segmentation_map.cpu() == 3).item() <= 13900
    )
