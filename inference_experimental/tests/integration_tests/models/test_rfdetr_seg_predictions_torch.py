import numpy as np
import pytest
import torch


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_stretch_against_numpy_input(
    snake_image_numpy: np.ndarray,
    snakes_rfdetr_seg_torch_stretch_package: str,
) -> None:
    # given
    from inference_exp.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_stretch_package,
    )

    # when
    predictions = model(snake_image_numpy)

    # then
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array([[136, 330, 1279, 554]]), atol=1
    )
    assert 206000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 207000


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_stretch_against_numpy_list_input(
    snake_image_numpy: np.ndarray,
    snakes_rfdetr_seg_torch_stretch_package: str,
) -> None:
    # given
    from inference_exp.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_stretch_package,
    )

    # when
    predictions = model([snake_image_numpy, snake_image_numpy])

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array([[136, 330, 1279, 554]]), atol=1
    )
    assert 206000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 207000
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array([[136, 330, 1279, 554]]), atol=1
    )
    assert 206000 <= np.sum(predictions[1].mask.cpu().numpy()) <= 207000


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_stretch_against_torch_input(
    snake_image_torch: torch.Tensor,
    snakes_rfdetr_seg_torch_stretch_package: str,
) -> None:
    # given
    from inference_exp.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_stretch_package,
    )

    # when
    predictions = model(snake_image_torch)

    # then
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array([[128, 333, 1259, 560]]), atol=1
    )
    assert 206000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 207000


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_stretch_against_torch_list_input(
    snake_image_torch: torch.Tensor,
    snakes_rfdetr_seg_torch_stretch_package: str,
) -> None:
    # given
    from inference_exp.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_stretch_package,
    )

    # when
    predictions = model([snake_image_torch, snake_image_torch])

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array([[128, 333, 1259, 560]]), atol=1
    )
    assert 206000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 207000
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array([[128, 333, 1259, 560]]), atol=1
    )
    assert 206000 <= np.sum(predictions[1].mask.cpu().numpy()) <= 207000


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_stretch_against_torch_batch_input(
    snake_image_torch: torch.Tensor,
    snakes_rfdetr_seg_torch_stretch_package: str,
) -> None:
    # given
    from inference_exp.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_stretch_package,
    )

    # when
    predictions = model(torch.stack([snake_image_torch, snake_image_torch], dim=0))

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array([[128, 333, 1259, 560]]), atol=1
    )
    assert 206000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 207000
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array([[128, 333, 1259, 560]]), atol=1
    )
    assert 206000 <= np.sum(predictions[1].mask.cpu().numpy()) <= 207000


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_letterbox_against_numpy_input(
    snake_image_numpy: np.ndarray,
    snakes_rfdetr_seg_torch_letterbox_package: str,
) -> None:
    # given
    from inference_exp.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_letterbox_package,
    )

    # when
    predictions = model(snake_image_numpy)

    # then
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array([[137, 330, 1270, 560]]), atol=1
    )
    assert 204000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 205000


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_letterbox_against_numpy_list_input(
    snake_image_numpy: np.ndarray,
    snakes_rfdetr_seg_torch_letterbox_package: str,
) -> None:
    # given
    from inference_exp.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_letterbox_package,
    )

    # when
    predictions = model([snake_image_numpy, snake_image_numpy])

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array([[137, 330, 1270, 560]]), atol=1
    )
    assert 204000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 205000
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array([[137, 330, 1270, 560]]), atol=1
    )
    assert 204000 <= np.sum(predictions[1].mask.cpu().numpy()) <= 205000


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_letterbox_against_torch_input(
    snake_image_torch: torch.Tensor,
    snakes_rfdetr_seg_torch_letterbox_package: str,
) -> None:
    # given
    from inference_exp.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_letterbox_package,
    )

    # when
    predictions = model(snake_image_torch)

    # then
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array([[137, 328, 1273, 560]]), atol=1
    )
    assert 204000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 205000


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_letterbox_against_torch_list_input(
    snake_image_torch: torch.Tensor,
    snakes_rfdetr_seg_torch_letterbox_package: str,
) -> None:
    # given
    from inference_exp.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_letterbox_package,
    )

    # when
    predictions = model([snake_image_torch, snake_image_torch])

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array([[137, 328, 1273, 560]]), atol=1
    )
    assert 204000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 205000
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array([[137, 328, 1273, 560]]), atol=1
    )
    assert 204000 <= np.sum(predictions[1].mask.cpu().numpy()) <= 205000


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_letterbox_against_torch_batch_input(
    snake_image_torch: torch.Tensor,
    snakes_rfdetr_seg_torch_letterbox_package: str,
) -> None:
    # given
    from inference_exp.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_letterbox_package,
    )

    # when
    predictions = model(torch.stack([snake_image_torch, snake_image_torch], dim=0))

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array([[137, 328, 1273, 560]]), atol=1
    )
    assert 204000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 205000
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array([[137, 328, 1273, 560]]), atol=1
    )
    assert 204000 <= np.sum(predictions[1].mask.cpu().numpy()) <= 205000


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_center_crop_against_numpy_input(
    snake_image_numpy: np.ndarray,
    snakes_rfdetr_seg_torch_center_crop_package: str,
) -> None:
    # given
    from inference_exp.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_center_crop_package,
    )

    # when
    predictions = model(snake_image_numpy)

    # then
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array([[423, 332, 857, 558]]), atol=1
    )
    assert 82000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 83000


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_center_crop_against_numpy_list_input(
    snake_image_numpy: np.ndarray,
    snakes_rfdetr_seg_torch_center_crop_package: str,
) -> None:
    # given
    from inference_exp.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_center_crop_package,
    )

    # when
    predictions = model([snake_image_numpy, snake_image_numpy])

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array([[423, 332, 857, 558]]), atol=1
    )
    assert 82000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 83000
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array([[423, 332, 857, 558]]), atol=1
    )
    assert 82000 <= np.sum(predictions[1].mask.cpu().numpy()) <= 83000


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_center_crop_against_torch_input(
    snake_image_torch: torch.Tensor,
    snakes_rfdetr_seg_torch_center_crop_package: str,
) -> None:
    # given
    from inference_exp.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_center_crop_package,
    )

    # when
    predictions = model(snake_image_torch)

    # then
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array([[423, 332, 857, 558]]), atol=1
    )
    assert 82000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 83000


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_center_crop_against_torch_list_input(
    snake_image_torch: torch.Tensor,
    snakes_rfdetr_seg_torch_center_crop_package: str,
) -> None:
    # given
    from inference_exp.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_center_crop_package,
    )

    # when
    predictions = model([snake_image_torch, snake_image_torch])

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array([[423, 332, 857, 558]]), atol=1
    )
    assert 82000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 83000
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array([[423, 332, 857, 558]]), atol=1
    )
    assert 82000 <= np.sum(predictions[1].mask.cpu().numpy()) <= 83000


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_center_crop_against_torch_batch_input(
    snake_image_torch: torch.Tensor,
    snakes_rfdetr_seg_torch_center_crop_package: str,
) -> None:
    # given
    from inference_exp.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_center_crop_package,
    )

    # when
    predictions = model(torch.stack([snake_image_torch, snake_image_torch], dim=0))

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array([[423, 332, 857, 558]]), atol=1
    )
    assert 82000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 83000
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array([[423, 332, 857, 558]]), atol=1
    )
    assert 82000 <= np.sum(predictions[1].mask.cpu().numpy()) <= 83000


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_static_crop_stretch_against_numpy_input(
    snake_image_numpy: np.ndarray,
    snakes_rfdetr_seg_torch_static_crop_stretch_package: str,
) -> None:
    # given
    from inference_exp.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_static_crop_stretch_package,
    )

    # when
    predictions = model(snake_image_numpy)

    # then
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array([[322, 330, 959, 561]]), atol=1
    )
    assert 121000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 122000


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_static_crop_stretch_against_numpy_list_input(
    snake_image_numpy: np.ndarray,
    snakes_rfdetr_seg_torch_static_crop_stretch_package: str,
) -> None:
    # given
    from inference_exp.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_static_crop_stretch_package,
    )

    # when
    predictions = model([snake_image_numpy, snake_image_numpy])

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array([[322, 330, 959, 561]]), atol=1
    )
    assert 121000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 122000
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array([[322, 330, 959, 561]]), atol=1
    )
    assert 121000 <= np.sum(predictions[1].mask.cpu().numpy()) <= 122000


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_static_crop_stretch_against_torch_input(
    snake_image_torch: torch.Tensor,
    snakes_rfdetr_seg_torch_static_crop_stretch_package: str,
) -> None:
    # given
    from inference_exp.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_static_crop_stretch_package,
    )

    # when
    predictions = model(snake_image_torch)

    # then
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array([[319, 332, 962, 561]]), atol=1
    )
    assert 120000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 122000


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_static_crop_stretch_against_torch_list_input(
    snake_image_torch: torch.Tensor,
    snakes_rfdetr_seg_torch_static_crop_stretch_package: str,
) -> None:
    # given
    from inference_exp.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_static_crop_stretch_package,
    )

    # when
    predictions = model([snake_image_torch, snake_image_torch])

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array([[319, 332, 962, 561]]), atol=1
    )
    assert 120000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 122000
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array([[319, 332, 962, 561]]), atol=1
    )
    assert 120000 <= np.sum(predictions[1].mask.cpu().numpy()) <= 122000


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_static_crop_stretch_against_torch_batch_input(
    snake_image_torch: torch.Tensor,
    snakes_rfdetr_seg_torch_static_crop_stretch_package: str,
) -> None:
    # given
    from inference_exp.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_static_crop_stretch_package,
    )

    # when
    predictions = model(torch.stack([snake_image_torch, snake_image_torch], dim=0))

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array([[319, 332, 962, 561]]), atol=1
    )
    assert 120000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 122000
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array([[319, 332, 962, 561]]), atol=1
    )
    assert 120000 <= np.sum(predictions[1].mask.cpu().numpy()) <= 122000


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_static_crop_letterbox_against_numpy_input(
    snake_image_numpy: np.ndarray,
    snakes_rfdetr_seg_torch_static_crop_letterbox_package: str,
) -> None:
    # given
    from inference_exp.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_static_crop_letterbox_package,
    )

    # when
    predictions = model(snake_image_numpy)

    # then
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array([[320, 329, 962, 559]]), atol=1
    )
    assert 119000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 120000


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_static_crop_letterbox_against_numpy_list_input(
    snake_image_numpy: np.ndarray,
    snakes_rfdetr_seg_torch_static_crop_letterbox_package: str,
) -> None:
    # given
    from inference_exp.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_static_crop_letterbox_package,
    )

    # when
    predictions = model([snake_image_numpy, snake_image_numpy])

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array([[320, 329, 962, 559]]), atol=1
    )
    assert 119000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 120000
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array([[320, 329, 962, 559]]), atol=1
    )
    assert 119000 <= np.sum(predictions[1].mask.cpu().numpy()) <= 120000


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_static_crop_letterbox_against_torch_input(
    snake_image_torch: torch.Tensor,
    snakes_rfdetr_seg_torch_static_crop_letterbox_package: str,
) -> None:
    # given
    from inference_exp.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_static_crop_letterbox_package,
    )

    # when
    predictions = model(snake_image_torch)

    # then
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array([[317, 330, 961, 557]]), atol=1
    )
    assert 119000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 120000


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_static_crop_letterbox_against_torch_list_input(
    snake_image_torch: torch.Tensor,
    snakes_rfdetr_seg_torch_static_crop_letterbox_package: str,
) -> None:
    # given
    from inference_exp.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_static_crop_letterbox_package,
    )

    # when
    predictions = model([snake_image_torch, snake_image_torch])

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array([[317, 330, 961, 557]]), atol=1
    )
    assert 119000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 120000
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array([[317, 330, 961, 557]]), atol=1
    )
    assert 119000 <= np.sum(predictions[1].mask.cpu().numpy()) <= 120000


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_static_crop_letterbox_against_torch_batch_input(
    snake_image_torch: torch.Tensor,
    snakes_rfdetr_seg_torch_static_crop_letterbox_package: str,
) -> None:
    # given
    from inference_exp.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_static_crop_letterbox_package,
    )

    # when
    predictions = model(torch.stack([snake_image_torch, snake_image_torch], dim=0))

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array([[317, 330, 961, 557]]), atol=1
    )
    assert 119000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 120000
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array([[317, 330, 961, 557]]), atol=1
    )
    assert 119000 <= np.sum(predictions[1].mask.cpu().numpy()) <= 120000


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_static_crop_center_crop_against_numpy_input(
    snake_image_numpy: np.ndarray,
    snakes_rfdetr_seg_torch_static_crop_center_crop_package: str,
) -> None:
    # given
    from inference_exp.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_static_crop_center_crop_package,
    )

    # when
    predictions = model(snake_image_numpy)

    # then
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array([[320, 329, 962, 559]]), atol=1
    )
    assert 119000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 120000


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_static_crop_center_crop_against_numpy_input(
    snake_image_numpy: np.ndarray,
    snakes_rfdetr_seg_torch_static_crop_center_crop_package: str,
) -> None:
    # given
    from inference_exp.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_static_crop_center_crop_package,
    )

    # when
    predictions = model(snake_image_numpy)

    # then
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array([[427, 331, 852, 552]]), atol=1
    )
    assert 80000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 81000


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_static_crop_center_crop_against_numpy_list_input(
    snake_image_numpy: np.ndarray,
    snakes_rfdetr_seg_torch_static_crop_center_crop_package: str,
) -> None:
    # given
    from inference_exp.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_static_crop_center_crop_package,
    )

    # when
    predictions = model([snake_image_numpy, snake_image_numpy])

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array([[427, 331, 852, 552]]), atol=1
    )
    assert 80000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 81000
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array([[427, 331, 852, 552]]), atol=1
    )
    assert 80000 <= np.sum(predictions[1].mask.cpu().numpy()) <= 81000


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_static_crop_center_crop_against_torch_input(
    snake_image_torch: torch.Tensor,
    snakes_rfdetr_seg_torch_static_crop_center_crop_package: str,
) -> None:
    # given
    from inference_exp.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_static_crop_center_crop_package,
    )

    # when
    predictions = model(snake_image_torch)

    # then
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array([[427, 331, 852, 552]]), atol=1
    )
    assert 80000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 81000


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_static_crop_center_crop_against_torch_list_input(
    snake_image_torch: torch.Tensor,
    snakes_rfdetr_seg_torch_static_crop_center_crop_package: str,
) -> None:
    # given
    from inference_exp.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_static_crop_center_crop_package,
    )

    # when
    predictions = model([snake_image_torch, snake_image_torch])

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array([[427, 331, 852, 552]]), atol=1
    )
    assert 80000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 81000
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array([[427, 331, 852, 552]]), atol=1
    )
    assert 80000 <= np.sum(predictions[1].mask.cpu().numpy()) <= 81000


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_static_crop_center_crop_against_torch_batch_input(
    snake_image_torch: torch.Tensor,
    snakes_rfdetr_seg_torch_static_crop_center_crop_package: str,
) -> None:
    # given
    from inference_exp.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_static_crop_center_crop_package,
    )

    # when
    predictions = model(torch.stack([snake_image_torch, snake_image_torch], dim=0))

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array([[427, 331, 852, 552]]), atol=1
    )
    assert 80000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 81000
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array([[427, 331, 852, 552]]), atol=1
    )
    assert 80000 <= np.sum(predictions[1].mask.cpu().numpy()) <= 81000
