import numpy as np
import pytest
import torch

from inference_models.models.common.rle_utils import coco_rle_masks_to_torch_mask


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_stretch_against_numpy_input(
    snake_image_numpy: np.ndarray,
    snakes_rfdetr_seg_torch_stretch_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_stretch_package,
    )

    # when
    predictions = model(snake_image_numpy, confidence=0.5)

    # then
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array(
        [
            [144, 337, 1265, 560],
        ],
    ), atol=1
    )
    assert 205614 <= np.sum(predictions[0].mask.cpu().numpy()) <= 206614


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_stretch_against_numpy_input_rle_variant(
    snake_image_numpy: np.ndarray,
    snakes_rfdetr_seg_torch_stretch_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_stretch_package,
    )

    # when
    predictions = model(snake_image_numpy, confidence=0.5, mask_format="rle")
    predictions_ref = model(snake_image_numpy, confidence=0.5, mask_format="dense")
    decoded_mask = coco_rle_masks_to_torch_mask(
        instances_masks=predictions[0].mask, device=torch.device("cpu")
    )

    # then
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array(
        [
            [144, 337, 1265, 560],
        ],
    ), atol=1
    )
    assert 205614 <= np.sum(decoded_mask.cpu().numpy()) <= 206614
    assert np.allclose(
        decoded_mask.cpu().numpy(), predictions_ref[0].mask.cpu().numpy()
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_stretch_against_numpy_list_input(
    snake_image_numpy: np.ndarray,
    snakes_rfdetr_seg_torch_stretch_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_stretch_package,
    )

    # when
    predictions = model([snake_image_numpy, snake_image_numpy], confidence=0.5)

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array(
        [
            [144, 337, 1265, 560],
        ],
    ), atol=1
    )
    assert 205614 <= np.sum(predictions[0].mask.cpu().numpy()) <= 206614
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array(
        [
            [144, 337, 1265, 560],
        ],
    ), atol=1
    )
    assert 205614 <= np.sum(predictions[1].mask.cpu().numpy()) <= 206614


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_stretch_against_numpy_list_input_rle_variant(
    snake_image_numpy: np.ndarray,
    snakes_rfdetr_seg_torch_stretch_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_stretch_package,
    )

    # when
    predictions = model(
        [snake_image_numpy, snake_image_numpy], confidence=0.5, mask_format="rle"
    )
    decoded_mask_1 = coco_rle_masks_to_torch_mask(
        instances_masks=predictions[0].mask, device=torch.device("cpu")
    )
    decoded_mask_2 = coco_rle_masks_to_torch_mask(
        instances_masks=predictions[1].mask, device=torch.device("cpu")
    )
    predictions_ref = model(
        [snake_image_numpy, snake_image_numpy], confidence=0.5, mask_format="dense"
    )

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array(
        [
            [144, 337, 1265, 560],
        ],
    ), atol=1
    )
    assert 205614 <= np.sum(decoded_mask_1.cpu().numpy()) <= 206614
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array(
        [
            [144, 337, 1265, 560],
        ],
    ), atol=1
    )
    assert 205614 <= np.sum(decoded_mask_2.cpu().numpy()) <= 206614
    assert np.allclose(
        decoded_mask_1.cpu().numpy(), predictions_ref[0].mask.cpu().numpy()
    )
    assert np.allclose(
        decoded_mask_2.cpu().numpy(), predictions_ref[1].mask.cpu().numpy()
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_stretch_against_torch_input(
    snake_image_torch: torch.Tensor,
    snakes_rfdetr_seg_torch_stretch_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_stretch_package,
    )

    # when
    predictions = model(snake_image_torch, confidence=0.5)

    # then
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array(
        [
            [127, 332, 1254, 561],
        ],
    ), atol=20
    )
    assert 205555 <= np.sum(predictions[0].mask.cpu().numpy()) <= 206555


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_stretch_against_torch_list_input(
    snake_image_torch: torch.Tensor,
    snakes_rfdetr_seg_torch_stretch_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_stretch_package,
    )

    # when
    predictions = model([snake_image_torch, snake_image_torch], confidence=0.5)

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array(
        [
            [127, 332, 1254, 561],
        ],
    ), atol=20
    )
    assert 205555 <= np.sum(predictions[0].mask.cpu().numpy()) <= 206555
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array(
        [
            [127, 332, 1254, 561],
        ],
    ), atol=20
    )
    assert 205555 <= np.sum(predictions[1].mask.cpu().numpy()) <= 206555


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_stretch_against_torch_batch_input(
    snake_image_torch: torch.Tensor,
    snakes_rfdetr_seg_torch_stretch_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_stretch_package,
    )

    # when
    predictions = model(
        torch.stack([snake_image_torch, snake_image_torch], dim=0),
        confidence=0.5,
    )

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array(
        [
            [127, 332, 1254, 561],
        ],
    ), atol=20
    )
    assert 205555 <= np.sum(predictions[0].mask.cpu().numpy()) <= 206555
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array(
        [
            [127, 332, 1254, 561],
        ],
    ), atol=20
    )
    assert 205555 <= np.sum(predictions[1].mask.cpu().numpy()) <= 206555


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_letterbox_against_numpy_input(
    snake_image_numpy: np.ndarray,
    snakes_rfdetr_seg_torch_letterbox_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_letterbox_package,
    )

    # when
    predictions = model(snake_image_numpy, confidence=0.5)

    # then
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array(
        [
            [126, 331, 1265, 560],
        ],
    ), atol=1
    )
    assert 206196 <= np.sum(predictions[0].mask.cpu().numpy()) <= 207196


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_letterbox_against_numpy_list_input(
    snake_image_numpy: np.ndarray,
    snakes_rfdetr_seg_torch_letterbox_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_letterbox_package,
    )

    # when
    predictions = model([snake_image_numpy, snake_image_numpy], confidence=0.5)

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array(
        [
            [126, 331, 1265, 560],
        ],
    ), atol=1
    )
    assert 206196 <= np.sum(predictions[0].mask.cpu().numpy()) <= 207196
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array(
        [
            [126, 331, 1265, 560],
        ],
    ), atol=1
    )
    assert 206196 <= np.sum(predictions[1].mask.cpu().numpy()) <= 207196


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_letterbox_against_torch_input(
    snake_image_torch: torch.Tensor,
    snakes_rfdetr_seg_torch_letterbox_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_letterbox_package,
    )

    # when
    predictions = model(snake_image_torch, confidence=0.5)

    # then
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array(
        [
            [133, 330, 1277, 557],
        ],
    ), atol=5
    )
    assert 207193 <= np.sum(predictions[0].mask.cpu().numpy()) <= 208193


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_letterbox_against_torch_list_input(
    snake_image_torch: torch.Tensor,
    snakes_rfdetr_seg_torch_letterbox_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_letterbox_package,
    )

    # when
    predictions = model([snake_image_torch, snake_image_torch], confidence=0.5)

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array(
        [
            [133, 330, 1277, 557],
        ],
    ), atol=5
    )
    assert 207193 <= np.sum(predictions[0].mask.cpu().numpy()) <= 208193
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array(
        [
            [133, 330, 1277, 557],
        ],
    ), atol=5
    )
    assert 207193 <= np.sum(predictions[1].mask.cpu().numpy()) <= 208193


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_letterbox_against_torch_batch_input(
    snake_image_torch: torch.Tensor,
    snakes_rfdetr_seg_torch_letterbox_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_letterbox_package,
    )

    # when
    predictions = model(
        torch.stack([snake_image_torch, snake_image_torch], dim=0),
        confidence=0.5,
    )

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array(
        [
            [133, 330, 1277, 557],
        ],
    ), atol=5
    )
    assert 207193 <= np.sum(predictions[0].mask.cpu().numpy()) <= 208193
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array(
        [
            [133, 330, 1277, 557],
        ],
    ), atol=5
    )
    assert 207193 <= np.sum(predictions[1].mask.cpu().numpy()) <= 208193


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_center_crop_against_numpy_input(
    snake_image_numpy: np.ndarray,
    snakes_rfdetr_seg_torch_center_crop_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_center_crop_package,
    )

    # when
    predictions = model(snake_image_numpy, confidence=0.5)

    # then
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array(
        [
            [130, 334, 1264, 557],
        ],
    ), atol=1
    )
    assert 205279 <= np.sum(predictions[0].mask.cpu().numpy()) <= 206279


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_center_crop_against_numpy_list_input(
    snake_image_numpy: np.ndarray,
    snakes_rfdetr_seg_torch_center_crop_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_center_crop_package,
    )

    # when
    predictions = model([snake_image_numpy, snake_image_numpy], confidence=0.5)

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array(
        [
            [130, 334, 1264, 557],
        ],
    ), atol=1
    )
    assert 205279 <= np.sum(predictions[0].mask.cpu().numpy()) <= 206279
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array(
        [
            [130, 334, 1264, 557],
        ],
    ), atol=1
    )
    assert 205279 <= np.sum(predictions[1].mask.cpu().numpy()) <= 206279


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_center_crop_against_torch_input(
    snake_image_torch: torch.Tensor,
    snakes_rfdetr_seg_torch_center_crop_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_center_crop_package,
    )

    # when
    predictions = model(snake_image_torch, confidence=0.5)

    # then
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array(
        [
            [123, 333, 1281, 558],
        ],
    ), atol=1
    )
    assert 205692 <= np.sum(predictions[0].mask.cpu().numpy()) <= 206692


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_center_crop_against_torch_list_input(
    snake_image_torch: torch.Tensor,
    snakes_rfdetr_seg_torch_center_crop_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_center_crop_package,
    )

    # when
    predictions = model([snake_image_torch, snake_image_torch], confidence=0.5)

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array(
        [
            [123, 333, 1281, 558],
        ],
    ), atol=1
    )
    assert 205692 <= np.sum(predictions[0].mask.cpu().numpy()) <= 206692
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array(
        [
            [123, 333, 1281, 558],
        ],
    ), atol=1
    )
    assert 205692 <= np.sum(predictions[1].mask.cpu().numpy()) <= 206692


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_center_crop_against_torch_batch_input(
    snake_image_torch: torch.Tensor,
    snakes_rfdetr_seg_torch_center_crop_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_center_crop_package,
    )

    # when
    predictions = model(
        torch.stack([snake_image_torch, snake_image_torch], dim=0), confidence=0.5
    )

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array(
        [
            [123, 333, 1281, 558],
        ],
    ), atol=1
    )
    assert 205692 <= np.sum(predictions[0].mask.cpu().numpy()) <= 206692
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array(
        [
            [123, 333, 1281, 558],
        ],
    ), atol=1
    )
    assert 205692 <= np.sum(predictions[1].mask.cpu().numpy()) <= 206692


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_static_crop_stretch_against_numpy_input(
    snake_image_numpy: np.ndarray,
    snakes_rfdetr_seg_torch_static_crop_stretch_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_static_crop_stretch_package,
    )

    # when
    predictions = model(snake_image_numpy, confidence=0.5)

    # then
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array(
        [
            [321, 331, 962, 561],
        ],
    ), atol=1
    )
    assert 120347 <= np.sum(predictions[0].mask.cpu().numpy()) <= 121347


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_static_crop_stretch_against_numpy_list_input(
    snake_image_numpy: np.ndarray,
    snakes_rfdetr_seg_torch_static_crop_stretch_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_static_crop_stretch_package,
    )

    # when
    predictions = model([snake_image_numpy, snake_image_numpy], confidence=0.5)

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array(
        [
            [321, 331, 962, 561],
        ],
    ), atol=1
    )
    assert 120347 <= np.sum(predictions[0].mask.cpu().numpy()) <= 121347
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array(
        [
            [321, 331, 962, 561],
        ],
    ), atol=1
    )
    assert 120347 <= np.sum(predictions[1].mask.cpu().numpy()) <= 121347


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_static_crop_stretch_against_torch_input(
    snake_image_torch: torch.Tensor,
    snakes_rfdetr_seg_torch_static_crop_stretch_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_static_crop_stretch_package,
    )

    # when
    predictions = model(snake_image_torch, confidence=0.5)

    # then
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array(
        [
            [320, 331, 961, 560],
        ],
    ), atol=1
    )
    assert 120534 <= np.sum(predictions[0].mask.cpu().numpy()) <= 121534


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_static_crop_stretch_against_torch_list_input(
    snake_image_torch: torch.Tensor,
    snakes_rfdetr_seg_torch_static_crop_stretch_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_static_crop_stretch_package,
    )

    # when
    predictions = model([snake_image_torch, snake_image_torch], confidence=0.5)

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array(
        [
            [320, 331, 961, 560],
        ],
    ), atol=1
    )
    assert 120534 <= np.sum(predictions[0].mask.cpu().numpy()) <= 121534
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array(
        [
            [320, 331, 961, 560],
        ],
    ), atol=1
    )
    assert 120534 <= np.sum(predictions[1].mask.cpu().numpy()) <= 121534


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_static_crop_stretch_against_torch_batch_input(
    snake_image_torch: torch.Tensor,
    snakes_rfdetr_seg_torch_static_crop_stretch_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_static_crop_stretch_package,
    )

    # when
    predictions = model(
        torch.stack([snake_image_torch, snake_image_torch], dim=0),
        confidence=0.5,
    )

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array(
        [
            [320, 331, 961, 560],
        ],
    ), atol=1
    )
    assert 120534 <= np.sum(predictions[0].mask.cpu().numpy()) <= 121534
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array(
        [
            [320, 331, 961, 560],
        ],
    ), atol=1
    )
    assert 120534 <= np.sum(predictions[1].mask.cpu().numpy()) <= 121534


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_static_crop_letterbox_against_numpy_input(
    snake_image_numpy: np.ndarray,
    snakes_rfdetr_seg_torch_static_crop_letterbox_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_static_crop_letterbox_package,
    )

    # when
    predictions = model(snake_image_numpy, confidence=0.5)

    # then
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array(
        [
            [320, 331, 962, 558],
        ],
    ), atol=1
    )
    assert 119874 <= np.sum(predictions[0].mask.cpu().numpy()) <= 120874


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_static_crop_letterbox_against_numpy_list_input(
    snake_image_numpy: np.ndarray,
    snakes_rfdetr_seg_torch_static_crop_letterbox_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_static_crop_letterbox_package,
    )

    # when
    predictions = model([snake_image_numpy, snake_image_numpy], confidence=0.5)

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array(
        [
            [320, 331, 962, 558],
        ],
    ), atol=1
    )
    assert 119874 <= np.sum(predictions[0].mask.cpu().numpy()) <= 120874
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array(
        [
            [320, 331, 962, 558],
        ],
    ), atol=1
    )
    assert 119874 <= np.sum(predictions[1].mask.cpu().numpy()) <= 120874


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_static_crop_letterbox_against_torch_input(
    snake_image_torch: torch.Tensor,
    snakes_rfdetr_seg_torch_static_crop_letterbox_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_static_crop_letterbox_package,
    )

    # when
    predictions = model(snake_image_torch, confidence=0.5)

    # then
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array(
        [
            [322, 333, 959, 559],
        ],
    ), atol=1
    )
    assert 120243 <= np.sum(predictions[0].mask.cpu().numpy()) <= 121243


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_static_crop_letterbox_against_torch_list_input(
    snake_image_torch: torch.Tensor,
    snakes_rfdetr_seg_torch_static_crop_letterbox_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_static_crop_letterbox_package,
    )

    # when
    predictions = model([snake_image_torch, snake_image_torch], confidence=0.5)

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array(
        [
            [322, 333, 959, 559],
        ],
    ), atol=1
    )
    assert 120243 <= np.sum(predictions[0].mask.cpu().numpy()) <= 121243
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array(
        [
            [322, 333, 959, 559],
        ],
    ), atol=1
    )
    assert 120243 <= np.sum(predictions[1].mask.cpu().numpy()) <= 121243


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_static_crop_letterbox_against_torch_batch_input(
    snake_image_torch: torch.Tensor,
    snakes_rfdetr_seg_torch_static_crop_letterbox_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_static_crop_letterbox_package,
    )

    # when
    predictions = model(
        torch.stack([snake_image_torch, snake_image_torch], dim=0),
        confidence=0.5,
    )

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array(
        [
            [322, 333, 959, 559],
        ],
    ), atol=1
    )
    assert 120243 <= np.sum(predictions[0].mask.cpu().numpy()) <= 121243
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array(
        [
            [322, 333, 959, 559],
        ],
    ), atol=1
    )
    assert 120243 <= np.sum(predictions[1].mask.cpu().numpy()) <= 121243


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_static_crop_center_crop_against_numpy_input(
    snake_image_numpy: np.ndarray,
    snakes_rfdetr_seg_torch_static_crop_center_crop_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_static_crop_center_crop_package,
    )

    # when
    predictions = model(snake_image_numpy, confidence=0.5)

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
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_static_crop_center_crop_package,
    )

    # when
    predictions = model(snake_image_numpy, confidence=0.5)

    # then
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array(
        [
            [321, 330, 956, 559],
        ],
    ), atol=1
    )
    assert 119330 <= np.sum(predictions[0].mask.cpu().numpy()) <= 120330


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_static_crop_center_crop_against_numpy_list_input(
    snake_image_numpy: np.ndarray,
    snakes_rfdetr_seg_torch_static_crop_center_crop_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_static_crop_center_crop_package,
    )

    # when
    predictions = model([snake_image_numpy, snake_image_numpy], confidence=0.5)

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array(
        [
            [321, 330, 956, 559],
        ],
    ), atol=1
    )
    assert 119330 <= np.sum(predictions[0].mask.cpu().numpy()) <= 120330
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array(
        [
            [321, 330, 956, 559],
        ],
    ), atol=1
    )
    assert 119330 <= np.sum(predictions[1].mask.cpu().numpy()) <= 120330


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_static_crop_center_crop_against_torch_input(
    snake_image_torch: torch.Tensor,
    snakes_rfdetr_seg_torch_static_crop_center_crop_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_static_crop_center_crop_package,
    )

    # when
    predictions = model(snake_image_torch, confidence=0.5)

    # then
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array(
        [
            [326, 329, 958, 558],
        ],
    ), atol=1
    )
    assert 119616 <= np.sum(predictions[0].mask.cpu().numpy()) <= 120616


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_static_crop_center_crop_against_torch_list_input(
    snake_image_torch: torch.Tensor,
    snakes_rfdetr_seg_torch_static_crop_center_crop_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_static_crop_center_crop_package,
    )

    # when
    predictions = model([snake_image_torch, snake_image_torch], confidence=0.5)

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array(
        [
            [326, 329, 958, 558],
        ],
    ), atol=1
    )
    assert 119616 <= np.sum(predictions[0].mask.cpu().numpy()) <= 120616
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array(
        [
            [326, 329, 958, 558],
        ],
    ), atol=1
    )
    assert 119616 <= np.sum(predictions[1].mask.cpu().numpy()) <= 120616


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_static_crop_center_crop_against_torch_batch_input(
    snake_image_torch: torch.Tensor,
    snakes_rfdetr_seg_torch_static_crop_center_crop_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_static_crop_center_crop_package,
    )

    # when
    predictions = model(
        torch.stack([snake_image_torch, snake_image_torch], dim=0),
        confidence=0.5,
    )

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array(
        [
            [326, 329, 958, 558],
        ],
    ), atol=1
    )
    assert 119616 <= np.sum(predictions[0].mask.cpu().numpy()) <= 120616
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array(
        [
            [326, 329, 958, 558],
        ],
    ), atol=1
    )
    assert 119616 <= np.sum(predictions[1].mask.cpu().numpy()) <= 120616


_NONSQUARE_LETTERBOX_SEG_TORCH_EXPECTED_XYXY_NUMPY = np.array([[125, 312, 1265, 528]])
_NONSQUARE_LETTERBOX_SEG_TORCH_EXPECTED_XYXY_TORCH = np.array([[126, 312, 1266, 531]])
_NONSQUARE_LETTERBOX_SEG_TORCH_EXPECTED_MASK_SUM_NUMPY = 212536
_NONSQUARE_LETTERBOX_SEG_TORCH_EXPECTED_MASK_SUM_TORCH = 211146


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_nonsquare_letterbox_against_numpy_input(
    snake_image_numpy: np.ndarray,
    snakes_rfdetr_seg_torch_static_bs_nonsquare_letterbox_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_static_bs_nonsquare_letterbox_package,
    )

    # when
    predictions = model(snake_image_numpy, confidence=0.5)

    # then
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(),
        torch.tensor(
        [
            [121, 312, 1263, 532],
        ],
        dtype=torch.int32,
    ),
        atol=2,
    )
    assert (
        _NONSQUARE_LETTERBOX_SEG_TORCH_EXPECTED_MASK_SUM_NUMPY - 500
        <= np.sum(predictions[0].mask.cpu().numpy())
        <= _NONSQUARE_LETTERBOX_SEG_TORCH_EXPECTED_MASK_SUM_NUMPY + 500
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_nonsquare_letterbox_against_numpy_list_input(
    snake_image_numpy: np.ndarray,
    snakes_rfdetr_seg_torch_static_bs_nonsquare_letterbox_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_static_bs_nonsquare_letterbox_package,
    )

    # when
    predictions = model([snake_image_numpy, snake_image_numpy], confidence=0.5)

    # then
    assert len(predictions) == 2
    for pred in predictions:
        assert np.allclose(
            pred.xyxy.cpu().numpy(),
            torch.tensor(
        [
            [121, 312, 1263, 532],
        ],
        dtype=torch.int32,
    ),
            atol=2,
        )
        assert (
            _NONSQUARE_LETTERBOX_SEG_TORCH_EXPECTED_MASK_SUM_NUMPY - 500
            <= np.sum(pred.mask.cpu().numpy())
            <= _NONSQUARE_LETTERBOX_SEG_TORCH_EXPECTED_MASK_SUM_NUMPY + 500
        )


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_nonsquare_letterbox_against_torch_input(
    snake_image_torch: torch.Tensor,
    snakes_rfdetr_seg_torch_static_bs_nonsquare_letterbox_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_static_bs_nonsquare_letterbox_package,
    )

    # when
    predictions = model(snake_image_torch, confidence=0.5)

    # then
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(),
        torch.tensor(
        [
            [125, 317, 1262, 536],
        ],
        dtype=torch.int32,
    ),
        atol=2,
    )
    assert (
        _NONSQUARE_LETTERBOX_SEG_TORCH_EXPECTED_MASK_SUM_TORCH - 500
        <= np.sum(predictions[0].mask.cpu().numpy())
        <= _NONSQUARE_LETTERBOX_SEG_TORCH_EXPECTED_MASK_SUM_TORCH + 500
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_nonsquare_letterbox_against_torch_list_input(
    snake_image_torch: torch.Tensor,
    snakes_rfdetr_seg_torch_static_bs_nonsquare_letterbox_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_static_bs_nonsquare_letterbox_package,
    )

    # when
    predictions = model([snake_image_torch, snake_image_torch], confidence=0.5)

    # then
    assert len(predictions) == 2
    for pred in predictions:
        assert np.allclose(
            pred.xyxy.cpu().numpy(),
            torch.tensor(
        [
            [125, 317, 1262, 536],
        ],
        dtype=torch.int32,
    ),
            atol=2,
        )
        assert (
            _NONSQUARE_LETTERBOX_SEG_TORCH_EXPECTED_MASK_SUM_TORCH - 500
            <= np.sum(pred.mask.cpu().numpy())
            <= _NONSQUARE_LETTERBOX_SEG_TORCH_EXPECTED_MASK_SUM_TORCH + 500
        )


@pytest.mark.slow
@pytest.mark.torch_models
def test_package_with_nonsquare_letterbox_against_torch_batch_input(
    snake_image_torch: torch.Tensor,
    snakes_rfdetr_seg_torch_static_bs_nonsquare_letterbox_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_static_bs_nonsquare_letterbox_package,
    )

    # when
    predictions = model(
        torch.stack([snake_image_torch, snake_image_torch], dim=0),
        confidence=0.5,
    )

    # then
    assert len(predictions) == 2
    for pred in predictions:
        assert np.allclose(
            pred.xyxy.cpu().numpy(),
            torch.tensor(
        [
            [125, 317, 1262, 536],
        ],
        dtype=torch.int32,
    ),
            atol=2,
        )
        assert (
            _NONSQUARE_LETTERBOX_SEG_TORCH_EXPECTED_MASK_SUM_TORCH - 500
            <= np.sum(pred.mask.cpu().numpy())
            <= _NONSQUARE_LETTERBOX_SEG_TORCH_EXPECTED_MASK_SUM_TORCH + 500
        )


@pytest.mark.slow
@pytest.mark.torch_models
def test_torch_per_class_confidence_filters_detections(
    snake_image_numpy: np.ndarray,
    snakes_rfdetr_seg_torch_stretch_package: str,
) -> None:
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
        RFDetrForInstanceSegmentationTorch,
    )
    from inference_models.weights_providers.entities import RecommendedParameters

    model = RFDetrForInstanceSegmentationTorch.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_torch_stretch_package,
    )
    model.recommended_parameters = RecommendedParameters(
        confidence=0.3,
        per_class_confidence={name: 1.01 for name in model.class_names},
    )
    predictions = model(snake_image_numpy, confidence="best")
    assert predictions[0].class_id.numel() == 0
