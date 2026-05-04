import numpy as np
import pytest
import torch

from inference_models.models.common.rle_utils import coco_rle_masks_to_torch_mask


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_package_with_stretch_against_numpy_input(
    snake_image_numpy: np.ndarray,
    snakes_rfdetr_seg_onnx_static_bs_stretch_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_onnx import (
        RFDetrForInstanceSegmentationOnnx,
    )

    model = RFDetrForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_onnx_static_bs_stretch_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(snake_image_numpy, confidence=0.5)

    # then
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array(
        [
            [130, 328, 1265, 557],
        ],
    ), atol=1
    )
    assert 205828 <= np.sum(predictions[0].mask.cpu().numpy()) <= 206828


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_package_with_stretch_against_numpy_input_rle_variant(
    snake_image_numpy: np.ndarray,
    snakes_rfdetr_seg_onnx_static_bs_stretch_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_onnx import (
        RFDetrForInstanceSegmentationOnnx,
    )

    model = RFDetrForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_onnx_static_bs_stretch_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(snake_image_numpy, confidence=0.5, mask_format="rle")
    decoded_mask = coco_rle_masks_to_torch_mask(
        instances_masks=predictions[0].mask, device=torch.device("cpu")
    )
    predictions_ref = model(snake_image_numpy, confidence=0.5, mask_format="dense")

    # then
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array(
        [
            [130, 328, 1265, 557],
        ],
    ), atol=1
    )
    assert 205828 <= np.sum(decoded_mask.cpu().numpy()) <= 206828
    assert np.allclose(
        decoded_mask.cpu().numpy(), predictions_ref[0].mask.cpu().numpy()
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_package_with_stretch_against_numpy_list_input(
    snake_image_numpy: np.ndarray,
    snakes_rfdetr_seg_onnx_static_bs_stretch_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_onnx import (
        RFDetrForInstanceSegmentationOnnx,
    )

    model = RFDetrForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_onnx_static_bs_stretch_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([snake_image_numpy, snake_image_numpy], confidence=0.5)

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array(
        [
            [130, 328, 1265, 557],
        ],
    ), atol=1
    )
    assert 205828 <= np.sum(predictions[0].mask.cpu().numpy()) <= 206828
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array(
        [
            [130, 328, 1265, 557],
        ],
    ), atol=1
    )
    assert 205828 <= np.sum(predictions[1].mask.cpu().numpy()) <= 206828


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_package_with_stretch_against_numpy_list_input_rle_variant(
    snake_image_numpy: np.ndarray,
    snakes_rfdetr_seg_onnx_static_bs_stretch_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_onnx import (
        RFDetrForInstanceSegmentationOnnx,
    )

    model = RFDetrForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_onnx_static_bs_stretch_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(
        [snake_image_numpy, snake_image_numpy], confidence=0.5, mask_format="rle"
    )
    predictions_ref = model([snake_image_numpy, snake_image_numpy], confidence=0.5)
    decoded_mask_1 = coco_rle_masks_to_torch_mask(
        instances_masks=predictions[0].mask, device=torch.device("cpu")
    )
    decoded_mask_2 = coco_rle_masks_to_torch_mask(
        instances_masks=predictions[1].mask, device=torch.device("cpu")
    )

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array(
        [
            [130, 328, 1265, 557],
        ],
    ), atol=1
    )
    assert 205828 <= np.sum(decoded_mask_1.cpu().numpy()) <= 206828
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array(
        [
            [130, 328, 1265, 557],
        ],
    ), atol=1
    )
    assert 205828 <= np.sum(decoded_mask_2.cpu().numpy()) <= 206828
    assert np.allclose(
        decoded_mask_1.cpu().numpy(), predictions_ref[0].mask.cpu().numpy()
    )
    assert np.allclose(
        decoded_mask_2.cpu().numpy(), predictions_ref[1].mask.cpu().numpy()
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_package_with_stretch_against_torch_input(
    snake_image_torch: torch.Tensor,
    snakes_rfdetr_seg_onnx_static_bs_stretch_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_onnx import (
        RFDetrForInstanceSegmentationOnnx,
    )

    model = RFDetrForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_onnx_static_bs_stretch_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(snake_image_torch, confidence=0.5)

    # then
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array(
        [
            [131, 320, 1261, 559],
        ],
    ), atol=5
    )
    assert 205435 <= np.sum(predictions[0].mask.cpu().numpy()) <= 206435


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_package_with_stretch_against_torch_list_input(
    snake_image_torch: torch.Tensor,
    snakes_rfdetr_seg_onnx_static_bs_stretch_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_onnx import (
        RFDetrForInstanceSegmentationOnnx,
    )

    model = RFDetrForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_onnx_static_bs_stretch_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([snake_image_torch, snake_image_torch], confidence=0.5)

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array(
        [
            [131, 320, 1261, 559],
        ],
    ), atol=5
    )
    assert 205435 <= np.sum(predictions[0].mask.cpu().numpy()) <= 206435
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array(
        [
            [131, 320, 1261, 559],
        ],
    ), atol=5
    )
    assert 205435 <= np.sum(predictions[1].mask.cpu().numpy()) <= 206435


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_package_with_stretch_against_torch_batch_input(
    snake_image_torch: torch.Tensor,
    snakes_rfdetr_seg_onnx_static_bs_stretch_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_onnx import (
        RFDetrForInstanceSegmentationOnnx,
    )

    model = RFDetrForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_onnx_static_bs_stretch_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
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
            [131, 320, 1261, 559],
        ],
    ), atol=5
    )
    assert 205435 <= np.sum(predictions[0].mask.cpu().numpy()) <= 206435
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array(
        [
            [131, 320, 1261, 559],
        ],
    ), atol=5
    )
    assert 205435 <= np.sum(predictions[1].mask.cpu().numpy()) <= 206435


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_package_with_letterbox_against_numpy_input(
    snake_image_numpy: np.ndarray,
    snakes_rfdetr_seg_onnx_static_bs_letterbox_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_onnx import (
        RFDetrForInstanceSegmentationOnnx,
    )

    model = RFDetrForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_onnx_static_bs_letterbox_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(snake_image_numpy, confidence=0.5)

    # then
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array(
        [
            [129, 333, 1270, 556],
        ],
    ), atol=1
    )
    assert 205875 <= np.sum(predictions[0].mask.cpu().numpy()) <= 206875


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_package_with_letterbox_against_numpy_list_input(
    snake_image_numpy: np.ndarray,
    snakes_rfdetr_seg_onnx_static_bs_letterbox_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_onnx import (
        RFDetrForInstanceSegmentationOnnx,
    )

    model = RFDetrForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_onnx_static_bs_letterbox_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([snake_image_numpy, snake_image_numpy], confidence=0.5)

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array(
        [
            [129, 333, 1270, 556],
        ],
    ), atol=1
    )
    assert 205875 <= np.sum(predictions[0].mask.cpu().numpy()) <= 206875
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array(
        [
            [129, 333, 1270, 556],
        ],
    ), atol=1
    )
    assert 205875 <= np.sum(predictions[1].mask.cpu().numpy()) <= 206875


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_package_with_letterbox_against_torch_input(
    snake_image_torch: torch.Tensor,
    snakes_rfdetr_seg_onnx_static_bs_letterbox_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_onnx import (
        RFDetrForInstanceSegmentationOnnx,
    )

    model = RFDetrForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_onnx_static_bs_letterbox_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(snake_image_torch, confidence=0.5)

    # then
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array(
        [
            [129, 332, 1274, 556],
        ],
    ), atol=1
    )
    assert 206703 <= np.sum(predictions[0].mask.cpu().numpy()) <= 207703


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_package_with_letterbox_against_torch_list_input(
    snake_image_torch: torch.Tensor,
    snakes_rfdetr_seg_onnx_static_bs_letterbox_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_onnx import (
        RFDetrForInstanceSegmentationOnnx,
    )

    model = RFDetrForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_onnx_static_bs_letterbox_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([snake_image_torch, snake_image_torch], confidence=0.5)

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array(
        [
            [129, 332, 1274, 556],
        ],
    ), atol=1
    )
    assert 206703 <= np.sum(predictions[0].mask.cpu().numpy()) <= 207703
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array(
        [
            [129, 332, 1274, 556],
        ],
    ), atol=1
    )
    assert 206703 <= np.sum(predictions[1].mask.cpu().numpy()) <= 207703


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_package_with_letterbox_against_torch_batch_input(
    snake_image_torch: torch.Tensor,
    snakes_rfdetr_seg_onnx_static_bs_letterbox_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_onnx import (
        RFDetrForInstanceSegmentationOnnx,
    )

    model = RFDetrForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_onnx_static_bs_letterbox_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
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
            [129, 332, 1274, 556],
        ],
    ), atol=1
    )
    assert 206703 <= np.sum(predictions[0].mask.cpu().numpy()) <= 207703
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array(
        [
            [129, 332, 1274, 556],
        ],
    ), atol=1
    )
    assert 206703 <= np.sum(predictions[1].mask.cpu().numpy()) <= 207703


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_package_with_center_crop_against_numpy_input(
    snake_image_numpy: np.ndarray,
    snakes_rfdetr_seg_onnx_static_bs_center_crop_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_onnx import (
        RFDetrForInstanceSegmentationOnnx,
    )

    model = RFDetrForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_onnx_static_bs_center_crop_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(snake_image_numpy, confidence=0.5)

    # then
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array(
        [
            [132, 330, 1264, 560],
        ],
    ), atol=1
    )
    assert 207308 <= np.sum(predictions[0].mask.cpu().numpy()) <= 208308


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_package_with_center_crop_against_numpy_list_input(
    snake_image_numpy: np.ndarray,
    snakes_rfdetr_seg_onnx_static_bs_center_crop_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_onnx import (
        RFDetrForInstanceSegmentationOnnx,
    )

    model = RFDetrForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_onnx_static_bs_center_crop_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([snake_image_numpy, snake_image_numpy], confidence=0.5)

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array(
        [
            [132, 330, 1264, 560],
        ],
    ), atol=1
    )
    assert 207308 <= np.sum(predictions[0].mask.cpu().numpy()) <= 208308
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array(
        [
            [132, 330, 1264, 560],
        ],
    ), atol=1
    )
    assert 207308 <= np.sum(predictions[1].mask.cpu().numpy()) <= 208308


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_package_with_center_crop_against_torch_input(
    snake_image_torch: torch.Tensor,
    snakes_rfdetr_seg_onnx_static_bs_center_crop_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_onnx import (
        RFDetrForInstanceSegmentationOnnx,
    )

    model = RFDetrForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_onnx_static_bs_center_crop_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(snake_image_torch, confidence=0.5)

    # then
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array(
        [
            [140, 330, 1271, 558],
        ],
    ), atol=1
    )
    assert 205984 <= np.sum(predictions[0].mask.cpu().numpy()) <= 206984


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_package_with_center_crop_against_torch_list_input(
    snake_image_torch: torch.Tensor,
    snakes_rfdetr_seg_onnx_static_bs_center_crop_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_onnx import (
        RFDetrForInstanceSegmentationOnnx,
    )

    model = RFDetrForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_onnx_static_bs_center_crop_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([snake_image_torch, snake_image_torch], confidence=0.5)

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array(
        [
            [140, 330, 1271, 558],
        ],
    ), atol=1
    )
    assert 205984 <= np.sum(predictions[0].mask.cpu().numpy()) <= 206984
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array(
        [
            [140, 330, 1271, 558],
        ],
    ), atol=1
    )
    assert 205984 <= np.sum(predictions[1].mask.cpu().numpy()) <= 206984


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_package_with_center_crop_against_torch_batch_input(
    snake_image_torch: torch.Tensor,
    snakes_rfdetr_seg_onnx_static_bs_center_crop_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_onnx import (
        RFDetrForInstanceSegmentationOnnx,
    )

    model = RFDetrForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_onnx_static_bs_center_crop_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
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
            [140, 330, 1271, 558],
        ],
    ), atol=1
    )
    assert 205984 <= np.sum(predictions[0].mask.cpu().numpy()) <= 206984
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array(
        [
            [140, 330, 1271, 558],
        ],
    ), atol=1
    )
    assert 205984 <= np.sum(predictions[1].mask.cpu().numpy()) <= 206984


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_package_with_static_crop_stretch_against_numpy_input(
    snake_image_numpy: np.ndarray,
    snakes_rfdetr_seg_onnx_static_bs_static_crop_stretch_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_onnx import (
        RFDetrForInstanceSegmentationOnnx,
    )

    model = RFDetrForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_onnx_static_bs_static_crop_stretch_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(
        snake_image_numpy,
        confidence=0.5,
    )

    # then
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array([[321, 329, 961, 560]]), atol=1
    )
    assert 120000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 122000


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_package_with_static_crop_stretch_against_numpy_list_input(
    snake_image_numpy: np.ndarray,
    snakes_rfdetr_seg_onnx_static_bs_static_crop_stretch_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_onnx import (
        RFDetrForInstanceSegmentationOnnx,
    )

    model = RFDetrForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_onnx_static_bs_static_crop_stretch_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(
        [snake_image_numpy, snake_image_numpy],
        confidence=0.5,
    )

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array(
        [
            [320, 332, 957, 561],
        ],
    ), atol=1
    )
    assert 120621 <= np.sum(predictions[0].mask.cpu().numpy()) <= 121621
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array(
        [
            [320, 332, 957, 561],
        ],
    ), atol=1
    )
    assert 120621 <= np.sum(predictions[1].mask.cpu().numpy()) <= 121621


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_package_with_static_crop_stretch_against_torch_input(
    snake_image_torch: torch.Tensor,
    snakes_rfdetr_seg_onnx_static_bs_static_crop_stretch_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_onnx import (
        RFDetrForInstanceSegmentationOnnx,
    )

    model = RFDetrForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_onnx_static_bs_static_crop_stretch_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(
        snake_image_torch,
        confidence=0.5,
    )

    # then
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array(
        [
            [320, 330, 958, 560],
        ],
    ), atol=5
    )
    assert 120433 <= np.sum(predictions[0].mask.cpu().numpy()) <= 121433


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_package_with_static_crop_stretch_against_torch_list_input(
    snake_image_torch: torch.Tensor,
    snakes_rfdetr_seg_onnx_static_bs_static_crop_stretch_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_onnx import (
        RFDetrForInstanceSegmentationOnnx,
    )

    model = RFDetrForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_onnx_static_bs_static_crop_stretch_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(
        [snake_image_torch, snake_image_torch],
        confidence=0.5,
    )

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array(
        [
            [320, 330, 958, 560],
        ],
    ), atol=5
    )
    assert 120433 <= np.sum(predictions[0].mask.cpu().numpy()) <= 121433
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array(
        [
            [320, 330, 958, 560],
        ],
    ), atol=5
    )
    assert 120433 <= np.sum(predictions[1].mask.cpu().numpy()) <= 121433


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_package_with_static_crop_stretch_against_torch_stack_input(
    snake_image_torch: torch.Tensor,
    snakes_rfdetr_seg_onnx_static_bs_static_crop_stretch_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_onnx import (
        RFDetrForInstanceSegmentationOnnx,
    )

    model = RFDetrForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_onnx_static_bs_static_crop_stretch_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
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
            [320, 330, 958, 560],
        ],
    ), atol=5
    )
    assert 120433 <= np.sum(predictions[0].mask.cpu().numpy()) <= 121433
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array(
        [
            [320, 330, 958, 560],
        ],
    ), atol=5
    )
    assert 120433 <= np.sum(predictions[1].mask.cpu().numpy()) <= 121433


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_package_with_static_crop_stretch_against_numpy_input(
    snake_image_numpy: np.ndarray,
    snakes_rfdetr_seg_onnx_static_bs_static_crop_stretch_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_onnx import (
        RFDetrForInstanceSegmentationOnnx,
    )

    model = RFDetrForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_onnx_static_bs_static_crop_stretch_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(snake_image_numpy, confidence=0.5)

    # then
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array(
        [
            [320, 332, 957, 561],
        ],
    ), atol=1
    )
    assert 120621 <= np.sum(predictions[0].mask.cpu().numpy()) <= 121621


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_package_with_static_crop_letterbox_against_numpy_input(
    snake_image_numpy: np.ndarray,
    snakes_rfdetr_seg_onnx_static_bs_static_crop_letterbox_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_onnx import (
        RFDetrForInstanceSegmentationOnnx,
    )

    model = RFDetrForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_onnx_static_bs_static_crop_letterbox_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(snake_image_numpy, confidence=0.5)

    # then
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array([[319, 332, 963, 561]]), atol=1
    )
    assert 119000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 120000


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_package_with_static_crop_letterbox_against_numpy_list_input(
    snake_image_numpy: np.ndarray,
    snakes_rfdetr_seg_onnx_static_bs_static_crop_letterbox_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_onnx import (
        RFDetrForInstanceSegmentationOnnx,
    )

    model = RFDetrForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_onnx_static_bs_static_crop_letterbox_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([snake_image_numpy, snake_image_numpy], confidence=0.5)

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array([[319, 332, 963, 561]]), atol=1
    )
    assert 119000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 120000
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array([[319, 332, 963, 561]]), atol=1
    )
    assert 119000 <= np.sum(predictions[1].mask.cpu().numpy()) <= 120000


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_package_with_static_crop_letterbox_against_torch_input(
    snake_image_torch: torch.Tensor,
    snakes_rfdetr_seg_onnx_static_bs_static_crop_letterbox_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_onnx import (
        RFDetrForInstanceSegmentationOnnx,
    )

    model = RFDetrForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_onnx_static_bs_static_crop_letterbox_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(snake_image_torch, confidence=0.5)

    # then
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array([[319, 332, 963, 561]]), atol=1
    )
    assert 119000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 120000


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_package_with_static_crop_letterbox_against_torch_list_input(
    snake_image_torch: torch.Tensor,
    snakes_rfdetr_seg_onnx_static_bs_static_crop_letterbox_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_onnx import (
        RFDetrForInstanceSegmentationOnnx,
    )

    model = RFDetrForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_onnx_static_bs_static_crop_letterbox_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([snake_image_torch, snake_image_torch], confidence=0.5)

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array([[319, 332, 963, 561]]), atol=1
    )
    assert 119000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 120000
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array([[319, 332, 963, 561]]), atol=1
    )
    assert 119000 <= np.sum(predictions[1].mask.cpu().numpy()) <= 120000


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_package_with_static_crop_letterbox_against_torch_batch_input(
    snake_image_torch: torch.Tensor,
    snakes_rfdetr_seg_onnx_static_bs_static_crop_letterbox_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_onnx import (
        RFDetrForInstanceSegmentationOnnx,
    )

    model = RFDetrForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_onnx_static_bs_static_crop_letterbox_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(
        torch.stack([snake_image_torch, snake_image_torch], dim=0),
        confidence=0.5,
    )

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(), np.array([[319, 332, 963, 561]]), atol=1
    )
    assert 119000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 120000
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array([[319, 332, 963, 561]]), atol=1
    )
    assert 119000 <= np.sum(predictions[1].mask.cpu().numpy()) <= 120000


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_package_with_static_crop_letterbox_against_numpy_input(
    snake_image_numpy: np.ndarray,
    snakes_rfdetr_seg_onnx_static_bs_static_crop_center_crop_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_onnx import (
        RFDetrForInstanceSegmentationOnnx,
    )

    model = RFDetrForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_onnx_static_bs_static_crop_center_crop_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
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
@pytest.mark.onnx_extras
def test_package_with_static_crop_letterbox_against_numpy_list_input(
    snake_image_numpy: np.ndarray,
    snakes_rfdetr_seg_onnx_static_bs_static_crop_center_crop_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_onnx import (
        RFDetrForInstanceSegmentationOnnx,
    )

    model = RFDetrForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_onnx_static_bs_static_crop_center_crop_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
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
@pytest.mark.onnx_extras
def test_package_with_static_crop_letterbox_against_torch_input(
    snake_image_torch: torch.Tensor,
    snakes_rfdetr_seg_onnx_static_bs_static_crop_center_crop_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_onnx import (
        RFDetrForInstanceSegmentationOnnx,
    )

    model = RFDetrForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_onnx_static_bs_static_crop_center_crop_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
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
@pytest.mark.onnx_extras
def test_package_with_static_crop_letterbox_against_torch_list_input(
    snake_image_torch: torch.Tensor,
    snakes_rfdetr_seg_onnx_static_bs_static_crop_center_crop_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_onnx import (
        RFDetrForInstanceSegmentationOnnx,
    )

    model = RFDetrForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_onnx_static_bs_static_crop_center_crop_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
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
@pytest.mark.onnx_extras
def test_package_with_static_crop_letterbox_against_torch_batch_input(
    snake_image_torch: torch.Tensor,
    snakes_rfdetr_seg_onnx_static_bs_static_crop_center_crop_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_onnx import (
        RFDetrForInstanceSegmentationOnnx,
    )

    model = RFDetrForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_onnx_static_bs_static_crop_center_crop_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
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


_NONSQUARE_LETTERBOX_SEG_ONNX_EXPECTED_XYXY_NUMPY = np.array([[119, 318, 1261, 547]])
_NONSQUARE_LETTERBOX_SEG_ONNX_EXPECTED_XYXY_TORCH = np.array([[119, 317, 1261, 547]])
_NONSQUARE_LETTERBOX_SEG_ONNX_EXPECTED_MASK_SUM_NUMPY = 212527
_NONSQUARE_LETTERBOX_SEG_ONNX_EXPECTED_MASK_SUM_TORCH = 212249


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_package_with_nonsquare_letterbox_against_numpy_input(
    snake_image_numpy: np.ndarray,
    snakes_rfdetr_seg_onnx_static_bs_nonsquare_letterbox_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_onnx import (
        RFDetrForInstanceSegmentationOnnx,
    )

    model = RFDetrForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_onnx_static_bs_nonsquare_letterbox_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(snake_image_numpy, confidence=0.5)

    # then
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(),
        torch.tensor(
        [
            [120, 318, 1260, 548],
        ],
        dtype=torch.int32,
    ),
        atol=2,
    )
    assert (
        _NONSQUARE_LETTERBOX_SEG_ONNX_EXPECTED_MASK_SUM_NUMPY - 500
        <= np.sum(predictions[0].mask.cpu().numpy())
        <= _NONSQUARE_LETTERBOX_SEG_ONNX_EXPECTED_MASK_SUM_NUMPY + 500
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_package_with_nonsquare_letterbox_against_numpy_list_input(
    snake_image_numpy: np.ndarray,
    snakes_rfdetr_seg_onnx_static_bs_nonsquare_letterbox_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_onnx import (
        RFDetrForInstanceSegmentationOnnx,
    )

    model = RFDetrForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_onnx_static_bs_nonsquare_letterbox_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
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
            [120, 318, 1260, 548],
        ],
        dtype=torch.int32,
    ),
            atol=2,
        )
        assert (
            _NONSQUARE_LETTERBOX_SEG_ONNX_EXPECTED_MASK_SUM_NUMPY - 500
            <= np.sum(pred.mask.cpu().numpy())
            <= _NONSQUARE_LETTERBOX_SEG_ONNX_EXPECTED_MASK_SUM_NUMPY + 500
        )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_package_with_nonsquare_letterbox_against_torch_input(
    snake_image_torch: torch.Tensor,
    snakes_rfdetr_seg_onnx_static_bs_nonsquare_letterbox_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_onnx import (
        RFDetrForInstanceSegmentationOnnx,
    )

    model = RFDetrForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_onnx_static_bs_nonsquare_letterbox_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(snake_image_torch, confidence=0.5)

    # then
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(),
        torch.tensor(
        [
            [121, 318, 1267, 549],
        ],
        dtype=torch.int32,
    ),
        atol=2,
    )
    assert (
        _NONSQUARE_LETTERBOX_SEG_ONNX_EXPECTED_MASK_SUM_TORCH - 500
        <= np.sum(predictions[0].mask.cpu().numpy())
        <= _NONSQUARE_LETTERBOX_SEG_ONNX_EXPECTED_MASK_SUM_TORCH + 500
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_package_with_nonsquare_letterbox_against_torch_list_input(
    snake_image_torch: torch.Tensor,
    snakes_rfdetr_seg_onnx_static_bs_nonsquare_letterbox_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_onnx import (
        RFDetrForInstanceSegmentationOnnx,
    )

    model = RFDetrForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_onnx_static_bs_nonsquare_letterbox_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
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
            [121, 318, 1267, 549],
        ],
        dtype=torch.int32,
    ),
            atol=2,
        )
        assert (
            _NONSQUARE_LETTERBOX_SEG_ONNX_EXPECTED_MASK_SUM_TORCH - 500
            <= np.sum(pred.mask.cpu().numpy())
            <= _NONSQUARE_LETTERBOX_SEG_ONNX_EXPECTED_MASK_SUM_TORCH + 500
        )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_package_with_nonsquare_letterbox_against_torch_batch_input(
    snake_image_torch: torch.Tensor,
    snakes_rfdetr_seg_onnx_static_bs_nonsquare_letterbox_package: str,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_onnx import (
        RFDetrForInstanceSegmentationOnnx,
    )

    model = RFDetrForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_onnx_static_bs_nonsquare_letterbox_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(
        torch.stack([snake_image_torch, snake_image_torch], dim=0), confidence=0.5
    )

    # then
    assert len(predictions) == 2
    for pred in predictions:
        assert np.allclose(
            pred.xyxy.cpu().numpy(),
            torch.tensor(
        [
            [121, 318, 1267, 549],
        ],
        dtype=torch.int32,
    ),
            atol=2,
        )
        assert (
            _NONSQUARE_LETTERBOX_SEG_ONNX_EXPECTED_MASK_SUM_TORCH - 500
            <= np.sum(pred.mask.cpu().numpy())
            <= _NONSQUARE_LETTERBOX_SEG_ONNX_EXPECTED_MASK_SUM_TORCH + 500
        )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_per_class_confidence_blocks_all_classes(
    snake_image_numpy: np.ndarray,
    snakes_rfdetr_seg_onnx_static_bs_stretch_package: str,
) -> None:
    """Baseline (see `test_package_with_stretch_against_numpy_input` above)
    returns 1 detection. Setting a 0.99 per-class threshold on every class
    leaves no detections."""
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_onnx import (
        RFDetrForInstanceSegmentationOnnx,
    )
    from inference_models.weights_providers.entities import RecommendedParameters

    model = RFDetrForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=snakes_rfdetr_seg_onnx_static_bs_stretch_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    model.recommended_parameters = RecommendedParameters(
        confidence=0.3,
        per_class_confidence={name: 1.01 for name in model.class_names},
    )
    predictions = model(snake_image_numpy, confidence="best")
    assert predictions[0].class_id.numel() == 0
