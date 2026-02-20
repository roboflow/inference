import numpy as np
import pytest
import torch


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
        predictions[0].xyxy.cpu().numpy(), np.array([[134, 324, 1265, 558]]), atol=1
    )
    assert 206000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 207000


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
        predictions[0].xyxy.cpu().numpy(), np.array([[134, 324, 1265, 558]]), atol=1
    )
    assert 206000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 207000
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array([[134, 324, 1265, 558]]), atol=1
    )
    assert 206000 <= np.sum(predictions[1].mask.cpu().numpy()) <= 207000


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
        predictions[0].xyxy.cpu().numpy(), np.array([[138, 325, 1262, 556]]), atol=5
    )
    assert 205000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 207000


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
        predictions[0].xyxy.cpu().numpy(), np.array([[138, 325, 1262, 556]]), atol=5
    )
    assert 205000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 207000
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array([[138, 325, 1262, 556]]), atol=5
    )
    assert 205000 <= np.sum(predictions[1].mask.cpu().numpy()) <= 207000


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
        predictions[0].xyxy.cpu().numpy(), np.array([[138, 325, 1262, 556]]), atol=5
    )
    assert 205000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 207000
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array([[138, 325, 1262, 556]]), atol=5
    )
    assert 205000 <= np.sum(predictions[1].mask.cpu().numpy()) <= 207000


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
        predictions[0].xyxy.cpu().numpy(), np.array([[133, 326, 1275, 556]]), atol=1
    )
    assert 204000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 205000


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
        predictions[0].xyxy.cpu().numpy(), np.array([[133, 326, 1275, 556]]), atol=1
    )
    assert 204000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 205000
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array([[133, 326, 1275, 556]]), atol=1
    )
    assert 204000 <= np.sum(predictions[1].mask.cpu().numpy()) <= 205000


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
        predictions[0].xyxy.cpu().numpy(), np.array([[138, 328, 1271, 555]]), atol=1
    )
    assert 204000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 205000


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
        predictions[0].xyxy.cpu().numpy(), np.array([[138, 328, 1271, 555]]), atol=1
    )
    assert 204000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 205000
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array([[138, 328, 1271, 555]]), atol=1
    )
    assert 204000 <= np.sum(predictions[1].mask.cpu().numpy()) <= 205000


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
        predictions[0].xyxy.cpu().numpy(), np.array([[138, 328, 1271, 555]]), atol=1
    )
    assert 204000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 205000
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array([[138, 328, 1271, 555]]), atol=1
    )
    assert 204000 <= np.sum(predictions[1].mask.cpu().numpy()) <= 205000


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
        predictions[0].xyxy.cpu().numpy(), np.array([[424, 332, 858, 558]]), atol=1
    )
    assert 82000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 83000


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
        predictions[0].xyxy.cpu().numpy(), np.array([[424, 332, 858, 558]]), atol=1
    )
    assert 82000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 83000
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array([[424, 332, 858, 558]]), atol=1
    )
    assert 82000 <= np.sum(predictions[1].mask.cpu().numpy()) <= 83000


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
        predictions[0].xyxy.cpu().numpy(), np.array([[424, 332, 858, 558]]), atol=1
    )
    assert 82000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 83000


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
        predictions[0].xyxy.cpu().numpy(), np.array([[424, 332, 858, 558]]), atol=1
    )
    assert 82000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 83000
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array([[424, 332, 858, 558]]), atol=1
    )
    assert 82000 <= np.sum(predictions[1].mask.cpu().numpy()) <= 83000


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
        predictions[0].xyxy.cpu().numpy(), np.array([[424, 332, 858, 558]]), atol=1
    )
    assert 82000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 83000
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array([[424, 332, 858, 558]]), atol=1
    )
    assert 82000 <= np.sum(predictions[1].mask.cpu().numpy()) <= 83000


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
        predictions[0].xyxy.cpu().numpy(), np.array([[321, 329, 961, 560]]), atol=1
    )
    assert 120000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 122000
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array([[321, 329, 961, 560]]), atol=1
    )
    assert 120000 <= np.sum(predictions[1].mask.cpu().numpy()) <= 122000


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
        predictions[0].xyxy.cpu().numpy(), np.array([[321, 331, 963, 561]]), atol=5
    )
    assert 120000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 122000


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
        predictions[0].xyxy.cpu().numpy(), np.array([[321, 331, 963, 561]]), atol=5
    )
    assert 120000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 122000
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array([[321, 331, 963, 561]]), atol=5
    )
    assert 120000 <= np.sum(predictions[1].mask.cpu().numpy()) <= 122000


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
        predictions[0].xyxy.cpu().numpy(), np.array([[321, 331, 963, 561]]), atol=5
    )
    assert 120000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 122000
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array([[321, 331, 963, 561]]), atol=5
    )
    assert 120000 <= np.sum(predictions[1].mask.cpu().numpy()) <= 122000


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
        predictions[0].xyxy.cpu().numpy(), np.array([[321, 329, 961, 560]]), atol=1
    )
    assert 120000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 122000


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
        predictions[0].xyxy.cpu().numpy(), np.array([[427, 331, 852, 552]]), atol=1
    )
    assert 80000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 81000


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
        predictions[0].xyxy.cpu().numpy(), np.array([[427, 331, 852, 552]]), atol=1
    )
    assert 80000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 81000
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array([[427, 331, 852, 552]]), atol=1
    )
    assert 80000 <= np.sum(predictions[1].mask.cpu().numpy()) <= 81000


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
        predictions[0].xyxy.cpu().numpy(), np.array([[427, 331, 852, 552]]), atol=1
    )
    assert 80000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 81000


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
        predictions[0].xyxy.cpu().numpy(), np.array([[427, 331, 852, 552]]), atol=1
    )
    assert 80000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 81000
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array([[427, 331, 852, 552]]), atol=1
    )
    assert 80000 <= np.sum(predictions[1].mask.cpu().numpy()) <= 81000


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
        predictions[0].xyxy.cpu().numpy(), np.array([[427, 331, 852, 552]]), atol=1
    )
    assert 80000 <= np.sum(predictions[0].mask.cpu().numpy()) <= 81000
    assert np.allclose(
        predictions[1].xyxy.cpu().numpy(), np.array([[427, 331, 852, 552]]), atol=1
    )
    assert 80000 <= np.sum(predictions[1].mask.cpu().numpy()) <= 81000
