import numpy as np
import pytest
import torch


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_batch_size_and_letterbox_numpy(
    coin_counting_yolov5_onnx_static_bs_letterbox_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yolov5.yolov5_object_detection_onnx import (
        YOLOv5ForObjectDetectionOnnx,
    )

    model = YOLOv5ForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=coin_counting_yolov5_onnx_static_bs_letterbox_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(coins_counting_image_numpy)

    # then
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].class_id.cpu().numpy(),
        [
            2,
            2,
            0,
            2,
            2,
            0,
            0,
            0,
            0,
            2,
            0,
            0,
            2,
            3,
            2,
            2,
            0,
            3,
            3,
            3,
            0,
            2,
            3,
            3,
            2,
            2,
            3,
            3,
            3,
            3,
        ],
    )
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), 0.6125, atol=0.001)
    assert np.allclose(
        predictions[0].xyxy[0].cpu().numpy(), [1256, 2054, 1419, 2244], atol=1
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_batch_size_and_letterbox_list_of_numpy(
    coin_counting_yolov5_onnx_static_bs_letterbox_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yolov5.yolov5_object_detection_onnx import (
        YOLOv5ForObjectDetectionOnnx,
    )

    model = YOLOv5ForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=coin_counting_yolov5_onnx_static_bs_letterbox_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([coins_counting_image_numpy, coins_counting_image_numpy])

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].class_id.cpu().numpy(),
        [
            2,
            2,
            0,
            2,
            2,
            0,
            0,
            0,
            0,
            2,
            0,
            0,
            2,
            3,
            2,
            2,
            0,
            3,
            3,
            3,
            0,
            2,
            3,
            3,
            2,
            2,
            3,
            3,
            3,
            3,
        ],
    )
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), 0.6125, atol=0.001)
    assert np.allclose(
        predictions[0].xyxy[0].cpu().numpy(), [1256, 2054, 1419, 2244], atol=1
    )
    assert np.allclose(
        predictions[1].class_id.cpu().numpy(),
        [
            2,
            2,
            0,
            2,
            2,
            0,
            0,
            0,
            0,
            2,
            0,
            0,
            2,
            3,
            2,
            2,
            0,
            3,
            3,
            3,
            0,
            2,
            3,
            3,
            2,
            2,
            3,
            3,
            3,
            3,
        ],
    )
    assert np.allclose(predictions[1].confidence[0].cpu().numpy(), 0.6125, atol=0.001)
    assert np.allclose(
        predictions[1].xyxy[0].cpu().numpy(), [1256, 2054, 1419, 2244], atol=1
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_batch_size_and_letterbox_torch(
    coin_counting_yolov5_onnx_static_bs_letterbox_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.yolov5.yolov5_object_detection_onnx import (
        YOLOv5ForObjectDetectionOnnx,
    )

    model = YOLOv5ForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=coin_counting_yolov5_onnx_static_bs_letterbox_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(coins_counting_image_torch)

    # then
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].class_id.cpu().numpy(),
        [
            2,
            2,
            0,
            2,
            2,
            0,
            0,
            0,
            0,
            2,
            0,
            0,
            2,
            3,
            2,
            2,
            0,
            3,
            3,
            3,
            0,
            2,
            3,
            3,
            2,
            2,
            3,
            3,
            3,
            3,
        ],
    )
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), 0.6125, atol=0.002)
    assert np.allclose(
        predictions[0].xyxy[0].cpu().numpy(), [1256, 2054, 1419, 2244], atol=1
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_batch_size_and_letterbox_torch_list(
    coin_counting_yolov5_onnx_static_bs_letterbox_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.yolov5.yolov5_object_detection_onnx import (
        YOLOv5ForObjectDetectionOnnx,
    )

    model = YOLOv5ForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=coin_counting_yolov5_onnx_static_bs_letterbox_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([coins_counting_image_torch, coins_counting_image_torch])

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].class_id.cpu().numpy(),
        [
            2,
            2,
            0,
            2,
            2,
            0,
            0,
            0,
            0,
            2,
            0,
            0,
            2,
            3,
            2,
            2,
            0,
            3,
            3,
            3,
            0,
            2,
            3,
            3,
            2,
            2,
            3,
            3,
            3,
            3,
        ],
    )
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), 0.6125, atol=0.002)
    assert np.allclose(
        predictions[0].xyxy[0].cpu().numpy(), [1256, 2054, 1419, 2244], atol=1
    )
    assert np.allclose(
        predictions[1].class_id.cpu().numpy(),
        [
            2,
            2,
            0,
            2,
            2,
            0,
            0,
            0,
            0,
            2,
            0,
            0,
            2,
            3,
            2,
            2,
            0,
            3,
            3,
            3,
            0,
            2,
            3,
            3,
            2,
            2,
            3,
            3,
            3,
            3,
        ],
    )
    assert np.allclose(predictions[1].confidence[0].cpu().numpy(), 0.6125, atol=0.002)
    assert np.allclose(
        predictions[1].xyxy[0].cpu().numpy(), [1256, 2054, 1419, 2244], atol=1
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_batch_size_and_letterbox_torch_batch(
    coin_counting_yolov5_onnx_static_bs_letterbox_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.yolov5.yolov5_object_detection_onnx import (
        YOLOv5ForObjectDetectionOnnx,
    )

    model = YOLOv5ForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=coin_counting_yolov5_onnx_static_bs_letterbox_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(
        torch.stack([coins_counting_image_torch, coins_counting_image_torch], dim=0)
    )

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].class_id.cpu().numpy(),
        [
            2,
            2,
            0,
            2,
            2,
            0,
            0,
            0,
            0,
            2,
            0,
            0,
            2,
            3,
            2,
            2,
            0,
            3,
            3,
            3,
            0,
            2,
            3,
            3,
            2,
            2,
            3,
            3,
            3,
            3,
        ],
    )
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), 0.6125, atol=0.002)
    assert np.allclose(
        predictions[0].xyxy[0].cpu().numpy(), [1256, 2054, 1419, 2244], atol=1
    )
    assert np.allclose(
        predictions[1].class_id.cpu().numpy(),
        [
            2,
            2,
            0,
            2,
            2,
            0,
            0,
            0,
            0,
            2,
            0,
            0,
            2,
            3,
            2,
            2,
            0,
            3,
            3,
            3,
            0,
            2,
            3,
            3,
            2,
            2,
            3,
            3,
            3,
            3,
        ],
    )
    assert np.allclose(predictions[1].confidence[0].cpu().numpy(), 0.6125, atol=0.002)
    assert np.allclose(
        predictions[1].xyxy[0].cpu().numpy(), [1256, 2054, 1419, 2244], atol=1
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_dynamic_batch_size_and_letterbox_numpy(
    coin_counting_yolov5_onnx_dynamic_bs_letterbox_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yolov5.yolov5_object_detection_onnx import (
        YOLOv5ForObjectDetectionOnnx,
    )

    model = YOLOv5ForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=coin_counting_yolov5_onnx_dynamic_bs_letterbox_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(coins_counting_image_numpy)

    # then
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].class_id.cpu().numpy(),
        [
            2,
            2,
            0,
            2,
            2,
            0,
            0,
            0,
            0,
            2,
            0,
            0,
            2,
            3,
            2,
            2,
            0,
            3,
            3,
            3,
            0,
            2,
            3,
            3,
            2,
            2,
            3,
            3,
            3,
            3,
        ],
    )
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), 0.6125, atol=0.001)
    assert np.allclose(
        predictions[0].xyxy[0].cpu().numpy(), [1256, 2054, 1419, 2244], atol=1
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_dynamic_batch_size_and_letterbox_list_of_numpy(
    coin_counting_yolov5_onnx_dynamic_bs_letterbox_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yolov5.yolov5_object_detection_onnx import (
        YOLOv5ForObjectDetectionOnnx,
    )

    model = YOLOv5ForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=coin_counting_yolov5_onnx_dynamic_bs_letterbox_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([coins_counting_image_numpy, coins_counting_image_numpy])

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].class_id.cpu().numpy(),
        [
            2,
            2,
            0,
            2,
            2,
            0,
            0,
            0,
            0,
            2,
            0,
            0,
            2,
            3,
            2,
            2,
            0,
            3,
            3,
            3,
            0,
            2,
            3,
            3,
            2,
            2,
            3,
            3,
            3,
            3,
        ],
    )
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), 0.6125, atol=0.001)
    assert np.allclose(
        predictions[0].xyxy[0].cpu().numpy(), [1256, 2054, 1419, 2244], atol=1
    )
    assert np.allclose(
        predictions[1].class_id.cpu().numpy(),
        [
            2,
            2,
            0,
            2,
            2,
            0,
            0,
            0,
            0,
            2,
            0,
            0,
            2,
            3,
            2,
            2,
            0,
            3,
            3,
            3,
            0,
            2,
            3,
            3,
            2,
            2,
            3,
            3,
            3,
            3,
        ],
    )
    assert np.allclose(predictions[1].confidence[0].cpu().numpy(), 0.6125, atol=0.001)
    assert np.allclose(
        predictions[1].xyxy[0].cpu().numpy(), [1256, 2054, 1419, 2244], atol=1
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_dynamic_batch_size_and_letterbox_torch(
    coin_counting_yolov5_onnx_dynamic_bs_letterbox_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.yolov5.yolov5_object_detection_onnx import (
        YOLOv5ForObjectDetectionOnnx,
    )

    model = YOLOv5ForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=coin_counting_yolov5_onnx_dynamic_bs_letterbox_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(coins_counting_image_torch)

    # then
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].class_id.cpu().numpy(),
        [
            2,
            2,
            0,
            2,
            2,
            0,
            0,
            0,
            0,
            2,
            0,
            0,
            2,
            3,
            2,
            2,
            0,
            3,
            3,
            3,
            0,
            2,
            3,
            3,
            2,
            2,
            3,
            3,
            3,
            3,
        ],
    )
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), 0.6125, atol=0.002)
    assert np.allclose(
        predictions[0].xyxy[0].cpu().numpy(), [1256, 2054, 1419, 2244], atol=1
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_dynamic_batch_size_and_letterbox_torch_list(
    coin_counting_yolov5_onnx_dynamic_bs_letterbox_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.yolov5.yolov5_object_detection_onnx import (
        YOLOv5ForObjectDetectionOnnx,
    )

    model = YOLOv5ForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=coin_counting_yolov5_onnx_dynamic_bs_letterbox_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([coins_counting_image_torch, coins_counting_image_torch])

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].class_id.cpu().numpy(),
        [
            2,
            2,
            0,
            2,
            2,
            0,
            0,
            0,
            0,
            2,
            0,
            0,
            2,
            3,
            2,
            2,
            0,
            3,
            3,
            3,
            0,
            2,
            3,
            3,
            2,
            2,
            3,
            3,
            3,
            3,
        ],
    )
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), 0.6125, atol=0.002)
    assert np.allclose(
        predictions[0].xyxy[0].cpu().numpy(), [1256, 2054, 1419, 2244], atol=1
    )
    assert np.allclose(
        predictions[1].class_id.cpu().numpy(),
        [
            2,
            2,
            0,
            2,
            2,
            0,
            0,
            0,
            0,
            2,
            0,
            0,
            2,
            3,
            2,
            2,
            0,
            3,
            3,
            3,
            0,
            2,
            3,
            3,
            2,
            2,
            3,
            3,
            3,
            3,
        ],
    )
    assert np.allclose(predictions[1].confidence[0].cpu().numpy(), 0.6125, atol=0.002)
    assert np.allclose(
        predictions[1].xyxy[0].cpu().numpy(), [1256, 2054, 1419, 2244], atol=1
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_dynamic_batch_size_and_letterbox_torch_batch(
    coin_counting_yolov5_onnx_dynamic_bs_letterbox_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.yolov5.yolov5_object_detection_onnx import (
        YOLOv5ForObjectDetectionOnnx,
    )

    model = YOLOv5ForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=coin_counting_yolov5_onnx_dynamic_bs_letterbox_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(
        torch.stack([coins_counting_image_torch, coins_counting_image_torch], dim=0)
    )

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].class_id.cpu().numpy(),
        [
            2,
            2,
            0,
            2,
            2,
            0,
            0,
            0,
            0,
            2,
            0,
            0,
            2,
            3,
            2,
            2,
            0,
            3,
            3,
            3,
            0,
            2,
            3,
            3,
            2,
            2,
            3,
            3,
            3,
            3,
        ],
    )
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), 0.6125, atol=0.002)
    assert np.allclose(
        predictions[0].xyxy[0].cpu().numpy(), [1256, 2054, 1419, 2244], atol=1
    )
    assert np.allclose(
        predictions[1].class_id.cpu().numpy(),
        [
            2,
            2,
            0,
            2,
            2,
            0,
            0,
            0,
            0,
            2,
            0,
            0,
            2,
            3,
            2,
            2,
            0,
            3,
            3,
            3,
            0,
            2,
            3,
            3,
            2,
            2,
            3,
            3,
            3,
            3,
        ],
    )
    assert np.allclose(predictions[1].confidence[0].cpu().numpy(), 0.6125, atol=0.002)
    assert np.allclose(
        predictions[1].xyxy[0].cpu().numpy(), [1256, 2054, 1419, 2244], atol=1
    )
