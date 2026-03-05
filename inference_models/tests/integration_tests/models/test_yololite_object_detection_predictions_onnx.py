import numpy as np
import pytest
import torch


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_dynamic_batch_size_and_letterbox_numpy(
    coin_counting_yololite_n_onnx_dynamic_bs_letterbox_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yololite.yololite_object_detection_onnx import (
        YOLOLiteForObjectDetectionOnnx,
    )

    model = YOLOLiteForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=coin_counting_yololite_n_onnx_dynamic_bs_letterbox_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(
        coins_counting_image_numpy,
        confidence=0.25,
        iou_threshold=0.45,
        max_detections=100,
    )

    # then
    assert isinstance(predictions, list)
    assert len(predictions) == 1
    assert predictions[0].xyxy.shape[1] == 4
    assert predictions[0].xyxy.dtype == torch.int32
    assert predictions[0].class_id.dtype == torch.int32
    assert predictions[0].confidence.dtype == torch.float32
    assert len(predictions[0].confidence) > 0
    assert torch.all(predictions[0].confidence >= 0.25)
    assert torch.all(predictions[0].confidence <= 1.0)


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_dynamic_batch_size_and_letterbox_batch_numpy(
    coin_counting_yololite_n_onnx_dynamic_bs_letterbox_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yololite.yololite_object_detection_onnx import (
        YOLOLiteForObjectDetectionOnnx,
    )

    model = YOLOLiteForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=coin_counting_yololite_n_onnx_dynamic_bs_letterbox_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(
        [coins_counting_image_numpy, coins_counting_image_numpy],
        confidence=0.25,
        iou_threshold=0.45,
        max_detections=100,
    )

    # then
    assert isinstance(predictions, list)
    assert len(predictions) == 2
    for pred in predictions:
        assert pred.xyxy.shape[1] == 4
        assert pred.xyxy.dtype == torch.int32
        assert len(pred.confidence) > 0
        assert torch.all(pred.confidence >= 0.25)
        assert torch.all(pred.confidence <= 1.0)
    # Both images are identical, so detections should match
    assert predictions[0].xyxy.shape == predictions[1].xyxy.shape
    assert torch.allclose(
        predictions[0].confidence, predictions[1].confidence, atol=0.01
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_dynamic_batch_size_and_letterbox_torch(
    coin_counting_yololite_n_onnx_dynamic_bs_letterbox_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.yololite.yololite_object_detection_onnx import (
        YOLOLiteForObjectDetectionOnnx,
    )

    model = YOLOLiteForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=coin_counting_yololite_n_onnx_dynamic_bs_letterbox_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(
        coins_counting_image_torch,
        confidence=0.25,
        iou_threshold=0.45,
        max_detections=100,
    )

    # then
    assert isinstance(predictions, list)
    assert len(predictions) == 1
    assert predictions[0].xyxy.shape[1] == 4
    assert predictions[0].xyxy.dtype == torch.int32
    assert len(predictions[0].confidence) > 0
    assert torch.all(predictions[0].confidence >= 0.25)



@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_high_confidence_threshold_returns_fewer_detections(
    coin_counting_yololite_n_onnx_dynamic_bs_letterbox_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yololite.yololite_object_detection_onnx import (
        YOLOLiteForObjectDetectionOnnx,
    )

    model = YOLOLiteForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=coin_counting_yololite_n_onnx_dynamic_bs_letterbox_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    low_conf_predictions = model(
        coins_counting_image_numpy, confidence=0.1, iou_threshold=0.45
    )
    high_conf_predictions = model(
        coins_counting_image_numpy, confidence=0.8, iou_threshold=0.45
    )

    # then
    assert len(low_conf_predictions[0].confidence) >= len(
        high_conf_predictions[0].confidence
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_class_agnostic_nms(
    coin_counting_yololite_n_onnx_dynamic_bs_letterbox_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yololite.yololite_object_detection_onnx import (
        YOLOLiteForObjectDetectionOnnx,
    )

    model = YOLOLiteForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=coin_counting_yololite_n_onnx_dynamic_bs_letterbox_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    standard_predictions = model(
        coins_counting_image_numpy,
        confidence=0.25,
        iou_threshold=0.45,
        class_agnostic_nms=False,
    )
    agnostic_predictions = model(
        coins_counting_image_numpy,
        confidence=0.25,
        iou_threshold=0.45,
        class_agnostic_nms=True,
    )

    # then - class-agnostic NMS should suppress more overlapping boxes
    assert len(agnostic_predictions[0].confidence) <= len(
        standard_predictions[0].confidence
    )
