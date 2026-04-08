import numpy as np
import pytest
import torch


# ── Static non-fused (batch=1) ──────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_static_non_fused_numpy(
    coin_counting_yololite_edge_n_onnx_static_bs_stretch_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.yololite.yololite_object_detection_onnx import (
        YOLOLiteForObjectDetectionOnnx,
    )

    model = YOLOLiteForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=coin_counting_yololite_edge_n_onnx_static_bs_stretch_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    predictions = model(
        coins_counting_image_numpy, confidence=0.25, iou_threshold=0.45
    )

    assert len(predictions) == 1
    assert predictions[0].xyxy.shape[1] == 4
    assert predictions[0].xyxy.dtype == torch.int32
    assert predictions[0].class_id.dtype == torch.int32
    assert predictions[0].confidence.dtype == torch.float32
    assert len(predictions[0].confidence) > 0
    assert torch.all(predictions[0].confidence >= 0.25)
    assert torch.all(predictions[0].confidence <= 1.0)


# ── Dynamic non-fused ────────────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_dynamic_non_fused_numpy(
    coin_counting_yololite_edge_n_onnx_dynamic_bs_stretch_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.yololite.yololite_object_detection_onnx import (
        YOLOLiteForObjectDetectionOnnx,
    )

    model = YOLOLiteForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=coin_counting_yololite_edge_n_onnx_dynamic_bs_stretch_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    predictions = model(
        coins_counting_image_numpy,
        confidence=0.25,
        iou_threshold=0.45,
        max_detections=100,
    )

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
def test_dynamic_non_fused_batch_numpy(
    coin_counting_yololite_edge_n_onnx_dynamic_bs_stretch_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.yololite.yololite_object_detection_onnx import (
        YOLOLiteForObjectDetectionOnnx,
    )

    model = YOLOLiteForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=coin_counting_yololite_edge_n_onnx_dynamic_bs_stretch_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    predictions = model(
        [coins_counting_image_numpy, coins_counting_image_numpy],
        confidence=0.25,
        iou_threshold=0.45,
        max_detections=100,
    )

    assert len(predictions) == 2
    for pred in predictions:
        assert pred.xyxy.shape[1] == 4
        assert len(pred.confidence) > 0
        assert torch.all(pred.confidence >= 0.25)
    assert predictions[0].xyxy.shape == predictions[1].xyxy.shape
    assert torch.allclose(
        predictions[0].confidence, predictions[1].confidence, atol=0.01
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_dynamic_non_fused_torch(
    coin_counting_yololite_edge_n_onnx_dynamic_bs_stretch_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    from inference_models.models.yololite.yololite_object_detection_onnx import (
        YOLOLiteForObjectDetectionOnnx,
    )

    model = YOLOLiteForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=coin_counting_yololite_edge_n_onnx_dynamic_bs_stretch_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    predictions = model(
        coins_counting_image_torch, confidence=0.25, iou_threshold=0.45
    )

    assert len(predictions) == 1
    assert predictions[0].xyxy.shape[1] == 4
    assert len(predictions[0].confidence) > 0
    assert torch.all(predictions[0].confidence >= 0.25)


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_dynamic_non_fused_high_confidence_returns_fewer(
    coin_counting_yololite_edge_n_onnx_dynamic_bs_stretch_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.yololite.yololite_object_detection_onnx import (
        YOLOLiteForObjectDetectionOnnx,
    )

    model = YOLOLiteForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=coin_counting_yololite_edge_n_onnx_dynamic_bs_stretch_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    low = model(coins_counting_image_numpy, confidence=0.1, iou_threshold=0.45)
    high = model(coins_counting_image_numpy, confidence=0.8, iou_threshold=0.45)

    assert len(low[0].confidence) >= len(high[0].confidence)


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_dynamic_non_fused_class_agnostic_nms(
    coin_counting_yololite_edge_n_onnx_dynamic_bs_stretch_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.yololite.yololite_object_detection_onnx import (
        YOLOLiteForObjectDetectionOnnx,
    )

    model = YOLOLiteForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=coin_counting_yololite_edge_n_onnx_dynamic_bs_stretch_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    standard = model(
        coins_counting_image_numpy, confidence=0.25, iou_threshold=0.45,
        class_agnostic_nms=False,
    )
    agnostic = model(
        coins_counting_image_numpy, confidence=0.25, iou_threshold=0.45,
        class_agnostic_nms=True,
    )

    assert len(agnostic[0].confidence) <= len(standard[0].confidence)


# ── NMS-fused ────────────────────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_fused_nms_numpy(
    coin_counting_yololite_edge_n_onnx_dynamic_bs_stretch_fused_nms_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.yololite.yololite_object_detection_onnx import (
        YOLOLiteForObjectDetectionOnnx,
    )

    model = YOLOLiteForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=coin_counting_yololite_edge_n_onnx_dynamic_bs_stretch_fused_nms_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    predictions = model(coins_counting_image_numpy, confidence=0.25)

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
def test_fused_nms_batch_numpy(
    coin_counting_yololite_edge_n_onnx_dynamic_bs_stretch_fused_nms_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.yololite.yololite_object_detection_onnx import (
        YOLOLiteForObjectDetectionOnnx,
    )

    model = YOLOLiteForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=coin_counting_yololite_edge_n_onnx_dynamic_bs_stretch_fused_nms_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    predictions = model(
        [coins_counting_image_numpy, coins_counting_image_numpy], confidence=0.25,
    )

    assert len(predictions) == 2
    for pred in predictions:
        assert pred.xyxy.shape[1] == 4
        assert len(pred.confidence) > 0
        assert torch.all(pred.confidence >= 0.25)
    assert predictions[0].xyxy.shape == predictions[1].xyxy.shape
    assert torch.allclose(
        predictions[0].confidence, predictions[1].confidence, atol=0.01
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_fused_nms_high_confidence_returns_fewer(
    coin_counting_yololite_edge_n_onnx_dynamic_bs_stretch_fused_nms_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.yololite.yololite_object_detection_onnx import (
        YOLOLiteForObjectDetectionOnnx,
    )

    model = YOLOLiteForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=coin_counting_yololite_edge_n_onnx_dynamic_bs_stretch_fused_nms_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    low = model(coins_counting_image_numpy, confidence=0.1)
    high = model(coins_counting_image_numpy, confidence=0.8)

    assert len(low[0].confidence) >= len(high[0].confidence)
