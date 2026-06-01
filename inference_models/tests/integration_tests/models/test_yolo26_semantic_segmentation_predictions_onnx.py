"""
Integration tests for the public yolo26{n}-sem-1024 pretrained packages,
exercising the YOLO26ForSemanticSegmentationOnnx class end-to-end with the
Cityscapes-class bus.jpg test image.

Class IDs follow the Roboflow convention with class 0 = background; the
underlying Cityscapes 19 classes are shifted to indices 1..19.

Numpy vs torch input pathways disagree by ~870 pixels on the dominant
class. cv2.imread (BGR via libjpeg) and torchvision.io.read_image (RGB via
libjpeg-turbo) decode the same JPEG to slightly different uint8 values;
for a 19-class semantic head with many near-tie argmax decisions the drift
amplifies. The model + post-processing are deterministic — only the
decoded input bytes differ.

On top of that, the torch pathway also diverges between CPU and CUDA
execution providers by another ~1500 pixels; the numpy pathway happens to
be stable across hardware. Per-path bounds keep regressions catchable
without forcing a single loose union range; the torch bounds are
deliberately wider to cover both hardware backends.
"""

from typing import List, Tuple

import numpy as np
import pytest
import torch


EXPECTED_CLASSES_PRESENT: List[int] = [0, 1, 2, 3, 5, 6, 8, 9, 12, 14, 15, 17]

# cv2.imread (BGR) input pathway.
NUMPY_MEAN_CONFIDENCE = 0.9355
NUMPY_TRAIN_PIXEL_RANGE = (256150, 256250)  # class 17 dominates bus body
NUMPY_ROAD_PIXEL_RANGE = (221140, 221240)  # class 1 ground plane

# torchvision.io.read_image (RGB) input pathway. Bounds are wider than the
# numpy path because CPU and CUDA execution providers disagree by ~1500
# pixels here (CPU local: train=257073, road=221256; GPU CI: train=255529).
# Same model graph, same input bytes — the drift comes from execution-provider
# precision differences amplified by 19-class near-tie argmax decisions.
TORCH_MEAN_CONFIDENCE = 0.9358
TORCH_TRAIN_PIXEL_RANGE = (255500, 257150)
TORCH_ROAD_PIXEL_RANGE = (221050, 221350)


def _assert_bus_predictions(
    prediction,
    mean_confidence: float,
    train_pixel_range: Tuple[int, int],
    road_pixel_range: Tuple[int, int],
) -> None:
    seg = prediction.segmentation_map
    conf = prediction.confidence
    assert seg.shape == (1080, 810)
    assert conf.shape == (1080, 810)
    assert sorted(torch.unique(seg).cpu().tolist()) == EXPECTED_CLASSES_PRESENT
    assert torch.allclose(
        torch.mean(conf).cpu(),
        torch.tensor(mean_confidence).cpu(),
        atol=0.005,
    )
    train_lo, train_hi = train_pixel_range
    train_px = int(torch.sum(seg.cpu() == 17).item())
    assert train_lo <= train_px <= train_hi, train_px
    road_lo, road_hi = road_pixel_range
    road_px = int(torch.sum(seg.cpu() == 1).item())
    assert road_lo <= road_px <= road_hi, road_px


def _assert_bus_predictions_numpy(prediction) -> None:
    _assert_bus_predictions(
        prediction,
        mean_confidence=NUMPY_MEAN_CONFIDENCE,
        train_pixel_range=NUMPY_TRAIN_PIXEL_RANGE,
        road_pixel_range=NUMPY_ROAD_PIXEL_RANGE,
    )


def _assert_bus_predictions_torch(prediction) -> None:
    _assert_bus_predictions(
        prediction,
        mean_confidence=TORCH_MEAN_CONFIDENCE,
        train_pixel_range=TORCH_TRAIN_PIXEL_RANGE,
        road_pixel_range=TORCH_ROAD_PIXEL_RANGE,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_static_package_numpy(
    yolo26n_sem_onnx_static_package: str,
    bus_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yolo26.yolo26_semantic_segmentation_onnx import (
        YOLO26ForSemanticSegmentationOnnx,
    )

    model = YOLO26ForSemanticSegmentationOnnx.from_pretrained(
        model_name_or_path=yolo26n_sem_onnx_static_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(bus_image_numpy, confidence=0.5)

    # then
    assert len(predictions) == 1
    _assert_bus_predictions_numpy(predictions[0])


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_static_package_batch_numpy(
    yolo26n_sem_onnx_static_package: str,
    bus_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yolo26.yolo26_semantic_segmentation_onnx import (
        YOLO26ForSemanticSegmentationOnnx,
    )

    model = YOLO26ForSemanticSegmentationOnnx.from_pretrained(
        model_name_or_path=yolo26n_sem_onnx_static_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([bus_image_numpy, bus_image_numpy], confidence=0.5)

    # then
    assert len(predictions) == 2
    _assert_bus_predictions_numpy(predictions[0])
    _assert_bus_predictions_numpy(predictions[1])


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_static_package_torch(
    yolo26n_sem_onnx_static_package: str,
    bus_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.yolo26.yolo26_semantic_segmentation_onnx import (
        YOLO26ForSemanticSegmentationOnnx,
    )

    model = YOLO26ForSemanticSegmentationOnnx.from_pretrained(
        model_name_or_path=yolo26n_sem_onnx_static_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(bus_image_torch, confidence=0.5)

    # then
    assert len(predictions) == 1
    _assert_bus_predictions_torch(predictions[0])


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_dynamic_package_numpy(
    yolo26n_sem_onnx_dynamic_package: str,
    bus_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yolo26.yolo26_semantic_segmentation_onnx import (
        YOLO26ForSemanticSegmentationOnnx,
    )

    model = YOLO26ForSemanticSegmentationOnnx.from_pretrained(
        model_name_or_path=yolo26n_sem_onnx_dynamic_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(bus_image_numpy, confidence=0.5)

    # then
    assert len(predictions) == 1
    _assert_bus_predictions_numpy(predictions[0])


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_dynamic_package_batch_numpy(
    yolo26n_sem_onnx_dynamic_package: str,
    bus_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yolo26.yolo26_semantic_segmentation_onnx import (
        YOLO26ForSemanticSegmentationOnnx,
    )

    model = YOLO26ForSemanticSegmentationOnnx.from_pretrained(
        model_name_or_path=yolo26n_sem_onnx_dynamic_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(
        [bus_image_numpy, bus_image_numpy, bus_image_numpy], confidence=0.5
    )

    # then
    assert len(predictions) == 3
    for prediction in predictions:
        _assert_bus_predictions_numpy(prediction)


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_static_package_below_confidence_collapses_to_background(
    yolo26n_sem_onnx_static_package: str,
    bus_image_numpy: np.ndarray,
) -> None:
    """At confidence=1.0 every pixel falls back to the background_class_id."""
    # given
    from inference_models.models.yolo26.yolo26_semantic_segmentation_onnx import (
        YOLO26ForSemanticSegmentationOnnx,
    )

    model = YOLO26ForSemanticSegmentationOnnx.from_pretrained(
        model_name_or_path=yolo26n_sem_onnx_static_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(bus_image_numpy, confidence=1.0)

    seg = predictions[0].segmentation_map
    assert torch.all(seg == 0)
