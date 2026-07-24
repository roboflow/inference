"""
Integration tests for the public yolo26n-depth-768 pretrained ONNX packages,
exercising YOLO26ForDepthEstimationOnnx end-to-end with the bus.jpg test image.

Expected values come from the Ultralytics PT reference (square 768 letterbox);
exported packages reproduce it within ~0.3% mean abs-rel. The numpy (cv2 BGR)
and torch (torchvision RGB) pathways decode the same JPEG to slightly
different uint8 values, so their expectations differ marginally; tolerances
additionally cover CPU vs CUDA execution-provider drift.
"""

from typing import Tuple

import numpy as np
import pytest
import torch

EXPECTED_SHAPE: Tuple[int, int] = (1080, 810)
EXPECTED_MEAN_METERS = 5.457
EXPECTED_MIN_METERS = 1.34
EXPECTED_MAX_METERS = 13.89
MEAN_ATOL = 0.1
RANGE_ATOL = 0.5


def _assert_bus_depth_prediction(prediction: torch.Tensor) -> None:
    assert prediction.shape == EXPECTED_SHAPE
    assert prediction.dtype == torch.float32
    prediction = prediction.cpu()
    assert abs(prediction.mean().item() - EXPECTED_MEAN_METERS) < MEAN_ATOL
    assert abs(prediction.min().item() - EXPECTED_MIN_METERS) < RANGE_ATOL
    assert abs(prediction.max().item() - EXPECTED_MAX_METERS) < RANGE_ATOL


@pytest.mark.slow
def test_onnx_static_package_numpy(
    yolo26n_depth_onnx_static_package: str,
    bus_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yolo26.yolo26_depth_estimation_onnx import (
        YOLO26ForDepthEstimationOnnx,
    )

    model = YOLO26ForDepthEstimationOnnx.from_pretrained(
        model_name_or_path=yolo26n_depth_onnx_static_package,
    )

    # when
    predictions = model(bus_image_numpy)

    # then
    assert len(predictions) == 1
    _assert_bus_depth_prediction(predictions[0])


@pytest.mark.slow
def test_onnx_static_package_batch_numpy(
    yolo26n_depth_onnx_static_package: str,
    bus_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yolo26.yolo26_depth_estimation_onnx import (
        YOLO26ForDepthEstimationOnnx,
    )

    model = YOLO26ForDepthEstimationOnnx.from_pretrained(
        model_name_or_path=yolo26n_depth_onnx_static_package,
    )

    # when
    predictions = model([bus_image_numpy, bus_image_numpy])

    # then
    assert len(predictions) == 2
    _assert_bus_depth_prediction(predictions[0])
    _assert_bus_depth_prediction(predictions[1])


@pytest.mark.slow
def test_onnx_static_package_torch(
    yolo26n_depth_onnx_static_package: str,
    bus_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.yolo26.yolo26_depth_estimation_onnx import (
        YOLO26ForDepthEstimationOnnx,
    )

    model = YOLO26ForDepthEstimationOnnx.from_pretrained(
        model_name_or_path=yolo26n_depth_onnx_static_package,
    )

    # when
    predictions = model(bus_image_torch)

    # then
    assert len(predictions) == 1
    _assert_bus_depth_prediction(predictions[0])


@pytest.mark.slow
def test_onnx_dynamic_package_numpy(
    yolo26n_depth_onnx_dynamic_package: str,
    bus_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yolo26.yolo26_depth_estimation_onnx import (
        YOLO26ForDepthEstimationOnnx,
    )

    model = YOLO26ForDepthEstimationOnnx.from_pretrained(
        model_name_or_path=yolo26n_depth_onnx_dynamic_package,
    )

    # when
    predictions = model(bus_image_numpy)

    # then
    assert len(predictions) == 1
    _assert_bus_depth_prediction(predictions[0])


@pytest.mark.slow
def test_onnx_dynamic_package_batch_torch(
    yolo26n_depth_onnx_dynamic_package: str,
    bus_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.yolo26.yolo26_depth_estimation_onnx import (
        YOLO26ForDepthEstimationOnnx,
    )

    model = YOLO26ForDepthEstimationOnnx.from_pretrained(
        model_name_or_path=yolo26n_depth_onnx_dynamic_package,
    )

    # when
    predictions = model([bus_image_torch, bus_image_torch])

    # then
    assert len(predictions) == 2
    _assert_bus_depth_prediction(predictions[0])
    _assert_bus_depth_prediction(predictions[1])
