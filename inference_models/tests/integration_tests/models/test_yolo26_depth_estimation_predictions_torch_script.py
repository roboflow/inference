"""
Integration tests for the public yolo26n-depth-768 pretrained TorchScript
package, exercising YOLO26ForDepthEstimationTorchScript end-to-end with the
bus.jpg test image.

Expected values come from the Ultralytics PT reference (square 768 letterbox);
the exported package reproduces it within ~0.3% mean abs-rel. See the ONNX
test module for notes on numpy vs torch input pathway differences.
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
def test_torch_script_package_numpy(
    yolo26n_depth_torch_script_package: str,
    bus_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yolo26.yolo26_depth_estimation_torch_script import (
        YOLO26ForDepthEstimationTorchScript,
    )

    model = YOLO26ForDepthEstimationTorchScript.from_pretrained(
        model_name_or_path=yolo26n_depth_torch_script_package,
    )

    # when
    predictions = model(bus_image_numpy)

    # then
    assert len(predictions) == 1
    _assert_bus_depth_prediction(predictions[0])


@pytest.mark.slow
def test_torch_script_package_batch_numpy(
    yolo26n_depth_torch_script_package: str,
    bus_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yolo26.yolo26_depth_estimation_torch_script import (
        YOLO26ForDepthEstimationTorchScript,
    )

    model = YOLO26ForDepthEstimationTorchScript.from_pretrained(
        model_name_or_path=yolo26n_depth_torch_script_package,
    )

    # when
    predictions = model([bus_image_numpy, bus_image_numpy])

    # then
    assert len(predictions) == 2
    _assert_bus_depth_prediction(predictions[0])
    _assert_bus_depth_prediction(predictions[1])


@pytest.mark.slow
def test_torch_script_package_torch(
    yolo26n_depth_torch_script_package: str,
    bus_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.yolo26.yolo26_depth_estimation_torch_script import (
        YOLO26ForDepthEstimationTorchScript,
    )

    model = YOLO26ForDepthEstimationTorchScript.from_pretrained(
        model_name_or_path=yolo26n_depth_torch_script_package,
    )

    # when
    predictions = model(bus_image_torch)

    # then
    assert len(predictions) == 1
    _assert_bus_depth_prediction(predictions[0])
