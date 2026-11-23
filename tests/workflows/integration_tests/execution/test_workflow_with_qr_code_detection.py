import cv2
import numpy as np
import pytest
import supervision as sv

from inference.core.env import (
    ENABLE_TENSOR_DATA_REPRESENTATION,
    WORKFLOWS_MAX_CONCURRENT_STEPS,
)
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.constants import DETECTED_CODE_KEY
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference_models.models.base.object_detection import Detections as NativeDetections

# Under ENABLE_TENSOR_DATA_REPRESENTATION the QR Code Detection block emits a native
# inference_models.Detections instead of sv.Detections, so each mode has its own test
# asserting the same facts.
_NUMPY_ONLY = pytest.mark.skipif(
    ENABLE_TENSOR_DATA_REPRESENTATION,
    reason="sv.Detections output; native under ENABLE_TENSOR_DATA_REPRESENTATION",
)
_TENSOR_ONLY = pytest.mark.skipif(
    not ENABLE_TENSOR_DATA_REPRESENTATION,
    reason="tensor-native variant; runs only with ENABLE_TENSOR_DATA_REPRESENTATION=True",
)

SMALL_CODE_TEXT = "https://inference.roboflow.com"

WORKFLOW_WITH_QR_CODE_DETECTION_V2 = {
    "version": "1.0",
    "inputs": [
        {"type": "InferenceImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "qr_scale_factor", "default_value": 1.0},
    ],
    "steps": [
        {
            "type": "roboflow_core/qr_code_detector@v2",
            "name": "qr_code_detector",
            "images": "$inputs.image",
            "scale_factor": "$inputs.qr_scale_factor",
        }
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "predictions",
            "selector": "$steps.qr_code_detector.predictions",
        }
    ],
}


@pytest.fixture
def small_qr_code_image() -> np.ndarray:
    # A code drawn at 2 pixels per module, too small for OpenCV to read as it is.
    code = cv2.QRCodeEncoder.create().encode(SMALL_CODE_TEXT)
    code = cv2.resize(code, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    canvas = np.full((480, 640), 255, dtype=np.uint8)
    canvas[150 : 150 + code.shape[0], 260 : 260 + code.shape[1]] = code
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


@_NUMPY_ONLY
def test_workflow_with_qr_code_detection_v2_reading_small_code(
    model_manager: ModelManager,
    small_qr_code_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_QR_CODE_DETECTION_V2,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    unzoomed = execution_engine.run(
        runtime_parameters={"image": small_qr_code_image, "qr_scale_factor": 1.0}
    )
    zoomed = execution_engine.run(
        runtime_parameters={"image": small_qr_code_image, "qr_scale_factor": 2.0}
    )

    # then
    assert len(unzoomed) == 1, "One image provided, so one output expected"
    assert (
        len(unzoomed[0]["predictions"]) == 0
    ), "Expected the small code to be unreadable without zooming in"
    predictions = zoomed[0]["predictions"]
    assert isinstance(predictions, sv.Detections)
    assert list(predictions[DETECTED_CODE_KEY]) == [
        SMALL_CODE_TEXT
    ], "Expected the code to be read at scale_factor 2.0"
    assert np.allclose(
        predictions.xyxy, [_box_around_dark_pixels(small_qr_code_image)], atol=1.5
    ), "Expected the box in the original image's coordinates"


@_TENSOR_ONLY
def test_workflow_with_qr_code_detection_v2_reading_small_code_tensor_native(
    model_manager: ModelManager,
    small_qr_code_image: np.ndarray,
    image_as_workflow_input,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_QR_CODE_DETECTION_V2,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    unzoomed = execution_engine.run(
        runtime_parameters={
            "image": [image_as_workflow_input(small_qr_code_image)],
            "qr_scale_factor": 1.0,
        }
    )
    zoomed = execution_engine.run(
        runtime_parameters={
            "image": [image_as_workflow_input(small_qr_code_image)],
            "qr_scale_factor": 2.0,
        }
    )

    # then
    assert len(unzoomed) == 1, "One image provided, so one output expected"
    assert tuple(unzoomed[0]["predictions"].xyxy.shape) == (
        0,
        4,
    ), "Expected the small code to be unreadable without zooming in"
    predictions = zoomed[0]["predictions"]
    assert isinstance(predictions, NativeDetections)
    assert [entry[DETECTED_CODE_KEY] for entry in predictions.bboxes_metadata] == [
        SMALL_CODE_TEXT
    ], "Expected the code to be read at scale_factor 2.0"
    assert np.allclose(
        predictions.xyxy.cpu().numpy(),
        [_box_around_dark_pixels(small_qr_code_image)],
        atol=1.5,
    ), "Expected the box in the original image's coordinates"


def _box_around_dark_pixels(image: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(image[:, :, 0] < 128)
    box = np.array([xs.min() - 0.5, ys.min() - 0.5, xs.max() + 0.5, ys.max() + 0.5])

    return box
