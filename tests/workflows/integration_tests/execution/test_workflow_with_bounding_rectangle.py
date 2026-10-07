import cv2 as cv
import numpy as np
import pytest
import supervision as sv

from inference.core.env import (
    ENABLE_TENSOR_DATA_REPRESENTATION,
    USE_INFERENCE_MODELS,
    WORKFLOWS_MAX_CONCURRENT_STEPS,
)
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.errors import RuntimeInputError, StepExecutionError
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference_models.models.base.instance_segmentation import (
    InstanceDetections as NativeInstanceDetections,
)
from tests.workflows.integration_tests.execution.tensor_input_utils import (
    numpy_image_as_tensor,
)
from tests.workflows.integration_tests.execution.workflows_gallery_collector.decorators import (
    add_to_workflows_gallery,
)

# Under ENABLE_TENSOR_DATA_REPRESENTATION the bounding_rect block emits a native
# inference_models.InstanceDetections instead of sv.Detections. The numpy-shaped
# assertions below run only with the flag off; an equivalent *_tensor_native test
# asserts the same facts against the native carrier with the flag on.
_NUMPY_ONLY = pytest.mark.skipif(
    ENABLE_TENSOR_DATA_REPRESENTATION,
    reason="sv.Detections output; native under ENABLE_TENSOR_DATA_REPRESENTATION "
    "— see the *_tensor_native parity test",
)
_TENSOR_ONLY = pytest.mark.skipif(
    not ENABLE_TENSOR_DATA_REPRESENTATION,
    reason="tensor-native variant; runs only with ENABLE_TENSOR_DATA_REPRESENTATION=True",
)

BOUNDNG_RECTANGLE_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
    ],
    "steps": [
        {
            "type": "InstanceSegmentationModel",
            "name": "detection",
            "image": "$inputs.image",
            "model_id": "yolov8n-seg-640",
        },
        {
            "type": "roboflow_core/bounding_rect@v1",
            "name": "bounding_rect",
            "predictions": "$steps.detection.predictions",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "result",
            "selector": "$steps.bounding_rect.detections_with_rect",
        }
    ],
}


_EXPECTED_RECTANGLES = [
    (np.array([[322, 402], [325, 224], [586, 228], [583, 406]]), 5.0),
    (np.array([[219, 82], [352, 57], [409, 363], [276, 388]]), 6.0),
]


@_NUMPY_ONLY
@add_to_workflows_gallery(
    category="Basic Workflows",
    use_case_title="Workflow with bounding rect",
    use_case_description="""
This is the basic workflow that only contains a single object detection model and bounding rectangle extraction.
    """,
    workflow_definition=BOUNDNG_RECTANGLE_WORKFLOW,
    workflow_name_in_app="fit-bounding-rectangle",
)
def test_rectangle_bounding_workflow(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=BOUNDNG_RECTANGLE_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [dogs_image],
        }
    )

    # then
    assert len(result) == 1, "One set ot outputs expected"
    assert "result" in result[0], "Output must contain key 'result'"
    assert isinstance(
        result[0]["result"], sv.Detections
    ), "Output must be instance of sv.Detections"
    assert len(result[0]["result"]) == 2, "Two dogs on the image"
    assert (
        "rect" in result[0]["result"].data
    ), "'rect' data field must expected to be found in result"
    assert (
        "width" in result[0]["result"].data
    ), "'width' data field must expected to be found in result"
    assert (
        "height" in result[0]["result"].data
    ), "'height' data field must expected to be found in result"
    assert (
        "angle" in result[0]["result"].data
    ), "'angle' data field must expected to be found in result"

    for i, (expected_rect, atol) in enumerate(_EXPECTED_RECTANGLES):
        _assert_rectangle_geometry(
            result[0]["result"]["rect"][i],
            width=result[0]["result"]["width"][i],
            height=result[0]["result"]["height"][i],
            angle=result[0]["result"]["angle"][i],
            expected_rect=expected_rect,
            atol=atol,
        )


@_TENSOR_ONLY
def test_rectangle_bounding_workflow_tensor_native(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
    image_as_workflow_input,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=BOUNDNG_RECTANGLE_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [image_as_workflow_input(dogs_image)],
        }
    )

    # then
    assert len(result) == 1, "One set ot outputs expected"
    assert "result" in result[0], "Output must contain key 'result'"
    detections = result[0]["result"]
    assert isinstance(
        detections, NativeInstanceDetections
    ), "Output must be instance of native inference_models.InstanceDetections"
    assert detections.xyxy.shape[0] == 2, "Two dogs on the image"

    # On the native carrier the rect/width/height/angle geometry that lived in the
    # vectorized sv.Detections.data columns is stored per-box in bboxes_metadata[i].
    assert detections.bboxes_metadata is not None, "Should have per-box metadata"
    for i in range(2):
        meta = detections.bboxes_metadata[i]
        assert "rect" in meta, "'rect' geometry must be found in per-box metadata"
        assert "width" in meta, "'width' geometry must be found in per-box metadata"
        assert "height" in meta, "'height' geometry must be found in per-box metadata"
        assert "angle" in meta, "'angle' geometry must be found in per-box metadata"

    for i, (expected_rect, atol) in enumerate(_EXPECTED_RECTANGLES):
        metadata = detections.bboxes_metadata[i]
        _assert_rectangle_geometry(
            np.array(metadata["rect"]),
            width=metadata["width"],
            height=metadata["height"],
            angle=metadata["angle"],
            expected_rect=expected_rect,
            atol=atol,
        )


@_TENSOR_ONLY
def test_rectangle_bounding_workflow_with_tensor_input(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    # Same as test_rectangle_bounding_workflow_tensor_native, but the image arrives ALREADY
    # materialised as a CHW RGB device tensor (is_tensor_materialised() == True), so the
    # instance-seg block runs its on-device tensor path. Results must match the numpy-input
    # variant.
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=BOUNDNG_RECTANGLE_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when — feed the fixture as a pre-materialised tensor
    result = execution_engine.run(
        runtime_parameters={
            "image": [numpy_image_as_tensor(dogs_image)],
        }
    )

    # then
    assert len(result) == 1, "One set ot outputs expected"
    assert "result" in result[0], "Output must contain key 'result'"
    detections = result[0]["result"]
    assert isinstance(
        detections, NativeInstanceDetections
    ), "Output must be instance of native inference_models.InstanceDetections"
    assert detections.xyxy.shape[0] == 2, "Two dogs on the image"

    # On the native carrier the rect/width/height/angle geometry that lived in the
    # vectorized sv.Detections.data columns is stored per-box in bboxes_metadata[i].
    assert detections.bboxes_metadata is not None, "Should have per-box metadata"
    for i in range(2):
        meta = detections.bboxes_metadata[i]
        assert "rect" in meta, "'rect' geometry must be found in per-box metadata"
        assert "width" in meta, "'width' geometry must be found in per-box metadata"
        assert "height" in meta, "'height' geometry must be found in per-box metadata"
        assert "angle" in meta, "'angle' geometry must be found in per-box metadata"

    for i, (expected_rect, atol) in enumerate(_EXPECTED_RECTANGLES):
        metadata = detections.bboxes_metadata[i]
        _assert_rectangle_geometry(
            np.array(metadata["rect"]),
            width=metadata["width"],
            height=metadata["height"],
            angle=metadata["angle"],
            expected_rect=expected_rect,
            atol=atol,
        )


def _assert_rectangle_geometry(
    rect: np.ndarray,
    *,
    width: float,
    height: float,
    angle: float,
    expected_rect: np.ndarray,
    atol: float,
) -> None:
    # OpenCV can swap axes and shift the starting corner of equivalent rectangles.
    _assert_rectangle_corners(rect, expected_rect=expected_rect, atol=atol)
    assert width > 0
    assert height > 0

    reconstructed = cv.boxPoints(
        (
            tuple(rect.astype(float).mean(axis=0)),
            (float(width), float(height)),
            float(angle),
        )
    )
    # The block truncates corners to integers and stores metadata as float16.
    _assert_rectangle_corners(rect, expected_rect=reconstructed, atol=1.0)


def _assert_rectangle_corners(
    rect: np.ndarray, *, expected_rect: np.ndarray, atol: float
) -> None:
    candidates = np.stack([np.roll(rect, shift, axis=0) for shift in range(4)])
    closest = np.argmin(np.max(np.abs(candidates - expected_rect), axis=(1, 2)))
    np.testing.assert_allclose(candidates[closest], expected_rect, atol=atol, rtol=0)
