import cv2
import numpy as np

from inference.core.env import (
    ENABLE_TENSOR_DATA_REPRESENTATION,
    WORKFLOWS_MAX_CONCURRENT_STEPS,
)
from inference.core.workflows.execution_engine.core import ExecutionEngine
from tests.workflows.integration_tests.execution.tensor_input_utils import (
    numpy_image_as_tensor,
)

IMAGE_STACK_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "InferenceImage", "name": "image"},
        {"type": "InferenceParameter", "name": "stack_size", "default_value": 3},
        {"type": "InferenceParameter", "name": "clear", "default_value": False},
    ],
    "steps": [
        {
            "type": "roboflow_core/image_stack@v1",
            "name": "image_stack",
            "image": "$inputs.image",
            "stack_size": "$inputs.stack_size",
            "clear": "$inputs.clear",
        }
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "frames",
            "selector": "$steps.image_stack.frames",
        },
        {
            "type": "JsonField",
            "name": "frames_count",
            "selector": "$steps.image_stack.frames_count",
        },
    ],
}


def test_image_stack_accumulates_frames(
    dogs_image: np.ndarray,
) -> None:
    # given
    execution_engine = ExecutionEngine.init(
        workflow_definition=IMAGE_STACK_WORKFLOW,
        init_parameters={},
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when — feed 3 frames
    for _ in range(3):
        result = execution_engine.run(runtime_parameters={"image": dogs_image})

    # then
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["frames_count"] == 3
    assert len(result[0]["frames"]) == 3
    for frame in result[0]["frames"]:
        assert isinstance(frame, bytes)
        assert frame[:2] == b"\xff\xd8"  # JPEG magic bytes


def test_image_stack_evicts_oldest_when_full(
    dogs_image: np.ndarray,
) -> None:
    # given
    execution_engine = ExecutionEngine.init(
        workflow_definition=IMAGE_STACK_WORKFLOW,
        init_parameters={},
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when — feed 5 frames with stack_size=3
    for _ in range(5):
        result = execution_engine.run(
            runtime_parameters={"image": dogs_image, "stack_size": 3}
        )

    # then — only 3 most recent kept
    assert result[0]["frames_count"] == 3
    assert len(result[0]["frames"]) == 3


def test_image_stack_shrink_keeps_latest_frames() -> None:
    """Changing the stack size through a selector keeps the current and prior frame."""
    execution_engine = ExecutionEngine.init(
        workflow_definition=IMAGE_STACK_WORKFLOW,
        init_parameters={},
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    try:
        for value, stack_size in ((32, 3), (64, 3), (96, 3), (128, 2)):
            image = np.full((64, 64, 3), value, dtype=np.uint8)
            result = execution_engine.run(
                runtime_parameters={
                    "image": (
                        numpy_image_as_tensor(image)
                        if ENABLE_TENSOR_DATA_REPRESENTATION
                        else image
                    ),
                    "stack_size": stack_size,
                }
            )[0]
    finally:
        execution_engine.close()

    assert result["frames_count"] == 2
    frame_values = []
    for frame in result["frames"]:
        decoded = cv2.imdecode(
            np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_GRAYSCALE
        )
        assert decoded is not None
        frame_values.append(int(decoded[32, 32]))
    assert frame_values == [128, 96]


def test_image_stack_clear_resets_buffer(
    dogs_image: np.ndarray,
) -> None:
    # given
    execution_engine = ExecutionEngine.init(
        workflow_definition=IMAGE_STACK_WORKFLOW,
        init_parameters={},
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when — fill buffer
    for _ in range(3):
        execution_engine.run(runtime_parameters={"image": dogs_image})

    # then — clear and add one
    result = execution_engine.run(
        runtime_parameters={"image": dogs_image, "clear": True}
    )
    assert result[0]["frames_count"] == 1
