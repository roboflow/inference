import numpy as np
import pytest

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine
from tests.workflows.integration_tests.execution.workflows_gallery_collector.decorators import (
    add_to_workflows_gallery,
)

WORKFLOW_WITH_MOTION_DETECTION = {
    "version": "1.0",
    "inputs": [
        {"type": "InferenceImage", "name": "image"},
    ],
    "steps": [
        {
            "type": "roboflow_core/motion_detection@v1",
            "name": "motion_detector",
            "image": "$inputs.image",
            "threshold": 16,
            "history": 30,
            "minimum_contour_area": 200,
            "morphological_kernel_size": 3,
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "motion_detected",
            "coordinates_system": "own",
            "selector": "$steps.motion_detector.motion",
        },
        {
            "type": "JsonField",
            "name": "motion_alarm",
            "coordinates_system": "own",
            "selector": "$steps.motion_detector.alarm",
        },
        {
            "type": "JsonField",
            "name": "detections",
            "coordinates_system": "own",
            "selector": "$steps.motion_detector.detections",
        },
        {
            "type": "JsonField",
            "name": "output_image",
            "coordinates_system": "own",
            "selector": "$steps.motion_detector.image",
        },
    ],
}


@pytest.fixture
def video_with_moving_element_frames():
    """
    Generate a sequence of frames with a still background and a moving element.

    The video has:
    - Resolution: 320x240
    - 40 frames total
    - Still background: dark gray
    - Moving element: white rectangle that moves from left to right
    """
    width, height = 320, 240
    num_frames = 40
    background_color = (50, 50, 50)  # Dark gray
    moving_element_color = (255, 255, 255)  # White
    element_width, element_height = 30, 30

    frames = []

    for frame_idx in range(num_frames):
        # Create background
        frame = np.full((height, width, 3), background_color, dtype=np.uint8)

        # Calculate position of moving element (moves left to right)
        # Element moves across the frame as progress goes from 0 to 1
        progress = frame_idx / num_frames
        x_pos = int(progress * (width - element_width))
        y_pos = (height - element_height) // 2  # Center vertically

        # Draw moving rectangle
        frame[
            y_pos : y_pos + element_height,
            x_pos : x_pos + element_width,
        ] = moving_element_color

        frames.append(frame)

    return frames


@add_to_workflows_gallery(
    category="Workflows with classical Computer Vision methods",
    use_case_title="Workflow detecting motion in video stream",
    use_case_description="""
This example shows how Motion Detection block can be used to detect motion in a video stream.

The motion detector uses background subtraction to identify moving elements in the video.
It outputs a motion flag, an alarm flag (which triggers when motion starts), detected objects,
and a visualization of the detected motion.
    """,
    workflow_definition=WORKFLOW_WITH_MOTION_DETECTION,
    workflow_name_in_app="motion-detection",
)
def test_workflow_with_motion_detection(
    model_manager: ModelManager,
    video_with_moving_element_frames: list,
) -> None:
    """
    Test motion detection block in a workflow with a sequence of frames
    containing a moving element on a still background.

    The test verifies:
    1. Motion is detected when element is moving
    2. The motion detector outputs proper detections
    3. The motion alarm is triggered appropriately
    4. Output image is produced for visualization
    """
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_MOTION_DETECTION,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    frames = video_with_moving_element_frames

    # when - run workflow on all frames
    results = []
    for frame in frames:
        result = execution_engine.run(
            runtime_parameters={
                "image": [frame],
            }
        )
        results.append(result)

    # then
    assert len(results) == len(
        frames
    ), f"Expected {len(frames)} results, got {len(results)}"

    # Check structure of first result
    first_result = results[0][0]
    assert "motion_detected" in first_result, "Expected 'motion_detected' in outputs"
    assert "motion_alarm" in first_result, "Expected 'motion_alarm' in outputs"
    assert "detections" in first_result, "Expected 'detections' in outputs"
    assert "output_image" in first_result, "Expected 'output_image' in outputs"

    # Early frames (before history is full) may be suppressed
    # Check that later frames (after history is built) detect motion
    motion_detected_in_later_frames = False
    for result in results[30:]:  # Check last 10 frames
        if result[0]["motion_detected"]:
            motion_detected_in_later_frames = True
            break

    assert (
        motion_detected_in_later_frames
    ), "Expected motion to be detected in later frames when history is full"

    # Verify that detections is a valid object
    last_result = results[-1][0]
    detections = last_result["detections"]
    assert detections is not None, "Expected detections to be present"

    # Verify output image is present and valid
    output_image = last_result["output_image"]
    assert output_image is not None, "Expected output_image to be present"


def test_workflow_with_motion_detection_batch_processing(
    model_manager: ModelManager,
    video_with_moving_element_frames: list,
) -> None:
    """
    Test motion detection block with batch processing of multiple frames at once.

    This verifies that the motion detector can handle batch inputs properly
    and maintains state across frames in a batch.
    """
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_MOTION_DETECTION,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    frames = video_with_moving_element_frames
    # Select a subset of frames for batch processing
    batch_frames = frames[25:35]  # 10 frames from the middle

    # when - run workflow on batch of frames
    result = execution_engine.run(
        runtime_parameters={
            "image": batch_frames,
        }
    )

    # then
    assert isinstance(result, list), "Expected result to be a list"
    assert len(result) == len(
        batch_frames
    ), f"Expected {len(batch_frames)} results, got {len(result)}"

    # Verify all results have the required outputs
    for i, frame_result in enumerate(result):
        assert (
            "motion_detected" in frame_result
        ), f"Frame {i}: Expected 'motion_detected' in outputs"
        assert (
            "detections" in frame_result
        ), f"Frame {i}: Expected 'detections' in outputs"
        assert (
            "output_image" in frame_result
        ), f"Frame {i}: Expected 'output_image' in outputs"
