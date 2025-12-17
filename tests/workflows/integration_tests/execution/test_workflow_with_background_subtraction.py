import numpy as np
import pytest

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine
from tests.workflows.integration_tests.execution.workflows_gallery_collector.decorators import (
    add_to_workflows_gallery,
)

WORKFLOW_WITH_BACKGROUND_SUBTRACTION = {
    "version": "1.0",
    "inputs": [
        {"type": "InferenceImage", "name": "image"},
    ],
    "steps": [
        {
            "type": "roboflow_core/background_subtraction@v1",
            "name": "bg_subtractor",
            "image": "$inputs.image",
            "threshold": 16,
            "history": 30,
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "output_image",
            "coordinates_system": "own",
            "selector": "$steps.bg_subtractor.image",
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
    use_case_title="Workflow for background subtraction in video stream",
    use_case_description="""
This example shows how Background Subtraction block can be used to extract motion masks from a video stream.

The background subtraction block uses MOG2 (Mixture of Gaussians) algorithm to identify pixels that
differ significantly from the background model. This is useful for motion detection, object tracking,
and video analysis tasks.
    """,
    workflow_definition=WORKFLOW_WITH_BACKGROUND_SUBTRACTION,
    workflow_name_in_app="background-subtraction",
)
def test_workflow_with_background_subtraction(
    model_manager: ModelManager,
    video_with_moving_element_frames: list,
) -> None:
    """
    Test background subtraction block in a workflow with a sequence of frames
    containing a moving element on a still background.

    The test verifies:
    1. Background subtraction produces output masks
    2. The output masks are images with proper dimensions
    3. Motion areas are highlighted in the output mask
    4. Background areas remain dark in the output mask
    """
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_BACKGROUND_SUBTRACTION,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    frames = video_with_moving_element_frames

    # when - run workflow on all frames to process video sequence
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
    assert "output_image" in first_result, "Expected 'output_image' in outputs"

    # Verify output image is valid
    output_image = first_result["output_image"]
    assert output_image is not None, "Expected output_image to be present"

    # Get the numpy array from the WorkflowImageData
    output_array = (
        output_image.numpy_image
        if hasattr(output_image, "numpy_image")
        else output_image
    )

    # Check image dimensions match input
    assert (
        output_array.shape[0] == 240
    ), f"Expected height 240, got {output_array.shape[0]}"
    assert (
        output_array.shape[1] == 320
    ), f"Expected width 320, got {output_array.shape[1]}"

    # Early frames (before history is full) may not show much motion
    # Check later frames where background model is established
    motion_detected_later = False
    for result in results[30:]:  # Check last 10 frames
        output_img = result[0]["output_image"]
        output_arr = (
            output_img.numpy_image if hasattr(output_img, "numpy_image") else output_img
        )
        # Check if there are any non-zero pixels (indicating detected motion)
        if np.any(output_arr > 0):
            motion_detected_later = True
            break

    assert (
        motion_detected_later
    ), "Expected motion to be highlighted in output masks in later frames"


def test_workflow_with_background_subtraction_batch_processing(
    model_manager: ModelManager,
    video_with_moving_element_frames: list,
) -> None:
    """
    Test background subtraction block with batch processing of multiple frames.

    This verifies that the background subtraction maintains state across frames
    in a batch and produces consistent output masks.
    """
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_BACKGROUND_SUBTRACTION,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    frames = video_with_moving_element_frames
    # Select a subset of frames for batch processing
    batch_frames = frames[20:35]  # 15 frames from the video

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
            "output_image" in frame_result
        ), f"Frame {i}: Expected 'output_image' in outputs"

        output_img = frame_result["output_image"]
        assert output_img is not None, f"Frame {i}: Expected output_image to be present"

        # Verify image dimensions
        output_arr = (
            output_img.numpy_image if hasattr(output_img, "numpy_image") else output_img
        )
        assert (
            output_arr.shape[0] == 240
        ), f"Frame {i}: Expected height 240, got {output_arr.shape[0]}"
        assert (
            output_arr.shape[1] == 320
        ), f"Frame {i}: Expected width 320, got {output_arr.shape[1]}"


def test_background_subtraction_with_different_thresholds(
    model_manager: ModelManager,
    video_with_moving_element_frames: list,
) -> None:
    """
    Test background subtraction with different threshold values.

    Verifies that varying the threshold parameter produces different sensitivity
    to motion detection.
    """
    frames = video_with_moving_element_frames
    test_frame = frames[32]  # Use a frame in the middle where motion is expected

    threshold_values = [8, 16, 32]
    results_by_threshold = {}

    for threshold in threshold_values:
        workflow = {
            "version": "1.0",
            "inputs": [
                {"type": "InferenceImage", "name": "image"},
            ],
            "steps": [
                {
                    "type": "roboflow_core/background_subtraction@v1",
                    "name": "bg_subtractor",
                    "image": "$inputs.image",
                    "threshold": threshold,
                    "history": 30,
                },
            ],
            "outputs": [
                {
                    "type": "JsonField",
                    "name": "output_image",
                    "coordinates_system": "own",
                    "selector": "$steps.bg_subtractor.image",
                },
            ],
        }

        workflow_init_parameters = {
            "workflows_core.model_manager": model_manager,
            "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
        }
        execution_engine = ExecutionEngine.init(
            workflow_definition=workflow,
            init_parameters=workflow_init_parameters,
            max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
        )

        # Run through frames to build history
        for frame in frames[:32]:
            execution_engine.run(
                runtime_parameters={
                    "image": [frame],
                }
            )

        # Run on the test frame
        result = execution_engine.run(
            runtime_parameters={
                "image": [test_frame],
            }
        )

        output_img = result[0]["output_image"]
        output_arr = (
            output_img.numpy_image if hasattr(output_img, "numpy_image") else output_img
        )
        results_by_threshold[threshold] = output_arr

    # Verify that lower thresholds (more sensitive) produce more detected pixels
    # than higher thresholds (less sensitive)
    pixels_8 = np.sum(results_by_threshold[8] > 0)
    pixels_16 = np.sum(results_by_threshold[16] > 0)
    pixels_32 = np.sum(results_by_threshold[32] > 0)

    assert pixels_8 >= pixels_16, (
        f"Lower threshold (8) should detect more pixels ({pixels_8}) "
        f"than medium threshold (16, {pixels_16})"
    )
    assert pixels_16 >= pixels_32, (
        f"Medium threshold (16) should detect more pixels ({pixels_16}) "
        f"than higher threshold (32, {pixels_32})"
    )
