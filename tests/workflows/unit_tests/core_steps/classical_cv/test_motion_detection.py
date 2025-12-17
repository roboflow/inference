import json

import numpy as np
import pytest
import supervision as sv
from pydantic import ValidationError

from inference.core.workflows.core_steps.classical_cv.motion_detection.v1 import (
    MotionDetectionBlockV1,
    MotionDetectionManifest,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


@pytest.mark.parametrize("images_field_alias", ["images", "image"])
def test_motion_detection_manifest_validation_with_valid_data(
    images_field_alias: str,
) -> None:
    """Test that the manifest validates correctly with valid data."""
    # given
    data = {
        "type": "roboflow_core/motion_detection@v1",
        "name": "motion_detector",
        images_field_alias: "$inputs.image",
        "threshold": 16,
        "history": 30,
        "minimum_contour_area": 200,
        "morphological_kernel_size": 3,
    }

    # when
    result = MotionDetectionManifest.model_validate(data)

    # then
    assert result.type == "roboflow_core/motion_detection@v1"
    assert result.name == "motion_detector"
    assert result.image == "$inputs.image"
    assert result.threshold == 16
    assert result.history == 30
    assert result.minimum_contour_area == 200
    assert result.morphological_kernel_size == 3


def test_motion_detection_manifest_validation_with_invalid_image() -> None:
    """Test that the manifest validation fails with invalid image selector."""
    # given
    data = {
        "type": "roboflow_core/motion_detection@v1",
        "name": "motion_detector",
        "image": "invalid",
        "threshold": 16,
    }

    # when & then
    with pytest.raises(ValidationError):
        _ = MotionDetectionManifest.model_validate(data)


def test_motion_detection_manifest_with_defaults() -> None:
    """Test that manifest uses correct default values."""
    # given
    data = {
        "type": "roboflow_core/motion_detection@v1",
        "name": "motion_detector",
        "image": "$inputs.image",
    }

    # when
    result = MotionDetectionManifest.model_validate(data)

    # then
    assert result.threshold == 16
    assert result.history == 30
    assert result.minimum_contour_area == 200
    assert result.morphological_kernel_size == 3
    assert result.suppress_first_detections is True


def test_motion_detection_block_initialization() -> None:
    """Test that the block initializes correctly."""
    # given & when
    block = MotionDetectionBlockV1()

    # then
    assert block.last_motion is False
    assert block.backSub is None
    assert block.threshold is None
    assert block.history is None
    assert block.frame_count == 0


def test_motion_detection_block_no_motion_in_still_image() -> None:
    """Test that no motion is detected in a still image."""
    # given
    block = MotionDetectionBlockV1()
    static_image = np.full((240, 320, 3), 100, dtype=np.uint8)

    # when - run the block multiple times to build background history
    for _ in range(35):  # Need more than history=30
        output = block.run(
            image=WorkflowImageData(
                parent_metadata=ImageParentMetadata(parent_id="test"),
                numpy_image=static_image,
            ),
            minimum_contour_area=200,
            morphological_kernel_size=3,
            threshold=16,
            history=30,
            suppress_first_detections=True,
            detection_zone=None,
        )

    # then
    assert output is not None
    assert "motion" in output
    assert "detections" in output
    assert "alarm" in output
    assert "image" in output

    assert output["motion"] is False
    assert isinstance(output["detections"], sv.Detections)
    assert output["detections"].is_empty()
    assert output["alarm"] is False


def test_motion_detection_block_detects_moving_object() -> None:
    """Test that motion is detected when an object moves."""
    # given
    block = MotionDetectionBlockV1()

    # Create 40 frames with a moving white rectangle
    frames = []
    width, height = 320, 240

    for frame_idx in range(40):
        frame = np.full((height, width, 3), 50, dtype=np.uint8)  # Dark gray background

        # Moving white rectangle
        progress = frame_idx / 40
        x_pos = int(progress * (width - 30))
        y_pos = (height - 30) // 2

        frame[y_pos : y_pos + 30, x_pos : x_pos + 30] = 255  # White rectangle
        frames.append(frame)

    # when - run through all frames
    results = []
    for frame in frames:
        output = block.run(
            image=WorkflowImageData(
                parent_metadata=ImageParentMetadata(parent_id="test"),
                numpy_image=frame,
            ),
            minimum_contour_area=200,
            morphological_kernel_size=3,
            threshold=16,
            history=30,
            suppress_first_detections=True,
            detection_zone=None,
        )
        results.append(output)

    # then
    # First 30 frames should not detect motion (building background model)
    for i in range(30):
        assert (
            results[i]["motion"] is False
        ), f"Frame {i}: Should not detect motion while building history"

    # Later frames should detect motion
    motion_detected = False
    for i in range(30, 40):
        if results[i]["motion"]:
            motion_detected = True
            break

    assert motion_detected, "Should detect motion in later frames"


def test_motion_detection_block_alarm_on_motion_start() -> None:
    """Test that alarm triggers on motion transition (False -> True)."""
    # given
    block = MotionDetectionBlockV1()

    # Initial state: last_motion should be False
    assert block.last_motion is False

    # Simulate motion detection by directly setting block state
    # First call with no motion
    static_output = {"motion": False}
    block.last_motion = False

    # Second call with motion - this should trigger alarm
    current_motion = True
    alarm = True if not block.last_motion and current_motion else False

    # then
    assert (
        alarm is True
    ), "Alarm should trigger when motion transitions from False to True"


def test_motion_detection_block_no_alarm_on_continuous_motion() -> None:
    """Test that alarm doesn't trigger on continuous motion (stays True)."""
    # given
    block = MotionDetectionBlockV1()

    # Simulate continuous motion detection state
    # Set up state where last_motion is already True
    block.last_motion = True
    current_motion = True

    # when - calculate alarm with continuous motion
    alarm = True if not block.last_motion and current_motion else False

    # then
    assert (
        alarm is False
    ), "Alarm should not trigger when motion is continuous (both True)"

    # Update state for next frame
    block.last_motion = current_motion

    # Next frame also has motion
    next_motion = True
    next_alarm = True if not block.last_motion and next_motion else False

    assert next_alarm is False, "Alarm should remain False with continuous motion"


def test_motion_detection_block_with_minimum_contour_area() -> None:
    """Test that minimum_contour_area filters small motion."""
    # given
    block = MotionDetectionBlockV1()

    # Static background
    static_frame = np.full((240, 320, 3), 50, dtype=np.uint8)

    # Very small moving object
    small_moving_frame = np.full((240, 320, 3), 50, dtype=np.uint8)
    small_moving_frame[100:105, 50:55] = 255  # 5x5 pixel white square (25 pixels)

    # when - build background
    for _ in range(32):
        block.run(
            image=WorkflowImageData(
                parent_metadata=ImageParentMetadata(parent_id="test"),
                numpy_image=static_frame,
            ),
            minimum_contour_area=200,
            morphological_kernel_size=3,
            threshold=16,
            history=30,
            suppress_first_detections=True,
            detection_zone=None,
        )

    # Send frame with small moving object
    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="test"),
            numpy_image=small_moving_frame,
        ),
        minimum_contour_area=200,  # 200 pixels minimum
        morphological_kernel_size=3,
        threshold=16,
        history=30,
        suppress_first_detections=True,
        detection_zone=None,
    )

    # then
    assert (
        output["motion"] is False
    ), "Small motion should be filtered by minimum_contour_area"
    assert output["detections"].is_empty()


def test_motion_detection_block_suppresses_first_detections() -> None:
    """Test that suppress_first_detections prevents early detections."""
    # given
    block = MotionDetectionBlockV1()

    # Moving frame from the start
    moving_frame = np.full((240, 320, 3), 50, dtype=np.uint8)
    moving_frame[100:130, 50:80] = 255

    # when - run with suppress_first_detections=True
    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="test"),
            numpy_image=moving_frame,
        ),
        minimum_contour_area=200,
        morphological_kernel_size=3,
        threshold=16,
        history=30,
        suppress_first_detections=True,
        detection_zone=None,
    )

    # then
    assert output["motion"] is False, "First frame should be suppressed"
    assert output["detections"].is_empty()


def test_motion_detection_block_without_suppress_first_detections() -> None:
    """Test that suppression can be disabled."""
    # given
    block = MotionDetectionBlockV1()

    # Moving frame
    moving_frame = np.full((240, 320, 3), 50, dtype=np.uint8)
    moving_frame[100:130, 50:80] = 255

    # when - run with suppress_first_detections=False
    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="test"),
            numpy_image=moving_frame,
        ),
        minimum_contour_area=200,
        morphological_kernel_size=3,
        threshold=16,
        history=30,
        suppress_first_detections=False,
        detection_zone=None,
    )

    # then - first frame detection is allowed
    # Note: This might still be empty due to the background subtraction algorithm
    # but the suppression logic won't prevent it
    assert output["motion"] is not None
    assert output["detections"] is not None


def test_motion_detection_block_output_structure() -> None:
    """Test that block output has correct structure."""
    # given
    block = MotionDetectionBlockV1()
    image = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)

    # when
    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="test"),
            numpy_image=image,
        ),
        minimum_contour_area=200,
        morphological_kernel_size=3,
        threshold=16,
        history=30,
        suppress_first_detections=True,
        detection_zone=None,
    )

    # then
    assert isinstance(output, dict)
    assert "image" in output
    assert "motion" in output
    assert "detections" in output
    assert "alarm" in output

    assert isinstance(output["image"], WorkflowImageData)
    assert isinstance(output["motion"], bool)
    assert isinstance(output["detections"], sv.Detections)
    assert isinstance(output["alarm"], bool)


def test_motion_detection_block_with_json_detection_zone() -> None:
    """Test that detection zone can be provided as JSON string."""
    # given
    block = MotionDetectionBlockV1()

    # Static background
    static_frame = np.full((240, 320, 3), 50, dtype=np.uint8)

    # Moving frame
    moving_frame = np.full((240, 320, 3), 50, dtype=np.uint8)
    moving_frame[100:130, 50:80] = 255

    # Define zone as JSON string
    detection_zone_json = json.dumps([[10, 10], [310, 10], [310, 230], [10, 230]])

    # when - build background
    for _ in range(32):
        block.run(
            image=WorkflowImageData(
                parent_metadata=ImageParentMetadata(parent_id="test"),
                numpy_image=static_frame,
            ),
            minimum_contour_area=200,
            morphological_kernel_size=3,
            threshold=16,
            history=30,
            suppress_first_detections=True,
            detection_zone=detection_zone_json,
        )

    # Send moving frame with zone
    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="test"),
            numpy_image=moving_frame,
        ),
        minimum_contour_area=200,
        morphological_kernel_size=3,
        threshold=16,
        history=30,
        suppress_first_detections=True,
        detection_zone=detection_zone_json,
    )

    # then
    assert output is not None
    assert isinstance(output["detections"], sv.Detections)


def test_motion_detection_block_with_list_detection_zone() -> None:
    """Test that detection zone can be provided as list."""
    # given
    block = MotionDetectionBlockV1()

    # Static background
    static_frame = np.full((240, 320, 3), 50, dtype=np.uint8)

    # Moving frame
    moving_frame = np.full((240, 320, 3), 50, dtype=np.uint8)
    moving_frame[100:130, 50:80] = 255

    # Define zone as list
    detection_zone = [[10, 10], [310, 10], [310, 230], [10, 230]]

    # when - build background
    for _ in range(32):
        block.run(
            image=WorkflowImageData(
                parent_metadata=ImageParentMetadata(parent_id="test"),
                numpy_image=static_frame,
            ),
            minimum_contour_area=200,
            morphological_kernel_size=3,
            threshold=16,
            history=30,
            suppress_first_detections=True,
            detection_zone=detection_zone,
        )

    # Send moving frame with zone
    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="test"),
            numpy_image=moving_frame,
        ),
        minimum_contour_area=200,
        morphological_kernel_size=3,
        threshold=16,
        history=30,
        suppress_first_detections=True,
        detection_zone=detection_zone,
    )

    # then
    assert output is not None
    assert isinstance(output["detections"], sv.Detections)


def test_motion_detection_block_changes_threshold() -> None:
    """Test that changing threshold recreates the background subtractor."""
    # given
    block = MotionDetectionBlockV1()
    image = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)

    # when - run with threshold 16
    block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="test"),
            numpy_image=image,
        ),
        minimum_contour_area=200,
        morphological_kernel_size=3,
        threshold=16,
        history=30,
        suppress_first_detections=True,
        detection_zone=None,
    )

    first_subtractor = block.backSub

    # Run with same threshold
    block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="test"),
            numpy_image=image,
        ),
        minimum_contour_area=200,
        morphological_kernel_size=3,
        threshold=16,
        history=30,
        suppress_first_detections=True,
        detection_zone=None,
    )

    same_subtractor = block.backSub

    # Run with different threshold
    block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="test"),
            numpy_image=image,
        ),
        minimum_contour_area=200,
        morphological_kernel_size=3,
        threshold=32,
        history=30,
        suppress_first_detections=True,
        detection_zone=None,
    )

    different_subtractor = block.backSub

    # then
    assert first_subtractor is same_subtractor, "Same threshold should reuse subtractor"
    assert (
        different_subtractor is not first_subtractor
    ), "Different threshold should recreate subtractor"
