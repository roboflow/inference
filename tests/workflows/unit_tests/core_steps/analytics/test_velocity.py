import datetime

import numpy as np
import pytest
import supervision as sv

from inference.core.workflows.core_steps.analytics.velocity.v1 import VelocityBlockV1
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)


def test_velocity_block_basic_calculation() -> None:
    # given
    velocity_block = VelocityBlockV1()

    # Initial frame detections
    frame1_detections = sv.Detections(
        xyxy=np.array(
            [
                [100, 100, 110, 110],  # Object 1
                [200, 200, 210, 210],  # Object 2
            ]
        ),
        tracker_id=np.array([1, 2]),
    )

    metadata1 = VideoMetadata(
        video_identifier="vid_1",
        frame_number=1,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570800).astimezone(
            tz=datetime.timezone.utc
        ),
    )
    image1 = WorkflowImageData(
        parent_metadata=None,
        numpy_image=np.array([1]),
        video_metadata=metadata1,
    )

    # Run on first frame
    frame1_result = velocity_block.run(
        detections=frame1_detections,
        image=image1,
        smoothing_alpha=0.5,
        pixels_per_meter=1000,  # 1000 pixels = 1 meter
    )

    # Since this is the first frame, velocities should be zero
    expected_data_frame1 = {
        "velocity": np.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        ),
        "speed": np.array(
            [
                0.0,
                0.0,
            ]
        ),
        "smoothed_velocity": np.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        ),
        "smoothed_speed": np.array(
            [
                0.0,
                0.0,
            ]
        ),
    }
    assert frame1_result == {"velocity_detections": frame1_detections}
    assert np.allclose(
        frame1_result["velocity_detections"].data["velocity"],
        expected_data_frame1["velocity"],
    )
    assert np.allclose(
        frame1_result["velocity_detections"].data["speed"],
        expected_data_frame1["speed"],
    )
    assert np.allclose(
        frame1_result["velocity_detections"].data["smoothed_velocity"],
        expected_data_frame1["smoothed_velocity"],
    )
    assert np.allclose(
        frame1_result["velocity_detections"].data["smoothed_speed"],
        expected_data_frame1["smoothed_speed"],
    )

    # Second frame detections with movement
    frame2_detections = sv.Detections(
        xyxy=np.array(
            [
                [105, 100, 115, 110],  # Object 1 moved +5 px right
                [200, 205, 210, 215],  # Object 2 moved +5 px down
            ]
        ),
        tracker_id=np.array([1, 2]),
    )

    metadata2 = VideoMetadata(
        video_identifier="vid_1",
        frame_number=2,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570801).astimezone(
            tz=datetime.timezone.utc
        ),  # 1 second later
    )
    image2 = WorkflowImageData(
        parent_metadata=None,
        numpy_image=np.array([1]),
        video_metadata=metadata2,
    )

    # Run on second frame
    frame2_result = velocity_block.run(
        detections=frame2_detections,
        image=image2,
        smoothing_alpha=0.5,
        pixels_per_meter=1000,  # 1000 pixels = 1 meter
    )

    # Expected velocities:
    # Object 1: [5 px/s, 0 px/s] => [0.005 m/s, 0.0 m/s]
    # Object 2: [0 px/s, 5 px/s] => [0.0 m/s, 0.005 m/s]

    # Expected smoothed velocities:
    # Object 1: 0.5 * [0.005, 0.0] + 0.5 * [0.0, 0.0] = [0.0025, 0.0]
    # Object 2: 0.5 * [0.0, 0.005] + 0.5 * [0.0, 0.0] = [0.0, 0.0025]

    expected_data_frame2 = {
        "velocity": np.array(
            [
                [0.005, 0.0],
                [0.0, 0.005],
            ]
        ),
        "speed": np.array(
            [
                0.005,
                0.005,
            ]
        ),
        "smoothed_velocity": np.array(
            [
                [0.0025, 0.0],
                [0.0, 0.0025],
            ]
        ),
        "smoothed_speed": np.array(
            [
                0.0025,
                0.0025,
            ]
        ),
    }
    assert frame2_result == {"velocity_detections": frame2_detections}
    assert np.allclose(
        frame2_result["velocity_detections"].data["velocity"],
        expected_data_frame2["velocity"],
    )
    assert np.allclose(
        frame2_result["velocity_detections"].data["speed"],
        expected_data_frame2["speed"],
    )
    assert np.allclose(
        frame2_result["velocity_detections"].data["smoothed_velocity"],
        expected_data_frame2["smoothed_velocity"],
    )
    assert np.allclose(
        frame2_result["velocity_detections"].data["smoothed_speed"],
        expected_data_frame2["smoothed_speed"],
    )


def test_velocity_block_new_tracker_id() -> None:
    # given
    velocity_block = VelocityBlockV1()

    # Frame 1 detections
    frame1_detections = sv.Detections(
        xyxy=np.array(
            [
                [100, 100, 110, 110],  # Object 1
            ]
        ),
        tracker_id=np.array([1]),
    )

    metadata1 = VideoMetadata(
        video_identifier="vid_1",
        frame_number=1,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570800).astimezone(
            tz=datetime.timezone.utc
        ),
    )
    image1 = WorkflowImageData(
        parent_metadata=None,
        numpy_image=np.array([1]),
        video_metadata=metadata1,
    )

    # Run on first frame
    velocity_block.run(
        detections=frame1_detections,
        image=image1,
        smoothing_alpha=0.5,
        pixels_per_meter=1000,  # 1000 pixels = 1 meter
    )

    # Second frame detections with a new tracker_id
    frame2_detections = sv.Detections(
        xyxy=np.array(
            [
                [105, 100, 115, 110],  # Object 1 moved +5 px right
                [200, 200, 210, 210],  # New Object 2
            ]
        ),
        tracker_id=np.array([1, 2]),
    )

    metadata2 = VideoMetadata(
        video_identifier="vid_1",
        frame_number=2,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570801).astimezone(
            tz=datetime.timezone.utc
        ),
    )
    image2 = WorkflowImageData(
        parent_metadata=None,
        numpy_image=np.array([1]),
        video_metadata=metadata2,
    )

    # Run on second frame
    frame2_result = velocity_block.run(
        detections=frame2_detections,
        image=image2,
        smoothing_alpha=0.5,
        pixels_per_meter=1000,
    )

    # Expected velocities:
    # Object 1: [5 px/s, 0 px/s] => [0.005 m/s, 0.0 m/s]
    # Object 2: [0 px/s, 0 px/s] => [0.0 m/s, 0.0 m/s] (first appearance)

    # Expected smoothed velocities:
    # Object 1: 0.5 * [0.005, 0.0] + 0.5 * [0.0, 0.0] = [0.0025, 0.0]
    # Object 2: [0.0, 0.0] (first appearance)

    expected_data_frame2 = {
        "velocity": np.array(
            [
                [0.005, 0.0],
                [0.0, 0.0],
            ]
        ),
        "speed": np.array(
            [
                0.005,
                0.0,
            ]
        ),
        "smoothed_velocity": np.array(
            [
                [0.0025, 0.0],
                [0.0, 0.0],
            ]
        ),
        "smoothed_speed": np.array(
            [
                0.0025,
                0.0,
            ]
        ),
    }
    assert np.allclose(
        frame2_result["velocity_detections"].data["velocity"],
        expected_data_frame2["velocity"],
    )
    assert np.allclose(
        frame2_result["velocity_detections"].data["speed"],
        expected_data_frame2["speed"],
    )
    assert np.allclose(
        frame2_result["velocity_detections"].data["smoothed_velocity"],
        expected_data_frame2["smoothed_velocity"],
    )
    assert np.allclose(
        frame2_result["velocity_detections"].data["smoothed_speed"],
        expected_data_frame2["smoothed_speed"],
    )


def test_velocity_block_missing_tracker_id() -> None:
    # given
    velocity_block = VelocityBlockV1()

    # Detections without tracker_id
    detections = sv.Detections(
        xyxy=np.array([[100, 100, 110, 110]]),
        # tracker_id is missing
    )

    metadata = VideoMetadata(
        video_identifier="vid_1",
        frame_number=1,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570800).astimezone(
            tz=datetime.timezone.utc
        ),
    )
    image = WorkflowImageData(
        parent_metadata=None,
        numpy_image=np.array([1]),
        video_metadata=metadata,
    )

    # when / then
    with pytest.raises(
        ValueError,
        match="tracker_id not initialized, VelocityBlock requires detections to be tracked",
    ):
        velocity_block.run(
            detections=detections,
            image=image,
            smoothing_alpha=0.5,
            pixels_per_meter=1000,
        )


def test_velocity_block_invalid_smoothing_alpha() -> None:
    # given
    velocity_block = VelocityBlockV1()

    detections = sv.Detections(
        xyxy=np.array([[100, 100, 110, 110]]),
        tracker_id=np.array([1]),
    )

    metadata = VideoMetadata(
        video_identifier="vid_1",
        frame_number=1,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570800).astimezone(
            tz=datetime.timezone.utc
        ),
    )
    image = WorkflowImageData(
        parent_metadata=None,
        numpy_image=np.array([1]),
        video_metadata=metadata,
    )

    # when / then: smoothing_alpha <= 0
    with pytest.raises(
        ValueError,
        match="smoothing_alpha must be between 0 \\(exclusive\\) and 1 \\(inclusive\\)",
    ):
        velocity_block.run(
            detections=detections,
            image=image,
            smoothing_alpha=0.0,
            pixels_per_meter=1000,
        )

    # when / then: smoothing_alpha > 1
    with pytest.raises(
        ValueError,
        match="smoothing_alpha must be between 0 \\(exclusive\\) and 1 \\(inclusive\\)",
    ):
        velocity_block.run(
            detections=detections,
            image=image,
            smoothing_alpha=1.5,
            pixels_per_meter=1000,
        )


def test_velocity_block_invalid_pixels_per_meter() -> None:
    # given
    velocity_block = VelocityBlockV1()

    detections = sv.Detections(
        xyxy=np.array([[100, 100, 110, 110]]),
        tracker_id=np.array([1]),
    )

    metadata = VideoMetadata(
        video_identifier="vid_1",
        frame_number=1,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570800).astimezone(
            tz=datetime.timezone.utc
        ),
    )
    image = WorkflowImageData(
        parent_metadata=None,
        numpy_image=np.array([1]),
        video_metadata=metadata,
    )

    # when / then: pixels_per_meter <= 0
    with pytest.raises(
        ValueError,
        match="pixels_per_meter must be greater than 0",
    ):
        velocity_block.run(
            detections=detections,
            image=image,
            smoothing_alpha=0.5,
            pixels_per_meter=0.0,
        )

    with pytest.raises(
        ValueError,
        match="pixels_per_meter must be greater than 0",
    ):
        velocity_block.run(
            detections=detections,
            image=image,
            smoothing_alpha=0.5,
            pixels_per_meter=-1000,
        )


def test_velocity_block_zero_delta_time() -> None:
    # given
    velocity_block = VelocityBlockV1()

    # Frame 1 detections
    frame1_detections = sv.Detections(
        xyxy=np.array([[100, 100, 110, 110]]),
        tracker_id=np.array([1]),
    )

    metadata1 = VideoMetadata(
        video_identifier="vid_1",
        frame_number=1,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570800).astimezone(
            tz=datetime.timezone.utc
        ),
    )
    image1 = WorkflowImageData(
        parent_metadata=None,
        numpy_image=np.array([1]),
        video_metadata=metadata1,
    )

    # Run on first frame
    velocity_block.run(
        detections=frame1_detections,
        image=image1,
        smoothing_alpha=0.5,
        pixels_per_meter=1000,
    )

    # Frame 2 with same timestamp (delta_time = 0)
    frame2_detections = sv.Detections(
        xyxy=np.array([[105, 100, 115, 110]]),  # Moved +5 px right
        tracker_id=np.array([1]),
    )

    metadata2 = VideoMetadata(
        video_identifier="vid_1",
        frame_number=2,
        frame_timestamp=metadata1.frame_timestamp,
    )
    image2 = WorkflowImageData(
        parent_metadata=None,
        numpy_image=np.array([1]),
        video_metadata=metadata2,
    )

    frame2_result = velocity_block.run(
        detections=frame2_detections,
        image=image2,
        smoothing_alpha=0.5,
        pixels_per_meter=1000,
    )

    # Expected velocities: [0.0, 0.0] due to delta_time = 0
    # Smoothed velocities: remains [0.0, 0.0]

    expected_data_frame2 = {
        "velocity": np.array(
            [
                [0.0, 0.0],
            ]
        ),
        "speed": np.array(
            [
                0.0,
            ]
        ),
        "smoothed_velocity": np.array(
            [
                [0.0, 0.0],
            ]
        ),
        "smoothed_speed": np.array(
            [
                0.0,
            ]
        ),
    }
    assert np.allclose(
        frame2_result["velocity_detections"].data["velocity"],
        expected_data_frame2["velocity"],
    )
    assert np.allclose(
        frame2_result["velocity_detections"].data["speed"],
        expected_data_frame2["speed"],
    )
    assert np.allclose(
        frame2_result["velocity_detections"].data["smoothed_velocity"],
        expected_data_frame2["smoothed_velocity"],
    )
    assert np.allclose(
        frame2_result["velocity_detections"].data["smoothed_speed"],
        expected_data_frame2["smoothed_speed"],
    )


def test_velocity_block_multiple_objects_with_movement() -> None:
    # given
    velocity_block = VelocityBlockV1()

    # Frame 1 detections
    frame1_detections = sv.Detections(
        xyxy=np.array(
            [
                [100, 100, 110, 110],  # Object 1
                [200, 200, 210, 210],  # Object 2
                [300, 300, 310, 310],  # Object 3
            ]
        ),
        tracker_id=np.array([1, 2, 3]),
    )

    metadata1 = VideoMetadata(
        video_identifier="vid_1",
        frame_number=1,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570800).astimezone(
            tz=datetime.timezone.utc
        ),
    )
    image1 = WorkflowImageData(
        parent_metadata=None,
        numpy_image=np.array([1]),
        video_metadata=metadata1,
    )

    # Run on first frame
    velocity_block.run(
        detections=frame1_detections,
        image=image1,
        smoothing_alpha=0.3,
        pixels_per_meter=1000,  # 1000 pixels = 1 meter
    )

    # Frame 2 detections with movements
    frame2_detections = sv.Detections(
        xyxy=np.array(
            [
                [105, 100, 115, 110],  # Object 1 moved +5 px right
                [200, 205, 210, 215],  # Object 2 moved +5 px down
                [295, 295, 305, 305],  # Object 3 moved -5 px left and up
            ]
        ),
        tracker_id=np.array([1, 2, 3]),
    )

    metadata2 = VideoMetadata(
        video_identifier="vid_1",
        frame_number=2,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570801).astimezone(
            tz=datetime.timezone.utc
        ),  # 1 second later
    )
    image2 = WorkflowImageData(
        parent_metadata=None,
        numpy_image=np.array([1]),
        video_metadata=metadata2,
    )

    frame2_result = velocity_block.run(
        detections=frame2_detections,
        image=image2,
        smoothing_alpha=0.3,
        pixels_per_meter=1000,
    )

    # Expected velocities:
    # Object 1: [5 px/s, 0 px/s] => [0.005 m/s, 0.0 m/s]
    # Object 2: [0 px/s, 5 px/s] => [0.0 m/s, 0.005 m/s]
    # Object 3: [-5 px/s, -5 px/s] => [-0.005 m/s, -0.005 m/s]

    # Expected smoothed velocities:
    # Object 1: 0.3 * [0.005, 0.0] + 0.7 * [0.0, 0.0] = [0.0015, 0.0]
    # Object 2: 0.3 * [0.0, 0.005] + 0.7 * [0.0, 0.0] = [0.0, 0.0015]
    # Object 3: 0.3 * [-0.005, -0.005] + 0.7 * [0.0, 0.0] = [-0.0015, -0.0015]

    expected_data_frame2 = {
        "velocity": np.array(
            [
                [0.005, 0.0],
                [0.0, 0.005],
                [-0.005, -0.005],
            ]
        ),
        "speed": np.array(
            [
                0.005,
                0.005,
                0.0070710678118654755,  # sqrt(0.005^2 + 0.005^2)
            ]
        ),
        "smoothed_velocity": np.array(
            [
                [0.0015, 0.0],
                [0.0, 0.0015],
                [-0.0015, -0.0015],
            ]
        ),
        "smoothed_speed": np.array(
            [
                0.0015,
                0.0015,
                0.002121320343559643,  # sqrt(0.0015^2 + 0.0015^2)
            ]
        ),
    }
    assert np.allclose(
        frame2_result["velocity_detections"].data["velocity"],
        expected_data_frame2["velocity"],
    )
    assert np.allclose(
        frame2_result["velocity_detections"].data["speed"],
        expected_data_frame2["speed"],
    )
    assert np.allclose(
        frame2_result["velocity_detections"].data["smoothed_velocity"],
        expected_data_frame2["smoothed_velocity"],
    )
    assert np.allclose(
        frame2_result["velocity_detections"].data["smoothed_speed"],
        expected_data_frame2["smoothed_speed"],
    )
    assert np.allclose(
        frame2_result["velocity_detections"].data["velocity"],
        expected_data_frame2["velocity"],
    )
    assert np.allclose(
        frame2_result["velocity_detections"].data["speed"],
        expected_data_frame2["speed"],
    )
    assert np.allclose(
        frame2_result["velocity_detections"].data["smoothed_velocity"],
        expected_data_frame2["smoothed_velocity"],
    )
    assert np.allclose(
        frame2_result["velocity_detections"].data["smoothed_speed"],
        expected_data_frame2["smoothed_speed"],
    )


def test_velocity_block_inconsistent_tracker_ids() -> None:
    # given
    velocity_block = VelocityBlockV1()

    # Frame 1 detections
    frame1_detections = sv.Detections(
        xyxy=np.array(
            [
                [100, 100, 110, 110],  # Object 1
                [200, 200, 210, 210],  # Object 2
            ]
        ),
        tracker_id=np.array([1, 2]),
    )

    metadata1 = VideoMetadata(
        video_identifier="vid_1",
        frame_number=1,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570800).astimezone(
            tz=datetime.timezone.utc
        ),
    )
    image1 = WorkflowImageData(
        parent_metadata=None,
        numpy_image=np.array([1]),
        video_metadata=metadata1,
    )

    velocity_block.run(
        detections=frame1_detections,
        image=image1,
        smoothing_alpha=0.5,
        pixels_per_meter=1000,
    )

    # Frame 2 detections with missing tracker_id=2 and new tracker_id=3
    frame2_detections = sv.Detections(
        xyxy=np.array(
            [
                [105, 100, 115, 110],  # Object 1 moved +5 px right
                [300, 300, 310, 310],  # New Object 3
            ]
        ),
        tracker_id=np.array([1, 3]),
    )

    metadata2 = VideoMetadata(
        video_identifier="vid_1",
        frame_number=2,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570801).astimezone(
            tz=datetime.timezone.utc
        ),
    )
    image2 = WorkflowImageData(
        parent_metadata=None,
        numpy_image=np.array([1]),
        video_metadata=metadata2,
    )

    frame2_result = velocity_block.run(
        detections=frame2_detections,
        image=image2,
        smoothing_alpha=0.5,
        pixels_per_meter=1000,
    )

    # Expected velocities:
    # Object 1: [5 px/s, 0 px/s] => [0.005 m/s, 0.0 m/s]
    # Object 3: [0 px/s, 0 px/s] => [0.0 m/s, 0.0 m/s] (first appearance)

    expected_data_frame2 = {
        "velocity": np.array(
            [
                [0.005, 0.0],
                [0.0, 0.0],
            ]
        ),
        "speed": np.array(
            [
                0.005,
                0.0,
            ]
        ),
        "smoothed_velocity": np.array(
            [
                [0.0025, 0.0],
                [0.0, 0.0],
            ]
        ),
        "smoothed_speed": np.array(
            [
                0.0025,
                0.0,
            ]
        ),
    }
    assert np.allclose(
        frame2_result["velocity_detections"].data["velocity"],
        expected_data_frame2["velocity"],
    )
    assert np.allclose(
        frame2_result["velocity_detections"].data["speed"],
        expected_data_frame2["speed"],
    )
    assert np.allclose(
        frame2_result["velocity_detections"].data["smoothed_velocity"],
        expected_data_frame2["smoothed_velocity"],
    )
    assert np.allclose(
        frame2_result["velocity_detections"].data["smoothed_speed"],
        expected_data_frame2["smoothed_speed"],
    )


def test_velocity_block_large_movement() -> None:
    # given
    velocity_block = VelocityBlockV1()

    # Frame 1 detections
    frame1_detections = sv.Detections(
        xyxy=np.array([[100, 100, 110, 110]]),
        tracker_id=np.array([1]),
    )

    metadata1 = VideoMetadata(
        video_identifier="vid_1",
        frame_number=1,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570800).astimezone(
            tz=datetime.timezone.utc
        ),
    )
    image1 = WorkflowImageData(
        parent_metadata=None,
        numpy_image=np.array([1]),
        video_metadata=metadata1,
    )

    # Run on first frame
    velocity_block.run(
        detections=frame1_detections,
        image=image1,
        smoothing_alpha=0.5,
        pixels_per_meter=1000,  # 1000 pixels = 1 meter
    )

    # Frame 2 detections with large movement
    frame2_detections = sv.Detections(
        xyxy=np.array([[2000, 2000, 2010, 2010]]),  # Moved +1900 px right and down
        tracker_id=np.array([1]),
    )

    metadata2 = VideoMetadata(
        video_identifier="vid_1",
        frame_number=2,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570801).astimezone(
            tz=datetime.timezone.utc
        ),
    )
    image2 = WorkflowImageData(
        parent_metadata=None,
        numpy_image=np.array([1]),
        video_metadata=metadata2,
    )

    frame2_result = velocity_block.run(
        detections=frame2_detections,
        image=image2,
        smoothing_alpha=0.5,
        pixels_per_meter=1000,
    )

    # Expected velocities:
    # [1900 px/s, 1900 px/s] => [1.9 m/s, 1.9 m/s]

    # Expected smoothed velocities:
    # 0.5 * [1.9, 1.9] + 0.5 * [0.0, 0.0] = [0.95, 0.95]

    expected_data_frame2 = {
        "velocity": np.array(
            [
                [1.9, 1.9],
            ]
        ),
        "speed": np.array(
            [
                2.68700577,  # sqrt(1.9^2 + 1.9^2)
            ]
        ),
        "smoothed_velocity": np.array(
            [
                [0.95, 0.95],
            ]
        ),
        "smoothed_speed": np.array(
            [
                1.34350288,  # sqrt(0.95^2 + 0.95^2)
            ]
        ),
    }
    assert np.allclose(
        frame2_result["velocity_detections"].data["velocity"],
        expected_data_frame2["velocity"],
    )
    assert np.allclose(
        frame2_result["velocity_detections"].data["speed"],
        expected_data_frame2["speed"],
    )
    assert np.allclose(
        frame2_result["velocity_detections"].data["smoothed_velocity"],
        expected_data_frame2["smoothed_velocity"],
    )
    assert np.allclose(
        frame2_result["velocity_detections"].data["smoothed_speed"],
        expected_data_frame2["smoothed_speed"],
    )


# --- tensor-native sibling ---------------------------------------------------
# These tests pin numerical parity with the numpy block above plus the tensor
# block's device-resident state semantics.


def _tensor_velocity_imports():
    torch = pytest.importorskip("torch")
    from inference.core.workflows.core_steps.analytics.velocity.v1_tensor import (
        VelocityBlockV1 as TensorVelocityBlockV1,
    )
    from inference_models.models.base.object_detection import (
        Detections as NativeDetections,
    )

    return torch, TensorVelocityBlockV1, NativeDetections


def _tensor_image(video_id: str, frame_number: int, fps: float = 10.0):
    metadata = VideoMetadata(
        video_identifier=video_id,
        frame_number=frame_number,
        frame_timestamp=datetime.datetime.fromtimestamp(1690000000 + frame_number),
        fps=fps,
        comes_from_video_file=True,
    )
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=np.zeros((100, 100, 3), dtype=np.uint8),
        video_metadata=metadata,
    )


def _native_detections(torch, NativeDetections, xyxy_rows, tracker_ids):
    return NativeDetections(
        xyxy=torch.tensor(xyxy_rows, dtype=torch.float32),
        class_id=torch.zeros(len(xyxy_rows), dtype=torch.long),
        confidence=torch.full((len(xyxy_rows),), 0.9),
        image_metadata={"class_names": {0: "object"}},
        bboxes_metadata=[
            {"detection_id": f"d{i}", "tracker_id": tid}
            for i, tid in enumerate(tracker_ids)
        ],
    )


def test_tensor_velocity_two_frame_calculation_matches_numpy_math() -> None:
    # given - one object moving +10px in x over 1 frame @ 10fps (0.1 s)
    torch, TensorVelocityBlockV1, NativeDetections = _tensor_velocity_imports()
    block = TensorVelocityBlockV1()

    # when
    block.run(
        image=_tensor_image("vid", 1),
        detections=_native_detections(
            torch, NativeDetections, [[0.0, 0.0, 10.0, 10.0]], [1]
        ),
        smoothing_alpha=0.5,
        pixels_per_meter=1.0,
    )
    result = block.run(
        image=_tensor_image("vid", 2),
        detections=_native_detections(
            torch, NativeDetections, [[10.0, 0.0, 20.0, 10.0]], [1]
        ),
        smoothing_alpha=0.5,
        pixels_per_meter=1.0,
    )

    # then - dx=10px / 0.1s = 100 px/s; EMA(0.5) of [0,0] -> [100,0] = [50,0]
    metadata = result["velocity_detections"].bboxes_metadata[0]
    assert np.allclose(metadata["velocity"], [100.0, 0.0], atol=1e-3)
    assert np.allclose(metadata["speed"], 100.0, atol=1e-3)
    assert np.allclose(metadata["smoothed_velocity"], [50.0, 0.0], atol=1e-3)
    assert np.allclose(metadata["smoothed_speed"], 50.0, atol=1e-3)


def test_tensor_velocity_new_tracker_and_units() -> None:
    # given
    torch, TensorVelocityBlockV1, NativeDetections = _tensor_velocity_imports()
    block = TensorVelocityBlockV1()

    # when - first sighting, with a pixels_per_meter conversion in play
    result = block.run(
        image=_tensor_image("vid", 1),
        detections=_native_detections(
            torch, NativeDetections, [[0.0, 0.0, 10.0, 10.0]], [7]
        ),
        smoothing_alpha=0.5,
        pixels_per_meter=100.0,
    )

    # then - no previous position: everything is zero
    metadata = result["velocity_detections"].bboxes_metadata[0]
    assert metadata["velocity"] == [0.0, 0.0]
    assert metadata["speed"] == 0.0
    assert metadata["smoothed_velocity"] == [0.0, 0.0]
    assert metadata["smoothed_speed"] == 0.0


def test_tensor_velocity_absent_tracker_reappears_with_preserved_state() -> None:
    # given - tracker 1 seen at frame 1, absent at frame 2, back at frame 3;
    # velocity on reappearance must use the frame-1 position and timestamp
    torch, TensorVelocityBlockV1, NativeDetections = _tensor_velocity_imports()
    block = TensorVelocityBlockV1()
    block.run(
        image=_tensor_image("vid", 1),
        detections=_native_detections(
            torch, NativeDetections, [[0.0, 0.0, 10.0, 10.0]], [1]
        ),
        smoothing_alpha=1.0,
        pixels_per_meter=1.0,
    )
    block.run(
        image=_tensor_image("vid", 2),
        detections=_native_detections(
            torch, NativeDetections, [[100.0, 100.0, 110.0, 110.0]], [2]
        ),
        smoothing_alpha=1.0,
        pixels_per_meter=1.0,
    )

    # when - tracker 1 reappears at frame 3, +20px in x since frame 1 (0.2 s)
    result = block.run(
        image=_tensor_image("vid", 3),
        detections=_native_detections(
            torch, NativeDetections, [[20.0, 0.0, 30.0, 10.0]], [1]
        ),
        smoothing_alpha=1.0,
        pixels_per_meter=1.0,
    )

    # then - 20px / 0.2s = 100 px/s
    metadata = result["velocity_detections"].bboxes_metadata[0]
    assert np.allclose(metadata["velocity"], [100.0, 0.0], atol=1e-3)


def test_tensor_velocity_zero_delta_time_yields_zero_velocity() -> None:
    # given - the same frame timestamp twice
    torch, TensorVelocityBlockV1, NativeDetections = _tensor_velocity_imports()
    block = TensorVelocityBlockV1()
    detections = _native_detections(
        torch, NativeDetections, [[0.0, 0.0, 10.0, 10.0]], [1]
    )
    block.run(
        image=_tensor_image("vid", 5),
        detections=detections,
        smoothing_alpha=0.5,
        pixels_per_meter=1.0,
    )

    # when
    result = block.run(
        image=_tensor_image("vid", 5),
        detections=_native_detections(
            torch, NativeDetections, [[50.0, 0.0, 60.0, 10.0]], [1]
        ),
        smoothing_alpha=0.5,
        pixels_per_meter=1.0,
    )

    # then
    metadata = result["velocity_detections"].bboxes_metadata[0]
    assert metadata["velocity"] == [0.0, 0.0]
    assert metadata["speed"] == 0.0


def test_tensor_velocity_validations() -> None:
    # given
    torch, TensorVelocityBlockV1, NativeDetections = _tensor_velocity_imports()
    block = TensorVelocityBlockV1()
    untracked = _native_detections(
        torch, NativeDetections, [[0.0, 0.0, 10.0, 10.0]], [None]
    )

    # when / then
    with pytest.raises(ValueError, match="tracker_id"):
        block.run(
            image=_tensor_image("vid", 1),
            detections=untracked,
            smoothing_alpha=0.5,
            pixels_per_meter=1.0,
        )
    tracked = _native_detections(torch, NativeDetections, [[0.0, 0.0, 10.0, 10.0]], [1])
    with pytest.raises(ValueError, match="smoothing_alpha"):
        block.run(
            image=_tensor_image("vid", 1),
            detections=tracked,
            smoothing_alpha=0.0,
            pixels_per_meter=1.0,
        )
    with pytest.raises(ValueError, match="pixels_per_meter"):
        block.run(
            image=_tensor_image("vid", 1),
            detections=tracked,
            smoothing_alpha=0.5,
            pixels_per_meter=0.0,
        )


def test_tensor_velocity_state_stays_on_device_when_mps_available() -> None:
    # given
    torch, TensorVelocityBlockV1, NativeDetections = _tensor_velocity_imports()
    if not torch.backends.mps.is_available():
        pytest.skip("device-residency check needs MPS")
    device = torch.device("mps")
    block = TensorVelocityBlockV1()

    def on_device(xyxy_rows, tracker_ids):
        detections = _native_detections(torch, NativeDetections, xyxy_rows, tracker_ids)
        detections.xyxy = detections.xyxy.to(device)
        detections.class_id = detections.class_id.to(device)
        detections.confidence = detections.confidence.to(device)
        return detections

    # when
    block.run(
        image=_tensor_image("vid", 1),
        detections=on_device([[0.0, 0.0, 10.0, 10.0]], [1]),
        smoothing_alpha=0.5,
        pixels_per_meter=1.0,
    )
    result = block.run(
        image=_tensor_image("vid", 2),
        detections=on_device([[10.0, 0.0, 20.0, 10.0]], [1]),
        smoothing_alpha=0.5,
        pixels_per_meter=1.0,
    )

    # then - the tracking state lives on the device
    state = block._states["vid"]
    assert state.positions.device.type == "mps"
    assert state.smoothed_velocities.device.type == "mps"
    metadata = result["velocity_detections"].bboxes_metadata[0]
    assert np.allclose(metadata["velocity"], [100.0, 0.0], atol=1e-2)
