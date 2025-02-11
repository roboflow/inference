import datetime

import numpy as np
import pytest
import supervision as sv

from inference.core.workflows.core_steps.analytics.velocity.v1 import VelocityBlockV1
from inference.core.workflows.execution_engine.entities.base import (
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
        "velocity": {
            1: [0.0, 0.0],
            2: [0.0, 0.0],
        },
        "speed": {
            1: 0.0,
            2: 0.0,
        },
        "smoothed_velocity": {
            1: [0.0, 0.0],
            2: [0.0, 0.0],
        },
        "smoothed_speed": {
            1: 0.0,
            2: 0.0,
        },
    }
    assert frame1_result == {"velocity_detections": frame1_detections}
    assert (
        frame1_result["velocity_detections"].data["velocity"]
        == expected_data_frame1["velocity"]
    )
    assert (
        frame1_result["velocity_detections"].data["speed"]
        == expected_data_frame1["speed"]
    )
    assert (
        frame1_result["velocity_detections"].data["smoothed_velocity"]
        == expected_data_frame1["smoothed_velocity"]
    )
    assert (
        frame1_result["velocity_detections"].data["smoothed_speed"]
        == expected_data_frame1["smoothed_speed"]
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
        "velocity": {
            1: [0.005, 0.0],
            2: [0.0, 0.005],
        },
        "speed": {
            1: 0.005,
            2: 0.005,
        },
        "smoothed_velocity": {
            1: [0.0025, 0.0],
            2: [0.0, 0.0025],
        },
        "smoothed_speed": {
            1: 0.0025,
            2: 0.0025,
        },
    }
    assert frame2_result == {"velocity_detections": frame2_detections}
    assert (
        frame2_result["velocity_detections"].data["velocity"]
        == expected_data_frame2["velocity"]
    )
    assert (
        frame2_result["velocity_detections"].data["speed"]
        == expected_data_frame2["speed"]
    )
    assert (
        frame2_result["velocity_detections"].data["smoothed_velocity"]
        == expected_data_frame2["smoothed_velocity"]
    )
    assert (
        frame2_result["velocity_detections"].data["smoothed_speed"]
        == expected_data_frame2["smoothed_speed"]
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
        "velocity": {
            1: [0.005, 0.0],
            2: [0.0, 0.0],
        },
        "speed": {
            1: 0.005,
            2: 0.0,
        },
        "smoothed_velocity": {
            1: [0.0025, 0.0],
            2: [0.0, 0.0],
        },
        "smoothed_speed": {
            1: 0.0025,
            2: 0.0,
        },
    }
    assert (
        frame2_result["velocity_detections"].data["velocity"]
        == expected_data_frame2["velocity"]
    )
    assert (
        frame2_result["velocity_detections"].data["speed"]
        == expected_data_frame2["speed"]
    )
    assert (
        frame2_result["velocity_detections"].data["smoothed_velocity"]
        == expected_data_frame2["smoothed_velocity"]
    )
    assert (
        frame2_result["velocity_detections"].data["smoothed_speed"]
        == expected_data_frame2["smoothed_speed"]
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
        "velocity": {
            1: [0.0, 0.0],
        },
        "speed": {
            1: 0.0,
        },
        "smoothed_velocity": {
            1: [0.0, 0.0],
        },
        "smoothed_speed": {
            1: 0.0,
        },
    }
    assert (
        frame2_result["velocity_detections"].data["velocity"]
        == expected_data_frame2["velocity"]
    )
    assert (
        frame2_result["velocity_detections"].data["speed"]
        == expected_data_frame2["speed"]
    )
    assert (
        frame2_result["velocity_detections"].data["smoothed_velocity"]
        == expected_data_frame2["smoothed_velocity"]
    )
    assert (
        frame2_result["velocity_detections"].data["smoothed_speed"]
        == expected_data_frame2["smoothed_speed"]
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
        "velocity": {
            1: [0.005, 0.0],
            2: [0.0, 0.005],
            3: [-0.005, -0.005],
        },
        "speed": {
            1: 0.005,
            2: 0.005,
            3: 0.0070710678118654755,  # sqrt(0.005^2 + 0.005^2)
        },
        "smoothed_velocity": {
            1: [0.0015, 0.0],
            2: [0.0, 0.0015],
            3: [-0.0015, -0.0015],
        },
        "smoothed_speed": {
            1: 0.0015,
            2: 0.0015,
            3: 0.002121320343559643,  # sqrt(0.0015^2 + 0.0015^2)
        },
    }
    assert (
        frame2_result["velocity_detections"].data["velocity"]
        == expected_data_frame2["velocity"]
    )
    assert frame2_result["velocity_detections"].data["speed"] == pytest.approx(
        expected_data_frame2["speed"], rel=1e-5
    )
    assert frame2_result["velocity_detections"].data[
        "smoothed_velocity"
    ] == pytest.approx(expected_data_frame2["smoothed_velocity"], rel=1e-5)
    assert frame2_result["velocity_detections"].data["smoothed_speed"] == pytest.approx(
        expected_data_frame2["smoothed_speed"], rel=1e-5
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
        "velocity": {
            1: [0.005, 0.0],
            3: [0.0, 0.0],
        },
        "speed": {
            1: 0.005,
            3: 0.0,
        },
        "smoothed_velocity": {
            1: [0.0025, 0.0],
            3: [0.0, 0.0],
        },
        "smoothed_speed": {
            1: 0.0025,
            3: 0.0,
        },
    }
    assert (
        frame2_result["velocity_detections"].data["velocity"]
        == expected_data_frame2["velocity"]
    )
    assert (
        frame2_result["velocity_detections"].data["speed"]
        == expected_data_frame2["speed"]
    )
    assert (
        frame2_result["velocity_detections"].data["smoothed_velocity"]
        == expected_data_frame2["smoothed_velocity"]
    )
    assert (
        frame2_result["velocity_detections"].data["smoothed_speed"]
        == expected_data_frame2["smoothed_speed"]
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
        "velocity": {
            1: [1.9, 1.9],
        },
        "speed": {
            1: 2.68675135,  # sqrt(1.9^2 + 1.9^2)
        },
        "smoothed_velocity": {
            1: [0.95, 0.95],
        },
        "smoothed_speed": {
            1: 1.343375675,  # sqrt(0.95^2 + 0.95^2)
        },
    }
    assert (
        frame2_result["velocity_detections"].data["velocity"]
        == expected_data_frame2["velocity"]
    )
    assert frame2_result["velocity_detections"].data["speed"] == pytest.approx(
        expected_data_frame2["speed"], rel=1e-4
    )
    assert frame2_result["velocity_detections"].data[
        "smoothed_velocity"
    ] == pytest.approx(expected_data_frame2["smoothed_velocity"], rel=1e-4)
    assert frame2_result["velocity_detections"].data["smoothed_speed"] == pytest.approx(
        expected_data_frame2["smoothed_speed"], rel=1e-4
    )
