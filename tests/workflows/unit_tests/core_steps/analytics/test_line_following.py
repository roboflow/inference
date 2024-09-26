import datetime

import numpy as np
import pytest
import supervision as sv

from inference.core.workflows.core_steps.analytics.line_following.v1 import (
    LineFollowingAnalyticsBlockV1,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)


def test_line_following_exact_path():
    # Given
    reference_path = [(0, 0), (1, 1), (2, 2), (3, 3)]
    frames = [0, 1, 2, 3]
    detections_list = []
    metadata_list = []

    for i in frames:
        # Create detection at reference_path[i]
        detections = sv.Detections(
            xyxy=np.array(
                [
                    [
                        reference_path[i][0] - 1,
                        reference_path[i][1] - 1,
                        reference_path[i][0] + 1,
                        reference_path[i][1] + 1,
                    ]
                ]
            ),
            tracker_id=np.array([1]),
        )
        detections_list.append(detections)

        # Create metadata
        metadata = VideoMetadata(
            video_identifier="vid_1",
            frame_number=i,
            fps=1,  # Assume 1 fps
            frame_timestamp=datetime.datetime.fromtimestamp(1726570875 + i).astimezone(
                tz=datetime.timezone.utc
            ),
            comes_from_video_file=True,
        )
        metadata_list.append(metadata)

    # Initialize the block
    line_following_block = LineFollowingAnalyticsBlockV1()

    # Run the block for each frame and collect frechet_distance
    frechet_distances = []

    for i in range(len(frames)):
        result = line_following_block.run(
            detections=detections_list[i],
            metadata=metadata_list[i],
            triggering_anchor=sv.Position.CENTER,  # Default
            reference_path=reference_path,
        )
        frechet_distance = result["frechet_distance"]
        frechet_distances.append(frechet_distance)

    # Then

    # Additional assertions to check intermediate results
    assert len(frechet_distances) == len(frames)
    for distance in frechet_distances:
        assert distance == pytest.approx(0.0, abs=1e-6)

    # Since the object follows exactly the reference path, the frechet distance should be zero.
    assert frechet_distances[-1] == pytest.approx(0.0, abs=1e-6)


def test_line_following_with_deviation():
    # Given
    reference_path = [(0, 0), (1, 1), (2, 2), (3, 3)]
    object_path = [(0, 0), (1, 1), (2, 2), (4, 4)]  # Deviates at the last point
    frames = [0, 1, 2, 3]
    detections_list = []
    metadata_list = []

    for i in frames:
        # Create detection at object_path[i]
        detections = sv.Detections(
            xyxy=np.array(
                [
                    [
                        object_path[i][0],
                        object_path[i][1],
                        object_path[i][0] + 1,
                        object_path[i][1] + 1,
                    ]
                ]
            ),
            tracker_id=np.array([1]),
        )
        detections_list.append(detections)

        # Create metadata
        metadata = VideoMetadata(
            video_identifier="vid_1",
            frame_number=i,
            fps=1,  # Assume 1 fps
            frame_timestamp=datetime.datetime.fromtimestamp(1726570875 + i).astimezone(
                tz=datetime.timezone.utc
            ),
            comes_from_video_file=True,
        )
        metadata_list.append(metadata)

    # Initialize the block
    line_following_block = LineFollowingAnalyticsBlockV1()

    # Run the block for each frame and collect frechet_distance
    frechet_distances = []

    for i in range(len(frames)):
        result = line_following_block.run(
            detections=detections_list[i],
            metadata=metadata_list[i],
            triggering_anchor=sv.Position.CENTER,  # Default
            reference_path=reference_path,
        )
        frechet_distance = result["frechet_distance"]
        frechet_distances.append(frechet_distance)

    # Then
    # Since the object deviates from the reference path, the frechet distance should be greater than zero.
    assert frechet_distances[-1] > 0.0


def test_line_following_multiple_objects():
    # Given
    reference_path = [(0, 0), (1, 1), (2, 2), (3, 3)]
    frames = [0, 1, 2, 3]
    detections_list = []
    metadata_list = []

    for i in frames:
        # Create detections for two objects
        detections = sv.Detections(
            xyxy=np.array(
                [
                    [i, i, i + 1, i + 1],  # Object 1 following the path
                    [i + 1, i, i + 2, i + 1],  # Object 2 deviating
                ]
            ),
            tracker_id=np.array([1, 2]),
        )
        detections_list.append(detections)

        # Create metadata
        metadata = VideoMetadata(
            video_identifier="vid_1",
            frame_number=i,
            fps=1,  # Assume 1 fps
            frame_timestamp=datetime.datetime.fromtimestamp(1726570875 + i).astimezone(
                tz=datetime.timezone.utc
            ),
            comes_from_video_file=True,
        )
        metadata_list.append(metadata)

    # Initialize the block
    line_following_block = LineFollowingAnalyticsBlockV1()

    # Run the block for each frame and collect frechet_distance
    frechet_distances = []

    for i in range(len(frames)):
        result = line_following_block.run(
            detections=detections_list[i],
            metadata=metadata_list[i],
            triggering_anchor=sv.Position.CENTER,  # Default
            reference_path=reference_path,
        )
        frechet_distance = result["frechet_distance"]
        frechet_distances.append(frechet_distance)

    # Then
    # Since one object follows the path exactly, and one deviates,
    # the frechet_distance should be determined by the object with the maximum distance
    # In this case, object 2 deviates, so the frechet distance should be greater than zero.
    assert frechet_distances[-1] > 0.0


def test_line_following_no_tracker_id():
    # Given
    reference_path = [(0, 0), (1, 1), (2, 2), (3, 3)]
    detections = sv.Detections(
        xyxy=np.array([[0, 0, 1, 1]]),
        tracker_id=None,  # No tracker_id
    )
    metadata = VideoMetadata(
        video_identifier="vid_1",
        frame_number=0,
        fps=1,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570875).astimezone(
            tz=datetime.timezone.utc
        ),
        comes_from_video_file=True,
    )
    line_following_block = LineFollowingAnalyticsBlockV1()

    # When / Then
    with pytest.raises(
        ValueError,
        match="tracker_id not initialized, LineFollowingAnalyticsBlockV1 requires detections to be tracked",
    ):
        _ = line_following_block.run(
            detections=detections,
            metadata=metadata,
            triggering_anchor=sv.Position.CENTER,
            reference_path=reference_path,
        )


def test_line_following_invalid_reference_path():
    # Given
    reference_path = "invalid_reference_path"  # Not a list of points
    detections = sv.Detections(
        xyxy=np.array([[0, 0, 1, 1]]),
        tracker_id=np.array([1]),
    )
    metadata = VideoMetadata(
        video_identifier="vid_1",
        frame_number=0,
        fps=1,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570875).astimezone(
            tz=datetime.timezone.utc
        ),
        comes_from_video_file=True,
    )
    line_following_block = LineFollowingAnalyticsBlockV1()

    # When / Then
    with pytest.raises(TypeError):
        _ = line_following_block.run(
            detections=detections,
            metadata=metadata,
            triggering_anchor=sv.Position.CENTER,
            reference_path=reference_path,
        )
