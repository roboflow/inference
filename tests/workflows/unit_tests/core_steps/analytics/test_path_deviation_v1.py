import datetime

import numpy as np
import pytest
import supervision as sv

from inference.core.workflows.core_steps.analytics.path_deviation.v1 import (
    PathDeviationAnalyticsBlockV1,
)
from inference.core.workflows.execution_engine.entities.base import VideoMetadata


def test_path_deviation_exact_path():
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
    path_deviation_block = PathDeviationAnalyticsBlockV1()

    # Run the block for each frame and collect frechet_distance
    frechet_distances = []

    for i in range(len(frames)):
        result = path_deviation_block.run(
            detections=detections_list[i],
            metadata=metadata_list[i],
            triggering_anchor=sv.Position.CENTER,  # Default
            reference_path=reference_path,
        )
        frechet_distance = result["path_deviation_detections"]["path_deviation"][0]
        frechet_distances.append(frechet_distance)

    # Then
    assert len(frechet_distances) == len(frames)
    # Optional: Check that frechet distances decrease over time
    assert all(
        frechet_distances[i] >= frechet_distances[i + 1]
        for i in range(len(frechet_distances) - 1)
    )

    # Since the object follows exactly the reference path,
    # the frechet distance should be zero at the final frame.
    assert frechet_distances[-1] == pytest.approx(0.0, abs=1e-6)


def test_path_deviation_with_deviation():
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
                        object_path[i][0] - 1,
                        object_path[i][1] - 1,
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
    path_deviation_block = PathDeviationAnalyticsBlockV1()

    # Run the block for each frame and collect frechet_distance
    frechet_distances = []

    for i in range(len(frames)):
        result = path_deviation_block.run(
            detections=detections_list[i],
            metadata=metadata_list[i],
            triggering_anchor=sv.Position.CENTER,  # Default
            reference_path=reference_path,
        )
        frechet_distance = result["path_deviation_detections"]["path_deviation"][0]
        frechet_distances.append(frechet_distance)

    # Then
    # Since the object deviates from the reference path, the frechet distance should be greater than zero.
    assert frechet_distances[-1] > 0.0


def test_path_deviation_multiple_objects():
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
    path_deviation_block = PathDeviationAnalyticsBlockV1()

    # Run the block for each frame and collect frechet_distance
    frechet_distances = []

    for i in range(len(frames)):
        result = path_deviation_block.run(
            detections=detections_list[i],
            metadata=metadata_list[i],
            triggering_anchor=sv.Position.CENTER,  # Default
            reference_path=reference_path,
        )
        frechet_distance = result["path_deviation_detections"]["path_deviation"][0]
        frechet_distances.append(frechet_distance)

    # Then
    # Since one object follows the path exactly, and one deviates,
    # the frechet_distance should be determined by the object with the maximum distance
    # In this case, object 2 deviates, so the frechet distance should be greater than zero.
    assert frechet_distances[-1] > 0.0


def test_path_deviation_no_tracker_id():
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
    path_deviation_block = PathDeviationAnalyticsBlockV1()

    # When / Then
    with pytest.raises(
        ValueError,
        match="tracker_id not initialized, PathDeviationAnalyticsBlockV1 requires detections to be tracked",
    ):
        _ = path_deviation_block.run(
            detections=detections,
            metadata=metadata,
            triggering_anchor=sv.Position.CENTER,
            reference_path=reference_path,
        )


def test_path_deviation_invalid_reference_path():
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
    path_deviation_block = PathDeviationAnalyticsBlockV1()

    # When / Then
    with pytest.raises(TypeError):
        _ = path_deviation_block.run(
            detections=detections,
            metadata=metadata,
            triggering_anchor=sv.Position.CENTER,
            reference_path=reference_path,
        )
