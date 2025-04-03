import datetime

import numpy as np
import pytest
import supervision as sv

from inference.core.workflows.core_steps.analytics.line_counter.v2 import (
    LineCounterBlockV2,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)


def test_line_counter() -> None:
    # given
    line_segment = [[15, 0], [15, 1000]]
    frame1_detections = sv.Detections(
        xyxy=np.array(
            [
                [10, 10, 11, 11],
                [20, 20, 21, 21],
                [100, 100, 101, 101],
                [200, 200, 201, 201],
            ]
        ),
        tracker_id=np.array([1, 2, 3, 4]),
    )
    frame2_detections = sv.Detections(
        xyxy=np.array(
            [[20, 10, 21, 21], [10, 20, 11, 11], [90, 90, 91, 91], [5, 5, 6, 6]]
        ),
        tracker_id=np.array([1, 2, 3, 5]),
    )
    metadata = VideoMetadata(
        video_identifier="vid_1",
        frame_number=10,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570875).astimezone(
            tz=datetime.timezone.utc
        ),
    )
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
        video_metadata=metadata,
    )
    line_counter_block = LineCounterBlockV2()

    # when
    frame1_result = line_counter_block.run(
        detections=frame1_detections,
        image=image,
        line_segment=line_segment,
        triggering_anchor="TOP_LEFT",
    )
    frame2_result = line_counter_block.run(
        detections=frame2_detections,
        image=image,
        line_segment=line_segment,
        triggering_anchor="TOP_LEFT",
    )

    # then
    assert frame1_result == {
        "count_in": 0,
        "count_out": 0,
        "detections_in": frame1_detections[[False, False, False, False]],
        "detections_out": frame1_detections[[False, False, False, False]],
    }
    assert frame2_result == {
        "count_in": 1,
        "count_out": 1,
        "detections_in": frame2_detections[[True, False, False, False]],
        "detections_out": frame2_detections[[False, True, False, False]],
    }


def test_line_counter_no_trackers() -> None:
    # given
    line_segment = [[15, 0], [15, 1000]]
    detections = sv.Detections(
        xyxy=np.array([[10, 10, 11, 11]]),
    )
    metadata = VideoMetadata(
        video_identifier="vid_1",
        frame_number=10,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570875).astimezone(
            tz=datetime.timezone.utc
        ),
    )
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
        video_metadata=metadata,
    )
    line_counter_block = LineCounterBlockV2()

    # when
    with pytest.raises(
        ValueError,
        match="tracker_id not initialized, LineCounterBlockV2 requires detections to be tracked",
    ):
        _ = line_counter_block.run(
            detections=detections,
            image=image,
            line_segment=line_segment,
            triggering_anchor="TOP_LEFT",
        )


def test_line_counter_too_short_line_segment() -> None:
    # given
    line_segment = [[15, 0]]
    detections = sv.Detections(
        xyxy=np.array([[10, 10, 11, 11]]),
        tracker_id=np.array([1]),
    )
    metadata = VideoMetadata(
        video_identifier="vid_1",
        frame_number=10,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570875).astimezone(
            tz=datetime.timezone.utc
        ),
    )
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
        video_metadata=metadata,
    )
    line_counter_block = LineCounterBlockV2()

    # when
    with pytest.raises(
        ValueError,
        match="LineCounterBlockV2 requires line zone to be a list containing exactly 2 points",
    ):
        _ = line_counter_block.run(
            detections=detections,
            image=image,
            line_segment=line_segment,
            triggering_anchor="TOP_LEFT",
        )


def test_line_counter_too_long_line_segment() -> None:
    # given
    line_segment = [[15, 0], [15, 1000], [3, 3]]
    detections = sv.Detections(
        xyxy=np.array([[10, 10, 11, 11]]),
        tracker_id=np.array([1]),
    )
    metadata = VideoMetadata(
        video_identifier="vid_1",
        frame_number=10,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570875).astimezone(
            tz=datetime.timezone.utc
        ),
    )
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
        video_metadata=metadata,
    )
    line_counter_block = LineCounterBlockV2()

    # when
    with pytest.raises(
        ValueError,
        match="LineCounterBlockV2 requires line zone to be a list containing exactly 2 points",
    ):
        _ = line_counter_block.run(
            detections=detections,
            image=image,
            line_segment=line_segment,
            triggering_anchor="TOP_LEFT",
        )


def test_line_counter_line_segment_not_points() -> None:
    # given
    line_segment = [1, 2]
    detections = sv.Detections(
        xyxy=np.array([[10, 10, 11, 11]]),
        tracker_id=np.array([1]),
    )
    metadata = VideoMetadata(
        video_identifier="vid_1",
        frame_number=10,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570875).astimezone(
            tz=datetime.timezone.utc
        ),
    )
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
        video_metadata=metadata,
    )
    line_counter_block = LineCounterBlockV2()

    # when
    with pytest.raises(
        ValueError,
        match="LineCounterBlockV2 requires each point of line zone to be a list containing exactly 2 coordinates",
    ):
        _ = line_counter_block.run(
            detections=detections,
            image=image,
            line_segment=line_segment,
            triggering_anchor="TOP_LEFT",
        )


def test_line_counter_line_segment_coordianates_not_numeric() -> None:
    # given
    line_segment = [["a", 1], [2, 3]]
    detections = sv.Detections(
        xyxy=np.array([[10, 10, 11, 11]]),
        tracker_id=np.array([1]),
    )
    metadata = VideoMetadata(
        video_identifier="vid_1",
        frame_number=10,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570875).astimezone(
            tz=datetime.timezone.utc
        ),
    )
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
        video_metadata=metadata,
    )
    line_counter_block = LineCounterBlockV2()

    # when
    with pytest.raises(
        ValueError,
        match="LineCounterBlockV2 requires each coordinate of line zone to be a number",
    ):
        _ = line_counter_block.run(
            detections=detections,
            image=image,
            line_segment=line_segment,
            triggering_anchor="TOP_LEFT",
        )
