import datetime

import numpy as np
import pytest
import supervision as sv

from inference.core.workflows.core_steps.analytics.time_in_zone.v1 import (
    TimeInZoneBlockV1,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)


def test_time_in_zone_keep_out_of_zone_detections() -> None:
    # given
    zone = [[10, 10], [10, 20], [20, 20], [20, 10]]
    frame1_detections = sv.Detections(
        xyxy=np.array(
            [[9, 15, 10, 16], [10, 15, 11, 16], [11, 15, 12, 16], [15, 15, 16, 16]]
        ),
        tracker_id=np.array([1, 2, 3, 4]),
    )
    frame2_detections = sv.Detections(
        xyxy=np.array(
            [[10, 15, 11, 16], [11, 15, 12, 16], [12, 15, 13, 16], [16, 16, 17, 17]]
        ),
        tracker_id=np.array([1, 2, 3, 5]),
    )
    frame3_detections = sv.Detections(
        xyxy=np.array([[11, 15, 12, 16], [20, 15, 21, 16], [21, 15, 22, 16]]),
        tracker_id=np.array([1, 2, 3]),
    )
    frame1_metadata = VideoMetadata(
        video_identifier="vid_1",
        frame_number=10,
        fps=1,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570875).astimezone(
            tz=datetime.timezone.utc
        ),
        comes_from_video_file=True,
    )
    frame2_metadata = VideoMetadata(
        video_identifier="vid_1",
        frame_number=11,
        fps=1,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570875).astimezone(
            tz=datetime.timezone.utc
        ),
        comes_from_video_file=True,
    )
    frame3_metadata = VideoMetadata(
        video_identifier="vid_1",
        frame_number=12,
        fps=1,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570875).astimezone(
            tz=datetime.timezone.utc
        ),
        comes_from_video_file=True,
    )
    time_in_zone_block = TimeInZoneBlockV1()

    parent_metadata = ImageParentMetadata(parent_id="img1")
    image = np.zeros((720, 1280, 3))
    image_data = WorkflowImageData(
        parent_metadata=parent_metadata,
        numpy_image=image,
    )

    # when
    frame1_result = time_in_zone_block.run(
        image=image_data,
        detections=frame1_detections,
        metadata=frame1_metadata,
        zone=zone,
        triggering_anchor="TOP_LEFT",
        remove_out_of_zone_detections=False,
        reset_out_of_zone_detections=False,
    )
    frame2_result = time_in_zone_block.run(
        image=image_data,
        detections=frame2_detections,
        metadata=frame2_metadata,
        zone=zone,
        triggering_anchor="TOP_LEFT",
        remove_out_of_zone_detections=False,
        reset_out_of_zone_detections=False,
    )
    frame3_result = time_in_zone_block.run(
        image=image_data,
        detections=frame3_detections,
        metadata=frame3_metadata,
        zone=zone,
        triggering_anchor="TOP_LEFT",
        remove_out_of_zone_detections=False,
        reset_out_of_zone_detections=False,
    )

    # then
    assert (
        frame1_result["timed_detections"].xyxy
        == np.array(
            [[9, 15, 10, 16], [10, 15, 11, 16], [11, 15, 12, 16], [15, 15, 16, 16]]
        )
    ).all()
    assert (
        frame1_result["timed_detections"]["time_in_zone"] == np.array([0, 0, 0, 0])
    ).all()

    assert (
        frame2_result["timed_detections"].xyxy
        == np.array(
            [[10, 15, 11, 16], [11, 15, 12, 16], [12, 15, 13, 16], [16, 16, 17, 17]]
        )
    ).all()
    assert (
        frame2_result["timed_detections"]["time_in_zone"] == np.array([0, 1, 1, 0])
    ).all()

    assert (
        frame3_result["timed_detections"].xyxy
        == np.array([[11, 15, 12, 16], [20, 15, 21, 16], [21, 15, 22, 16]])
    ).all()
    assert (
        frame3_result["timed_detections"]["time_in_zone"] == np.array([1, 2, 0])
    ).all()


def test_time_in_zone_remove_out_of_zone_detections() -> None:
    # given
    zone = [[10, 10], [10, 20], [20, 20], [20, 10]]
    frame1_detections = sv.Detections(
        xyxy=np.array(
            [[9, 15, 10, 16], [10, 15, 11, 16], [11, 15, 12, 16], [15, 15, 16, 16]]
        ),
        tracker_id=np.array([1, 2, 3, 4]),
    )
    frame2_detections = sv.Detections(
        xyxy=np.array(
            [[10, 15, 11, 16], [11, 15, 12, 16], [12, 15, 13, 16], [16, 16, 17, 17]]
        ),
        tracker_id=np.array([1, 2, 3, 5]),
    )
    frame3_detections = sv.Detections(
        xyxy=np.array([[11, 15, 12, 16], [20, 15, 21, 16], [21, 15, 22, 16]]),
        tracker_id=np.array([1, 2, 3]),
    )
    frame1_metadata = VideoMetadata(
        video_identifier="vid_1",
        frame_number=10,
        fps=1,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570875).astimezone(
            tz=datetime.timezone.utc
        ),
        comes_from_video_file=True,
    )
    frame2_metadata = VideoMetadata(
        video_identifier="vid_1",
        frame_number=11,
        fps=1,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570875).astimezone(
            tz=datetime.timezone.utc
        ),
        comes_from_video_file=True,
    )
    frame3_metadata = VideoMetadata(
        video_identifier="vid_1",
        frame_number=12,
        fps=1,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570875).astimezone(
            tz=datetime.timezone.utc
        ),
        comes_from_video_file=True,
    )
    time_in_zone_block = TimeInZoneBlockV1()

    parent_metadata = ImageParentMetadata(parent_id="img1")
    image = np.zeros((720, 1280, 3))
    image_data = WorkflowImageData(
        parent_metadata=parent_metadata,
        numpy_image=image,
    )

    # when
    frame1_result = time_in_zone_block.run(
        image=image_data,
        detections=frame1_detections,
        metadata=frame1_metadata,
        zone=zone,
        triggering_anchor="TOP_LEFT",
        remove_out_of_zone_detections=True,
        reset_out_of_zone_detections=True,
    )
    frame2_result = time_in_zone_block.run(
        image=image_data,
        detections=frame2_detections,
        metadata=frame2_metadata,
        zone=zone,
        triggering_anchor="TOP_LEFT",
        remove_out_of_zone_detections=True,
        reset_out_of_zone_detections=True,
    )
    frame3_result = time_in_zone_block.run(
        image=image_data,
        detections=frame3_detections,
        metadata=frame3_metadata,
        zone=zone,
        triggering_anchor="TOP_LEFT",
        remove_out_of_zone_detections=True,
        reset_out_of_zone_detections=True,
    )

    # then
    assert (
        frame1_result["timed_detections"].xyxy
        == np.array([[10, 15, 11, 16], [11, 15, 12, 16], [15, 15, 16, 16]])
    ).all()
    assert (
        frame1_result["timed_detections"]["time_in_zone"] == np.array([0, 0, 0])
    ).all()

    assert (
        frame2_result["timed_detections"].xyxy
        == np.array(
            [[10, 15, 11, 16], [11, 15, 12, 16], [12, 15, 13, 16], [16, 16, 17, 17]]
        )
    ).all()
    assert (
        frame2_result["timed_detections"]["time_in_zone"] == np.array([0, 1, 1, 0])
    ).all()

    assert (
        frame3_result["timed_detections"].xyxy
        == np.array([[11, 15, 12, 16], [20, 15, 21, 16]])
    ).all()
    assert (frame3_result["timed_detections"]["time_in_zone"] == np.array([1, 2])).all()


def test_time_in_zone_remove_and_reset_out_of_zone_detections() -> None:
    # given
    zone = [[10, 10], [10, 20], [20, 20], [20, 10]]
    frame1_detections = sv.Detections(
        xyxy=np.array([[8, 8, 9, 9], [11, 11, 12, 12], [14, 14, 15, 15]]),
        tracker_id=np.array([1, 2, 3]),
    )
    frame2_detections = sv.Detections(
        xyxy=np.array([[8, 8, 9, 9], [11, 11, 12, 12], [14, 14, 15, 15]]),
        tracker_id=np.array([1, 2, 3]),
    )
    frame3_detections = sv.Detections(
        xyxy=np.array([[8, 8, 9, 9], [1, 1, 2, 2], [14, 14, 15, 15]]),
        tracker_id=np.array([1, 2, 3]),
    )
    frame4_detections = sv.Detections(
        xyxy=np.array([[8, 8, 9, 9], [11, 11, 12, 12], [14, 14, 15, 15]]),
        tracker_id=np.array([1, 2, 3]),
    )
    frame1_metadata = VideoMetadata(
        video_identifier="vid_1",
        frame_number=10,
        fps=1,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570875).astimezone(
            tz=datetime.timezone.utc
        ),
        comes_from_video_file=True,
    )
    frame2_metadata = VideoMetadata(
        video_identifier="vid_1",
        frame_number=11,
        fps=1,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570876).astimezone(
            tz=datetime.timezone.utc
        ),
        comes_from_video_file=True,
    )
    frame3_metadata = VideoMetadata(
        video_identifier="vid_1",
        frame_number=12,
        fps=1,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570877).astimezone(
            tz=datetime.timezone.utc
        ),
        comes_from_video_file=True,
    )
    frame4_metadata = VideoMetadata(
        video_identifier="vid_1",
        frame_number=13,
        fps=1,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570878).astimezone(
            tz=datetime.timezone.utc
        ),
        comes_from_video_file=True,
    )
    time_in_zone_block = TimeInZoneBlockV1()

    parent_metadata = ImageParentMetadata(parent_id="img1")
    image = np.zeros((720, 1280, 3))
    image_data = WorkflowImageData(
        parent_metadata=parent_metadata,
        numpy_image=image,
    )

    # when
    frame1_result = time_in_zone_block.run(
        image=image_data,
        detections=frame1_detections,
        metadata=frame1_metadata,
        zone=zone,
        triggering_anchor="TOP_LEFT",
        remove_out_of_zone_detections=True,
        reset_out_of_zone_detections=True,
    )
    frame2_result = time_in_zone_block.run(
        image=image_data,
        detections=frame2_detections,
        metadata=frame2_metadata,
        zone=zone,
        triggering_anchor="TOP_LEFT",
        remove_out_of_zone_detections=True,
        reset_out_of_zone_detections=True,
    )
    frame3_result = time_in_zone_block.run(
        image=image_data,
        detections=frame3_detections,
        metadata=frame3_metadata,
        zone=zone,
        triggering_anchor="TOP_LEFT",
        remove_out_of_zone_detections=True,
        reset_out_of_zone_detections=True,
    )
    frame4_result = time_in_zone_block.run(
        image=image_data,
        detections=frame4_detections,
        metadata=frame4_metadata,
        zone=zone,
        triggering_anchor="TOP_LEFT",
        remove_out_of_zone_detections=True,
        reset_out_of_zone_detections=True,
    )

    # then
    assert (
        frame1_result["timed_detections"].xyxy
        == np.array([[11, 11, 12, 12], [14, 14, 15, 15]])
    ).all()
    assert (frame1_result["timed_detections"]["time_in_zone"] == np.array([0, 0])).all()

    assert (
        frame2_result["timed_detections"].xyxy
        == np.array([[11, 11, 12, 12], [14, 14, 15, 15]])
    ).all()
    assert (frame2_result["timed_detections"]["time_in_zone"] == np.array([1, 1])).all()

    assert (
        frame3_result["timed_detections"].xyxy == np.array([[14, 14, 15, 15]])
    ).all()
    assert (frame3_result["timed_detections"]["time_in_zone"] == np.array([2])).all()

    assert (
        frame4_result["timed_detections"].xyxy
        == np.array([[11, 11, 12, 12], [14, 14, 15, 15]])
    ).all()
    assert (frame4_result["timed_detections"]["time_in_zone"] == np.array([0, 3])).all()


def test_time_in_zone_keep_and_reset_out_of_zone_detections() -> None:
    # given
    zone = [[10, 10], [10, 20], [20, 20], [20, 10]]
    frame1_detections = sv.Detections(
        xyxy=np.array([[8, 8, 9, 9], [11, 11, 12, 12], [14, 14, 15, 15]]),
        tracker_id=np.array([1, 2, 3]),
    )
    frame2_detections = sv.Detections(
        xyxy=np.array([[8, 8, 9, 9], [11, 11, 12, 12], [14, 14, 15, 15]]),
        tracker_id=np.array([1, 2, 3]),
    )
    frame3_detections = sv.Detections(
        xyxy=np.array([[8, 8, 9, 9], [1, 1, 2, 2], [14, 14, 15, 15]]),
        tracker_id=np.array([1, 2, 3]),
    )
    frame4_detections = sv.Detections(
        xyxy=np.array([[8, 8, 9, 9], [11, 11, 12, 12], [14, 14, 15, 15]]),
        tracker_id=np.array([1, 2, 3]),
    )
    frame1_metadata = VideoMetadata(
        video_identifier="vid_1",
        frame_number=10,
        fps=1,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570875).astimezone(
            tz=datetime.timezone.utc
        ),
        comes_from_video_file=True,
    )
    frame2_metadata = VideoMetadata(
        video_identifier="vid_1",
        frame_number=11,
        fps=1,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570876).astimezone(
            tz=datetime.timezone.utc
        ),
        comes_from_video_file=True,
    )
    frame3_metadata = VideoMetadata(
        video_identifier="vid_1",
        frame_number=12,
        fps=1,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570877).astimezone(
            tz=datetime.timezone.utc
        ),
        comes_from_video_file=True,
    )
    frame4_metadata = VideoMetadata(
        video_identifier="vid_1",
        frame_number=13,
        fps=1,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570878).astimezone(
            tz=datetime.timezone.utc
        ),
        comes_from_video_file=True,
    )
    time_in_zone_block = TimeInZoneBlockV1()

    parent_metadata = ImageParentMetadata(parent_id="img1")
    image = np.zeros((720, 1280, 3))
    image_data = WorkflowImageData(
        parent_metadata=parent_metadata,
        numpy_image=image,
    )

    # when
    frame1_result = time_in_zone_block.run(
        image=image_data,
        detections=frame1_detections,
        metadata=frame1_metadata,
        zone=zone,
        triggering_anchor="TOP_LEFT",
        remove_out_of_zone_detections=False,
        reset_out_of_zone_detections=True,
    )
    frame2_result = time_in_zone_block.run(
        image=image_data,
        detections=frame2_detections,
        metadata=frame2_metadata,
        zone=zone,
        triggering_anchor="TOP_LEFT",
        remove_out_of_zone_detections=False,
        reset_out_of_zone_detections=True,
    )
    frame3_result = time_in_zone_block.run(
        image=image_data,
        detections=frame3_detections,
        metadata=frame3_metadata,
        zone=zone,
        triggering_anchor="TOP_LEFT",
        remove_out_of_zone_detections=False,
        reset_out_of_zone_detections=True,
    )
    frame4_result = time_in_zone_block.run(
        image=image_data,
        detections=frame4_detections,
        metadata=frame4_metadata,
        zone=zone,
        triggering_anchor="TOP_LEFT",
        remove_out_of_zone_detections=False,
        reset_out_of_zone_detections=True,
    )

    # then
    assert (
        frame1_result["timed_detections"].xyxy
        == np.array([[8, 8, 9, 9], [11, 11, 12, 12], [14, 14, 15, 15]])
    ).all()
    assert (
        frame1_result["timed_detections"]["time_in_zone"] == np.array([0, 0, 0])
    ).all()

    assert (
        frame2_result["timed_detections"].xyxy
        == np.array([[8, 8, 9, 9], [11, 11, 12, 12], [14, 14, 15, 15]])
    ).all()
    assert (
        frame2_result["timed_detections"]["time_in_zone"] == np.array([0, 1, 1])
    ).all()

    assert (
        frame3_result["timed_detections"].xyxy
        == np.array([[8, 8, 9, 9], [1, 1, 2, 2], [14, 14, 15, 15]])
    ).all()
    assert (
        frame3_result["timed_detections"]["time_in_zone"] == np.array([0, 0, 2])
    ).all()

    assert (
        frame4_result["timed_detections"].xyxy
        == np.array([[8, 8, 9, 9], [11, 11, 12, 12], [14, 14, 15, 15]])
    ).all()
    assert (
        frame4_result["timed_detections"]["time_in_zone"] == np.array([0, 0, 3])
    ).all()


def test_time_in_zone_no_trackers() -> None:
    # given
    zone = [[15, 0], [15, 1000], [3, 3]]
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
    time_in_zone_block = TimeInZoneBlockV1()

    parent_metadata = ImageParentMetadata(parent_id="img1")
    image = np.zeros((720, 1280, 3))
    image_data = WorkflowImageData(
        parent_metadata=parent_metadata,
        numpy_image=image,
    )

    # when
    with pytest.raises(
        ValueError,
        match="tracker_id not initialized, TimeInZoneBlockV1 requires detections to be tracked",
    ):
        _ = time_in_zone_block.run(
            image=image_data,
            detections=detections,
            metadata=metadata,
            zone=zone,
            triggering_anchor="TOP_LEFT",
            remove_out_of_zone_detections=True,
            reset_out_of_zone_detections=True,
        )


def test_time_in_zone_list_of_points_too_short() -> None:
    # given
    zone = [[15, 0], [15, 1000]]
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
    time_in_zone_block = TimeInZoneBlockV1()

    parent_metadata = ImageParentMetadata(parent_id="img1")
    image = np.zeros((720, 1280, 3))
    image_data = WorkflowImageData(
        parent_metadata=parent_metadata,
        numpy_image=image,
    )

    # when
    with pytest.raises(
        ValueError,
        match="TimeInZoneBlockV1 requires zone to be a list containing more than 2 points",
    ):
        _ = time_in_zone_block.run(
            image=image_data,
            detections=detections,
            metadata=metadata,
            zone=zone,
            triggering_anchor="TOP_LEFT",
            remove_out_of_zone_detections=True,
            reset_out_of_zone_detections=True,
        )


def test_time_in_zone_elements_not_points() -> None:
    # given
    zone = [1, 2]
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
    time_in_zone_block = TimeInZoneBlockV1()

    parent_metadata = ImageParentMetadata(parent_id="img1")
    image = np.zeros((720, 1280, 3))
    image_data = WorkflowImageData(
        parent_metadata=parent_metadata,
        numpy_image=image,
    )

    # when
    with pytest.raises(
        ValueError,
        match="TimeInZoneBlockV1 requires zone to be a list containing more than 2 points",
    ):
        _ = time_in_zone_block.run(
            image=image_data,
            detections=detections,
            metadata=metadata,
            zone=zone,
            triggering_anchor="TOP_LEFT",
            remove_out_of_zone_detections=True,
            reset_out_of_zone_detections=True,
        )


def test_time_in_zone_coordianates_not_numeric() -> None:
    # given
    zone = [["a", 1], [2, 3], [4, 5]]
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
    time_in_zone_block = TimeInZoneBlockV1()

    parent_metadata = ImageParentMetadata(parent_id="img1")
    image = np.zeros((720, 1280, 3))
    image_data = WorkflowImageData(
        parent_metadata=parent_metadata,
        numpy_image=image,
    )

    # when
    with pytest.raises(
        ValueError,
        match="TimeInZoneBlockV1 requires each coordinate of zone to be a number",
    ):
        _ = time_in_zone_block.run(
            image=image_data,
            detections=detections,
            metadata=metadata,
            zone=zone,
            triggering_anchor="TOP_LEFT",
            remove_out_of_zone_detections=True,
            reset_out_of_zone_detections=True,
        )
