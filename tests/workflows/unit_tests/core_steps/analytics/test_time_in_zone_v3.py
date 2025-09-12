import datetime

import numpy as np
import pytest
import supervision as sv

from inference.core.workflows.core_steps.analytics.time_in_zone.v2 import (
    TimeInZoneBlockV3, calculate_nesting_depth,
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
    time_in_zone_block = TimeInZoneBlockV3()
    image = np.zeros((720, 1280, 3))

    # when
    frame1_result = time_in_zone_block.run(
        image=_wrap_with_workflow_image(image=image, metadata=frame1_metadata),
        detections=frame1_detections,
        zone=zone,
        triggering_anchor="TOP_LEFT",
        remove_out_of_zone_detections=False,
        reset_out_of_zone_detections=False,
    )
    frame2_result = time_in_zone_block.run(
        image=_wrap_with_workflow_image(image=image, metadata=frame2_metadata),
        detections=frame2_detections,
        zone=zone,
        triggering_anchor="TOP_LEFT",
        remove_out_of_zone_detections=False,
        reset_out_of_zone_detections=False,
    )
    frame3_result = time_in_zone_block.run(
        image=_wrap_with_workflow_image(image=image, metadata=frame3_metadata),
        detections=frame3_detections,
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
    time_in_zone_block = TimeInZoneBlockV3()
    image = np.zeros((720, 1280, 3))

    # when
    frame1_result = time_in_zone_block.run(
        image=_wrap_with_workflow_image(image=image, metadata=frame1_metadata),
        detections=frame1_detections,
        zone=zone,
        triggering_anchor="TOP_LEFT",
        remove_out_of_zone_detections=True,
        reset_out_of_zone_detections=True,
    )
    frame2_result = time_in_zone_block.run(
        image=_wrap_with_workflow_image(image=image, metadata=frame2_metadata),
        detections=frame2_detections,
        zone=zone,
        triggering_anchor="TOP_LEFT",
        remove_out_of_zone_detections=True,
        reset_out_of_zone_detections=True,
    )
    frame3_result = time_in_zone_block.run(
        image=_wrap_with_workflow_image(image=image, metadata=frame3_metadata),
        detections=frame3_detections,
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
    time_in_zone_block = TimeInZoneBlockV3()
    image = np.zeros((720, 1280, 3))

    # when
    frame1_result = time_in_zone_block.run(
        image=_wrap_with_workflow_image(image=image, metadata=frame1_metadata),
        detections=frame1_detections,
        zone=zone,
        triggering_anchor="TOP_LEFT",
        remove_out_of_zone_detections=True,
        reset_out_of_zone_detections=True,
    )
    frame2_result = time_in_zone_block.run(
        image=_wrap_with_workflow_image(image=image, metadata=frame2_metadata),
        detections=frame2_detections,
        zone=zone,
        triggering_anchor="TOP_LEFT",
        remove_out_of_zone_detections=True,
        reset_out_of_zone_detections=True,
    )
    frame3_result = time_in_zone_block.run(
        image=_wrap_with_workflow_image(image=image, metadata=frame3_metadata),
        detections=frame3_detections,
        zone=zone,
        triggering_anchor="TOP_LEFT",
        remove_out_of_zone_detections=True,
        reset_out_of_zone_detections=True,
    )
    frame4_result = time_in_zone_block.run(
        image=_wrap_with_workflow_image(image=image, metadata=frame4_metadata),
        detections=frame4_detections,
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
    time_in_zone_block = TimeInZoneBlockV3()
    image = np.zeros((720, 1280, 3))

    # when
    frame1_result = time_in_zone_block.run(
        image=_wrap_with_workflow_image(image=image, metadata=frame1_metadata),
        detections=frame1_detections,
        zone=zone,
        triggering_anchor="TOP_LEFT",
        remove_out_of_zone_detections=False,
        reset_out_of_zone_detections=True,
    )
    frame2_result = time_in_zone_block.run(
        image=_wrap_with_workflow_image(image=image, metadata=frame2_metadata),
        detections=frame2_detections,
        zone=zone,
        triggering_anchor="TOP_LEFT",
        remove_out_of_zone_detections=False,
        reset_out_of_zone_detections=True,
    )
    frame3_result = time_in_zone_block.run(
        image=_wrap_with_workflow_image(image=image, metadata=frame3_metadata),
        detections=frame3_detections,
        zone=zone,
        triggering_anchor="TOP_LEFT",
        remove_out_of_zone_detections=False,
        reset_out_of_zone_detections=True,
    )
    frame4_result = time_in_zone_block.run(
        image=_wrap_with_workflow_image(image=image, metadata=frame4_metadata),
        detections=frame4_detections,
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
    time_in_zone_block = TimeInZoneBlockV3()
    image = np.zeros((720, 1280, 3))

    # when
    with pytest.raises(
        ValueError,
        match="tracker_id not initialized, TimeInZoneBlockV3 requires detections to be tracked",
    ):
        _ = time_in_zone_block.run(
            image=_wrap_with_workflow_image(image=image, metadata=metadata),
            detections=detections,
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
    time_in_zone_block = TimeInZoneBlockV3()
    image = np.zeros((720, 1280, 3))

    # when
    with pytest.raises(
        ValueError,
        match="TimeInZoneBlockV3 requires zone to be a list containing more than 2 points",
    ):
        _ = time_in_zone_block.run(
            image=_wrap_with_workflow_image(image=image, metadata=metadata),
            detections=detections,
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
    time_in_zone_block = TimeInZoneBlockV3()
    image = np.zeros((720, 1280, 3))

    # when
    with pytest.raises(
        ValueError,
        match="TimeInZoneBlockV3 requires zone to be a list containing more than 2 points",
    ):
        _ = time_in_zone_block.run(
            image=_wrap_with_workflow_image(image=image, metadata=metadata),
            detections=detections,
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
    time_in_zone_block = TimeInZoneBlockV3()
    image = np.zeros((720, 1280, 3))

    # when
    with pytest.raises(
        ValueError,
        match="TimeInZoneBlockV3 requires each coordinate of zone to be a number",
    ):
        _ = time_in_zone_block.run(
            image=_wrap_with_workflow_image(image=image, metadata=metadata),
            detections=detections,
            zone=zone,
            triggering_anchor="TOP_LEFT",
            remove_out_of_zone_detections=True,
            reset_out_of_zone_detections=True,
        )


def _wrap_with_workflow_image(
    image: np.ndarray, metadata: VideoMetadata
) -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=image,
        video_metadata=metadata,
    )


def test_time_in_zone_multiple_zones() -> None:
    # given
    zones = [
        [(10, 10), (10, 20), (20, 20)],
        [(30, 30), (40, 40), (30, 40)],
    ]
    frame1_detections = sv.Detections(
        xyxy=np.array(
            [
                [12, 19, 13, 20],
                [31, 39, 32, 40],
                [5, 5, 6, 6],
            ]
        ),
        tracker_id=np.array([1, 2, 3]),
    )
    frame2_detections = sv.Detections(
        xyxy=np.array(
            [
                [12, 19, 13, 20],
                [31, 39, 32, 40],
                [12, 19, 13, 20],
                [100, 100, 101, 101],
            ]
        ),
        tracker_id=np.array([1, 2, 3, 4]),
    )
    frame1_metadata = VideoMetadata(
        video_identifier="vid_multi",
        frame_number=10,
        fps=1,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570875).astimezone(
            tz=datetime.timezone.utc
        ),
        comes_from_video_file=True,
    )
    frame2_metadata = VideoMetadata(
        video_identifier="vid_multi",
        frame_number=11,
        fps=1,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570876).astimezone(
            tz=datetime.timezone.utc
        ),
        comes_from_video_file=True,
    )
    time_in_zone_block = TimeInZoneBlockV3()
    image = np.zeros((720, 1280, 3))

    # when
    frame1_result = time_in_zone_block.run(
        image=_wrap_with_workflow_image(image=image, metadata=frame1_metadata),
        detections=frame1_detections,
        zone=zones,
        triggering_anchor="TOP_LEFT",
        remove_out_of_zone_detections=False,
        reset_out_of_zone_detections=False,
    )
    frame2_result = time_in_zone_block.run(
        image=_wrap_with_workflow_image(image=image, metadata=frame2_metadata),
        detections=frame2_detections,
        zone=zones,
        triggering_anchor="TOP_LEFT",
        remove_out_of_zone_detections=False,
        reset_out_of_zone_detections=False,
    )

    # then
    assert (
        frame1_result["timed_detections"]["time_in_zone"] == np.array([0, 0, 0])
    ).all()
    assert (
        frame2_result["timed_detections"]["time_in_zone"] == np.array([1, 1, 0, 0])
    ).all()


def test_time_in_zone_empty_zones_results_in_zero_time() -> None:
    # given
    zones: list[list[tuple[int, int]]] = []
    detections = sv.Detections(
        xyxy=np.array(
            [
                [12, 19, 13, 20],
                [31, 39, 32, 40],
                [100, 100, 101, 101],
            ]
        ),
        tracker_id=np.array([1, 2, 3]),
    )
    metadata = VideoMetadata(
        video_identifier="vid_empty_zones",
        frame_number=10,
        fps=1,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570875).astimezone(
            tz=datetime.timezone.utc
        ),
        comes_from_video_file=True,
    )
    time_in_zone_block = TimeInZoneBlockV3()
    image = np.zeros((720, 1280, 3))

    # when
    result = time_in_zone_block.run(
        image=_wrap_with_workflow_image(image=image, metadata=metadata),
        detections=detections,
        zone=zones,
        triggering_anchor="TOP_LEFT",
        remove_out_of_zone_detections=False,
        reset_out_of_zone_detections=False,
    )

    # then
    assert (result["timed_detections"]["time_in_zone"] == np.array([0, 0, 0])).all()


def test_time_in_zone_updates_zones_cache_between_runs() -> None:
    # given
    zones_initial = [
        [(10, 10), (10, 20), (20, 20)],
    ]
    zones_empty: list[list[tuple[int, int]]] = []
    frame1_detections = sv.Detections(
        xyxy=np.array([[11, 11, 12, 12]]),
        tracker_id=np.array([1]),
    )
    frame2_detections = sv.Detections(
        xyxy=np.array([[11, 11, 12, 12]]),
        tracker_id=np.array([1]),
    )
    frame1_metadata = VideoMetadata(
        video_identifier="vid_cache_update",
        frame_number=10,
        fps=1,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570875).astimezone(
            tz=datetime.timezone.utc
        ),
        comes_from_video_file=True,
    )
    frame2_metadata = VideoMetadata(
        video_identifier="vid_cache_update",
        frame_number=11,
        fps=1,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570876).astimezone(
            tz=datetime.timezone.utc
        ),
        comes_from_video_file=True,
    )
    time_in_zone_block = TimeInZoneBlockV3()
    image = np.zeros((720, 1280, 3))

    # when: first run with a zone that contains the detection
    frame1_result = time_in_zone_block.run(
        image=_wrap_with_workflow_image(image=image, metadata=frame1_metadata),
        detections=frame1_detections,
        zone=zones_initial,
        triggering_anchor="TOP_LEFT",
        remove_out_of_zone_detections=False,
        reset_out_of_zone_detections=True,
    )
    # when: second run with empty zones should not reuse previously cached polygons
    frame2_result = time_in_zone_block.run(
        image=_wrap_with_workflow_image(image=image, metadata=frame2_metadata),
        detections=frame2_detections,
        zone=zones_empty,
        triggering_anchor="TOP_LEFT",
        remove_out_of_zone_detections=True,
        reset_out_of_zone_detections=True,
    )

    # then
    assert len(frame1_result["timed_detections"]) == 1
    assert (frame1_result["timed_detections"]["time_in_zone"] == np.array([0])).all()
    assert len(frame2_result["timed_detections"]) == 0


def test_calculate_nesting_depth_when_scalar_value_provided() -> None:
    # when
    result = calculate_nesting_depth(zone=1, max_depth=3)

    # then
    assert result == 0


def test_calculate_nesting_depth_when_list_of_scalar_value_provided() -> None:
    # when
    result = calculate_nesting_depth(zone=[1, 2, 3], max_depth=3)

    # then
    assert result == 1


def test_calculate_nesting_depth_when_1d_np_array_provided() -> None:
    # when
    result = calculate_nesting_depth(zone=np.array([1, 2, 3]), max_depth=3)

    # then
    assert result == 1


def test_calculate_nesting_depth_when_2d_np_array_provided() -> None:
    # when
    result = calculate_nesting_depth(zone=np.array([[1, 2, 3], [1, 2, 3]]), max_depth=3)

    # then
    assert result == 2


def test_calculate_nesting_depth_when_3d_np_array_provided() -> None:
    # when
    result = calculate_nesting_depth(zone=np.array([[[1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]]]), max_depth=3)

    # then
    assert result == 3


def test_calculate_nesting_depth_when_3d_np_array_provided_and_max_depth_exceeded() -> None:
    # when
    with pytest.raises(ValueError):
        _ = calculate_nesting_depth(zone=np.array([[[1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]]]), max_depth=2)


def test_calculate_nesting_depth_when_list_of_lists_of_scalar_values_provided() -> None:
    # when
    result = calculate_nesting_depth(zone=[[1, 2], [3, 4]], max_depth=3)

    # then
    assert result == 2


def test_calculate_nesting_depth_when_list_of_lists_of_lists_of_scalar_values_provided() -> None:
    # when
    result = calculate_nesting_depth(zone=[[[1, 2], [3, 4]], [[1, 2], [3, 4]]], max_depth=3)

    # then
    assert result == 3


def test_calculate_nesting_depth_when_list_of_lists_of_lists_of_scalar_values_provided_along_with_more_nested_inputs() -> None:
    # when
    with pytest.raises(ValueError):
        _ = calculate_nesting_depth(zone=[[[1, 2], [3, 4]], [[1, 2], [3, [1, [1, [1]]]]]], max_depth=3)


def test_calculate_nesting_depth_when_list_of_lists_of_scalar_values_provided_with_more_nested_inputs_but_not_exceeding_max_depth() -> None:
    # when
    with pytest.raises(ValueError):
        _ = calculate_nesting_depth(zone=[[1, 2], [3, [1, 2, 3]]], max_depth=3)


def test_calculate_nesting_depth_when_tuple_of_scalar_value_provided() -> None:
    # when
    result = calculate_nesting_depth(zone=(1, 2, 3), max_depth=3)

    # then
    assert result == 1


def test_calculate_nesting_depth_when_tuple_of_tuples_of_scalar_value_provided() -> None:
    # when
    result = calculate_nesting_depth(zone=((1, 2), (2, 3)), max_depth=3)

    # then
    assert result == 2


def test_calculate_nesting_depth_when_tuple_of_tuples_of_tuples_of_scalar_value_provided() -> None:
    # when
    result = calculate_nesting_depth(zone=(((1, 2), (2, 3)), ((1, 2), (2, 3))), max_depth=3)

    # then
    assert result == 3


def test_calculate_nesting_depth_when_tuple_of_irregular_shape_provided() -> None:
    # when
    with pytest.raises(ValueError):
        _ = calculate_nesting_depth(zone=(((1, 2), (2, 3)), 2), max_depth=3)
