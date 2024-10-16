import numpy as np
import supervision as sv

from inference.core.workflows.core_steps.transformations.stabilize_detections.v1 import (
    BlockManifest,
    StabilizeTrackedDetectionsBlockV1,
    VelocityKalmanFilter,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)


def test_stabilize_detections_validation_when_valid_manifest_is_given():
    # given
    data = {
        "type": "roboflow_core/stabilize_detections@v1",
        "name": "some",
        "detections": "$steps.other.dets",
        "image": "$inputs.image",
        "smoothing_window_size": 5,
        "bbox_smoothing_coefficient": 0.5,
    }

    # when
    result = BlockManifest.model_validate(data)

    # then
    assert result == BlockManifest(
        type="roboflow_core/stabilize_detections@v1",
        name="some",
        detections="$steps.other.dets",
        image="$inputs.image",
        smoothing_window_size=5,
        bbox_smoothing_coefficient=0.5,
    )


def test_velocity_kalman_filter_stable_detections():
    # given
    kalman_filter = VelocityKalmanFilter(smoothing_window_size=3)
    frame_1_vel = {
        1: (10, 0),
        2: (0, 15),
        3: (15, 15),
        4: (-7, 0),
        5: (0, -7),
        6: (-7, -7),
    }
    frame_2_vel = {
        1: (15, 0),
        2: (0, 15),
        3: (15, 15),
        4: (-7, 0),
        5: (0, -7),
        6: (-7, -7),
    }
    frame_3_vel = {
        1: (20, 0),
        2: (0, 20),
        3: (20, 20),
        4: (-5, 0),
        5: (0, -5),
        6: (-5, -5),
    }
    kalman_filter.update(measurements=frame_1_vel)
    kalman_filter.update(measurements=frame_2_vel)
    kalman_filter.update(measurements=frame_3_vel)

    # when
    predictions_1 = kalman_filter.update(measurements={})
    predictions_2 = kalman_filter.update(measurements={})
    predictions_3 = kalman_filter.update(measurements={})
    predictions_4 = kalman_filter.update(measurements={})

    # then
    assert (
        len(predictions_1) == 6
    ), "6 predictions expected to be produced for first frame when predictions are missing and smoothing window contains 3 elements"
    assert (
        len(predictions_2) == 6
    ), "6 predictions expected to be produced for second frame when predictions are missing and smoothing window contains 3 elements"
    assert (
        len(predictions_3) == 6
    ), "6 predictions expected to be produced for third frame when predictions are missing and smoothing window contains 3 elements"
    assert (
        len(predictions_4) == 0
    ), "No predictions expected to be produced for fourth and subsequent frames"
    assert (
        len(kalman_filter.tracked_vectors) == 0
    ), "No tracked versions expected in kalman filter"


def test_stabilize_detections():
    # when
    frame_1_dets = sv.Detections(
        xyxy=np.array(
            [
                [10, 10, 100, 100],
                [110, 110, 200, 200],
            ]
        ),
        tracker_id=np.array([1, 2]),
    )
    frame_2_dets = sv.Detections(
        xyxy=np.array(
            [
                [30, 10, 120, 100],
                [110, 130, 200, 220],
            ]
        ),
        tracker_id=np.array([1, 2]),
    )
    frame_3_dets = sv.Detections(
        xyxy=np.array(
            [
                [60, 10, 150, 100],
                [110, 160, 200, 250],
            ]
        ),
        tracker_id=np.array([1, 2]),
    )
    frame_4_dets = sv.Detections(
        xyxy=np.array(
            [
                [90, 10, 180, 100],
                [110, 190, 200, 280],
            ]
        ),
        tracker_id=np.array([1, 2]),
    )
    empty_dets = sv.Detections.merge([])
    empty_dets.tracker_id = np.array([])

    block = StabilizeTrackedDetectionsBlockV1()
    parent_img_metadata = ImageParentMetadata("")
    video_metadata = VideoMetadata(
        video_identifier="1",
        frame_number=1,
        frame_timestamp=0,
    )
    img = WorkflowImageData(
        parent_metadata=parent_img_metadata,
        video_metadata=video_metadata,
        numpy_image=np.zeros((10, 10, 3)),
    )

    # then
    res_1 = block.run(
        image=img,
        detections=frame_1_dets,
        smoothing_window_size=3,
        bbox_smoothing_coefficient=0.3,
    )
    res_2 = block.run(
        image=img,
        detections=frame_2_dets,
        smoothing_window_size=3,
        bbox_smoothing_coefficient=0.3,
    )
    res_3 = block.run(
        image=img,
        detections=frame_3_dets,
        smoothing_window_size=3,
        bbox_smoothing_coefficient=0.3,
    )
    res_4 = block.run(
        image=img,
        detections=frame_4_dets,
        smoothing_window_size=3,
        bbox_smoothing_coefficient=0.3,
    )
    res_5 = block.run(
        image=img,
        detections=empty_dets,
        smoothing_window_size=3,
        bbox_smoothing_coefficient=0.3,
    )
    res_6 = block.run(
        image=img,
        detections=empty_dets,
        smoothing_window_size=3,
        bbox_smoothing_coefficient=0.3,
    )
    res_7 = block.run(
        image=img,
        detections=empty_dets,
        smoothing_window_size=3,
        bbox_smoothing_coefficient=0.3,
    )
    res_8 = block.run(
        image=img,
        detections=empty_dets,
        smoothing_window_size=3,
        bbox_smoothing_coefficient=0.3,
    )

    assert len(res_1["tracked_detections"]) == 2
    assert len(res_2["tracked_detections"]) == 2
    assert len(res_3["tracked_detections"]) == 2
    assert len(res_4["tracked_detections"]) == 2
    assert len(res_5["tracked_detections"]) == 2
    assert len(res_6["tracked_detections"]) == 2
    assert len(res_7["tracked_detections"]) == 2
    assert len(res_8["tracked_detections"]) == 0
