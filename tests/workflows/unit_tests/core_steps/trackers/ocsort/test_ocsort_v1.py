from inference.core.workflows.core_steps.trackers.ocsort.v1 import OCSORTBlockV1
from tests.workflows.unit_tests.core_steps.trackers.conftest import (
    FRAME1_XYXY,
    FRAME2_XYXY,
    FRAME3_XYXY,
    make_detections,
    make_metadata,
    wrap_with_workflow_image,
)


def test_ocsort_block() -> None:
    block = OCSORTBlockV1()
    block.run(
        image=wrap_with_workflow_image(make_metadata(10)),
        detections=make_detections(FRAME1_XYXY),
        minimum_consecutive_frames=1,
        high_conf_det_threshold=0.5,
    )
    frame2_result = block.run(
        image=wrap_with_workflow_image(make_metadata(11)),
        detections=make_detections(FRAME2_XYXY),
        minimum_consecutive_frames=1,
        high_conf_det_threshold=0.5,
    )
    frame3_result = block.run(
        image=wrap_with_workflow_image(make_metadata(12)),
        detections=make_detections(FRAME3_XYXY),
        minimum_consecutive_frames=1,
        high_conf_det_threshold=0.5,
    )

    frame2_ids = set(frame2_result["tracked_detections"].tracker_id.tolist())
    frame3_ids = set(frame3_result["tracked_detections"].tracker_id.tolist())

    assert len(frame2_ids) == 3
    assert len(frame3_ids) == 3
    assert frame3_ids == frame2_ids

    assert len(frame2_result["new_instances"]) == 3
    assert len(frame2_result["already_seen_instances"]) == 0
    assert len(frame3_result["new_instances"]) == 0
    assert len(frame3_result["already_seen_instances"]) == 3