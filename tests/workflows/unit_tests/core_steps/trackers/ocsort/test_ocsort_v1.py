from inference.core.workflows.core_steps.trackers.ocsort.v1 import (
    OCSORTBlockV1,
    OCSORTManifest,
)
from inference.core.workflows.core_steps.visualizations.common.base import (
    PredictionsVisualizationManifest,
)
from tests.workflows.unit_tests.core_steps.trackers.conftest import (
    FRAME1_XYXY,
    FRAME2_XYXY,
    FRAME3_XYXY,
    make_detections,
    make_metadata,
    manifest_accepted_kind_names,
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


def test_ocsort_input_and_output_kinds_match() -> None:
    """Trackers are pass-through: output kinds must equal input kinds."""
    input_kinds = manifest_accepted_kind_names(OCSORTManifest, field_name="detections")
    for output_def in OCSORTManifest.describe_outputs():
        output_kinds = {k.name for k in output_def.kind}
        assert (
            output_kinds == input_kinds
        ), f"Output '{output_def.name}' kinds {output_kinds} != input kinds {input_kinds}"


def test_ocsort_outputs_superset_of_predictions_visualization_kinds() -> None:
    viz_kinds = manifest_accepted_kind_names(PredictionsVisualizationManifest)
    for output_def in OCSORTManifest.describe_outputs():
        tracker_kinds = {k.name for k in output_def.kind}
        missing = viz_kinds - tracker_kinds
        assert not missing, f"Output '{output_def.name}' missing kinds {missing}"
