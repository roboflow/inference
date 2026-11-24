"""Compile real tracker -> zone counters -> drawings across independent sources.

The OpenCV oracle consumes asserted counter outputs; it never replaces the
counter steps or supplies counts as workflow inputs. Polygon geometry is shared
with time-in-zone, while line labels select the actual line-counter outputs.
"""

from datetime import datetime, timedelta

import cv2
import numpy as np
import pytest
import torch
from roboflow_workflows.execution_engine.entities.base import VideoMetadata
from roboflow_workflows.v2.blocks import create_catalogue
from roboflow_workflows.v2.blocks.analytics.line_counter.v2 import LineCounterBlock
from roboflow_workflows.v2.blocks.analytics.time_in_zone.v3 import TimeInZoneBlock
from roboflow_workflows.v2.blocks.visualization.annotations.line_counter_visualization.v1 import LineCounterVisualizationBlock
from roboflow_workflows.v2.blocks.visualization.annotations.polygon_zone_visualization.v1 import PolygonZoneVisualizationBlock
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.plan import CompileOptions
from roboflow_workflows.v2.engine.errors import StepExecutionError

from inference_models.models.base.object_detection import Detections

LINE = [[60, 60], [150, 60]]
ZONE = [[0, 70], [159, 70], [159, 159], [0, 159]]
BLOCKS = (
    LineCounterBlock,
    TimeInZoneBlock,
    PolygonZoneVisualizationBlock,
    LineCounterVisualizationBlock,
)


def _plan(*, mode, tracking=True):
    catalogue = create_catalogue()
    missing = [block for block in BLOCKS if catalogue.find(block.type) is None]
    definition = {
        "version": "2.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            {
                "type": "WorkflowBatchInput",
                "name": "predictions",
                "kind": ["object_detection_prediction"],
            },
            {"type": "WorkflowParameter", "name": "line", "kind": ["list_of_values"]},
            {"type": "WorkflowParameter", "name": "zone", "kind": ["list_of_values"]},
        ],
        "steps": [
            {
                "type": "roboflow_core/trackers_bytetrack@v1",
                "name": "track",
                "image": "$inputs.image",
                "predictions": "$inputs.predictions",
                "frame_rate": 10.0,
                "minimum_iou_threshold": 0.2,
                "minimum_consecutive_frames": 2,
            },
            {
                "type": LineCounterBlock.type,
                "name": "counter",
                "image": "$inputs.image",
                "predictions": "$steps.track.tracked_detections",
                "line_segment": "$inputs.line",
                "triggering_anchor": "CENTER",
            },
            {
                "type": TimeInZoneBlock.type,
                "name": "occupancy",
                "image": "$inputs.image",
                "predictions": "$steps.track.tracked_detections",
                "zone": "$inputs.zone",
            },
            {
                "type": PolygonZoneVisualizationBlock.type,
                "name": "polygon",
                "image": "$inputs.image",
                "zone": "$inputs.zone",
                "color": "RED",
                "opacity": 0.5,
            },
            {
                "type": LineCounterVisualizationBlock.type,
                "name": "drawn",
                "image": "$steps.polygon.image",
                "zone": "$inputs.line",
                "count_in": "$steps.counter.count_in",
                "count_out": "$steps.counter.count_out",
                "color": "#00ff00",
                "opacity": 0.5,
                "text_scale": 0.4,
            },
        ],
        "outputs": [
            {"type": "JsonField", "name": name, "selector": selector}
            for name, selector in (
                ("drawn", "$steps.drawn.image"),
                ("in", "$steps.counter.count_in"),
                ("out", "$steps.counter.count_out"),
                ("in_zone", "$steps.occupancy.count_in_zone"),
                ("timed", "$steps.occupancy.timed_detections"),
                ("tracked", "$steps.track.tracked_detections"),
            )
        ],
    }
    if not tracking:
        definition["steps"] = definition["steps"][1:]
        for step in definition["steps"]:
            if step.get("predictions") == "$steps.track.tracked_detections":
                step["predictions"] = "$inputs.predictions"
        definition["outputs"][-1]["selector"] = "$inputs.predictions"

    plan = compile_workflow(
        definition,
        catalogue=catalogue.with_blocks(missing),
        options=CompileOptions(block_execution=mode),
    )

    return plan


def _image(source, *, frame):
    image = ImageData.from_tensor(
        torch.full((3, 160, 160), 40 if source == "cam-a" else 80, dtype=torch.uint8),
        image_id=f"{source}-{frame}",
        video_metadata=VideoMetadata(
            video_identifier=source,
            frame_number=frame,
            frame_timestamp=datetime(2026, 10, 13, 12) + timedelta(seconds=frame / 10),
            fps=10.0,
            comes_from_video_file=True,
        ),
    )

    return image


def _predictions(image, *, frame, empty, pretracked):
    source = image.video_metadata.video_identifier
    y = 25 + 10 * frame if source == "cam-a" else 95 - 10 * frame
    predictions = Detections(
        xyxy=(
            torch.empty((0, 4))
            if empty
            else torch.tensor([[80.0, y - 10, 100.0, y + 10]])
        ),
        class_id=torch.empty(0, dtype=torch.int64) if empty else torch.tensor([0]),
        confidence=torch.empty(0) if empty else torch.tensor([0.9]),
        image_metadata={**image.prediction_metadata(), "class_names": {0: "person"}},
        bboxes_metadata=[{"tracker_id": 7}] if pretracked and not empty else None,
    )

    return predictions


def _pixels(image):
    pixels = image.tensor_image.permute(1, 2, 0).cpu().numpy()

    return pixels


def _opencv_picture(image, *, counts, line, zone):
    """Draw independently of V2 painters, including the exact count glyphs."""
    picture = _pixels(image).copy()
    if zone:
        mask = np.zeros_like(picture)
        cv2.fillPoly(mask, [np.asarray(zone, dtype=np.int32)], (255, 0, 0))
        picture = cv2.addWeighted(mask, 0.5, picture, 1, 0)
    mask = np.zeros_like(picture)
    cv2.line(mask, tuple(line[0]), tuple(line[1]), (0, 255, 0), 2)
    picture = cv2.addWeighted(mask, 0.5, picture, 1, 0)

    text = f"in: {counts[0]}, out: {counts[1]}"
    (width, height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    x, y = line[0][0] - width // 2, line[0][1] - height // 2
    cv2.rectangle(picture, (x, y), (x + width, y + height), (255, 255, 255), -1)
    cv2.putText(
        picture,
        text,
        (x, line[0][1] + height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

    return picture


def _run(session, *, frame, empty=False, line=LINE, zone=ZONE, pretracked=False):
    # Source identity must survive changing batch positions and new frame ids.
    sources = ("cam-a", "cam-b") if frame % 2 else ("cam-b", "cam-a")
    images = [_image(source, frame=frame) for source in sources]
    predictions = [
        _predictions(image, frame=frame, empty=empty, pretracked=pretracked)
        for image in images
    ]
    before = [_pixels(image).copy() for image in images]
    boxes_before = [prediction.xyxy.clone() for prediction in predictions]
    rows = session.run(
        {"image": images, "predictions": predictions, "line": line, "zone": zone}
    ).rows()

    for row, image, original, prediction, original_boxes in zip(
        rows, images, before, predictions, boxes_before
    ):
        drawn = row["drawn"]
        assert drawn.image_id == image.image_id
        assert drawn.video_metadata == image.video_metadata
        assert drawn.parent == image.parent and drawn.root == image.root
        assert drawn.tensor_image.device == image.tensor_image.device
        assert np.array_equal(_pixels(image), original)
        assert torch.equal(prediction.xyxy, original_boxes)
        assert np.array_equal(
            _pixels(drawn),
            _opencv_picture(
                image, counts=(row["in"], row["out"]), line=line, zone=zone
            ),
        )
    by_source = dict(zip(sources, rows))

    return by_source


@pytest.mark.parametrize("mode", ("run", "phases"))
def test_real_counter_outputs_draw_correct_labels_and_shared_zone_across_sources(mode):
    """Check crossing totals, zone membership and glyph pixels across seven frames."""
    session = _plan(mode=mode).create_session()
    try:
        frames = [_run(session, frame=frame) for frame in range(7)]
    finally:
        session.close()

    assert [(row["cam-a"]["in"], row["cam-a"]["out"]) for row in frames] == [
        (0, 0)
    ] * 4 + [(0, 1)] * 3
    assert [(row["cam-b"]["in"], row["cam-b"]["out"]) for row in frames] == [
        (0, 0)
    ] * 4 + [(1, 0)] * 3
    assert [row["cam-a"]["in_zone"] for row in frames] == [0, 0, 0, 0, 0, 1, 1]
    assert [row["cam-b"]["in_zone"] for row in frames] == [0, 1, 1, 0, 0, 0, 0]
    assert all(len(row["cam-a"]["tracked"]) == 1 for row in frames[1:])
    assert frames[6]["cam-a"]["timed"].bboxes_metadata[0][
        "time_in_zone"
    ] == pytest.approx(0.1)
    # Text consumes this frame's source-specific producer outputs, not prior totals.
    for source in ("cam-a", "cam-b"):
        assert not np.array_equal(
            _pixels(frames[3][source]["drawn"]), _pixels(frames[4][source]["drawn"])
        )


@pytest.mark.parametrize("mode", ("run", "phases"))
def test_empty_frames_keep_real_totals_and_selected_geometry_updates_both_consumers(
    mode,
):
    """Keep counts on empty frames, then reset the producer with a new line."""
    session = _plan(mode=mode).create_session()
    moved_line = [[60, 100], [150, 100]]
    try:
        for frame in range(7):
            _run(session, frame=frame)
        empty = _run(session, frame=7, empty=True)
        moved = _run(session, frame=8, line=moved_line, zone=[])
    finally:
        session.close()

    assert (empty["cam-a"]["in"], empty["cam-a"]["out"]) == (0, 1)
    assert (empty["cam-b"]["in"], empty["cam-b"]["out"]) == (1, 0)
    for source in ("cam-a", "cam-b"):
        assert empty[source]["in_zone"] == 0
        assert len(empty[source]["tracked"]) == 0
        assert (moved[source]["in"], moved[source]["out"]) == (0, 0)
        assert moved[source]["in_zone"] == 0
        assert len(moved[source]["timed"]) == 0
        assert (
            _pixels(moved[source]["drawn"])[150, 10].tolist()
            == [40 if source == "cam-a" else 80] * 3
        )


@pytest.mark.parametrize("mode", ("run", "phases"))
def test_same_tracker_id_in_distinct_sources_keeps_counts_and_labels_separate(mode):
    """Feed tracked rows with id 7 to real counters in both independent sources."""
    session = _plan(mode=mode, tracking=False).create_session()
    try:
        frames = [_run(session, frame=frame, pretracked=True) for frame in range(7)]
    finally:
        session.close()

    assert [(row["cam-a"]["in"], row["cam-a"]["out"]) for row in frames] == [
        (0, 0)
    ] * 4 + [(0, 1)] * 3
    assert [(row["cam-b"]["in"], row["cam-b"]["out"]) for row in frames] == [
        (0, 0)
    ] * 4 + [(1, 0)] * 3
    for source in ("cam-a", "cam-b"):
        assert all(
            row[source]["tracked"].bboxes_metadata == [{"tracker_id": 7}]
            for row in frames
        )


@pytest.mark.parametrize("mode", ("run", "phases"))
def test_equal_endpoint_shared_line_is_rejected_without_changing_counter_totals(mode):
    """Reject invalid selected geometry and retain the previous source totals."""
    session = _plan(mode=mode, tracking=False).create_session()
    try:
        for frame in range(7):
            _run(session, frame=frame, pretracked=True)
        with pytest.raises(StepExecutionError, match="needs two different points"):
            _run(session, frame=7, pretracked=True, line=[[60, 60], [60, 60]])
        after = _run(session, frame=8, empty=True, pretracked=True)
    finally:
        session.close()

    assert (after["cam-a"]["in"], after["cam-a"]["out"]) == (0, 1)
    assert (after["cam-b"]["in"], after["cam-b"]["out"]) == (1, 0)
