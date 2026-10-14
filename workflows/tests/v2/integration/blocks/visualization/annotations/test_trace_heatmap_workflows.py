"""Trace and heatmap blocks in compiled V2 workflows over video frames.

Real blocks end to end: ``roboflow_core/roboflow_instance_segmentation_model@v4`` with the painted-scene
fake model (``scene``; not model inference), ``roboflow_core/trackers_bytetrack@v1``, then
the trace and heatmap blocks. One session runs frame after frame, so state
lives in the session's block instances as in a video pipeline. The active
tracker confirms a new track on its second frame; its ids are shared across
tracker instances, so the checks compare continuity rather than a fixed id.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Sequence

import numpy as np
import pytest
import torch
from roboflow_workflows.execution_engine.entities.base import VideoMetadata
from roboflow_workflows.v2.blocks.models.cache import ModelCache
from roboflow_workflows.v2.blocks.visualization.annotations.heatmap_visualization.v1 import HeatmapVisualizationBlock
from roboflow_workflows.v2.blocks.visualization.annotations.trace_visualization.v1 import TraceVisualizationBlock
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.errors import StepExecutionError
from roboflow_workflows.v2.engine.plan import CompileOptions

from tests.v2.integration.blocks.visualization.annotations.scene import (
    RED,
    STREET_SIZE,
    SceneLoader,
    Shape,
    annotation_catalogue,
    changed,
    pixels_of,
    scene_pixels,
)

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
GREEN = (0, 255, 0)
MODES = ("run", "phases")
SEGMENT_AND_TRACK = [
    {
        "type": "roboflow_core/roboflow_instance_segmentation_model@v4",
        "name": "seg",
        "image": "$inputs.image",
        "model_id": "shapes-seg",
    },
    {
        "type": "roboflow_core/trackers_bytetrack@v1",
        "name": "tracker",
        "image": "$inputs.image",
        "predictions": "$steps.seg.predictions",
        "frame_rate": 30.0,
        "minimum_iou_threshold": 0.3,
        "minimum_consecutive_frames": 2,
        "track_activation_threshold": 0.5,
        "high_conf_det_threshold": 0.5,
    },
]
TRACE = {
    "type": "roboflow_core/trace_visualization@v1",
    "name": "trace",
    "image": "$inputs.image",
    "predictions": "$steps.tracker.tracked_detections",
    "trace_length": "$inputs.trace_length",
    "thickness": 2,
    "color_palette": "CUSTOM",
    "custom_colors": ["#00FF00"],
}
HEAT = {
    "type": "roboflow_core/heatmap_visualization@v1",
    "name": "heat",
    "image": "$steps.trace.image",
    "predictions": "$steps.tracker.tracked_detections",
    "ignore_stationary": False,
    "radius": 4,
    "kernel_size": 1,
    "opacity": 1.0,
}
OUTPUTS = {
    "tracked": "$steps.tracker.tracked_detections",
    "trace": "$steps.trace.image",
    "heat": "$steps.heat.image",
}


def compile_plan(mode: str = "run", **heat_parameters: Any):
    steps = [*SEGMENT_AND_TRACK, TRACE, {**HEAT, **heat_parameters}]
    plan = compile_steps(steps, OUTPUTS, mode=mode)

    return plan


def compile_steps(steps: List[Dict], outputs: Dict[str, str], *, mode: str = "run"):
    definition = {
        "version": "2.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            {"type": "WorkflowParameter", "name": "trace_length", "default_value": 10},
        ],
        "steps": steps,
        "outputs": [
            {"type": "JsonField", "name": name, "selector": selector}
            for name, selector in outputs.items()
        ],
    }
    catalogue = annotation_catalogue().with_blocks(
        [TraceVisualizationBlock, HeatmapVisualizationBlock]
    )
    plan = compile_workflow(
        definition, catalogue=catalogue, options=CompileOptions(block_execution=mode)
    )

    return plan


def video_frame(video: str, number: int, *, x: int, y: int = 10) -> ImageData:
    """A red ellipse whose box starts at ``(x, y)``, as frame ``number``."""
    shape = Shape("ellipse", (x, y, x + 22, y + 30), RED)
    pixels = torch.from_numpy(scene_pixels(STREET_SIZE, [shape])).permute(2, 0, 1)
    metadata = VideoMetadata(
        video_identifier=video,
        frame_number=number,
        frame_timestamp=T0 + timedelta(seconds=number / 30),
        fps=30.0,
    )
    image = ImageData.from_tensor(
        pixels.contiguous(), image_id=f"{video}-{number}", video_metadata=metadata
    )

    return image


def run_frames(
    session: Any, batches: Sequence[Sequence[ImageData]], **values: Any
) -> List[List[Dict[str, Any]]]:
    results = [
        session.run({"image": list(batch), **values}).rows() for batch in batches
    ]

    return results


def warm_up_tracker(session: Any, batch: Sequence[ImageData]) -> None:
    """Observe the real tracker's unconfirmed frame before measuring trails."""
    rows = run_frames(session, [batch])[0]
    assert all(len(row["tracked"]) == 0 for row in rows)
    assert all(not green(row["trace"]).any() for row in rows)
    assert all(not changed(row["trace"], row["heat"]).any() for row in rows)


def centre_of(tracked: Any) -> List[int]:
    x1, y1, x2, y2 = tracked.xyxy[0].tolist()
    centre = [int((x1 + x2) / 2), int((y1 + y2) / 2)]

    return centre


def green(image: ImageData) -> np.ndarray:
    mask = np.all(pixels_of(image) == GREEN, axis=-1)

    return mask


@pytest.mark.parametrize("mode", MODES)
def test_a_tracked_object_leaves_a_trail_and_a_heat_track(mode: str) -> None:
    session = compile_plan(mode).create_session({"models": ModelCache(SceneLoader())})
    warm_up_tracker(session, [video_frame("street", 0, x=8)])
    frames = [
        [video_frame("street", number + 1, x=8 + 8 * number)]
        for number in range(5)
    ]
    rows = [batch[0] for batch in run_frames(session, frames)]

    centres = [centre_of(row["tracked"]) for row in rows]
    assert all(len(row["tracked"]) == 1 for row in rows)
    tracker_ids = [row["tracked"].bboxes_metadata[0]["tracker_id"] for row in rows]
    assert tracker_ids[0] >= 0
    assert tracker_ids == [tracker_ids[0]] * 5
    assert not green(rows[0]["trace"]).any()
    last = green(rows[-1]["trace"])
    assert all(last[y, x] for x, y in centres)
    heat = changed(rows[-1]["trace"], rows[-1]["heat"])
    bottoms = [
        (x, int(row["tracked"].xyxy[0, 3])) for (x, _), row in zip(centres, rows)
    ]
    assert all(heat[min(y, STREET_SIZE[0] - 1), x] for x, y in bottoms)
    assert not heat[:, 90:].any()


def test_two_cameras_in_one_batch_keep_their_own_trails() -> None:
    session = compile_plan().create_session({"models": ModelCache(SceneLoader())})
    warm_up_tracker(
        session, [video_frame("left", 0, x=6), video_frame("right", 0, x=80)]
    )
    batches = [
        [
            video_frame("left", number + 1, x=6 + 6 * number),
            video_frame("right", number + 1, x=80 - 6 * number),
        ]
        for number in range(4)
    ]
    left, right = run_frames(session, batches)[-1]

    left_trail, right_trail = green(left["trace"]), green(right["trace"])
    assert left_trail.any() and right_trail.any()
    assert not right_trail[:, :40].any()
    assert not left_trail[:, 60:].any()


@pytest.mark.parametrize("drawing", [TRACE, HEAT])
def test_a_cropped_frame_is_refused_with_draw_before_cropping(drawing: Dict) -> None:
    crop = {
        "type": "roboflow_core/absolute_static_crop@v1",
        "name": "crop",
        "image": "$inputs.image",
        "x_center": 40,
        "y_center": 36,
        "width": 80,
        "height": 72,
    }
    step = {**drawing, "image": "$steps.crop.crops"}
    plan = compile_steps(
        [*SEGMENT_AND_TRACK, crop, step], {"drawn": f"$steps.{step['name']}.image"}
    )
    session = plan.create_session({"models": ModelCache(SceneLoader())})

    with pytest.raises(StepExecutionError, match="Draw before cropping"):
        run_frames(session, [[video_frame("street", 0, x=8)]])


def test_a_resized_frame_gets_its_trail_and_heat_at_its_own_scale() -> None:
    resize = {
        "type": "roboflow_core/image_preprocessing@v1",
        "name": "resize",
        "image": "$inputs.image",
        "task_type": "resize",
        "width": 56,
        "height": 36,
    }
    steps = [
        resize,
        *SEGMENT_AND_TRACK,
        {**TRACE, "image": "$steps.resize.image"},
        HEAT,
    ]
    plan = compile_steps(steps, OUTPUTS)
    session = plan.create_session({"models": ModelCache(SceneLoader())})
    warm_up_tracker(session, [video_frame("street", 0, x=8)])
    frames = [
        [video_frame("street", number + 1, x=8 + 8 * number)]
        for number in range(4)
    ]
    rows = [batch[0] for batch in run_frames(session, frames)]

    trace, heat = rows[-1]["trace"], rows[-1]["heat"]
    assert trace.size_hw == heat.size_hw == (36, 56)
    halves = [[value // 2 for value in centre_of(row["tracked"])] for row in rows]
    assert all(green(trace)[y, x] for x, y in halves)
    assert changed(trace, heat).any()


def test_a_selected_trace_length_is_checked_when_it_arrives() -> None:
    session = compile_plan().create_session({"models": ModelCache(SceneLoader())})

    with pytest.raises(
        StepExecutionError, match="trace_length must be an integer >= 1"
    ):
        run_frames(session, [[video_frame("street", 0, x=8)]], trace_length=0)


def test_a_literal_hue_outside_the_opencv_range_fails_compilation() -> None:
    with pytest.raises(Exception, match="top_hue must be an OpenCV hue from 0 to 179"):
        compile_plan(top_hue=200)
