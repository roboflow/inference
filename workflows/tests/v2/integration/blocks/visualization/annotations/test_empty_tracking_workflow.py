"""A tracker's zero-row frames reach a ``color_axis: TRACK`` drawing (Client M1).

One compiled workflow::

    detections ─▶ v2/trackers_botsort ─▶ v2/dot_visualization (TRACK)

With V1 defaults BoT-SORT reports no rows on frame 0 (tracks not yet
confirmed) and boxes A, B from frame 1. A detector frame with no rows also
gives no tracked rows. Zero-row frames must return the image unchanged, as
V1's supervision annotators do; confirmed frames draw one dot per track.
"""

from datetime import datetime

import numpy as np
import torch
from roboflow_workflows.execution_engine.entities.base import VideoMetadata
from roboflow_workflows.v2.blocks.tracking.botsort import BoTSORTTrackerBlock
from roboflow_workflows.v2.blocks.visualization.annotations.dot import (
    DotVisualizationBlock,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow

from inference_models.models.base.object_detection import Detections

CATALOGUE = Catalogue([BoTSORTTrackerBlock, DotVisualizationBlock])
START = torch.tensor(
    [[10, 10, 40, 40], [100, 20, 130, 50], [200, 60, 230, 90]], dtype=torch.float32
)
STEP = torch.tensor([[4, 0, 4, 0], [0, 3, 0, 3], [-5, 0, -5, 0]], dtype=torch.float32)
DEFINITION = {
    "version": "2.0",
    "inputs": [
        {"type": "WorkflowBatchInput", "name": "image", "kind": ["image"]},
        {
            "type": "WorkflowBatchInput",
            "name": "predictions",
            "kind": ["object_detection_prediction"],
        },
    ],
    "steps": [
        {
            "type": BoTSORTTrackerBlock.type,
            "name": "tracker",
            "image": "$inputs.image",
            "predictions": "$inputs.predictions",
        },
        {
            "type": DotVisualizationBlock.type,
            "name": "dots",
            "image": "$inputs.image",
            "predictions": "$steps.tracker.tracked_detections",
            "color_axis": "TRACK",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "tracked",
            "selector": "$steps.tracker.tracked_detections",
        },
        {"type": "JsonField", "name": "drawn", "selector": "$steps.dots.image"},
    ],
}


def _image() -> ImageData:
    image = ImageData.from_tensor(
        torch.zeros((3, 120, 240), dtype=torch.uint8),
        video_metadata=VideoMetadata(
            video_identifier="camera",
            fps=30,
            frame_number=0,
            frame_timestamp=datetime(2026, 10, 13),
        ),
    )

    return image


def _predictions(frame: int, *, rows: int = 3) -> Detections:
    predictions = Detections(
        xyxy=(START + frame * STEP)[:rows],
        confidence=torch.tensor([0.9, 0.85, 0.4])[:rows],
        class_id=torch.tensor([0, 1, 2])[:rows],
        image_metadata={"parent_id": f"frame-{frame}"},
        bboxes_metadata=[{"name": name} for name in "ABC"[:rows]],
    )

    return predictions


def test_botsort_zero_row_frames_draw_nothing_and_confirmed_tracks_draw_dots():
    session = compile_workflow(DEFINITION, catalogue=CATALOGUE).create_session()
    try:
        rows = [
            session.run({"image": [_image()], "predictions": [predictions]}).rows()[0]
            for predictions in (
                _predictions(0),
                _predictions(1),
                _predictions(2, rows=0),
            )
        ]
    finally:
        session.close()
    unconfirmed, confirmed, no_detections = rows

    for row in (unconfirmed, no_detections):
        assert len(row["tracked"].xyxy) == 0
        assert not row["drawn"].tensor_image.any()
    tracked = confirmed["tracked"]
    assert sorted(meta["name"] for meta in tracked.bboxes_metadata) == ["A", "B"]
    assert all(meta["tracker_id"] is not None for meta in tracked.bboxes_metadata)
    pixels = confirmed["drawn"].tensor_image.permute(1, 2, 0).numpy()
    centres = ((tracked.xyxy[:, :2] + tracked.xyxy[:, 2:]) / 2).round().int()
    for x, y in centres.tolist():
        assert np.any(pixels[y, x] != 0)
