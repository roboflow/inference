"""Focused contract tests for direct SIMD tracker workflow dispatch."""

from __future__ import annotations

import datetime
from typing import Any

import numpy as np
import pytest
import torch
import tracktors

import inference.core.workflows.core_steps.trackers._base_tensor as base_tensor
from inference.core.workflows.core_steps.trackers.batch_scheduler import (
    TrackerBatchScheduler,
)
from inference.core.workflows.core_steps.trackers.bytetrack.v1_tensor import (
    ByteTrackBlockV1,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)
from inference_models.models.base.object_detection import Detections


def _image(video_id: str) -> WorkflowImageData:
    """Build one workflow image carrying independent video state."""
    metadata = VideoMetadata(
        video_identifier=video_id,
        frame_number=1,
        frame_timestamp=datetime.datetime.fromtimestamp(
            1,
            tz=datetime.timezone.utc,
        ),
        fps=30,
    )
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id=video_id),
        numpy_image=np.zeros((8, 8, 3), dtype=np.uint8),
        video_metadata=metadata,
    )


def _detections(stream_index: int) -> Detections:
    """Build two native rows whose numeric fields remain tensors."""
    offset = float(stream_index * 10)
    return Detections(
        xyxy=torch.tensor(
            [
                [offset, 0.0, offset + 2.0, 2.0],
                [offset + 3.0, 0.0, offset + 5.0, 2.0],
            ]
        ),
        class_id=torch.zeros(2, dtype=torch.long),
        confidence=torch.full((2,), 0.95),
    )


def test_one_workflow_batch_calls_persistent_executor_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One direct Batch invokes update_batch once and never queues scalar work."""
    stream_count = 4
    indices = [(index,) for index in range(stream_count)]
    images = Batch.init(
        [_image(f"video-{index}") for index in range(stream_count)],
        indices,
    )
    detections = Batch.init(
        [_detections(index) for index in range(stream_count)],
        indices,
    )
    block = ByteTrackBlockV1()
    block._trackers = {f"video-{index}": object() for index in range(stream_count)}
    monkeypatch.setattr(block, "_can_batch_tracker_update", lambda _: True)
    scheduler = TrackerBatchScheduler(batch_window_ms=0.0, max_batch_size=8)
    monkeypatch.setattr(
        base_tensor,
        "get_tracker_batch_scheduler",
        lambda: scheduler,
    )

    def reject_scalar_update(*args: Any, **kwargs: Any) -> None:
        """Fail if direct SIMD dispatch enters the Future-based scalar queue."""
        raise AssertionError("direct Batch unexpectedly used scheduler.update")

    monkeypatch.setattr(scheduler, "update", reject_scalar_update)
    calls: list[tuple[int, Any]] = []

    def update_batch(
        trackers: list[Any],
        batch_detections: list[Any],
        **kwargs: Any,
    ) -> list[Any]:
        """Return deterministic IDs while recording the persistent executor."""
        calls.append((len(trackers), kwargs["executor"]))
        for stream_index, output in enumerate(batch_detections):
            output.tracker_id = torch.arange(len(output)) + stream_index * 10
        return batch_detections

    monkeypatch.setattr(tracktors, "update_batch", update_batch)
    persistent_executor = None
    try:
        results = block.run(
            image=images,
            detections=detections,
            minimum_consecutive_frames=1,
        )
        persistent_executor = scheduler._executor
    finally:
        scheduler.close()

    assert len(calls) == 1
    assert calls[0][0] == stream_count
    assert isinstance(calls[0][1], tracktors.CUDABatchExecutor)
    assert calls[0][1] is persistent_executor
    assert isinstance(results, list)
    assert len(results) == stream_count
    assert [result["tracked_detections"].tracker_id.tolist() for result in results] == [
        [0, 1],
        [10, 11],
        [20, 21],
        [30, 31],
    ]
