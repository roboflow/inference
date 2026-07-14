"""Integration coverage for tensor-native tracker workflow boundaries."""

from __future__ import annotations

from typing import Any

import pytest
import supervision as sv
import torch

import inference.core.workflows.core_steps.trackers._base_tensor as base_module
from inference.core.workflows.core_steps.trackers._base_tensor import (
    _TRACKER_ROW_INDEX_KEY,
    TrackerBlockBase,
)
from inference_models.models.base.object_detection import Detections
from tests.workflows.unit_tests.core_steps.trackers.conftest import (
    make_metadata,
    wrap_with_workflow_image,
)


class _TensorTrackerBlock(TrackerBlockBase):
    @classmethod
    def get_manifest(cls) -> Any:
        raise NotImplementedError

    def _create_tracker(self, fps: int, **kwargs: Any) -> object:
        return object()

    def run(self, *args: Any, **kwargs: Any) -> Any:
        return self._run_tracker(*args, **kwargs)


class _PackedIdScheduler:
    def __init__(self) -> None:
        self.calls: list[tuple[object, sv.Detections]] = []

    def update(
        self,
        tracker: object,
        detections: sv.Detections,
        **kwargs: Any,
    ) -> tuple[sv.Detections, list[int]]:
        self.calls.append((tracker, detections))
        output = sv.Detections(
            xyxy=detections.xyxy,
            confidence=detections.confidence,
            class_id=detections.class_id,
            tracker_id=torch.tensor([-1, 41], dtype=torch.long),
            data={_TRACKER_ROW_INDEX_KEY: detections.data[_TRACKER_ROW_INDEX_KEY]},
        )
        return output, [-1, 41]


class _MissingRowScheduler(_PackedIdScheduler):
    def update(
        self,
        tracker: object,
        detections: sv.Detections,
        **kwargs: Any,
    ) -> tuple[sv.Detections, list[int]]:
        self.calls.append((tracker, detections))
        output = sv.Detections(
            xyxy=detections.xyxy,
            confidence=detections.confidence,
            class_id=detections.class_id,
            tracker_id=torch.tensor([-1, 41], dtype=torch.long),
        )
        return output, [-1, 41]


def _native_detections() -> Detections:
    return Detections(
        xyxy=torch.tensor(
            [[10.0, 10.0, 20.0, 20.0], [30.0, 30.0, 40.0, 40.0]],
            dtype=torch.float32,
        ),
        class_id=torch.tensor([3, 7], dtype=torch.long),
        confidence=torch.tensor([0.9, 0.8], dtype=torch.float32),
        bboxes_metadata=[{"source": "first"}, {"source": "second"}],
    )


def test_packed_host_tracker_ids_follow_unmatched_track_filter(
    monkeypatch,
) -> None:
    scheduler = _PackedIdScheduler()
    block = _TensorTrackerBlock()
    monkeypatch.setattr(
        block,
        "_can_batch_tracker_update",
        lambda detections: True,
    )
    monkeypatch.setattr(
        base_module,
        "get_tracker_batch_scheduler",
        lambda: scheduler,
    )
    source = _native_detections()
    image = wrap_with_workflow_image(make_metadata(1))

    first = block._run_tracker(
        image=image,
        detections=source,
        instances_cache_size=16,
    )
    second = block._run_tracker(
        image=image,
        detections=source,
        instances_cache_size=16,
    )

    assert len(scheduler.calls) == 2
    assert torch.equal(
        first["tracked_detections"].xyxy,
        source.xyxy[[1]],
    )
    assert first["tracked_detections"].bboxes_metadata == [
        {"source": "second", "tracker_id": 41}
    ]
    assert len(first["new_instances"].xyxy) == 1
    assert len(first["already_seen_instances"].xyxy) == 0
    assert len(second["new_instances"].xyxy) == 0
    assert len(second["already_seen_instances"].xyxy) == 1
    assert source.bboxes_metadata == [{"source": "first"}, {"source": "second"}]


def test_packed_tracker_ids_require_surviving_row_lineage(monkeypatch) -> None:
    scheduler = _MissingRowScheduler()
    block = _TensorTrackerBlock()
    monkeypatch.setattr(
        block,
        "_can_batch_tracker_update",
        lambda detections: True,
    )
    monkeypatch.setattr(
        base_module,
        "get_tracker_batch_scheduler",
        lambda: scheduler,
    )

    with pytest.raises(
        RuntimeError,
        match="batched tracker IDs do not align with surviving input rows",
    ):
        block._run_tracker(
            image=wrap_with_workflow_image(make_metadata(1)),
            detections=_native_detections(),
            instances_cache_size=16,
        )
