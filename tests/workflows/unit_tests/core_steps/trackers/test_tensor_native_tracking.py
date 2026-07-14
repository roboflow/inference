"""Tests for device-resident tensor tracker workflow outputs."""

from __future__ import annotations

import datetime
from types import SimpleNamespace

import numpy as np
import pytest
import supervision as sv
import torch
import tracktors

from inference.core.workflows.core_steps.trackers._base_tensor import (
    _TRACKER_ROW_INDEX_KEY,
    InstanceCache,
    _bbox_component,
)
from inference.core.workflows.core_steps.trackers.bytetrack.v1_tensor import (
    ByteTrackBlockV1,
    ByteTrackManifest,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.profiling.core import (
    NullWorkflowsProfiler,
)
from inference.core.workflows.execution_engine.v1.executor.core import run_simd_step
from inference.core.workflows.execution_engine.v1.executor.execution_data_manager.step_input_assembler import (
    BatchModeSIMDStepInput,
)
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections


def _image(video_id: str) -> WorkflowImageData:
    """Build one workflow frame with independent video state."""
    metadata = VideoMetadata(
        video_identifier=video_id,
        frame_number=1,
        frame_timestamp=datetime.datetime.fromtimestamp(1).astimezone(
            datetime.timezone.utc
        ),
        fps=30,
    )
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id=video_id),
        numpy_image=np.zeros((8, 8, 3), dtype=np.uint8),
        video_metadata=metadata,
    )


def _objects(value: float) -> Detections:
    """Build one tensor-native object detection without ragged metadata."""
    return Detections(
        xyxy=torch.tensor([[value, 0.0, value + 1.0, 1.0]]),
        class_id=torch.tensor([0]),
        confidence=torch.tensor([0.9]),
    )


def test_instance_cache_vectorizes_unique_tracker_contract_and_wraparound() -> None:
    """Unique per-frame IDs preserve seen decisions and newest FIFO ring rows."""
    cache = InstanceCache(size=3)

    first_seen = cache.record_instances(torch.tensor([10, 11, 12]))
    second_seen = cache.record_instances(torch.tensor([11, 13]))
    third_seen = cache.record_instances(torch.tensor([10, 12, 14]))
    wrapped_seen = cache.record_instances(torch.tensor([20, 21, 22, 23]))

    assert first_seen.tolist() == [False, False, False]
    assert second_seen.tolist() == [True, False]
    assert third_seen.tolist() == [False, True, False]
    assert wrapped_seen.tolist() == [False, False, False, False]
    assert cache._ids[:3].tolist() == [23, 21, 22]


def test_instance_cache_does_not_call_cpu_or_tolist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Routine cache classification never materializes tracker IDs on the host."""
    cache = InstanceCache(size=8)

    def reject_boundary(*args, **kwargs):
        """Fail if the tensor-native cache enters an explicit host boundary."""
        raise AssertionError("unexpected tracker ID materialization")

    monkeypatch.setattr(torch.Tensor, "cpu", reject_boundary)
    monkeypatch.setattr(torch.Tensor, "tolist", reject_boundary)

    seen = cache.record_instances(torch.tensor([1, 2, 3]))

    assert torch.equal(seen, torch.tensor([False, False, False]))


@pytest.mark.parametrize("prediction_kind", ["object", "instance", "keypoint"])
def test_recover_tracker_output_keeps_ids_on_native_bbox_component(
    prediction_kind: str,
) -> None:
    """Object, instance, and keypoint bbox paths attach the same tensor IDs."""
    boxes = Detections(
        xyxy=torch.tensor([[0.0, 0.0, 2.0, 2.0], [4.0, 4.0, 6.0, 6.0]]),
        class_id=torch.tensor([0, 1]),
        confidence=torch.tensor([0.9, 0.8]),
    )
    if prediction_kind == "instance":
        prediction = InstanceDetections(
            xyxy=boxes.xyxy,
            class_id=boxes.class_id,
            confidence=boxes.confidence,
            mask=torch.ones((2, 8, 8), dtype=torch.bool),
        )
    elif prediction_kind == "keypoint":
        prediction = (
            KeyPoints(
                xy=torch.ones((2, 3, 2)),
                class_id=boxes.class_id,
                confidence=torch.ones((2, 3)),
            ),
            boxes,
        )
    else:
        prediction = boxes
    tracked_sv = sv.Detections(
        xyxy=boxes.xyxy,
        class_id=boxes.class_id,
        confidence=boxes.confidence,
        tracker_id=torch.tensor([31, 32]),
        data={_TRACKER_ROW_INDEX_KEY: torch.tensor([0, 1])},
    )

    tracked, tracker_ids = ByteTrackBlockV1._recover_tracker_output(
        detections=prediction,
        bbox=_bbox_component(prediction),
        tracked_sv=tracked_sv,
    )

    assert tracker_ids.device == boxes.xyxy.device
    assert _bbox_component(tracked).tracker_id.tolist() == [31, 32]
    assert _bbox_component(tracked).bboxes_metadata is None


def test_recover_tracker_output_filters_control_tensors_without_sv_row_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unconfirmed rows are removed before the one native prediction gather."""
    prediction = Detections(
        xyxy=torch.tensor(
            [
                [0.0, 0.0, 2.0, 2.0],
                [4.0, 4.0, 6.0, 6.0],
                [8.0, 8.0, 10.0, 10.0],
            ]
        ),
        class_id=torch.tensor([0, 1, 2]),
        confidence=torch.tensor([0.9, 0.8, 0.7]),
    )
    tracked_sv = sv.Detections(
        xyxy=prediction.xyxy,
        class_id=prediction.class_id,
        confidence=prediction.confidence,
        tracker_id=torch.tensor([-1, 42, -1]),
        data={_TRACKER_ROW_INDEX_KEY: torch.tensor([2, 0, 1])},
    )

    def reject_supervision_slice(*args, **kwargs):
        """The recovery boundary must not copy the temporary tracker output."""
        raise AssertionError("unexpected sv.Detections row selection")

    monkeypatch.setattr(sv.Detections, "__getitem__", reject_supervision_slice)

    tracked, tracker_ids = ByteTrackBlockV1._recover_tracker_output(
        detections=prediction,
        bbox=prediction,
        tracked_sv=tracked_sv,
    )

    assert torch.equal(tracker_ids, torch.tensor([42]))
    assert torch.equal(tracked.xyxy, prediction.xyxy[[0]])
    assert torch.equal(tracked.tracker_id, tracker_ids)


class _ExecutionDataManager:
    """Minimal batch-mode manager used to exercise the real SIMD dispatcher."""

    def __init__(self, parameters: dict) -> None:
        """Store one assembled SIMD input and capture registered outputs."""
        self.parameters = parameters
        self.outputs = None
        self.indices = [(index,) for index in range(4)]

    def get_simd_step_input(self, step_selector: str) -> BatchModeSIMDStepInput:
        """Return the preassembled four-source tracker input."""
        return BatchModeSIMDStepInput(
            indices=self.indices,
            parameters=self.parameters,
        )

    def register_simd_step_output(
        self,
        step_selector: str,
        indices: list,
        outputs: list,
    ) -> None:
        """Capture the dispatcher result and its exact dimensional indices."""
        self.indices = indices
        self.outputs = outputs


def test_simd_execution_calls_tracktors_update_batch_once_for_four_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The execution engine sends one four-video call to Tracktors batch update."""
    indices = [(index,) for index in range(4)]
    images = Batch.init([_image(f"video-{index}") for index in range(4)], indices)
    detections = Batch.init([_objects(float(index)) for index in range(4)], indices)
    block = ByteTrackBlockV1()
    block._trackers = {f"video-{index}": object() for index in range(4)}
    monkeypatch.setattr(block, "_can_batch_tracker_update", lambda _: True)
    calls: list[int] = []
    executors = []

    def update_batch(trackers, batch_detections, **kwargs):
        """Attach deterministic same-device IDs to each fake tracker output."""
        calls.append(len(trackers))
        executors.append(kwargs["executor"])
        for index, output in enumerate(batch_detections):
            output.tracker_id = torch.tensor(
                [100 + index],
                device=output.xyxy.device,
            )
        return batch_detections

    monkeypatch.setattr(tracktors, "update_batch", update_batch)
    manager = _ExecutionDataManager(
        parameters={"image": images, "detections": detections}
    )
    workflow = SimpleNamespace(
        steps={
            "tracker": SimpleNamespace(
                step=block,
                manifest=ByteTrackManifest,
            )
        }
    )

    run_simd_step(
        step_selector="tracker",
        workflow=workflow,
        execution_data_manager=manager,
        profiler=NullWorkflowsProfiler.init(),
    )

    assert calls == [4]
    assert isinstance(executors[0], tracktors.CUDABatchExecutor)
    assert manager.indices == indices
    assert len(manager.outputs) == 4
    assert [
        result["tracked_detections"].tracker_id.item() for result in manager.outputs
    ] == [100, 101, 102, 103]
    assert all(
        result["tracked_detections"].bboxes_metadata is None
        for result in manager.outputs
    )
