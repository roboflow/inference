"""Tests for device-resident tensor tracker workflow outputs."""

from __future__ import annotations

import datetime
from collections import deque
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
from inference.core.workflows.core_steps.trackers.instance_cache_kernels import (
    has_triton_instance_cache,
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


def _deque_record(
    history: deque[int],
    tracker_ids: list[int],
) -> list[bool]:
    """Apply the legacy sequential FIFO contract to a deque reference."""
    seen = []
    for tracker_id in tracker_ids:
        was_seen = tracker_id in history
        seen.append(was_seen)
        if not was_seen:
            history.append(tracker_id)
    return seen


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


def test_instance_cache_records_interleaved_new_ids_in_fifo_order() -> None:
    """Vector insertion stores the actual new rows, not their compacted ranks."""
    cache = InstanceCache(size=3)

    assert cache.record_instances(torch.tensor([10, 11, 12])).tolist() == [
        False,
        False,
        False,
    ]
    assert cache.record_instances(torch.tensor([11, 13])).tolist() == [True, False]
    assert cache._ids[:3].tolist() == [13, 11, 12]
    assert cache.record_instances(torch.tensor([13, 14])).tolist() == [True, False]
    assert cache._ids[:3].tolist() == [13, 14, 12]


@pytest.mark.parametrize(
    "frames",
    [
        pytest.param(
            [[10, 11, 12], [10, 13, 10], [12, 14, 13]],
            id="interleaved-eviction",
        ),
        pytest.param(
            [[1, 1, 2, 2], [3, 1, 3, 4], [2, 4, 2]],
            id="duplicates",
        ),
        pytest.param(
            [[-1, 0, 1], [-1, 2, 0], [-(2**63), 2**63 - 1, -1]],
            id="signed-extremes",
        ),
        pytest.param(
            [list(range(20)), [19, 0, 20, 1, 21], list(range(21, -1, -1))],
            id="multiple-wraps",
        ),
    ],
)
def test_instance_cache_matches_adversarial_deque_reference(
    frames: list[list[int]],
) -> None:
    """Scalar cache matches exact deque decisions under adversarial ordering."""
    size = 3
    cache = InstanceCache(size=size)
    reference: deque[int] = deque(maxlen=size)

    for frame in frames:
        actual = cache.record_instances(torch.tensor(frame, dtype=torch.long))
        expected = _deque_record(reference, frame)

        assert actual.tolist() == expected
        assert set(cache._ids[cache._valid].tolist()) == set(reference)


def test_batch_instance_cache_preserves_interleaved_eviction_order() -> None:
    """Batch result indices reflect mutations made by earlier rows in a frame."""
    block = ByteTrackBlockV1()

    def prediction(ids: list[int]) -> Detections:
        """Build one native prediction carrying the requested tracker IDs."""
        count = len(ids)
        return Detections(
            xyxy=torch.arange(count * 4, dtype=torch.float32).reshape(count, 4),
            class_id=torch.zeros(count, dtype=torch.long),
            confidence=torch.ones(count),
            tracker_id=torch.tensor(ids, dtype=torch.long),
        )

    initial = prediction([10, 11, 12])
    block._build_tracker_results_batch(
        video_ids=["video"],
        tracked_detections=[initial],
        tracker_ids=[initial.tracker_id],
        instances_cache_size=3,
    )
    interleaved = prediction([10, 13, 10])

    result = block._build_tracker_results_batch(
        video_ids=["video"],
        tracked_detections=[interleaved],
        tracker_ids=[interleaved.tracker_id],
        instances_cache_size=3,
    )[0]

    assert result["new_instances"].tracker_id.tolist() == [13, 10]
    assert result["already_seen_instances"].tracker_id.tolist() == [10]


def test_instance_cache_scalar_batch_scalar_interop() -> None:
    """Scalar and batch entry points share one persistent FIFO/hash state."""
    block = ByteTrackBlockV1()
    cache = InstanceCache(size=3)
    block._per_video_cache["video"] = cache
    assert cache.record_instances(torch.tensor([1, 2, 3])).tolist() == [
        False,
        False,
        False,
    ]
    prediction = Detections(
        xyxy=torch.zeros((3, 4)),
        class_id=torch.zeros(3, dtype=torch.long),
        confidence=torch.ones(3),
        tracker_id=torch.tensor([1, 4, 1]),
    )

    result = block._build_tracker_results_batch(
        video_ids=["video"],
        tracked_detections=[prediction],
        tracker_ids=[prediction.tracker_id],
        instances_cache_size=3,
    )[0]
    final_seen = cache.record_instances(torch.tensor([4, 5]))

    assert result["new_instances"].tracker_id.tolist() == [4, 1]
    assert result["already_seen_instances"].tracker_id.tolist() == [1]
    assert final_seen.tolist() == [True, False]


@pytest.mark.skipif(
    not torch.cuda.is_available() or not has_triton_instance_cache(),
    reason="CUDA Triton instance cache is unavailable",
)
def test_cuda_instance_cache_matches_long_deque_churn() -> None:
    """Fused CUDA hashing preserves exact deque decisions across long churn."""
    size = 17
    cache = InstanceCache(size=size)
    reference: deque[int] = deque(maxlen=size)
    generator = torch.Generator().manual_seed(9137)

    for _ in range(100):
        frame = torch.randint(
            -(2**20),
            2**20,
            (64,),
            generator=generator,
            dtype=torch.long,
        ).tolist()
        duplicate_positions = frame[1::7]
        frame[1::7] = frame[::7][: len(duplicate_positions)]
        expected = _deque_record(reference, frame)
        actual = cache.record_instances(
            torch.tensor(frame, dtype=torch.long, device="cuda")
        )

        assert actual.cpu().tolist() == expected
        assert set(cache._ids[cache._valid].cpu().tolist()) == set(reference)


def test_batch_instance_cache_matches_independent_scalar_caches() -> None:
    """Ragged SIMD cache classification preserves each video's FIFO history."""
    block = ByteTrackBlockV1()
    video_ids = ["video-0", "video-1", "video-2"]

    def prediction(ids: list[int]) -> Detections:
        count = len(ids)
        return Detections(
            xyxy=torch.arange(count * 4, dtype=torch.float32).reshape(count, 4),
            class_id=torch.zeros(count, dtype=torch.long),
            confidence=torch.ones(count),
            tracker_id=torch.tensor(ids, dtype=torch.long),
        )

    first_ids = [[10, 11], [20], [30, 31, 32]]
    first_predictions = [prediction(ids) for ids in first_ids]
    first = block._build_tracker_results_batch(
        video_ids=video_ids,
        tracked_detections=first_predictions,
        tracker_ids=[item.tracker_id for item in first_predictions],
        instances_cache_size=3,
    )
    assert [item["new_instances"].tracker_id.tolist() for item in first] == first_ids
    assert [len(item["already_seen_instances"]) for item in first] == [0, 0, 0]

    second_ids = [[11, 12], [21, 20], [32, 33]]
    second_predictions = [prediction(ids) for ids in second_ids]
    second = block._build_tracker_results_batch(
        video_ids=video_ids,
        tracked_detections=second_predictions,
        tracker_ids=[item.tracker_id for item in second_predictions],
        instances_cache_size=3,
    )
    assert [item["new_instances"].tracker_id.tolist() for item in second] == [
        [12],
        [21],
        [33],
    ]
    assert [item["already_seen_instances"].tracker_id.tolist() for item in second] == [
        [11],
        [20],
        [32],
    ]


def test_batch_instance_cache_preserves_duplicate_video_order() -> None:
    """Repeated video rows use sequential cache semantics instead of aliasing."""
    block = ByteTrackBlockV1()

    def prediction(tracker_id: int) -> Detections:
        return Detections(
            xyxy=torch.zeros((1, 4)),
            class_id=torch.zeros(1, dtype=torch.long),
            confidence=torch.ones(1),
            tracker_id=torch.tensor([tracker_id]),
        )

    predictions = [prediction(10), prediction(10)]
    results = block._build_tracker_results_batch(
        video_ids=["same-video", "same-video"],
        tracked_detections=predictions,
        tracker_ids=[item.tracker_id for item in predictions],
        instances_cache_size=8,
    )

    assert results[0]["new_instances"].tracker_id.tolist() == [10]
    assert len(results[0]["already_seen_instances"]) == 0
    assert len(results[1]["new_instances"]) == 0
    assert results[1]["already_seen_instances"].tracker_id.tolist() == [10]


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


def test_batch_recovery_stably_filters_ragged_unconfirmed_rows() -> None:
    """SIMD recovery preserves per-stream row order with one packed partition."""
    predictions = [
        Detections(
            xyxy=torch.arange(count * 4, dtype=torch.float32).reshape(count, 4),
            class_id=torch.arange(count),
            confidence=torch.ones(count),
        )
        for count in (3, 2, 1)
    ]
    output_rows = [torch.tensor([2, 0, 1]), torch.tensor([1, 0]), torch.tensor([0])]
    output_ids = [
        torch.tensor([-1, 40, 41]),
        torch.tensor([50, -1]),
        torch.tensor([-1]),
    ]
    tracked_outputs = [
        sv.Detections(
            xyxy=prediction.xyxy,
            class_id=prediction.class_id,
            confidence=prediction.confidence,
            tracker_id=ids,
            data={_TRACKER_ROW_INDEX_KEY: rows},
        )
        for prediction, rows, ids in zip(predictions, output_rows, output_ids)
    ]

    recovered = ByteTrackBlockV1._recover_tracker_outputs_batch(
        detections=predictions,
        bboxes=predictions,
        tracked_outputs=tracked_outputs,
    )

    assert [item[1].tolist() for item in recovered] == [[40, 41], [50], []]
    assert [item[0].class_id.tolist() for item in recovered] == [[0, 1], [1], []]


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
