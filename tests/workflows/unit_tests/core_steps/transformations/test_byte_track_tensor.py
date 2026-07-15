import datetime
from typing import Any

import numpy as np
import pytest
import supervision as sv
import torch

from inference.core.workflows.core_steps.transformations.byte_tracker._tensor_native import (
    recover_tensor_byte_tracker_output,
    update_tensor_byte_tracker,
)
from inference.core.workflows.core_steps.transformations.byte_tracker.v1_tensor import (
    ByteTrackerBlockV1,
)
from inference.core.workflows.core_steps.transformations.byte_tracker.v2_tensor import (
    ByteTrackerBlockV2,
)
from inference.core.workflows.core_steps.transformations.byte_tracker.v3_tensor import (
    ByteTrackerBlockV3,
    InstanceCache,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)
from inference_models.models.base.object_detection import Detections


class _ReorderingTracker:
    """Minimal tracker double that records and reverses its tensor input."""

    def __init__(self) -> None:
        self.received: sv.Detections | None = None

    def update_with_detections(self, detections: sv.Detections) -> sv.Detections:
        """Return reversed rows with deterministic tracker IDs."""
        self.received = detections
        indices = torch.arange(
            len(detections) - 1,
            -1,
            -1,
            device=detections.xyxy.device,
        )
        result = detections[indices]
        result.tracker_id = torch.arange(
            101,
            101 + len(result),
            dtype=torch.long,
            device=detections.xyxy.device,
        )
        return result


def _detections(device: torch.device | str = "cpu") -> Detections:
    """Build two tensor-native detections with distinguishable metadata."""
    return Detections(
        xyxy=torch.tensor(
            [[10.0, 10.0, 20.0, 20.0], [40.0, 10.0, 50.0, 20.0]],
            device=device,
        ),
        class_id=torch.tensor([3, 7], dtype=torch.long, device=device),
        confidence=torch.tensor([0.9, 0.8], device=device),
        image_metadata={"source": "camera"},
        bboxes_metadata=[{"name": "first"}, {"name": "second"}],
    )


def _video_metadata(frame_number: int = 0) -> VideoMetadata:
    """Build stable video metadata for one test frame."""
    return VideoMetadata(
        video_identifier="tensor-video",
        frame_number=frame_number,
        fps=30,
        frame_timestamp=datetime.datetime.fromtimestamp(
            1726570875 + frame_number,
            tz=datetime.timezone.utc,
        ),
        comes_from_video_file=True,
    )


def _workflow_image(frame_number: int = 0) -> WorkflowImageData:
    """Wrap test video metadata in a workflow image."""
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="root"),
        numpy_image=np.zeros((64, 64, 3), dtype=np.uint8),
        video_metadata=_video_metadata(frame_number),
    )


def test_tensor_byte_tracker_never_calls_numpy_and_preserves_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep tracker numeric input tensor-native and recover tracker output order."""
    detections = _detections()
    tracker = _ReorderingTracker()

    def reject_numpy(_tensor: torch.Tensor) -> Any:
        """Fail if the deprecated host-array boundary is reintroduced."""
        raise AssertionError("ByteTrack input must not call Tensor.numpy()")

    monkeypatch.setattr(torch.Tensor, "numpy", reject_numpy)

    kept_indices, tracker_ids = update_tensor_byte_tracker(tracker, detections)  # type: ignore[arg-type]
    tracked, partitions = recover_tensor_byte_tracker_output(
        detections=detections,
        kept_indices=kept_indices,
        tracker_ids=tracker_ids,
    )

    assert tracker.received is not None
    assert tracker.received.xyxy.device == detections.xyxy.device
    assert tracker.received.class_id.device == detections.class_id.device
    assert tracker.received.confidence.device == detections.confidence.device
    assert torch.equal(tracked.xyxy, detections.xyxy.flip(0))
    assert tracked.tracker_id is not None
    assert tracked.tracker_id.tolist() == [101, 102]
    assert tracked.bboxes_metadata == [
        {"name": "second", "tracker_id": 101},
        {"name": "first", "tracker_id": 102},
    ]
    assert detections.bboxes_metadata == [{"name": "first"}, {"name": "second"}]
    assert partitions is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_tensor_byte_tracker_keeps_cuda_input_and_output_on_device() -> None:
    """Keep boxes, row indices, IDs, and recovered numeric fields on CUDA."""
    detections = _detections("cuda")
    tracker = _ReorderingTracker()

    kept_indices, tracker_ids = update_tensor_byte_tracker(tracker, detections)  # type: ignore[arg-type]
    tracked, _ = recover_tensor_byte_tracker_output(
        detections=detections,
        kept_indices=kept_indices,
        tracker_ids=tracker_ids,
    )

    assert tracker.received is not None
    assert tracker.received.xyxy.is_cuda
    assert kept_indices.is_cuda
    assert tracker_ids.is_cuda
    assert tracked.xyxy.is_cuda
    assert tracked.tracker_id is not None and tracked.tracker_id.is_cuda


@pytest.mark.parametrize(
    "block_type",
    [
        pytest.param(ByteTrackerBlockV1, id="v1"),
        pytest.param(ByteTrackerBlockV2, id="v2"),
    ],
)
def test_legacy_tensor_blocks_preserve_ids_order_and_metadata(
    block_type: type[ByteTrackerBlockV1] | type[ByteTrackerBlockV2],
) -> None:
    """Preserve legacy V1/V2 row and metadata contracts with tensor tracker IDs."""
    detections = _detections()
    block = block_type()

    if isinstance(block, ByteTrackerBlockV1):
        result = block.run(metadata=_video_metadata(), detections=detections)
    else:
        result = block.run(image=_workflow_image(), detections=detections)
    tracked = result["tracked_detections"]

    assert torch.equal(tracked.xyxy, detections.xyxy)
    assert tracked.tracker_id is not None
    assert tracked.tracker_id.tolist() == [1, 2]
    assert [item["tracker_id"] for item in tracked.bboxes_metadata] == [1, 2]
    assert [item["name"] for item in tracked.bboxes_metadata] == ["first", "second"]


def test_legacy_tensor_v3_preserves_cache_and_metadata_contract() -> None:
    """Classify first and repeated V3 IDs on-device without changing public metadata."""
    block = ByteTrackerBlockV3()
    first_detections = _detections()
    second_detections = _detections()

    first = block.run(image=_workflow_image(0), detections=first_detections)
    second = block.run(image=_workflow_image(1), detections=second_detections)

    first_tracked = first["tracked_detections"]
    second_tracked = second["tracked_detections"]
    assert first_tracked.tracker_id is not None
    assert second_tracked.tracker_id is not None
    assert first_tracked.tracker_id.tolist() == second_tracked.tracker_id.tolist()
    assert first["new_instances"].tracker_id.tolist() == [1, 2]
    assert len(first["already_seen_instances"]) == 0
    assert len(second["new_instances"]) == 0
    assert second["already_seen_instances"].tracker_id.tolist() == [1, 2]
    assert [item["tracker_id"] for item in second_tracked.bboxes_metadata] == [1, 2]


def test_legacy_tensor_v3_cache_preserves_fifo_eviction() -> None:
    """Match the legacy cache's exact first-seen and FIFO eviction behavior."""
    cache = InstanceCache(size=2)

    first_seen = cache.record_instances(torch.tensor([1, 2]))
    third_seen = cache.record_instances(torch.tensor([3]))
    second_seen = cache.record_instances(torch.tensor([2]))
    evicted_seen = cache.record_instances(torch.tensor([1]))

    assert first_seen.tolist() == [False, False]
    assert third_seen.tolist() == [False]
    assert second_seen.tolist() == [True]
    assert evicted_seen.tolist() == [False]
