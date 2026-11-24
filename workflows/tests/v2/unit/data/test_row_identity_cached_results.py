"""Row identity on cached dense results shared between callers.

``StreamOrderedOnce`` publishes one dense result, for example a view's
``full_res()``. Its row-record ids are row tensors like the boxes: they are
validated once at publication and join every caller's stream ordering, so a
caller stream keeps their allocation alive after the producer lets go.

The CUDA cases are skipped without a device. A skip is not a pass; run this
file on a CUDA host to prove the stream lifetime behavior.
"""

import gc

import pytest
import torch
from roboflow_workflows.v2.data.errors import ContractError
from roboflow_workflows.v2.data.masks.grid import MaskGridGeometry, SegmentationView
from roboflow_workflows.v2.data.masks.stream_order import StreamOrderedOnce
from roboflow_workflows.v2.data.row_identity import (
    ROW_RECORD_KEY,
    row_record,
    with_row_identity,
)

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections

needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason=(
        "needs a CUDA device; proof request: run this file on a CUDA host for "
        "the cached full_res ids readiness and lifetime checks"
    ),
)

# About 0.1-1 s of GPU time, so a caller stream's reads stay pending while
# the host releases the result and reuses memory on the producer stream.
_SLEEP_CYCLES = 500_000_000


def _detections(rows=4, *, device="cpu"):
    return Detections(
        xyxy=torch.tensor([[0.0, 0.0, 2.0, 2.0]], device=device).repeat(rows, 1),
        class_id=torch.arange(rows, device=device),
        confidence=torch.full((rows,), 0.5, device=device),
        image_metadata={"parent_id": "frame"},
    )


def _view(detections):
    rows = int(detections.xyxy.shape[0])
    view = SegmentationView(
        detections=detections,
        scores=torch.randn(rows, 4, 4, device=detections.xyxy.device),
        score_type="logits",
        threshold=0.0,
        geometry=MaskGridGeometry.identity((4, 4)),
    )

    return view


def _dense(detections):
    rows = int(detections.xyxy.shape[0])
    predictions = InstanceDetections(
        xyxy=detections.xyxy,
        class_id=detections.class_id,
        confidence=detections.confidence,
        mask=torch.zeros(rows, 4, 4, dtype=torch.bool),
        image_metadata=detections.image_metadata,
    )

    return predictions


def test_cached_full_res_keeps_selected_ids_for_every_caller():
    picked = _view(with_row_identity(_detections())).select(torch.tensor([2, 2, 0]))

    first = picked.full_res()
    second = picked.full_res()

    assert second is first
    assert picked.materialization_count == 1
    assert row_record(second).host_ids() == [2, 2, 0]


def test_stale_record_publishes_nothing_and_a_later_call_retries():
    stale = with_row_identity(_detections())
    stale.image_metadata[ROW_RECORD_KEY]["rows"] = 5
    cache = StreamOrderedOnce()

    with pytest.raises(ContractError, match="describes 5 rows"):
        cache.get(lambda: _dense(stale))
    assert not cache.is_set

    valid = _dense(with_row_identity(_detections()))
    assert cache.get(lambda: valid) is valid
    assert cache.is_set


def test_ids_on_another_device_than_the_rows_are_rejected():
    # Meta tensors have no data, so validation reads shapes and devices only.
    detections = with_row_identity(_detections(device=torch.device("meta")))
    detections.image_metadata[ROW_RECORD_KEY]["ids"] = torch.arange(4)
    predictions = InstanceDetections(
        xyxy=detections.xyxy,
        class_id=detections.class_id,
        confidence=detections.confidence,
        mask=torch.zeros(4, 4, 4, dtype=torch.bool, device="meta"),
        image_metadata=detections.image_metadata,
    )
    cache = StreamOrderedOnce()

    with pytest.raises(ContractError, match="ids"):
        cache.get(lambda: predictions)
    assert not cache.is_set


@needs_cuda
def test_concurrent_cuda_consumer_reads_cached_ids_without_sync():
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(producer):
        # The ids are produced late on the producer stream; a consumer that
        # did not wait for the publication event would read them too early.
        torch.cuda._sleep(_SLEEP_CYCLES)
        source = _view(with_row_identity(_detections(device="cuda")))
        picked = source.select(torch.tensor([3, 1, 1], device="cuda"))
        published = picked.full_res()

    torch.cuda.set_sync_debug_mode("error")
    try:
        with torch.cuda.stream(consumer):
            shared = picked.full_res()
            ids = shared.image_metadata[ROW_RECORD_KEY]["ids"]
            read = ids.clone()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()

    assert shared is published
    assert picked.materialization_count == 1
    assert read.tolist() == [3, 1, 1]


@needs_cuda
def test_cached_cuda_ids_outlive_the_producer_on_a_caller_stream():
    consumer = torch.cuda.Stream()
    picked = _view(with_row_identity(_detections(device="cuda"))).select(
        torch.tensor([3, 1, 1], device="cuda")
    )
    picked.full_res()
    torch.cuda.synchronize()

    with torch.cuda.stream(consumer):
        # Only full_res() orders this stream; the view's inputs are not read
        # here, so the cached result alone must record the ids on it.
        shared = picked.full_res()
        ids = shared.image_metadata[ROW_RECORD_KEY]["ids"]
        torch.cuda._sleep(_SLEEP_CYCLES)
        read = ids.clone()
    size = ids.numel()
    del picked, shared, ids
    gc.collect()
    # Freed small blocks return to the producer stream's pool. Without
    # record_stream on the consumer, one of these overwrites the ids before
    # the pending clone reads them.
    overwrites = [
        torch.full((size,), -1, dtype=torch.int64, device="cuda") for _ in range(64)
    ]
    torch.cuda.synchronize()

    assert overwrites
    assert read.tolist() == [3, 1, 1]
