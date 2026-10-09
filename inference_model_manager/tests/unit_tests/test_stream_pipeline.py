from __future__ import annotations

import threading
import time
from concurrent.futures import Future
from typing import Any, List, Optional

import numpy as np
import pytest
import torch

from inference_models.models.base.async_handoff import (
    STREAM_PIPELINE_CONTEXT_ID_KWARG,
    attach_deferred_postprocess_handoff,
    get_async_response_context_id,
    get_async_response_future,
)
from inference_models.models.base.instance_segmentation import (
    InstanceDetections,
    InstanceSegmentationModel,
)
from inference_models.models.base.types import InstancesRLEMasks

from inference_model_manager import configuration as cfg
from inference_model_manager.stream_pipeline import (
    STREAM_PIPELINE_PRODUCER_ID_KWARG,
    StreamPipelinedModel,
    stream_pipeline_for,
)


def detections(frame: int) -> InstanceDetections:
    return InstanceDetections(
        xyxy=torch.tensor([[frame, frame, frame + 2, frame + 2]], dtype=torch.int32),
        class_id=torch.tensor([0], dtype=torch.int32),
        confidence=torch.tensor([0.9], dtype=torch.float32),
        mask=InstancesRLEMasks(image_size=(4, 6), masks=[b"0"]),
    )


class FakeFuture:
    def __init__(
        self, frame: int, release: threading.Event, metadata_available: bool = True
    ):
        self.frame = frame
        self.release = release
        self.preprocess_metadata = [{"frame": frame}] if metadata_available else None
        self.gpu_work_submissions = 0

    def submit_gpu_work(self, meta: Any = None) -> None:
        self.gpu_work_submissions += 1

    def done(self) -> bool:
        return self.release.is_set()

    def result(self) -> List[InstanceDetections]:
        assert self.release.wait(timeout=5)
        return [detections(self.frame)]


class FakeSegmentationModel(InstanceSegmentationModel):
    def __init__(self, supports_stream_pipeline: bool = True):
        self._supports = supports_stream_pipeline
        self.sync_calls: list = []
        self.async_calls: list = []
        self.futures: list = []
        self.events: list = []
        self.metadata_available = True
        self.block_async: Optional[threading.Event] = None
        self.async_started = threading.Event()

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs):
        return cls()

    @property
    def class_names(self) -> List[str]:
        return ["cat"]

    @property
    def supported_mask_formats(self):
        return {"dense", "rle"}

    @property
    def supports_stream_pipeline(self) -> bool:
        return self._supports

    def infer(self, images, **kwargs) -> List[InstanceDetections]:
        self.sync_calls.append((images, kwargs))
        self.events.append(("sync", [f.gpu_work_submissions for f in self.futures]))
        batch = images if isinstance(images, list) else [images]
        return [detections(len(self.sync_calls)) for _ in batch]

    def infer_async(self, images, **kwargs):
        self.async_calls.append((images, kwargs))
        self.events.append(("async-start", len(self.async_calls)))
        self.async_started.set()
        if self.block_async is not None:
            assert self.block_async.wait(timeout=5)
        future = FakeFuture(
            len(self.async_calls), threading.Event(), self.metadata_available
        )
        self.futures.append(future)
        self.events.append(("async-return", len(self.async_calls)))
        return future

    def pre_process(self, images, **kwargs):
        raise AssertionError("not used")

    def forward(self, pre_processed_images, **kwargs):
        raise AssertionError("not used")

    def post_process(self, model_results, pre_processing_meta, **kwargs):
        raise AssertionError("not used")


def image() -> np.ndarray:
    return np.zeros((4, 6, 3), dtype=np.uint8)


def ctx(context_id: str, producer_id: str = "producer-a") -> dict:
    return {
        STREAM_PIPELINE_CONTEXT_ID_KWARG: context_id,
        STREAM_PIPELINE_PRODUCER_ID_KWARG: producer_id,
    }


def queues_of(pipeline: StreamPipelinedModel) -> tuple:
    return (
        list(pipeline._pending_gpu_submissions),
        list(pipeline._pending_futures),
        list(pipeline._response_futures),
    )


def worker_threads_alive(executor) -> bool:
    for thread in list(executor._threads):
        thread.join(timeout=5)
    return any(thread.is_alive() for thread in executor._threads)


@pytest.fixture
def depth_two(monkeypatch):
    monkeypatch.setenv("RFDETR_PIPELINE_DEPTH", "2")


@pytest.fixture
def pipeline(depth_two):
    model = FakeSegmentationModel()
    pipelined = stream_pipeline_for(model)
    yield pipelined
    for future in model.futures:
        future.release.set()
    pipelined.shutdown_pipeline()


def test_depth_one_is_not_pipelined(monkeypatch):
    monkeypatch.setenv("RFDETR_PIPELINE_DEPTH", "1")

    assert stream_pipeline_for(FakeSegmentationModel()) is None


def test_model_without_stream_pipeline_support_is_not_pipelined(depth_two):
    assert (
        stream_pipeline_for(FakeSegmentationModel(supports_stream_pipeline=False))
        is None
    )


def test_non_segmentation_model_is_not_pipelined(depth_two):
    assert stream_pipeline_for(object()) is None


def test_depth_is_capped_at_the_maximum(monkeypatch):
    monkeypatch.setenv("RFDETR_PIPELINE_DEPTH", "7")

    pipelined = stream_pipeline_for(FakeSegmentationModel())

    assert isinstance(pipelined, StreamPipelinedModel)
    assert pipelined.pipeline_depth == 2
    assert pipelined.supports_stream_pipeline is True


def test_depth_two_returns_the_previous_frame_delayed_by_one(pipeline):
    model = pipeline.model

    first = pipeline.infer(image(), **ctx("ctx-1"))
    second = pipeline.infer(image(), **ctx("ctx-2"))

    assert len(first) == 1 and len(first[0]) == 0
    assert get_async_response_future(first[0]) is None
    assert get_async_response_context_id(first[0]) == "ctx-1"
    assert len(second) == 1 and len(second[0]) == 0
    future = get_async_response_future(second[0])
    assert isinstance(future, Future)
    assert get_async_response_context_id(second[0]) == "ctx-1"
    assert not future.done()
    model.futures[0].release.set()
    result = future.result(timeout=5)
    assert [int(r.xyxy[0][0]) for r in result] == [1]
    assert isinstance(result[0].xyxy, np.ndarray)
    assert model.sync_calls == []
    assert len(model.async_calls) == 2


def test_flush_returns_the_tail_frame_with_its_context_id(pipeline):
    model = pipeline.model
    pipeline.infer(image(), **ctx("ctx-1"))
    pipeline.infer(image(), **ctx("ctx-2"))
    for future in model.futures:
        future.release.set()

    flushed = pipeline.flush()

    assert [int(r.xyxy[0][0]) for r in flushed] == [2]
    assert get_async_response_context_id(flushed[0]) == "ctx-2"
    assert get_async_response_future(flushed[0]) is None
    assert pipeline.flush() == []


def test_flush_resolves_every_pending_frame_in_order_and_empties_the_queues(
    pipeline,
):
    model = pipeline.model
    pipeline.infer(image(), **ctx("ctx-1"))
    for future in model.futures:
        future.release.set()

    flushed = pipeline.flush()

    assert [get_async_response_context_id(r) for r in flushed] == ["ctx-1"]
    assert queues_of(pipeline) == ([], [], [])
    assert [f.gpu_work_submissions for f in model.futures] == [1]


def test_flush_on_an_idle_pipeline_returns_nothing(pipeline):
    assert pipeline.flush() == []


def test_batch_requests_bypass_the_pipeline_under_the_lock(pipeline):
    model = pipeline.model
    lock_held = []
    original_infer = model.infer

    def infer(images, **kwargs):
        lock_held.append(pipeline._lock.locked())
        return original_infer(images, **kwargs)

    model.infer = infer

    result = pipeline.infer([image(), image()], confidence=0.5, **ctx("ctx-1"))

    assert len(result) == 2
    assert model.async_calls == []
    assert model.sync_calls[0][1] == {"confidence": 0.5}
    assert lock_held == [True]
    assert queues_of(pipeline) == ([], [], [])


def test_call_without_context_id_is_synchronous(pipeline):
    model = pipeline.model

    result = pipeline.infer(image(), confidence=0.5)

    assert len(result) == 1 and len(result[0]) == 1
    assert get_async_response_future(result[0]) is None
    assert get_async_response_context_id(result[0]) is None
    assert model.async_calls == []
    assert model.sync_calls[0][1] == {"confidence": 0.5}
    assert queues_of(pipeline) == ([], [], [])


def test_synchronous_call_submits_the_pending_gpu_work_first(pipeline):
    model = pipeline.model
    model.metadata_available = False
    pipeline.infer(image(), **ctx("ctx-1"))
    assert model.futures[0].gpu_work_submissions == 0

    pipeline.infer(image())

    assert model.events[-1] == ("sync", [1])


def test_pipeline_id_kwargs_are_not_forwarded_to_the_model(pipeline):
    model = pipeline.model

    pipeline.infer(image(), confidence=0.5, **ctx("c"))
    pipeline.infer([image(), image()], **ctx("c"))

    async_kwargs = model.async_calls[0][1]
    assert STREAM_PIPELINE_CONTEXT_ID_KWARG not in async_kwargs
    assert STREAM_PIPELINE_PRODUCER_ID_KWARG not in async_kwargs
    assert async_kwargs["confidence"] == 0.5
    assert async_kwargs["defer_postprocess_sync"] is True
    assert async_kwargs["reuse_trt_graph_outputs"] is True
    assert STREAM_PIPELINE_CONTEXT_ID_KWARG not in model.sync_calls[0][1]
    assert STREAM_PIPELINE_PRODUCER_ID_KWARG not in model.sync_calls[0][1]


def test_call_with_a_context_id_but_no_producer_id_is_synchronous(pipeline):
    model = pipeline.model

    result = pipeline.infer(image(), **{STREAM_PIPELINE_CONTEXT_ID_KWARG: "c"})

    assert len(result) == 1 and len(result[0]) == 1
    assert get_async_response_future(result[0]) is None
    assert get_async_response_context_id(result[0]) is None
    assert model.async_calls == []
    assert STREAM_PIPELINE_CONTEXT_ID_KWARG not in model.sync_calls[0][1]
    assert queues_of(pipeline) == ([], [], [])


def test_another_producer_runs_synchronously_while_the_pipeline_is_owned(pipeline):
    model = pipeline.model
    pipeline.infer(image(), **ctx("a-1", "producer-a"))
    second = pipeline.infer(image(), **ctx("a-2", "producer-a"))
    _, pending_before, responses_before = queues_of(pipeline)

    other = pipeline.infer(image(), **ctx("b-1", "producer-b"))

    assert len(other) == 1 and len(other[0]) == 1
    assert int(other[0].xyxy[0][0]) == 1
    assert get_async_response_future(other[0]) is None
    assert get_async_response_context_id(other[0]) is None
    assert len(model.sync_calls) == 1 and len(model.async_calls) == 2
    assert queues_of(pipeline) == ([], pending_before, responses_before)
    for future in model.futures:
        future.release.set()
    delayed = get_async_response_future(second[0]).result(timeout=5)
    assert [int(r.xyxy[0][0]) for r in delayed] == [1]
    flushed = pipeline.flush()
    assert [get_async_response_context_id(r) for r in flushed] == ["a-2"]
    assert [int(r.xyxy[0][0]) for r in flushed] == [2]

    opened = pipeline.infer(image(), **ctx("b-2", "producer-b"))

    assert len(opened[0]) == 0
    assert get_async_response_context_id(opened[0]) == "b-2"
    assert len(model.async_calls) == 3
    followed = pipeline.infer(image(), **ctx("a-3", "producer-a"))
    assert get_async_response_future(followed[0]) is None
    assert get_async_response_context_id(followed[0]) is None
    assert len(model.sync_calls) == 2


def test_gpu_work_is_submitted_once_per_frame(pipeline):
    model = pipeline.model

    pipeline.infer(image(), **ctx("a"))
    pipeline.infer(image(), **ctx("b"))
    pipeline.infer(image(), **ctx("c"))

    assert [f.gpu_work_submissions for f in model.futures] == [1, 1, 1]


def test_deferred_postprocess_is_finalized_on_the_response_thread(pipeline):
    model = pipeline.model
    finalized = []

    def finalize():
        finalized.append(threading.current_thread().name)
        return detections(9)

    placeholder = detections(0)
    attach_deferred_postprocess_handoff(
        placeholder,
        done_event=None,
        trt_outputs_consumed_event=None,
        finalize=finalize,
    )
    pipeline.infer(image(), **ctx("a"))
    model.futures[0].result = lambda: [placeholder]
    second = pipeline.infer(image(), **ctx("b"))

    result = get_async_response_future(second[0]).result(timeout=5)

    assert int(result[0].xyxy[0][0]) == 9
    assert finalized and finalized[0] != threading.current_thread().name


def test_shutdown_without_flush_resolves_the_pending_frames_and_stops_the_worker(
    pipeline,
):
    model = pipeline.model
    pipeline.infer(image(), **ctx("a"))
    second = pipeline.infer(image(), **ctx("b"))
    handed_out = get_async_response_future(second[0])
    executor = pipeline._response_executor
    for future in model.futures:
        future.release.set()

    pipeline.shutdown_pipeline()

    assert handed_out.done()
    assert [int(r.xyxy[0][0]) for r in handed_out.result(timeout=0)] == [1]
    assert [f.gpu_work_submissions for f in model.futures] == [1, 1]
    assert queues_of(pipeline) == ([], [], [])
    assert executor._shutdown is True
    assert worker_threads_alive(executor) is False
    assert pipeline._response_executor is None


def test_shutdown_waits_a_bounded_time_for_a_stuck_frame(pipeline, monkeypatch):
    monkeypatch.setattr(cfg, "INFERENCE_PROCESS_TIMEOUT_S", 0.2)
    model = pipeline.model
    pipeline.infer(image(), **ctx("a"))
    second = pipeline.infer(image(), **ctx("b"))
    handed_out = get_async_response_future(second[0])
    started = time.monotonic()

    pipeline.shutdown_pipeline()

    assert time.monotonic() - started < 4
    assert not handed_out.done()
    assert queues_of(pipeline) == ([], [], [])
    assert pipeline._response_executor is None
    third = pipeline.infer(image(), **ctx("c"))
    fourth = pipeline.infer(image(), **ctx("d"))
    assert get_async_response_context_id(third[0]) == "c"
    assert get_async_response_future(third[0]) is None
    assert get_async_response_context_id(fourth[0]) == "c"
    assert pipeline._response_executor is not None
    assert model.sync_calls == []
    for future in model.futures:
        future.release.set()
    assert [int(r.xyxy[0][0]) for r in handed_out.result(timeout=5)] == [1]
    reopened = get_async_response_future(fourth[0]).result(timeout=5)
    assert [int(r.xyxy[0][0]) for r in reopened] == [3]


def test_infer_after_shutdown_is_pipelined_again_from_an_empty_pipeline(pipeline):
    model = pipeline.model
    pipeline.shutdown_pipeline()

    first = pipeline.infer(image(), **ctx("a"))
    second = pipeline.infer(image(), **ctx("b"))

    assert len(first) == 1 and len(first[0]) == 0
    assert get_async_response_future(first[0]) is None
    assert get_async_response_context_id(first[0]) == "a"
    assert get_async_response_context_id(second[0]) == "a"
    assert isinstance(get_async_response_future(second[0]), Future)
    assert len(model.async_calls) == 2 and model.sync_calls == []
    for future in model.futures:
        future.release.set()
    assert [get_async_response_context_id(r) for r in pipeline.flush()] == ["b"]


def test_double_shutdown_is_a_noop(pipeline):
    pipeline.infer(image(), **ctx("a"))
    pipeline.model.futures[0].release.set()
    pipeline.shutdown_pipeline()

    pipeline.shutdown_pipeline()

    assert pipeline._response_executor is None
    assert queues_of(pipeline) == ([], [], [])


def test_restart_after_shutdown_starts_from_an_empty_pipeline(pipeline):
    model = pipeline.model
    pipeline.infer(image(), **ctx("a"))
    model.futures[0].release.set()

    pipeline.shutdown_pipeline()
    second = pipeline.infer(image(), **ctx("b", "producer-b"))

    assert len(second) == 1 and len(second[0]) == 0
    assert get_async_response_future(second[0]) is None
    assert get_async_response_context_id(second[0]) == "b"
    assert len(model.async_calls) == 2 and model.sync_calls == []
    model.futures[1].release.set()
    flushed = pipeline.flush()
    assert [get_async_response_context_id(r) for r in flushed] == ["b"]
    assert [int(r.xyxy[0][0]) for r in flushed] == [2]


def test_shutdown_releases_the_response_executor(pipeline):
    pipeline.infer(image(), **ctx("a"))
    pipeline.infer(image(), **ctx("b"))
    executor = pipeline._response_executor
    for future in pipeline.model.futures:
        future.release.set()

    pipeline.shutdown_pipeline()

    assert executor._shutdown is True
    assert pipeline._response_executor is None


def test_second_producer_waits_for_the_first_handoff_then_runs_synchronously(
    pipeline,
):
    model = pipeline.model
    model.block_async = threading.Event()
    first_done = threading.Event()
    second_done = threading.Event()
    second_result = []

    def first_producer():
        pipeline.infer(image(), **ctx("a"))
        first_done.set()

    def second_producer():
        second_result.extend(pipeline.infer(image()))
        second_done.set()

    first = threading.Thread(target=first_producer)
    second = threading.Thread(target=second_producer)
    first.start()
    assert model.async_started.wait(timeout=5)
    second.start()
    try:
        assert not second_done.wait(timeout=0.3)
        assert model.sync_calls == []
    finally:
        model.block_async.set()
        first.join(timeout=5)
        second.join(timeout=5)

    assert first_done.is_set() and second_done.is_set()
    assert model.events == [
        ("async-start", 1),
        ("async-return", 1),
        ("sync", [1]),
    ]
    assert len(second_result) == 1 and len(second_result[0]) == 1
    assert get_async_response_future(second_result[0]) is None
    assert get_async_response_context_id(second_result[0]) is None
    model.futures[0].release.set()
    flushed = pipeline.flush()
    assert [get_async_response_context_id(r) for r in flushed] == ["a"]
