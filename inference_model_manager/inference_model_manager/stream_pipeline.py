"""Stream pipelining of instance-segmentation models with deferred GPU handoff.

`StreamPipelinedModel` overlaps consecutive frames of one video stream: the
call for frame N submits its GPU work and returns an empty placeholder carrying
the response future and context id of frame N-1, so the producer keeps feeding
frames while the GPU finishes the previous one.

Contract.

The wrapper owns the model's execution for pipelined and non-pipelined calls
alike. Every call to the underlying model (`infer`, `infer_async`, flush work)
runs under one wrapper lock. The lock is held from before the model call until
the GPU handoff of that call has been submitted: after the response scheduling
of a pipelined frame, or the return of `model.infer` otherwise. A second caller
waits. It never replays the graph while a pending future still references the
shared output buffers, because the pending GPU work is submitted first.

A call is pipelined only when it carries a string `stream_pipeline_context_id`
and a string `stream_pipeline_producer_id`, the depth is above one and the
batch is one. Every other call (HTTP requests, batches, warmup) is synchronous
and never touches the queues. Neither id is forwarded to the model.

The pipeline is owned by one producer at a time. The first pipelined call on
an empty pipeline binds the wrapper to its producer; the binding is released
when the queues are empty again (after `flush()` or `shutdown_pipeline()`, or
once the last pending frame has been handed out). A pipelined call from a
different producer while the pipeline is bound runs synchronously under the
lock, exactly like a call without ids.

`flush()` resolves every pending frame in order and leaves the queues empty.

`shutdown_pipeline()` drains. Under the lock it submits the GPU work and the
response build of every pending frame, so every handed-out response future
resolves and no caller blocks forever. It clears the three queues, shuts the
response executor down with a bounded wait and releases it. A handed-out
future whose finalisation outlives that bound stays pending until the worker
finishes it. The wrapper is not closed: the next pipelined call recreates the
executor lazily and starts from an empty pipeline, so its first placeholder is
bare. A second `shutdown_pipeline()` is a no-op. The wrapper is gone only when
the manager drops it (unload, eviction or shutdown).

A pipelined call that the wrapper answers synchronously (a non-owner producer,
or any other reason the result is not a placeholder) is reported to the block
through its own completion protocol: the provider attaches an already-completed
future resolving to the repacked synchronous responses, together with the
current context id, so the block's queued context is consumed. The provider
never leaves a block context without a future on the pipelined path.

A bare first placeholder (no future) from the wrapper is the authoritative
signal of a fresh pipeline: when the provider receives one, every older frame
it still holds for that registry id belongs to a dead pipeline and is dropped
before the current frame is recorded. The cached route generation check stays
as a secondary guard.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from typing import Any, Deque, List, Optional, Tuple
from weakref import finalize

import torch

from inference_model_manager import configuration as cfg
from inference_model_manager.marshalling import MODEL_CLOCK_KWARG, tensors_to_numpy
from inference_models.configuration import (
    MAX_RFDETR_PIPELINE_DEPTH,
    get_rfdetr_pipeline_depth,
)
from inference_models.models.base.async_handoff import (
    STREAM_PIPELINE_CONTEXT_ID_KWARG,
    adapter_gpu_work_submitted,
    attach_adapter_mapped_kwargs,
    attach_async_response_future,
    get_adapter_gpu_submit_generation,
    get_adapter_stream_pipeline_context_id,
    get_deferred_postprocess_finalizer,
    mark_adapter_gpu_work_submitted,
)
from inference_models.models.base.instance_segmentation import (
    InferenceFuture,
    InstanceDetections,
    InstanceSegmentationModel,
)
from inference_models.models.base.types import InstancesRLEMasks

STREAM_PIPELINE_PRODUCER_ID_KWARG = "stream_pipeline_producer_id"

_PendingFrame = Tuple[InferenceFuture, Any, dict]


def stream_pipeline_for(model: Any) -> Optional["StreamPipelinedModel"]:
    """Wrap a model in a stream pipeline when it qualifies for one.

    Args:
        model: The loaded model object.

    Returns:
        The pipelined wrapper for an instance-segmentation model that supports
        the deferred GPU handoff while the configured depth is above one,
        otherwise None.
    """
    if not isinstance(model, InstanceSegmentationModel):
        return None
    if not _supports_stream_pipeline(model):
        return None
    depth = min(get_rfdetr_pipeline_depth(), MAX_RFDETR_PIPELINE_DEPTH)
    if depth <= 1:
        return None

    pipelined = StreamPipelinedModel(model, depth=depth)

    return pipelined


def _supports_stream_pipeline(model: Any) -> bool:
    supports = getattr(model, "supports_stream_pipeline", False)
    if callable(supports):
        return bool(supports())
    return bool(supports)


class StreamPipelinedModel:
    """Depth-pipelined view of an instance-segmentation model.

    Args:
        model: Model whose `infer_async` returns a future with the deferred
            GPU handoff contract.
        depth: Number of in-flight stages; responses are delayed by depth - 1
            frames.
    """

    def __init__(self, model: InstanceSegmentationModel, *, depth: int) -> None:
        self.model = model
        self._pipeline_depth = depth
        self._response_delay = max(1, depth - 1)
        self._lock = threading.Lock()
        self._producer_id: Optional[str] = None
        self._pending_gpu_submissions: Deque[_PendingFrame] = deque()
        self._pending_futures: Deque[_PendingFrame] = deque()
        self._gpu_submit_generation = 0
        self._response_executor: Optional[ThreadPoolExecutor] = None
        self._response_executor_finalizer: Optional[finalize] = None
        self._response_futures: Deque[Tuple[Future, Optional[str]]] = deque()

    @property
    def pipeline_depth(self) -> int:
        return self._pipeline_depth

    @property
    def supports_stream_pipeline(self) -> bool:
        return self._pipeline_depth > 1

    def infer(self, images: Any, **kwargs: Any) -> List[InstanceDetections]:
        """Run one frame through the pipeline, or the model synchronously.

        Args:
            images: One image, or a batch; batches bypass the pipeline.
            **kwargs: Inference kwargs; `stream_pipeline_context_id` names the
                frame and `stream_pipeline_producer_id` its producer. Neither
                is forwarded to the model. `model_clock`, a one-element list,
                accumulates the seconds a synchronous model call took, lock
                wait excluded; it is not forwarded either.

        Returns:
            For a pipelined frame, one empty placeholder detection carrying the
            response future and context id of the oldest finished frame, or
            only the current frame's context id while no earlier frame has
            finished. Otherwise the model's synchronous result.
        """
        model_clock = kwargs.pop(MODEL_CLOCK_KWARG, None)
        context_id = kwargs.pop(STREAM_PIPELINE_CONTEXT_ID_KWARG, None)
        producer_id = kwargs.pop(STREAM_PIPELINE_PRODUCER_ID_KWARG, None)
        with self._lock:
            if self._is_pipelined(images, context_id, producer_id):
                result = self._infer_pipelined(images, kwargs, context_id, producer_id)
            else:
                self._submit_all_pending_gpu_work()
                model_started = time.perf_counter()
                try:
                    result = self.model.infer(images, **kwargs)
                finally:
                    if model_clock is not None:
                        model_clock[0] += time.perf_counter() - model_started

        return result

    def _is_pipelined(self, images: Any, context_id: Any, producer_id: Any) -> bool:
        if not isinstance(context_id, str) or not isinstance(producer_id, str):
            return False
        if self._pipeline_depth <= 1:
            return False
        if self._producer_id is not None and self._producer_id != producer_id:
            return False
        return _request_batch_size(images) == 1

    def _infer_pipelined(
        self, images: Any, kwargs: dict, context_id: str, producer_id: str
    ) -> List[InstanceDetections]:
        self._producer_id = producer_id
        mapped_kwargs = dict(kwargs)
        mapped_kwargs["defer_postprocess_sync"] = True
        mapped_kwargs["reuse_trt_graph_outputs"] = True
        self._submit_next_pending_gpu_work()
        future = self.model.infer_async(images, **mapped_kwargs)
        attach_adapter_mapped_kwargs(
            future, mapped_kwargs, stream_pipeline_context_id=context_id
        )
        meta = getattr(future, "preprocess_metadata", None)
        if meta is not None:
            self._submit_future_gpu_work(future, meta, mapped_kwargs)
        self._submit_ready_responses()

        self._pending_gpu_submissions.append((future, meta, mapped_kwargs))
        self._pending_futures.append((future, meta, mapped_kwargs))
        if len(self._pending_futures) > self._response_delay:
            self._submit_next_pending_gpu_work()
            self._submit_ready_responses()

        placeholder = _placeholder(images)
        if not self._response_futures:
            attach_async_response_future(placeholder, None, context_id)
            return [placeholder]
        response_future, response_context_id = self._response_futures.popleft()
        attach_async_response_future(placeholder, response_future, response_context_id)
        self._release_producer_if_idle()

        return [placeholder]

    def _release_producer_if_idle(self) -> None:
        if (
            not self._pending_gpu_submissions
            and not self._pending_futures
            and not self._response_futures
        ):
            self._producer_id = None

        return None

    def flush(self) -> List[InstanceDetections]:
        """Drain the in-flight frames.

        Callers running a depth above one must call this at stream end or the
        tail frames are dropped. Each returned detection carries the context id
        of its frame.

        Returns:
            The finished detections of every in-flight frame, oldest first.
        """
        with self._lock:
            self._submit_all_pending_gpu_work()
            self._submit_all_pending_responses()
            pending = list(self._response_futures)
            self._response_futures.clear()
            self._producer_id = None

        responses: List[InstanceDetections] = []
        for response_future, context_id in pending:
            for detection in _resolve_response_future(
                response_future, context="RF-DETR stream pipeline flush"
            ):
                attach_async_response_future(detection, None, context_id)
                responses.append(detection)

        return responses

    def shutdown_pipeline(self) -> None:
        """Drain the pending frames and release the response worker.

        Every handed-out response future is finished by the worker; one whose
        finalisation outlives the bounded wait stays pending until the worker
        completes it. The wrapper stays usable: the next pipelined call
        recreates the worker and starts from an empty pipeline.
        """
        with self._lock:
            self._submit_all_pending_gpu_work()
            self._submit_all_pending_responses()
            self._response_futures.clear()
            self._producer_id = None
            executor = self._response_executor
            finalizer = self._response_executor_finalizer
            self._response_executor = None
            self._response_executor_finalizer = None
            if executor is None:
                return None
            if finalizer is not None and finalizer.alive:
                finalizer.detach()
            drained = _wait_for_queued_work(executor, cfg.INFERENCE_PROCESS_TIMEOUT_S)
            executor.shutdown(wait=drained)

        return None

    def _get_response_executor(self) -> ThreadPoolExecutor:
        if self._response_executor is None:
            executor = ThreadPoolExecutor(max_workers=1)
            self._response_executor = executor
            self._response_executor_finalizer = finalize(
                self, executor.shutdown, wait=False
            )
        return self._response_executor

    def _submit_future_gpu_work(
        self, future: InferenceFuture, meta: Any, mapped_kwargs: dict
    ) -> None:
        if adapter_gpu_work_submitted(future):
            return None
        submit_gpu_work = getattr(future, "submit_gpu_work", None)
        if not callable(submit_gpu_work):
            return None
        submit_gpu_work(meta)
        self._gpu_submit_generation += 1
        mark_adapter_gpu_work_submitted(future, self._gpu_submit_generation)

        return None

    def _submit_next_pending_gpu_work(self) -> None:
        if not self._pending_gpu_submissions:
            return None
        self._submit_future_gpu_work(*self._pending_gpu_submissions.popleft())

        return None

    def _submit_all_pending_gpu_work(self) -> None:
        while self._pending_gpu_submissions:
            self._submit_future_gpu_work(*self._pending_gpu_submissions.popleft())

    def _submit_response_build(
        self, future: InferenceFuture, meta: Any, mapped_kwargs: dict
    ) -> None:
        response_future = self._get_response_executor().submit(_finalize_future, future)
        context_id = get_adapter_stream_pipeline_context_id(future)
        self._response_futures.append((response_future, context_id))

    def _submit_ready_responses(self) -> None:
        while self._pending_futures:
            future, meta, mapped_kwargs = self._pending_futures[0]
            submit_generation = get_adapter_gpu_submit_generation(future)
            if submit_generation is None:
                self._submit_future_gpu_work(future, meta, mapped_kwargs)
                submit_generation = get_adapter_gpu_submit_generation(future)
            if submit_generation is None:
                break
            if self._gpu_submit_generation < submit_generation + self._response_delay:
                break
            self._submit_response_build(*self._pending_futures.popleft())

    def _submit_all_pending_responses(self) -> None:
        while self._pending_futures:
            self._submit_response_build(*self._pending_futures.popleft())


def _finalize_future(future: InferenceFuture) -> List[InstanceDetections]:
    detections_list = []
    for detections in future.result():
        finalize_pending = get_deferred_postprocess_finalizer(detections)
        if callable(finalize_pending):
            detections = finalize_pending()
        detections_list.append(detections)

    finalized = tensors_to_numpy(detections_list)

    return finalized


def _wait_for_queued_work(executor: ThreadPoolExecutor, timeout_s: float) -> bool:
    try:
        executor.submit(lambda: None).result(timeout=timeout_s)
    except TimeoutError:
        return False
    return True


def _resolve_response_future(future: Future, *, context: str) -> Any:
    try:
        return future.result(timeout=cfg.INFERENCE_PROCESS_TIMEOUT_S)
    except TimeoutError as error:
        raise RuntimeError(f"Timed out while waiting for {context}.") from error


def _request_batch_size(images: Any) -> int:
    if isinstance(images, (list, tuple)):
        return len(images)
    shape = getattr(images, "shape", None)
    if shape is not None and len(shape) == 4:
        return int(shape[0])
    return 1


def _image_size(images: Any) -> Tuple[int, int]:
    image = images[0] if isinstance(images, (list, tuple)) else images
    shape = tuple(getattr(image, "shape", ()))
    if len(shape) == 2:
        return int(shape[0]), int(shape[1])
    if len(shape) == 3 and shape[-1] in (1, 3, 4):
        return int(shape[0]), int(shape[1])
    if len(shape) == 3:
        return int(shape[1]), int(shape[2])
    return 0, 0


def _placeholder(images: Any) -> InstanceDetections:
    placeholder = InstanceDetections(
        xyxy=torch.empty((0, 4), dtype=torch.int32),
        class_id=torch.empty((0,), dtype=torch.int32),
        confidence=torch.empty((0,), dtype=torch.float32),
        mask=InstancesRLEMasks(image_size=_image_size(images), masks=[]),
    )

    return placeholder
