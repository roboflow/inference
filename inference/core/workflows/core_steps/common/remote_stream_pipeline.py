"""Shared in-flight request bookkeeping for stream-pipelined remote model steps.

A model block executing remotely can keep several frames' HTTP requests in
flight while stateful downstream steps (trackers, visualizations) consume
results strictly in frame order through the stream-pipeline flush path. This
module owns the request state machine; a block adds the thin stream-pipeline
protocol methods (`is_stream_pipelined`, `stream_pipeline_depth`,
`defers_downstream_execution`, `flush_stream_pipeline_outputs`,
`close_stream_pipeline`) and delegates to a `RemoteStreamPipeline` instance.

Instances are not thread-safe by design: `submit_request`, `flush_oldest` and
`close` are only ever called from the stream inference thread; worker threads
only complete the futures.
"""

import weakref
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Deque, List, Tuple

from inference.core.env import WORKFLOWS_ASYNC_FUTURE_RESULT_TIMEOUT
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    WorkflowImageData,
)
from inference.core.workflows.prototypes.block import BlockResult


@dataclass(frozen=True)
class _PendingRemotePrediction:
    images: Batch[WorkflowImageData]
    result_future: Future


class RemoteStreamPipeline:

    def __init__(
        self,
        max_in_flight_requests: int,
        thread_name_prefix: str = "workflows_remote_stream_pipeline",
    ):
        self._pending_predictions: Deque[_PendingRemotePrediction] = deque()
        self._request_executor = ThreadPoolExecutor(
            max_workers=max(1, max_in_flight_requests),
            thread_name_prefix=thread_name_prefix,
        )
        self._request_executor_finalizer = weakref.finalize(
            self, self._request_executor.shutdown, False
        )

    @property
    def pending_requests(self) -> int:
        return len(self._pending_predictions)

    def submit_request(
        self,
        task: Callable[[], BlockResult],
        images: Batch[WorkflowImageData],
    ) -> Future:
        result_future = self._request_executor.submit(task)
        self._pending_predictions.append(
            _PendingRemotePrediction(images=images, result_future=result_future)
        )
        return result_future

    def flush_oldest(self) -> List[Tuple[List[Tuple[int, ...]], BlockResult]]:
        # Drains a single pending request per call so the stream dispatcher can
        # emit frames one at a time, in order, as their remote results land.
        if not self._pending_predictions:
            return []
        pending = self._pending_predictions.popleft()
        outputs = pending.result_future.result(
            timeout=WORKFLOWS_ASYNC_FUTURE_RESULT_TIMEOUT
        )
        return [(_batch_indices(images=pending.images), outputs)]

    def close(self) -> None:
        self._pending_predictions.clear()
        if self._request_executor_finalizer.alive:
            self._request_executor_finalizer.detach()
        self._request_executor.shutdown(wait=False)


def make_prediction_future(
    result_future: Future,
    image_index: int,
    output_field: str = "predictions",
) -> Future:
    # Chained via callback rather than executor.submit() — selector tasks
    # waiting on inference tasks in the same pool could exhaust its workers.
    prediction_future: Future = Future()

    def _propagate_result(done_future: Future) -> None:
        error = done_future.exception()
        if error is not None:
            prediction_future.set_exception(error)
            return
        prediction_future.set_result(done_future.result()[image_index][output_field])

    result_future.add_done_callback(_propagate_result)
    return prediction_future


def _batch_indices(images: Batch[WorkflowImageData]) -> List[Tuple[int, ...]]:
    indices = getattr(images, "indices", None)
    if indices is not None:
        return list(indices)
    return [(image_index,) for image_index in range(len(images))]
