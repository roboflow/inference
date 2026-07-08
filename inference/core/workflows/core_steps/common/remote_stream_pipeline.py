"""Async request launching for stream-pipelined remote model steps.

This is the block-side half of stream-lookahead execution (the scheduler
half — frontier computation, deferred/resume passes — lives in
``inference.core.workflows.execution_engine.v1.executor.core``). A remote
model block submits each frame's HTTP request to a long-lived worker pool
and returns future-bearing outputs; the execution engine registers those
futures in the frame's execution state and resolves them at ordered
emission time, when the remaining steps run.

Blocks add the thin stream-pipeline protocol methods (``is_stream_pipelined``,
``can_activate_stream_pipeline``, ``defers_downstream_execution``,
``stream_pipeline_depth``, ``close_stream_pipeline``) and delegate request
submission here.

Instances are not thread-safe by design: ``submit_request`` and ``close`` are
only ever called from the stream inference thread; worker threads only
complete the futures.
"""

import weakref
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable

from inference.core.workflows.prototypes.block import BlockResult


class RemoteStreamPipeline:

    def __init__(self, max_in_flight_requests: int):
        self._request_executor = ThreadPoolExecutor(
            max_workers=max(1, max_in_flight_requests),
            thread_name_prefix="workflows_remote_stream_pipeline",
        )
        self._request_executor_finalizer = weakref.finalize(
            self, self._request_executor.shutdown, False
        )

    def submit_request(self, task: Callable[[], BlockResult]) -> Future:
        return self._request_executor.submit(task)

    def close(self) -> None:
        if self._request_executor_finalizer.alive:
            self._request_executor_finalizer.detach()
        self._request_executor.shutdown(wait=False)


def make_prediction_future(result_future: Future, image_index: int) -> Future:
    # Chained via callback rather than executor.submit() — selector tasks
    # waiting on inference tasks in the same pool could exhaust its workers.
    prediction_future: Future = Future()

    def _propagate_result(done_future: Future) -> None:
        error = done_future.exception()
        if error is not None:
            prediction_future.set_exception(error)
            return
        try:
            prediction_future.set_result(
                done_future.result()[image_index]["predictions"]
            )
        except Exception as selection_error:
            # concurrent.futures swallows callback exceptions — without this
            # the chained future would never resolve and consumers would hang
            # until the future-resolution timeout.
            prediction_future.set_exception(selection_error)

    result_future.add_done_callback(_propagate_result)
    return prediction_future
