"""Async request launching for stream-pipelined remote model steps.

A model block that executes remotely can keep several frames' HTTP requests
in flight while stateful downstream steps (trackers, visualizations) consume
results strictly in frame order. The block submits each frame's request to a
long-lived worker pool and returns future-bearing outputs; the execution
engine registers those futures in the frame's execution state and resolves
them at ordered emission time, when the remaining steps run.

Blocks add the thin stream-pipeline protocol methods (`is_stream_pipelined`,
`stream_pipeline_depth`, `defers_downstream_execution`,
`close_stream_pipeline`) and delegate request submission here.

Instances are not thread-safe by design: `submit_request` and `close` are
only ever called from the stream inference thread; worker threads only
complete the futures.
"""

import weakref
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable

from inference.core.workflows.prototypes.block import BlockResult


class RemoteStreamPipeline:

    def __init__(
        self,
        max_in_flight_requests: int,
        thread_name_prefix: str = "workflows_remote_stream_pipeline",
    ):
        self._request_executor = ThreadPoolExecutor(
            max_workers=max(1, max_in_flight_requests),
            thread_name_prefix=thread_name_prefix,
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
