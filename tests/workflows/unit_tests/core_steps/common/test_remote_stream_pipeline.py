import threading
from typing import Callable, Dict, List

import pytest

from inference.core.workflows.core_steps.common.remote_stream_pipeline import (
    RemoteStreamPipeline,
    make_prediction_future,
)
from inference.core.workflows.execution_engine.entities.base import Batch
from inference.core.workflows.prototypes.block import BlockResult

_SUBMISSION_ORDER = ("a", "b", "c")


def _make_blocking_task(
    tag: str,
    started_events: Dict[str, threading.Event],
    release_events: Dict[str, threading.Event],
    completion_order: List[str],
) -> Callable[[], BlockResult]:
    def _task() -> BlockResult:
        started_events[tag].set()
        assert release_events[tag].wait(timeout=5), f"task {tag} never released"
        completion_order.append(tag)
        return [{"predictions": f"prediction-{tag}"}]

    return _task


def test_flush_oldest_preserves_submission_order_when_completion_order_is_reversed() -> (
    None
):
    # given
    pipeline = RemoteStreamPipeline(max_in_flight_requests=len(_SUBMISSION_ORDER))
    started_events = {tag: threading.Event() for tag in _SUBMISSION_ORDER}
    release_events = {tag: threading.Event() for tag in _SUBMISSION_ORDER}
    completion_order: List[str] = []
    release_oldest_timer = None

    try:
        # when
        result_futures, prediction_futures = {}, {}
        for index, tag in enumerate(_SUBMISSION_ORDER):
            result_future = pipeline.submit_request(
                task=_make_blocking_task(
                    tag=tag,
                    started_events=started_events,
                    release_events=release_events,
                    completion_order=completion_order,
                ),
                images=Batch.init(content=[f"image-{tag}"], indices=[(index,)]),
            )
            result_futures[tag] = result_future
            prediction_futures[tag] = make_prediction_future(
                result_future=result_future, image_index=0
            )
        for tag in _SUBMISSION_ORDER:
            assert started_events[tag].wait(timeout=5), f"task {tag} never started"
        # completing the newest requests first simulates the remote API
        # answering out of order; the oldest request stays in flight
        release_events["c"].set()
        assert prediction_futures["c"].result(timeout=5) == "prediction-c"
        release_events["b"].set()
        assert prediction_futures["b"].result(timeout=5) == "prediction-b"

        # then
        assert completion_order == ["c", "b"]
        assert result_futures["a"].done() is False
        assert pipeline.pending_requests == 3

        # when - flush_oldest must block until the oldest request lands
        release_oldest_timer = threading.Timer(0.05, release_events["a"].set)
        release_oldest_timer.start()
        first_flush = pipeline.flush_oldest()

        # then - FIFO order regardless of completion order
        assert result_futures["a"].done() is True
        assert first_flush == [([(0,)], [{"predictions": "prediction-a"}])]
        assert pipeline.flush_oldest() == [([(1,)], [{"predictions": "prediction-b"}])]
        assert pipeline.flush_oldest() == [([(2,)], [{"predictions": "prediction-c"}])]
        assert pipeline.flush_oldest() == []
        assert prediction_futures["a"].result(timeout=5) == "prediction-a"
        assert completion_order == ["c", "b", "a"]
    finally:
        if release_oldest_timer is not None:
            release_oldest_timer.cancel()
        for release_event in release_events.values():
            release_event.set()
        pipeline.close()


def test_per_image_prediction_futures_resolve_to_their_own_frames_payloads() -> None:
    # given
    pipeline = RemoteStreamPipeline(max_in_flight_requests=len(_SUBMISSION_ORDER))
    started_events = {tag: threading.Event() for tag in _SUBMISSION_ORDER}
    release_events = {tag: threading.Event() for tag in _SUBMISSION_ORDER}
    completion_order: List[str] = []

    try:
        # when
        prediction_futures = {}
        for index, tag in enumerate(_SUBMISSION_ORDER):
            result_future = pipeline.submit_request(
                task=_make_blocking_task(
                    tag=tag,
                    started_events=started_events,
                    release_events=release_events,
                    completion_order=completion_order,
                ),
                images=Batch.init(content=[f"image-{tag}"], indices=[(index,)]),
            )
            prediction_futures[tag] = make_prediction_future(
                result_future=result_future, image_index=0
            )
        for tag in reversed(_SUBMISSION_ORDER):
            release_events[tag].set()

        # then - no cross-frame payload mixing despite reversed completion
        for tag in _SUBMISSION_ORDER:
            assert prediction_futures[tag].result(timeout=5) == f"prediction-{tag}"
    finally:
        for release_event in release_events.values():
            release_event.set()
        pipeline.close()


def test_pending_requests_reflect_submissions_and_close_clears_them() -> None:
    # given
    pipeline = RemoteStreamPipeline(max_in_flight_requests=2)

    # when
    pipeline.submit_request(
        task=lambda: [{"predictions": "prediction-a"}],
        images=Batch.init(content=["image-a"], indices=[(0,)]),
    )
    pipeline.submit_request(
        task=lambda: [{"predictions": "prediction-b"}],
        images=Batch.init(content=["image-b"], indices=[(1,)]),
    )

    # then
    assert pipeline.pending_requests == 2
    pipeline.close()
    assert pipeline.pending_requests == 0
    assert pipeline.flush_oldest() == []


def test_make_prediction_future_propagates_task_error() -> None:
    # given
    pipeline = RemoteStreamPipeline(max_in_flight_requests=1)

    def _failing_task() -> BlockResult:
        raise RuntimeError("remote request failed")

    try:
        # when
        result_future = pipeline.submit_request(
            task=_failing_task,
            images=Batch.init(content=["image-a"], indices=[(0,)]),
        )
        prediction_future = make_prediction_future(
            result_future=result_future, image_index=0
        )

        # then
        with pytest.raises(RuntimeError, match="remote request failed"):
            prediction_future.result(timeout=5)
    finally:
        pipeline.close()
