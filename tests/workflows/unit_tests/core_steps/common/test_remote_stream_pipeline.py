import threading
from concurrent.futures import Future
from typing import Callable, Dict, List

import pytest

from inference.core.workflows.core_steps.common.remote_stream_pipeline import (
    RemoteStreamPipeline,
    make_prediction_future,
)
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


def test_submit_request_returns_future_resolving_to_task_result() -> None:
    # given
    pipeline = RemoteStreamPipeline(max_in_flight_requests=1)

    try:
        # when
        result_future = pipeline.submit_request(
            task=lambda: [{"predictions": "prediction-a"}]
        )

        # then
        assert result_future.result(timeout=5) == [{"predictions": "prediction-a"}]
    finally:
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
        for tag in _SUBMISSION_ORDER:
            result_future = pipeline.submit_request(
                task=_make_blocking_task(
                    tag=tag,
                    started_events=started_events,
                    release_events=release_events,
                    completion_order=completion_order,
                ),
            )
            prediction_futures[tag] = make_prediction_future(
                result_future=result_future, image_index=0
            )
        for tag in _SUBMISSION_ORDER:
            assert started_events[tag].wait(timeout=5), f"task {tag} never started"
        # completing the newest requests first simulates the remote API
        # answering out of order; waiting for each result sequences completion
        for tag in reversed(_SUBMISSION_ORDER):
            release_events[tag].set()
            assert prediction_futures[tag].result(timeout=5) == f"prediction-{tag}"

        # then - no cross-frame payload mixing despite reversed completion
        for tag in _SUBMISSION_ORDER:
            assert prediction_futures[tag].result(timeout=5) == f"prediction-{tag}"
        assert completion_order == list(reversed(_SUBMISSION_ORDER))
    finally:
        for release_event in release_events.values():
            release_event.set()
        pipeline.close()


def test_make_prediction_future_propagates_task_error() -> None:
    # given
    pipeline = RemoteStreamPipeline(max_in_flight_requests=1)

    def _failing_task() -> BlockResult:
        raise RuntimeError("remote request failed")

    try:
        # when
        result_future = pipeline.submit_request(task=_failing_task)
        prediction_future = make_prediction_future(
            result_future=result_future, image_index=0
        )

        # then
        with pytest.raises(RuntimeError, match="remote request failed"):
            prediction_future.result(timeout=5)
    finally:
        pipeline.close()


def test_make_prediction_future_resolves_to_indexed_predictions() -> None:
    # given
    result_future: Future = Future()
    prediction_future = make_prediction_future(
        result_future=result_future, image_index=1
    )

    # when
    result_future.set_result([{"predictions": "first"}, {"predictions": "second"}])

    # then
    assert prediction_future.result(timeout=5) == "second"


def test_make_prediction_future_surfaces_malformed_payload_instead_of_hanging() -> None:
    # given
    result_future: Future = Future()
    prediction_future = make_prediction_future(
        result_future=result_future, image_index=0
    )

    # when - the result payload lacks the expected output field
    result_future.set_result([{}])

    # then - the chained future resolves with the selection error
    with pytest.raises(KeyError):
        prediction_future.result(timeout=1)


def test_close_shuts_down_request_executor_and_detaches_finalizer() -> None:
    # given
    pipeline = RemoteStreamPipeline(max_in_flight_requests=1)
    assert pipeline._request_executor_finalizer.alive

    # when
    pipeline.close()

    # then
    assert pipeline._request_executor._shutdown
    assert not pipeline._request_executor_finalizer.alive
