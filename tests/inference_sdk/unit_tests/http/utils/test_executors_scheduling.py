import asyncio
import contextvars
import threading
import time
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import pytest
from aiohttp import ClientConnectionError, ClientResponseError, RequestInfo
from requests import HTTPError, Response

from inference_sdk.config import RemoteProcessingTimeCollector, remote_processing_times
from inference_sdk.http.utils import executors
from inference_sdk.http.utils.executors import (
    RequestMethod,
    execute_requests_packages,
    execute_requests_packages_async,
)
from inference_sdk.http.utils.request_building import RequestData

PROBE_VAR = contextvars.ContextVar("scheduling_probe", default=None)
WORKER_BLEED_VAR = contextvars.ContextVar("scheduling_worker_bleed", default=None)


def _request_data(index: int, model_id: Optional[str] = None) -> RequestData:
    payload = {"index": index}
    if model_id is not None:
        payload["model_id"] = model_id
    return RequestData(
        url=f"https://some.com/{index}",
        request_elements=1,
        headers=None,
        data=None,
        parameters=None,
        payload=payload,
        image_scaling_factors=[None],
    )


def _ok_response(index: int) -> Response:
    response = Response()
    response.status_code = 200
    response.url = f"https://some.com/{index}"
    return response


def _error_response(index: int) -> Response:
    response = Response()
    response.status_code = 500
    response.url = f"https://some.com/{index}?api_key=AAAABBBBCCCC"
    response.reason = "Server Error"
    return response


def _requests_data(count: int) -> List[RequestData]:
    return [_request_data(index) for index in range(count)]


def _start_sync_worker(
    requests_data: List[RequestData],
    cap: int,
) -> Tuple[Dict[str, object], threading.Thread]:
    """Run the sync executor on a worker thread, capturing result or error."""
    outcome: Dict[str, object] = {}

    def run() -> None:
        try:
            outcome["result"] = execute_requests_packages(
                requests_data=requests_data,
                request_method=RequestMethod.GET,
                max_concurrent_requests=cap,
            )
        except Exception as error:  # noqa: BLE001 - captured for assertion
            outcome["error"] = error

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    return outcome, worker


def _join_worker(worker: threading.Thread, timeout: float = 10.0) -> None:
    worker.join(timeout=timeout)
    assert not worker.is_alive()


class _PendingTracker:
    """Tracks submitted-but-unfinished futures for bounded-pending proofs."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.outstanding = 0
        self.max_outstanding = 0
        self.submitted = 0

    def record_submit(self, future) -> None:
        with self._lock:
            self.outstanding += 1
            self.submitted += 1
            self.max_outstanding = max(self.max_outstanding, self.outstanding)
        future.add_done_callback(self._record_done)

    def _record_done(self, _future) -> None:
        with self._lock:
            self.outstanding -= 1


@pytest.mark.parametrize(("total", "cap"), [(9, 3), (4, 4), (3, 10)])
def test_in_flight_and_pending_submissions_never_exceed_cap(
    monkeypatch: pytest.MonkeyPatch, total: int, cap: int
) -> None:
    # given
    expected = min(total, cap)
    tracker = _PendingTracker()
    release_all = threading.Event()
    active, max_active, lock = 0, 0, threading.Lock()

    def fake_make_request(request_data, request_method) -> Response:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            release_all.wait(timeout=15)
            return _ok_response(request_data.payload["index"])
        finally:
            with lock:
                active -= 1

    class _TrackingExecutor(executors.ThreadPoolExecutor):
        def submit(self, fn, *args, **kwargs):
            future = super().submit(fn, *args, **kwargs)
            tracker.record_submit(future)
            return future

    monkeypatch.setattr(executors, "make_request", fake_make_request)
    monkeypatch.setattr(executors, "ThreadPoolExecutor", _TrackingExecutor)
    outcome, worker = _start_sync_worker(_requests_data(total), cap)

    # when - wait until the window is full, then release everything
    deadline = time.monotonic() + 10
    while tracker.max_outstanding < expected and time.monotonic() < deadline:
        time.sleep(0.005)
    release_all.set()
    _join_worker(worker)

    # then - this intentionally verifies bounded submission, not merely
    # active concurrency: the tracker counts each future from submit() until
    # it finishes, so max_outstanding == cap proves no more than cap futures
    # were ever pending while submitted == total proves all work was admitted
    assert tracker.max_outstanding == expected
    assert tracker.submitted == total
    assert max_active == expected
    assert len(outcome["result"]) == total


def test_empty_input_returns_empty_without_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # given
    monkeypatch.setattr(executors, "ThreadPoolExecutor", MagicMock())

    # when
    result = execute_requests_packages(
        requests_data=[],
        request_method=RequestMethod.GET,
        max_concurrent_requests=4,
    )

    # then
    assert result == []
    executors.ThreadPoolExecutor.assert_not_called()


@pytest.mark.parametrize(("total", "cap"), [(3, 0), (3, 1), (3, -3), (1, 5)])
def test_serial_paths_run_on_caller_thread_without_an_executor(
    monkeypatch: pytest.MonkeyPatch, total: int, cap: int
) -> None:
    # given
    caller_thread = threading.get_ident()
    seen_threads: List[int] = []

    def fake_make_request(request_data, request_method) -> Response:
        seen_threads.append(threading.get_ident())
        return _ok_response(request_data.payload["index"])

    monkeypatch.setattr(executors, "make_request", fake_make_request)
    monkeypatch.setattr(executors, "ThreadPoolExecutor", MagicMock())

    # when
    result = execute_requests_packages(
        requests_data=_requests_data(total),
        request_method=RequestMethod.GET,
        max_concurrent_requests=cap,
    )

    # then
    assert [response.status_code for response in result] == [200] * total
    assert seen_threads == [caller_thread] * total
    executors.ThreadPoolExecutor.assert_not_called()


def test_http_error_status_does_not_stop_dispatch_and_raises_in_input_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # given - indices 1 and 3 return 500 responses (not transport exceptions)
    total = 4
    started = [threading.Event() for _ in range(total)]
    release_first = threading.Event()
    error_responses = {index: _error_response(index) for index in (1, total - 1)}

    def fake_make_request(request_data, request_method) -> Response:
        index = request_data.payload["index"]
        started[index].set()
        if index == 0:
            release_first.wait(timeout=15)
        if index == total - 1:
            release_first.set()
        if index in error_responses:
            return error_responses[index]
        return _ok_response(index)

    monkeypatch.setattr(executors, "make_request", fake_make_request)
    outcome, worker = _start_sync_worker(_requests_data(total), cap=2)

    # when / then - every request is admitted despite the 500s, and the
    # post-pass raises the error for the lowest input index
    assert started[total - 1].wait(timeout=10)
    assert all(event.is_set() for event in started)
    _join_worker(worker)
    assert isinstance(outcome["error"], HTTPError)
    assert outcome["error"].response is error_responses[1]
    assert "AAAABBBBCCCC" not in str(outcome["error"])
    assert "api_key=AA***CC" in str(outcome["error"])
    assert "result" not in outcome


def test_transport_error_stops_admission_and_raises_earliest_index(monkeypatch):
    release = threading.Event()
    initial_workers = threading.Barrier(3, timeout=5)
    started = []
    original_wait = executors.wait

    def request(request_data, request_method):
        index = request_data.payload["index"]
        started.append(index)
        if index < 3:
            initial_workers.wait()
        if index == 2:
            raise ConnectionError("error-2")
        if index < 2:
            assert release.wait(timeout=5)
            if index == 1:
                raise ConnectionError("error-1")
        return _ok_response(index)

    def wait_for_first_failure(futures, **kwargs):
        done, pending = original_wait(futures, timeout=5, **kwargs)
        release.set()
        assert done == {futures[2]}
        return done, pending

    monkeypatch.setattr(executors, "make_request", request)
    monkeypatch.setattr(executors, "wait", wait_for_first_failure)

    with pytest.raises(ConnectionError, match="error-1"):
        execute_requests_packages(
            requests_data=_requests_data(6),
            request_method=RequestMethod.GET,
            max_concurrent_requests=3,
        )
    assert sorted(started) == [0, 1, 2]


def test_simultaneous_transport_failures_raise_lowest_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # given - indices 0 and 2 fail while 1 succeeds; the wait() result is
    # forced to report every in-flight future in a single done batch so both
    # failures are observed simultaneously. The futures still complete for
    # real (future.result() blocks until then), so there is no timing race.
    started: List[int] = []

    def fake_make_request(request_data, request_method) -> Response:
        index = request_data.payload["index"]
        started.append(index)
        if index == 0:
            raise ConnectionError("error-0")
        if index == 2:
            raise ConnectionError("error-2")
        return _ok_response(index)

    monkeypatch.setattr(executors, "make_request", fake_make_request)
    monkeypatch.setattr(
        executors, "wait", lambda futures, **kwargs: (set(futures), set())
    )

    # when / then - the lowest input index wins within one observation batch
    # and no slot is refilled afterwards
    with pytest.raises(ConnectionError, match="error-0"):
        execute_requests_packages(
            requests_data=_requests_data(5),
            request_method=RequestMethod.GET,
            max_concurrent_requests=3,
        )
    assert sorted(started) == [0, 1, 2]


def test_unobserved_slow_failure_does_not_stop_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # given - cap=3, N=12; index 0 fails only after a gate is released, so
    # its failure cannot be observed while fast requests refill freed slots
    total, cap = 12, 3
    started = [threading.Event() for _ in range(total)]
    release_failure = threading.Event()

    def fake_make_request(request_data, request_method) -> Response:
        index = request_data.payload["index"]
        started[index].set()
        if index == 0:
            release_failure.wait(timeout=15)
            raise ConnectionError("slow-transport-failure")
        return _ok_response(index)

    monkeypatch.setattr(executors, "make_request", fake_make_request)
    outcome, worker = _start_sync_worker(_requests_data(total), cap)

    # when - every request is admitted while index 0 is still blocked, so no
    # failure has been observed yet
    try:
        for index, event in enumerate(started):
            assert event.wait(timeout=10), (
                f"request index {index} was not dispatched before the "
                "pending index-0 failure could be observed"
            )
        assert worker.is_alive()
    finally:
        release_failure.set()
    _join_worker(worker)

    # then - admission continued until the failure was observed, so all 12
    # requests were dispatched; the observed failure still propagates
    assert isinstance(outcome["error"], ConnectionError)
    assert str(outcome["error"]) == "slow-transport-failure"
    assert "result" not in outcome


def test_contextvars_propagate_to_rolling_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # given
    total = 4
    observed: List[object] = []
    token = PROBE_VAR.set("parent-value")

    def fake_make_request(request_data, request_method) -> Response:
        observed.append(PROBE_VAR.get())
        return _ok_response(request_data.payload["index"])

    monkeypatch.setattr(executors, "make_request", fake_make_request)
    try:
        # when
        result = execute_requests_packages(
            requests_data=_requests_data(total),
            request_method=RequestMethod.GET,
            max_concurrent_requests=2,
        )
    finally:
        PROBE_VAR.reset(token)

    # then
    assert len(result) == total
    assert sorted(observed) == ["parent-value"] * total
    assert PROBE_VAR.get() is None


def test_worker_contextvar_changes_do_not_leak_between_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # given - a variable absent from the parent context that request code sets
    total, cap = 8, 2
    observed: List[object] = []

    def fake_make_request(request_data, request_method) -> Response:
        observed.append(WORKER_BLEED_VAR.get())
        WORKER_BLEED_VAR.set(f"leaked-{request_data.payload['index']}")
        return _ok_response(request_data.payload["index"])

    monkeypatch.setattr(executors, "make_request", fake_make_request)

    # when - 8 requests over 2 workers guarantees thread reuse
    result = execute_requests_packages(
        requests_data=_requests_data(total),
        request_method=RequestMethod.GET,
        max_concurrent_requests=cap,
    )

    # then - no request observes a value set by an earlier request on a
    # reused worker thread, and the caller context is untouched
    assert len(result) == total
    assert observed == [None] * total
    assert WORKER_BLEED_VAR.get() is None


def test_remote_processing_time_pairs_by_input_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # given - index 0 is released by index 3, so it cannot complete first
    total = 4
    collector = RemoteProcessingTimeCollector()
    token = remote_processing_times.set(collector)
    gate_zero = threading.Event()
    requests_data = [
        _request_data(index, model_id=f"model-{index}") for index in range(total)
    ]

    def fake_make_request(request_data, request_method) -> Response:
        index = request_data.payload["index"]
        if index == 0:
            gate_zero.wait(timeout=15)
        if index == total - 1:
            gate_zero.set()
        response = _ok_response(index)
        response.headers["X-Processing-Time"] = f"{10 + index}.0"
        return response

    monkeypatch.setattr(executors, "make_request", fake_make_request)

    try:
        # when - index 0 completes after later indices have finished
        result = execute_requests_packages(
            requests_data=requests_data,
            request_method=RequestMethod.GET,
            max_concurrent_requests=2,
        )

        # then - each processing time is paired with its own input index
        assert len(result) == total
        assert collector.drain() == [
            (f"model-{index}", float(10 + index)) for index in range(total)
        ]
        assert collector.snapshot_model_ids() == {
            f"model-{index}" for index in range(total)
        }
    finally:
        remote_processing_times.reset(token)


@pytest.mark.parametrize("fail", [False, True], ids=["success", "failure"])
def test_worker_sessions_close_and_threads_join(monkeypatch, fail):
    caller = threading.current_thread()
    workers = []
    sessions = []

    def handle(url, **kwargs):
        index = int(url.rsplit("/", 1)[1])
        if fail and index == 1:
            raise ConnectionError("transport failure")
        return _ok_response(index)

    def session_factory():
        workers.append(threading.current_thread())
        session = MagicMock()
        session.get.side_effect = handle
        sessions.append(session)
        return session

    monkeypatch.setattr(executors.requests, "Session", session_factory)

    def run():
        result = execute_requests_packages(
            requests_data=_requests_data(4),
            request_method=RequestMethod.GET,
            max_concurrent_requests=2,
        )
        return result

    if fail:
        with pytest.raises(ConnectionError, match="transport failure"):
            run()
    else:
        assert len(run()) == len(sessions) == 4

    assert workers and all(thread is not caller for thread in workers)
    assert all(not thread.is_alive() for thread in workers)
    for session in sessions:
        session.close.assert_called_once()


# --- async scheduling tests -------------------------------------------------


class _FakeAsyncSession:
    def __init__(self, tracker: Dict[str, object]) -> None:
        self._tracker = tracker

    async def __aenter__(self) -> "_FakeAsyncSession":
        return self

    async def __aexit__(self, *exc_info) -> bool:
        self._tracker["session_closed"] = True
        return False


@pytest.fixture
def session_tracker(monkeypatch: pytest.MonkeyPatch) -> Dict[str, object]:
    """Patch aiohttp.ClientSession and return its close-state tracker."""
    tracker: Dict[str, object] = {"session_closed": False}
    monkeypatch.setattr(
        executors.aiohttp,
        "ClientSession",
        lambda: _FakeAsyncSession(tracker),
    )
    return tracker


async def _teardown_executor_task(task: asyncio.Task) -> None:
    """Cancel an executor task and await its cleanup within a bounded wait."""
    if task.done():
        return
    task.cancel()
    try:
        await asyncio.wait_for(task, timeout=5)
    except (asyncio.CancelledError, asyncio.TimeoutError):
        pass


async def _run_executor_with_watchdog(
    requests_data: List[RequestData],
    max_concurrent_requests: int,
    timeout: float = 15.0,
):
    """Await the async executor with a deadlock watchdog and task teardown."""
    task = asyncio.ensure_future(
        execute_requests_packages_async(
            requests_data=requests_data,
            request_method=RequestMethod.GET,
            max_concurrent_requests=max_concurrent_requests,
        )
    )
    try:
        return await asyncio.wait_for(task, timeout=timeout)
    finally:
        await _teardown_executor_task(task)


@pytest.mark.asyncio
async def test_async_rolling_dispatch_refills_slots_and_preserves_input_order(
    monkeypatch: pytest.MonkeyPatch, session_tracker
) -> None:
    # given - index 0 stays blocked until every other request has finished,
    # so freed slots must be refilled for later indices and completion order
    # differs from input order
    total, cap = 5, 2
    others_finished = asyncio.Event()
    finished_count = 0
    log: List[tuple] = []

    async def fake_request(rd, request_method, session):
        nonlocal finished_count
        index = rd.payload["index"]
        log.append(("start", index))
        if index == 0:
            await others_finished.wait()
        else:
            finished_count += 1
            if finished_count == total - 1:
                others_finished.set()
        log.append(("end", index))
        return 200, {"index": index}

    monkeypatch.setattr(executors, "make_request_async", fake_request)

    # when
    result = await _run_executor_with_watchdog(_requests_data(total), cap)

    # then - requests beyond the initial window started before the blocked
    # index 0 completed, and results are still returned in input order
    assert result == [{"index": index} for index in range(total)]
    start_pos = {index: pos for pos, (op, index) in enumerate(log) if op == "start"}
    end_zero = next(pos for pos, event in enumerate(log) if event == ("end", 0))
    assert all(start_pos[index] < end_zero for index in range(cap, total))
    assert log[-1] == ("end", 0)


@pytest.mark.asyncio
async def test_async_in_flight_and_pending_never_exceed_cap(
    monkeypatch: pytest.MonkeyPatch, session_tracker
) -> None:
    # given - owned tasks are also recorded so their cleanup after success
    # can be checked without ambient all_tasks() accounting
    total, cap = 8, 3
    release_all = asyncio.Event()
    cap_reached = asyncio.Event()
    active = max_active = 0
    owned_tasks: List[asyncio.Task] = []

    async def fake_request(rd, request_method, session):
        nonlocal active, max_active
        owned_tasks.append(asyncio.current_task())
        active += 1
        max_active = max(max_active, active)
        if active == cap:
            cap_reached.set()
        try:
            await release_all.wait()
            return 200, {"index": rd.payload["index"]}
        finally:
            active -= 1

    monkeypatch.setattr(executors, "make_request_async", fake_request)

    # when
    task = asyncio.create_task(
        execute_requests_packages_async(
            requests_data=_requests_data(total),
            request_method=RequestMethod.GET,
            max_concurrent_requests=cap,
        )
    )
    try:
        await asyncio.wait_for(cap_reached.wait(), timeout=5)
        release_all.set()
        result = await asyncio.wait_for(task, timeout=15)
    finally:
        release_all.set()
        await _teardown_executor_task(task)

    # then
    assert len(result) == total
    assert max_active == cap
    assert owned_tasks and all(t.done() for t in owned_tasks)


@pytest.mark.asyncio
async def test_async_empty_input_returns_empty_without_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # given
    session_cls = MagicMock()
    monkeypatch.setattr(executors.aiohttp, "ClientSession", session_cls)

    # when
    result = await _run_executor_with_watchdog(
        requests_data=[],
        max_concurrent_requests=4,
        timeout=5,
    )

    # then
    assert result == []
    session_cls.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("cap", [0, 1])
async def test_async_single_concurrency_runs_sequentially(
    monkeypatch: pytest.MonkeyPatch, session_tracker, cap: int
) -> None:
    # given
    total = 3
    started: List[int] = []

    async def fake_request(rd, request_method, session):
        index = rd.payload["index"]
        started.append(index)
        return 200, {"index": index}

    monkeypatch.setattr(executors, "make_request_async", fake_request)

    # when
    result = await _run_executor_with_watchdog(_requests_data(total), cap)

    # then
    assert result == [{"index": index} for index in range(total)]
    assert started == list(range(total))


@pytest.mark.asyncio
async def test_async_contextvars_propagate_to_request_tasks(
    monkeypatch: pytest.MonkeyPatch, session_tracker
) -> None:
    # given
    observed: List[object] = []
    token = PROBE_VAR.set("async-parent-value")

    async def fake_request(rd, request_method, session):
        observed.append(PROBE_VAR.get())
        return 200, {"index": rd.payload["index"]}

    monkeypatch.setattr(executors, "make_request_async", fake_request)

    try:
        # when
        result = await _run_executor_with_watchdog(_requests_data(3), 2)
    finally:
        PROBE_VAR.reset(token)

    # then
    assert len(result) == 3
    assert observed == ["async-parent-value"] * 3


def _client_response_error() -> ClientResponseError:
    return ClientResponseError(
        request_info=RequestInfo(url="https://some.com/1", method="GET", headers={}),
        history=(),
        status=500,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("cap", "error", "admitted"),
    [
        (3, ClientConnectionError("async-boom"), {0, 1, 2}),
        (2, _client_response_error(), {0, 1}),
    ],
    ids=["transport-error", "http-status-error"],
)
async def test_async_request_failure_stops_admission_and_closes_session(
    monkeypatch: pytest.MonkeyPatch,
    session_tracker,
    cap: int,
    error: Exception,
    admitted: set,
) -> None:
    # given - index 1 fails immediately; the other admitted requests block.
    # Unlike the sync path, both transport errors and HTTP error statuses
    # raised by make_request_async stop admission and propagate.
    gates = {index: asyncio.Event() for index in admitted - {1}}
    started: List[int] = []
    owned_tasks: List[asyncio.Task] = []

    async def fake_request(rd, request_method, session):
        index = rd.payload["index"]
        started.append(index)
        owned_tasks.append(asyncio.current_task())
        if index == 1:
            raise error
        if index in gates:
            await gates[index].wait()
        return 200, {"index": index}

    monkeypatch.setattr(executors, "make_request_async", fake_request)

    # when
    try:
        with pytest.raises(type(error)):
            await _run_executor_with_watchdog(_requests_data(6), cap)

        # then - admission stopped, owned tasks cancelled and awaited,
        # session closed
        assert set(started) <= admitted
        assert session_tracker["session_closed"] is True
        assert owned_tasks and all(t.done() for t in owned_tasks)
    finally:
        # release any orphaned request tasks so teardown is reliable
        for gate in gates.values():
            gate.set()
        await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_async_external_cancellation_cancels_owned_tasks(
    monkeypatch: pytest.MonkeyPatch, session_tracker
) -> None:
    # given - all admitted requests block until cancelled; owned tasks are
    # recorded directly so cleanup is checked without ambient all_tasks()
    cap = 2
    release_all = asyncio.Event()
    cap_reached = asyncio.Event()
    active = cancelled_count = 0
    owned_tasks: List[asyncio.Task] = []

    async def fake_request(rd, request_method, session):
        nonlocal active, cancelled_count
        owned_tasks.append(asyncio.current_task())
        active += 1
        if active == cap:
            cap_reached.set()
        try:
            await release_all.wait()
            return 200, {"index": rd.payload["index"]}
        except asyncio.CancelledError:
            cancelled_count += 1
            raise
        finally:
            active -= 1

    monkeypatch.setattr(executors, "make_request_async", fake_request)

    # when
    task = asyncio.create_task(
        execute_requests_packages_async(
            requests_data=_requests_data(4),
            request_method=RequestMethod.GET,
            max_concurrent_requests=cap,
        )
    )
    try:
        await asyncio.wait_for(cap_reached.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=15)
    finally:
        release_all.set()
        await _teardown_executor_task(task)

    # then - owned tasks cancelled and awaited, session closed, no orphans
    assert cancelled_count >= 1
    assert session_tracker["session_closed"] is True
    assert owned_tasks and all(t.done() for t in owned_tasks)


@pytest.mark.asyncio
async def test_async_raises_first_observed_failure_not_cleanup_error(
    monkeypatch: pytest.MonkeyPatch, session_tracker
) -> None:
    # given - index 1 fails immediately; index 0 is still blocked when that
    # failure is observed and surfaces a different error while being drained
    gate_zero = asyncio.Event()
    started: List[int] = []

    async def fake_request(rd, request_method, session):
        index = rd.payload["index"]
        started.append(index)
        if index == 0:
            try:
                await gate_zero.wait()
            except asyncio.CancelledError:
                raise ValueError("cleanup-error-zero")
        elif index == 1:
            raise ClientConnectionError("observed-error-one")
        return 200, {"index": index}

    monkeypatch.setattr(executors, "make_request_async", fake_request)

    try:
        # when
        with pytest.raises(ClientConnectionError, match="observed-error-one"):
            await _run_executor_with_watchdog(_requests_data(3), 2)

        # then - admission stopped before index 2 and the drain-time error
        # did not replace the first observed failure
        assert set(started) <= {0, 1}
    finally:
        gate_zero.set()
        await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_async_simultaneous_failures_raise_lowest_index(
    monkeypatch: pytest.MonkeyPatch, session_tracker
) -> None:
    # given - indices 0 and 2 fail while 1 succeeds; asyncio.wait is forced
    # to gather every pending task and report them all in a single done
    # batch so both failures are observed simultaneously, with no timing race
    started: List[int] = []
    owned_tasks: List[asyncio.Task] = []

    async def fake_request(rd, request_method, session):
        index = rd.payload["index"]
        started.append(index)
        owned_tasks.append(asyncio.current_task())
        if index == 0:
            raise ClientConnectionError("error-0")
        if index == 2:
            raise ClientConnectionError("error-2")
        return 200, {"index": index}

    async def force_all_done(tasks, **kwargs):
        await asyncio.gather(*tasks, return_exceptions=True)
        return set(tasks), set()

    monkeypatch.setattr(executors, "make_request_async", fake_request)
    monkeypatch.setattr(asyncio, "wait", force_all_done)

    # when / then - the lowest input index wins within one observation batch
    # and no slot is refilled afterwards
    with pytest.raises(ClientConnectionError, match="error-0"):
        await _run_executor_with_watchdog(_requests_data(5), 3)
    assert sorted(started) == [0, 1, 2]
    assert session_tracker["session_closed"] is True
    assert owned_tasks and all(t.done() for t in owned_tasks)


@pytest.mark.asyncio
async def test_async_unobserved_slow_failure_does_not_stop_admission(
    monkeypatch: pytest.MonkeyPatch, session_tracker
) -> None:
    # given - same blast-radius scenario as the sync test: index 0 fails
    # only after a gate is released, so its failure cannot be observed while
    # fast requests refill freed slots
    total, cap = 12, 3
    started = [asyncio.Event() for _ in range(total)]
    release_failure = asyncio.Event()

    async def fake_request(rd, request_method, session):
        index = rd.payload["index"]
        started[index].set()
        if index == 0:
            await release_failure.wait()
            raise ClientConnectionError("slow-transport-failure")
        return 200, {"index": index}

    monkeypatch.setattr(executors, "make_request_async", fake_request)

    # when - every request is admitted while index 0 is still blocked, so no
    # failure has been observed yet
    task = asyncio.ensure_future(
        execute_requests_packages_async(
            requests_data=_requests_data(total),
            request_method=RequestMethod.GET,
            max_concurrent_requests=cap,
        )
    )
    try:
        for event in started:
            await asyncio.wait_for(event.wait(), timeout=10)
        assert not task.done()
        assert session_tracker["session_closed"] is False
        release_failure.set()
        with pytest.raises(ClientConnectionError, match="slow-transport-failure"):
            await asyncio.wait_for(task, timeout=15)

        # then - admission continued until the failure was observed, so all
        # 12 requests were dispatched; the observed failure still propagates
        assert session_tracker["session_closed"] is True
    finally:
        release_failure.set()
        await _teardown_executor_task(task)
