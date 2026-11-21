import asyncio
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import backoff._async as backoff_async
import backoff._sync as backoff_sync
import pytest
from aiohttp import ClientResponseError
from aioresponses import aioresponses
from requests import ConnectionError, HTTPError, Response
from yarl import URL

from inference_sdk.http.utils import executors
from inference_sdk.http.utils.executors import RequestMethod
from inference_sdk.http.utils.request_building import RequestData

BASE_URL = "https://some.com"
WATCHDOG = 5


def _requests(count):
    return [
        RequestData(
            url=f"{BASE_URL}/{index}",
            request_elements=1,
            headers=None,
            data=None,
            parameters=None,
            payload={"index": index},
            image_scaling_factors=[None],
        )
        for index in range(count)
    ]


def _sync_session(monkeypatch, handle):
    def factory():
        session = MagicMock()
        session.get.side_effect = handle
        return session

    monkeypatch.setattr(executors.requests, "Session", factory)


def _response(url, status=200):
    response = Response()
    response.url = url
    response.status_code = status
    return response


def _run_sync():
    result = executors.execute_requests_packages(
        requests_data=_requests(4),
        request_method=RequestMethod.GET,
        max_concurrent_requests=2,
    )
    return result


@pytest.mark.parametrize("fail", [False, True], ids=["refill", "drain"])
def test_sync_scheduler_while_an_admitted_request_is_in_backoff(monkeypatch, fail):
    parked, release, checkpoint = (threading.Event() for _ in range(3))
    request_index = threading.local()
    attempts = Counter()
    lock = threading.Lock()

    def sleep(_seconds):
        if request_index.value == 0:
            parked.set()
            assert release.wait(2 * WATCHDOG)

    def handle(url, **_kwargs):
        index = int(url.rsplit("/", 1)[1])
        request_index.value = index
        with lock:
            attempts[index] += 1
            attempt = attempts[index]
        if index == 0 and attempt == 1:
            return _response(url, status=429)
        if fail and index == 1:
            assert parked.wait(WATCHDOG)
            raise ConnectionError("transport failure")
        if index == 3:
            checkpoint.set()
        return _response(url)

    _sync_session(monkeypatch, handle)
    monkeypatch.setattr(backoff_sync, "time", SimpleNamespace(sleep=sleep))
    if fail:
        original_wait = executors.wait

        def observe_failure(futures, **kwargs):
            done, pending = original_wait(futures, **kwargs)
            for future in pending:
                cancel = future.cancel

                def observe_cancel(cancel=cancel):
                    cancelled = cancel()
                    checkpoint.set()
                    return cancelled

                monkeypatch.setattr(future, "cancel", observe_cancel)
            return done, pending

        monkeypatch.setattr(executors, "wait", observe_failure)

    with ThreadPoolExecutor(max_workers=1) as caller:
        scheduled = caller.submit(_run_sync)
        try:
            assert parked.wait(WATCHDOG)
            assert checkpoint.wait(WATCHDOG)
            assert attempts[0] == 1
            assert not scheduled.done()
        finally:
            release.set()
        if fail:
            with pytest.raises(ConnectionError, match="transport failure"):
                scheduled.result(timeout=WATCHDOG)
            assert attempts == {0: 2, 1: 3}
        else:
            responses = scheduled.result(timeout=WATCHDOG)
            assert [response.url for response in responses] == [
                request.url for request in _requests(4)
            ]
            assert attempts == {0: 2, 1: 1, 2: 1, 3: 1}


def test_sync_scheduler_checks_retryable_status_after_exhaustion(monkeypatch):
    attempts = Counter()
    waits = []

    def handle(url, **_kwargs):
        attempts[url] += 1
        return _response(url, status=429 if url.endswith("/1") else 200)

    _sync_session(monkeypatch, handle)
    monkeypatch.setattr(backoff_sync, "time", SimpleNamespace(sleep=waits.append))

    with pytest.raises(HTTPError) as error:
        _run_sync()

    assert error.value.response.status_code == 429
    assert [attempts[request.url] for request in _requests(4)] == [1, 3, 1, 1]
    assert len(waits) == 2


@pytest.mark.asyncio
async def test_async_scheduler_cancels_owned_request_during_backoff(monkeypatch):
    parked, release = asyncio.Event(), asyncio.Event()
    cancelled = asyncio.Event()
    sessions = []
    session_factory = executors.aiohttp.ClientSession

    def track_session():
        session = session_factory()
        sessions.append(session)
        return session

    async def sleep(_seconds):
        parked.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    monkeypatch.setattr(executors.aiohttp, "ClientSession", track_session)
    monkeypatch.setattr(backoff_async, "asyncio", SimpleNamespace(sleep=sleep))

    with aioresponses() as http:
        http.get(f"{BASE_URL}/0", status=429)
        http.get(f"{BASE_URL}/1", payload={"index": 1})
        task = asyncio.create_task(
            executors.execute_requests_packages_async(
                requests_data=_requests(2),
                request_method=RequestMethod.GET,
                max_concurrent_requests=2,
            )
        )
        try:
            await asyncio.wait_for(parked.wait(), timeout=WATCHDOG)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=WATCHDOG)
            assert cancelled.is_set()
            assert len(sessions) == 1 and sessions[0].closed
            assert len(http.requests[("GET", URL(f"{BASE_URL}/0"))]) == 1
        finally:
            release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_async_scheduler_stops_admission_after_status_retries_exhaust(
    monkeypatch,
):
    waits = []

    async def sleep(seconds):
        waits.append(seconds)

    monkeypatch.setattr(backoff_async, "asyncio", SimpleNamespace(sleep=sleep))
    requests_data = _requests(2)
    url = f"{BASE_URL}/0?api_key=example-secret-key"
    requests_data[0] = replace(requests_data[0], url=url)
    with aioresponses() as http:
        http.get(url, status=429, repeat=True)
        with pytest.raises(ClientResponseError) as error:
            await asyncio.wait_for(
                executors.execute_requests_packages_async(
                    requests_data=requests_data,
                    request_method=RequestMethod.GET,
                    max_concurrent_requests=1,
                ),
                timeout=WATCHDOG,
            )

        assert error.value.status == 429
        assert "example-secret-key" not in str(error.value)
        assert "api_key=ex***ey" in str(error.value)
        assert list(http.requests) == [("GET", URL(url))]
        assert len(http.requests[("GET", URL(url))]) == 3
        assert len(waits) == 2
