import asyncio
from types import SimpleNamespace
from typing import List
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse, StreamingResponse
from starlette.routing import Route

from inference.core.interfaces.http.middlewares.concurrency_limit import (
    ConcurrencyLimitMiddleware,
)

LIMIT = 2
MESSAGE = "Server is at its concurrent request limit - retry request."


class Gate:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.entered = asyncio.Semaphore(0)
        self.release = asyncio.Event()


def build_app(gate: Gate, queue_timeout: float, limit: int = LIMIT):
    async def hold(request: Request):
        gate.entered.release()
        await gate.release.wait()
        return PlainTextResponse("held")

    async def quick(request: Request):
        return PlainTextResponse("quick")

    async def boom(request: Request):
        raise RuntimeError("boom")

    async def stream(request: Request):
        async def body():
            yield b"first"
            gate.entered.release()
            await gate.release.wait()
            yield b"second"

        return StreamingResponse(body())

    async def healthz(request: Request):
        return PlainTextResponse("ok")

    routes = [
        Route("/hold", hold),
        Route("/quick", quick),
        Route("/boom", boom),
        Route("/stream", stream),
        Route("/healthz", healthz),
    ]
    inner = Starlette(routes=routes)
    middleware = ConcurrencyLimitMiddleware(
        inner, max_concurrent_requests=limit, queue_timeout=queue_timeout
    )

    return middleware


def client_for(app) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)

    return httpx.AsyncClient(
        transport=transport, base_url="http://test", follow_redirects=True
    )


async def fill_slots(client: httpx.AsyncClient, gate: Gate, count: int, path="/hold"):
    tasks = [asyncio.ensure_future(client.get(path)) for _ in range(count)]
    for _ in range(count):
        await gate.entered.acquire()

    return tasks


async def assert_full_capacity(app, gate: Gate) -> None:
    gate.reset()
    app._queue_timeout = 0
    async with client_for(app) as client:
        tasks = await fill_slots(client, gate, LIMIT)
        rejected = await client.get("/quick")
        assert rejected.status_code == 503
        gate.release.set()
        responses = await asyncio.gather(*tasks)

    assert [r.status_code for r in responses] == [200] * LIMIT


async def run_asgi(app, path: str, root_path: str = "", scope_type: str = "http"):
    sent: List[dict] = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    scope = {
        "type": scope_type,
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": root_path + path,
        "raw_path": (root_path + path).encode(),
        "root_path": root_path,
        "query_string": b"",
        "headers": [],
        "server": ("test", 80),
        "client": ("test", 1),
    }
    await app(scope, receive, send)

    return sent


@pytest.mark.asyncio
async def test_request_over_limit_is_rejected_immediately_and_slot_reusable() -> None:
    gate = Gate()
    app = build_app(gate, queue_timeout=0)
    async with client_for(app) as client:
        tasks = await fill_slots(client, gate, LIMIT)

        rejected = await client.get("/quick")

        assert rejected.status_code == 503
        assert rejected.headers["Retry-After"] == "1"
        assert rejected.json() == {"message": MESSAGE}

        gate.release.set()
        await asyncio.gather(*tasks)
        accepted = await client.get("/quick")

    assert accepted.status_code == 200


@pytest.mark.asyncio
async def test_queued_request_gets_slot_when_one_frees() -> None:
    gate = Gate()
    app = build_app(gate, queue_timeout=5)
    async with client_for(app) as client:
        tasks = await fill_slots(client, gate, LIMIT)
        queued = asyncio.ensure_future(client.get("/quick"))
        await asyncio.sleep(0)
        assert not queued.done()

        gate.release.set()
        response = await queued
        await asyncio.gather(*tasks)

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_queued_request_timing_out_gets_503_and_leaks_no_permit() -> None:
    gate = Gate()
    app = build_app(gate, queue_timeout=0.05)
    async with client_for(app) as client:
        tasks = await fill_slots(client, gate, LIMIT)

        timed_out = await client.get("/quick")

        assert timed_out.status_code == 503
        gate.release.set()
        await asyncio.gather(*tasks)

    await assert_full_capacity(app, gate)


@pytest.mark.asyncio
async def test_cancelled_waiter_leaks_no_permit() -> None:
    gate = Gate()
    app = build_app(gate, queue_timeout=5)
    async with client_for(app) as client:
        tasks = await fill_slots(client, gate, LIMIT)
        waiter = asyncio.ensure_future(run_asgi(app, "/quick"))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not waiter.done()

        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter

        gate.release.set()
        await asyncio.gather(*tasks)

    await assert_full_capacity(app, gate)


@pytest.mark.asyncio
@pytest.mark.parametrize("yield_before_cancel", [False, True])
async def test_waiter_cancelled_right_after_slot_freed_leaks_no_permit(
    yield_before_cancel: bool,
) -> None:
    mw = build_app(Gate(), queue_timeout=5)
    for _ in range(LIMIT):
        await mw._semaphore.acquire()
    waiter = asyncio.ensure_future(mw._acquire_slot())
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    mw._semaphore.release()
    if yield_before_cancel:
        await asyncio.sleep(0)
    assert not waiter.done()

    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    assert mw._semaphore._value == 1


@pytest.mark.asyncio
async def test_queued_request_body_is_not_read_before_slot() -> None:
    gate = Gate()
    app = build_app(gate, queue_timeout=0.05)
    receive_calls: List[int] = []

    async def receive():
        receive_calls.append(1)
        return {"type": "http.request", "body": b"", "more_body": False}

    sent: List[dict] = []

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/quick",
        "root_path": "",
        "headers": [],
    }
    async with client_for(app) as client:
        tasks = await fill_slots(client, gate, LIMIT)

        await app(scope, receive, send)

        assert sent[0]["status"] == 503
        assert receive_calls == []
        gate.release.set()
        await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_options_request_bypasses_limit() -> None:
    gate = Gate()
    app = build_app(gate, queue_timeout=0)
    async with client_for(app) as client:
        tasks = await fill_slots(client, gate, LIMIT)

        response = await client.options("/quick")

        assert response.status_code != 503
        gate.release.set()
        await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_app_raising_releases_permit() -> None:
    gate = Gate()
    app = build_app(gate, queue_timeout=0)
    async with client_for(app) as client:
        for _ in range(LIMIT + 1):
            response = await client.get("/boom")
            assert response.status_code == 500

    await assert_full_capacity(app, gate)


@pytest.mark.asyncio
@pytest.mark.parametrize("root_path", ["", "/prefix"])
async def test_exempt_path_bypasses_limit(root_path: str) -> None:
    gate = Gate()
    app = build_app(gate, queue_timeout=0)
    async with client_for(app) as client:
        tasks = await fill_slots(client, gate, LIMIT)

        sent = await run_asgi(app, "/healthz", root_path=root_path)

        assert sent[0]["status"] == 200
        rejected = await client.get("/quick")
        assert rejected.status_code == 503
        gate.release.set()
        await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_exempt_match_is_exact_not_prefix() -> None:
    gate = Gate()
    app = build_app(gate, queue_timeout=0)
    async with client_for(app) as client:
        tasks = await fill_slots(client, gate, LIMIT)

        rejected = await client.get("/healthz/extra")

        assert rejected.status_code == 503
        gate.release.set()
        await asyncio.gather(*tasks)


@pytest.mark.asyncio
@pytest.mark.parametrize("scope_type", ["lifespan", "websocket"])
async def test_non_http_scope_passes_through(scope_type: str) -> None:
    seen: List[str] = []

    async def inner(scope, receive, send):
        seen.append(scope["type"])

    app = ConcurrencyLimitMiddleware(inner, max_concurrent_requests=1, queue_timeout=0)
    await app._semaphore.acquire()

    await app({"type": scope_type}, AsyncMock(), AsyncMock())

    assert seen == [scope_type]


@pytest.mark.asyncio
async def test_slot_held_until_streaming_body_fully_sent() -> None:
    gate = Gate()
    app = build_app(gate, queue_timeout=0, limit=1)
    async with client_for(app) as client:
        streaming = asyncio.ensure_future(client.get("/stream"))
        await gate.entered.acquire()

        rejected = await client.get("/quick")

        assert rejected.status_code == 503
        gate.release.set()
        response = await streaming
        accepted = await client.get("/quick")

    assert response.content == b"firstsecond"
    assert accepted.status_code == 200


def _build_interface(monkeypatch, limit: int):
    import inference.core.interfaces.http.http_api as http_api

    monkeypatch.setattr(
        http_api,
        "InferenceInstrumentator",
        lambda *args, **kwargs: SimpleNamespace(
            set_stream_manager_client=lambda client: None
        ),
    )
    monkeypatch.setattr(
        http_api.usage_collector, "async_push_usage_payloads", AsyncMock()
    )
    monkeypatch.setattr(http_api, "MAX_CONCURRENT_INFERENCE_REQUESTS", limit)
    model_manager = MagicMock()
    model_manager.pingback = None
    model_manager.num_errors = 0

    return http_api.HttpInterface(model_manager=model_manager)


@pytest.mark.parametrize("limit, expected_present", [(3, True), (0, False)])
def test_middleware_wired_only_when_limit_enabled(
    monkeypatch, limit: int, expected_present: bool
) -> None:
    interface = _build_interface(monkeypatch, limit=limit)

    classes = [m.cls for m in interface.app.user_middleware]

    assert (ConcurrencyLimitMiddleware in classes) is expected_present
    if expected_present:
        assert interface.app.user_middleware[0].cls is ConcurrencyLimitMiddleware
