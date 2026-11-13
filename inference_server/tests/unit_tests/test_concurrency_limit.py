import asyncio
import importlib
from typing import List

import httpx
import pytest
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route

from inference_server import configuration
from inference_server.middlewares.concurrency_limit import ConcurrencyLimitMiddleware

LIMIT = 2
MESSAGE = "Server is at its concurrent request limit - retry request."


class Gate:
    def __init__(self) -> None:
        self.entered = asyncio.Semaphore(0)
        self.release = asyncio.Event()


def build_app(gate: Gate, queue_timeout: float) -> ConcurrencyLimitMiddleware:
    async def hold(request):
        gate.entered.release()
        await gate.release.wait()
        return PlainTextResponse("held")

    async def quick(request):
        return PlainTextResponse("quick")

    async def boom(request):
        raise RuntimeError("boom")

    inner = Starlette(
        routes=[
            Route("/hold", hold),
            Route("/quick", quick),
            Route("/boom", boom),
            Route("/healthz", quick),
        ]
    )

    return ConcurrencyLimitMiddleware(
        inner, max_concurrent_requests=LIMIT, queue_timeout=queue_timeout
    )


def client_for(app) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)

    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def fill_slots(client: httpx.AsyncClient, gate: Gate) -> list:
    tasks = [asyncio.ensure_future(client.get("/hold")) for _ in range(LIMIT)]
    for _ in range(LIMIT):
        await gate.entered.acquire()

    return tasks


async def assert_full_capacity(app: ConcurrencyLimitMiddleware) -> None:
    assert app._semaphore._value == LIMIT


@pytest.mark.asyncio
async def test_over_limit_rejected_immediately_and_slot_reusable() -> None:
    gate = Gate()
    app = build_app(gate, queue_timeout=0)
    async with client_for(app) as client:
        tasks = await fill_slots(client, gate)

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
        tasks = await fill_slots(client, gate)
        queued = asyncio.ensure_future(client.get("/quick"))
        await asyncio.sleep(0)
        assert not queued.done()

        gate.release.set()
        response = await queued
        await asyncio.gather(*tasks)

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_queued_request_timing_out_gets_503_and_keeps_capacity() -> None:
    gate = Gate()
    app = build_app(gate, queue_timeout=0.05)
    async with client_for(app) as client:
        tasks = await fill_slots(client, gate)

        timed_out = await client.get("/quick")

        assert timed_out.status_code == 503
        gate.release.set()
        await asyncio.gather(*tasks)

    await assert_full_capacity(app)


@pytest.mark.asyncio
async def test_waiter_cancelled_right_after_slot_freed_leaks_no_permit() -> None:
    app = build_app(Gate(), queue_timeout=5)
    for _ in range(LIMIT):
        await app._semaphore.acquire()
    waiter = asyncio.ensure_future(app._acquire_slot())
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    app._semaphore.release()
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    assert app._semaphore._value == 1


@pytest.mark.asyncio
async def test_app_raising_releases_slot() -> None:
    app = build_app(Gate(), queue_timeout=0)
    async with client_for(app) as client:
        for _ in range(LIMIT + 1):
            response = await client.get("/boom")
            assert response.status_code == 500

    await assert_full_capacity(app)


@pytest.mark.asyncio
async def test_exempt_path_and_options_pass_while_full() -> None:
    gate = Gate()
    app = build_app(gate, queue_timeout=0)
    async with client_for(app) as client:
        tasks = await fill_slots(client, gate)

        health = await client.get("/healthz")
        preflight = await client.options("/quick")

        assert health.status_code == 200
        assert preflight.status_code != 503
        gate.release.set()
        await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_queued_request_body_is_not_read_before_slot() -> None:
    gate = Gate()
    app = build_app(gate, queue_timeout=0.05)
    receive_calls: List[int] = []
    sent: List[dict] = []

    async def receive():
        receive_calls.append(1)
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "method": "POST", "path": "/quick", "headers": []}
    async with client_for(app) as client:
        tasks = await fill_slots(client, gate)

        await app(scope, receive, send)

        assert sent[0]["status"] == 503
        assert receive_calls == []
        gate.release.set()
        await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_non_http_scope_passes_through() -> None:
    seen: List[str] = []

    async def inner(scope, receive, send):
        seen.append(scope["type"])

    app = ConcurrencyLimitMiddleware(inner, max_concurrent_requests=1, queue_timeout=0)
    await app._semaphore.acquire()

    await app({"type": "lifespan"}, None, None)

    assert seen == ["lifespan"]


@pytest.fixture
def reloaded_app(monkeypatch):
    import inference_server.app as app_mod

    monkeypatch.setattr(
        "inference_model_manager.watchdogs.start_enabled_watchdogs", lambda: []
    )
    monkeypatch.delenv("INFERENCE_PRELOAD_MODELS", raising=False)

    def _reload(**overrides):
        for name, value in overrides.items():
            monkeypatch.setattr(configuration, name, value)
        return importlib.reload(app_mod)

    yield _reload
    monkeypatch.undo()
    importlib.reload(app_mod)


def test_limiter_wired_outside_body_readers_only_when_enabled(reloaded_app) -> None:
    names = [m.cls.__name__ for m in reloaded_app().app.user_middleware]
    assert "ConcurrencyLimitMiddleware" not in names

    names = [
        m.cls.__name__
        for m in reloaded_app(MAX_CONCURRENT_INFERENCE_REQUESTS=3).app.user_middleware
    ]

    assert names.index("ConcurrencyLimitMiddleware") == (
        names.index("CorrelationIdMiddleware") - 1
    )
    assert names.index("ConcurrencyLimitMiddleware") < names.index("_AuthMiddleware")
