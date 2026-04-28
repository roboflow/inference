"""Integration tests for v2 server endpoints with real MMP.

Starts MMP in a background thread, manually inits app ZMQ + SHM globals
(simulating what lifespan does), then uses httpx AsyncClient as ASGI transport.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import uuid

import httpx
import pytest
import pytest_asyncio
import zmq
import zmq.asyncio

from inference_model_manager.model_manager_process import ModelManagerProcess
from multiprocessing.shared_memory import SharedMemory


_TEST_INPUT_MB = 0.1
_TIMEOUT_S = 5.0


class _MMPHarness:
    def __init__(self, n_slots: int = 8):
        self._ready = threading.Event()
        self.mmp = ModelManagerProcess(
            n_slots=n_slots,
            input_mb=_TEST_INPUT_MB,
        )
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="mmp-test"
        )
        self._thread.start()
        assert self._ready.wait(timeout=_TIMEOUT_S), "MMP did not start"
        self.addr = self.mmp.bound_addr
        self.shm_name = self.mmp._pool.name

    def _run_loop(self) -> None:
        asyncio.run(self.mmp.run(addr="tcp://127.0.0.1:0", ready_event=self._ready))

    def teardown(self) -> None:
        self.mmp.stop()
        self._thread.join(timeout=_TIMEOUT_S)


@pytest.fixture(scope="module")
def mmp_harness():
    h = _MMPHarness()
    yield h
    h.teardown()


@pytest_asyncio.fixture()
async def client(mmp_harness):
    """Async httpx client with manually initialized app globals."""
    import inference_server.app as app_mod

    app_mod._DEBUG_BENCHMARK_MODE = True
    app_mod._MMP_ADDR = mmp_harness.addr
    app_mod._SHM_NAME = mmp_harness.shm_name
    app_mod._SHM_DATA_SIZE = int(_TEST_INPUT_MB * 1024 * 1024)

    # Manually init globals (same as lifespan)
    identity = f"test_{os.getpid()}_{uuid.uuid4().hex[:8]}".encode()
    ctx = zmq.asyncio.Context()
    sock = ctx.socket(zmq.DEALER)
    sock.setsockopt(zmq.IDENTITY, identity)
    sock.setsockopt(zmq.SNDHWM, 0)
    sock.setsockopt(zmq.RCVHWM, 0)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(mmp_harness.addr)

    shm = SharedMemory(name=mmp_harness.shm_name, create=False)

    app_mod._ctx = ctx
    app_mod._sock = sock
    app_mod._shm = shm
    app_mod._pending = {}
    app_mod._recv_task = asyncio.create_task(app_mod._recv_loop(), name="zmq-recv-test")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app_mod.app),
        base_url="http://test",
    ) as c:
        yield c

    app_mod._recv_task.cancel()
    try:
        await app_mod._recv_task
    except asyncio.CancelledError:
        pass
    sock.close()
    ctx.term()
    shm.close()


@pytest.mark.asyncio
async def test_health_returns_ok(client):
    resp = await client.get("/v2/server/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_ready_no_preload(client):
    resp = await client.get("/v2/server/ready")
    assert resp.status_code == 200
    assert resp.json()["ready"] is True


@pytest.mark.asyncio
async def test_info_returns_models(client):
    resp = await client.get("/v2/server/info")
    assert resp.status_code == 200
    body = resp.json()
    assert "server" in body
    assert "models_loaded" in body
    assert body["models_loaded"] >= 0


@pytest.mark.asyncio
async def test_metrics_returns_json(client):
    resp = await client.get("/v2/server/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, dict)
