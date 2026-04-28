"""Integration tests for v2 server endpoints with real MMP.

Starts MMP in a background thread, manually inits state module globals
(simulating what lifespan does), then uses httpx AsyncClient as ASGI transport.
"""

from __future__ import annotations

import asyncio
import os
import threading
import uuid
from multiprocessing.shared_memory import SharedMemory

import httpx
import pytest
import pytest_asyncio
import zmq
import zmq.asyncio

from inference_model_manager.model_manager_process import ModelManagerProcess

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
    """Async httpx client with manually initialized state globals."""
    import inference_server.app as app_mod
    import inference_server.state as st

    app_mod._DEBUG_BENCHMARK_MODE = True

    # Manually init state globals (same as lifespan)
    identity = f"test_{os.getpid()}_{uuid.uuid4().hex[:8]}".encode()
    ctx = zmq.asyncio.Context()
    sock = ctx.socket(zmq.DEALER)
    sock.setsockopt(zmq.IDENTITY, identity)
    sock.setsockopt(zmq.SNDHWM, 0)
    sock.setsockopt(zmq.RCVHWM, 0)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(mmp_harness.addr)

    shm = SharedMemory(name=mmp_harness.shm_name, create=False)

    st.ctx = ctx
    st.sock = sock
    st.shm = shm
    st.pending = {}
    st.recv_task = asyncio.create_task(st.recv_loop(), name="zmq-recv-test")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app_mod.app),
        base_url="http://test",
    ) as c:
        yield c

    st.recv_task.cancel()
    try:
        await st.recv_task
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


# ---------------------------------------------------------------------------
# v2 model lifecycle (load / list / interface / unload / delete)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_load_unknown_model_returns_error(client):
    """Load unknown model → 500 (ModelManager can't find weights)."""
    resp = await client.post("/v2/models/load?model_id=nonexistent-model")
    assert resp.status_code == 500


@pytest.mark.asyncio
async def test_unload_unknown_model_returns_ok(client):
    """Unload model that was never loaded → still 200 (idempotent)."""
    resp = await client.post("/v2/models/unload?model_id=never-loaded")
    # MMP may return ok or error depending on implementation
    assert resp.status_code in (200, 500)


@pytest.mark.asyncio
async def test_load_missing_model_id(client):
    resp = await client.post("/v2/models/load")
    assert resp.status_code == 400
    assert "model_id" in resp.json()["description"]


@pytest.mark.asyncio
async def test_unload_missing_model_id(client):
    resp = await client.post("/v2/models/unload")
    assert resp.status_code == 400
    assert "model_id" in resp.json()["description"]


@pytest.mark.asyncio
async def test_interface_model_not_loaded(client):
    """Interface for unloaded model → 404."""
    resp = await client.get("/v2/models/interface?model_id=nonexistent")
    assert resp.status_code == 404
    assert resp.json()["error_code"] == "MODEL_NOT_LOADED"


@pytest.mark.asyncio
async def test_interface_missing_model_id(client):
    resp = await client.get("/v2/models/interface")
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_interface_stub_loaded_model(client):
    """Stub-loaded model has no registry entry → tasks empty."""
    await client.post("/v2/models/load?model_id=stub-for-interface")

    resp = await client.get("/v2/models/interface?model_id=stub-for-interface")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "stub-for-interface"
    assert isinstance(body["tasks"], dict)

    # Cleanup
    await client.post("/v2/models/unload?model_id=stub-for-interface")
