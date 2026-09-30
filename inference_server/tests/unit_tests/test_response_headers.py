from __future__ import annotations

import asyncio
import contextvars
import json
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from inference_server.framework.dispatch import handle_model_inference_request
from inference_server.framework.entities import (
    ModelHandlerDescription,
    ModelInterfaceDescription,
)
from inference_server.framework.registry import _HANDLERS
from inference_server.gateway import ModelManagerGateway
from inference_server.legacy.bridge import (
    LegacyModelBridge,
    LoopBridge,
    Route,
    SyncLegacyBridge,
)
from inference_server.response_headers import (
    MAX_LOAD_DETAILS_BYTES,
    RequestModelUsage,
    ResponseHeadersMiddleware,
    build_model_response_headers,
    record_model_used,
    resolve_request_id,
    track_model_usage,
)


class _LoadingManager:
    """In-process manager double that really tracks what is loaded."""

    executor = None

    def __init__(self) -> None:
        self.loaded: set[str] = set()
        self.load_calls: list[str] = []

    def __contains__(self, key: str) -> bool:
        return key in self.loaded

    def is_healthy(self, key: str) -> bool:
        return True

    def load(self, key: str, api_key: str, **kwargs) -> None:
        time.sleep(0.01)
        self.load_calls.append(key)
        self.loaded.add(key)


def test_headers_without_models_report_no_cold_start():
    headers = build_model_response_headers(RequestModelUsage())

    assert headers == {"X-Model-Cold-Start": "false", "X-Model-Cold-Start-Count": "0"}


def test_headers_for_warm_model_carry_model_id_only():
    usage = RequestModelUsage()
    usage.record_model_used("b/2")
    usage.record_model_used("a/1")
    usage.record_model_used("a/1")

    headers = build_model_response_headers(usage)

    assert headers == {
        "X-Model-Cold-Start": "false",
        "X-Model-Cold-Start-Count": "0",
        "X-Model-Id": "a/1,b/2",
    }


def test_headers_for_cold_start_carry_load_time_and_details():
    usage = RequestModelUsage()
    usage.record_model_used("a/1")
    usage.record_model_used("b/2")
    usage.record_model_load("a/1", 1.5)
    usage.record_model_load("b/2", 0.25)

    headers = build_model_response_headers(usage)

    assert headers["X-Model-Cold-Start"] == "true"
    assert headers["X-Model-Cold-Start-Count"] == "2"
    assert headers["X-Model-Load-Time"] == "1.75"
    assert json.loads(headers["X-Model-Load-Details"]) == [
        {"m": "a/1", "t": 1.5},
        {"m": "b/2", "t": 0.25},
    ]
    assert headers["X-Model-Id"] == "a/1,b/2"


def test_oversized_load_details_are_dropped_but_totals_kept():
    usage = RequestModelUsage()
    for index in range(200):
        usage.record_model_load(f"workspace/model-{index}", 0.1)

    headers = build_model_response_headers(usage)

    assert len(json.dumps([{"m": "workspace/model-0", "t": 0.1}] * 200)) > (
        MAX_LOAD_DETAILS_BYTES
    )
    assert headers["X-Model-Cold-Start-Count"] == "200"
    assert float(headers["X-Model-Load-Time"]) == pytest.approx(20.0)
    assert "X-Model-Load-Details" not in headers


@pytest.mark.parametrize(
    "incoming",
    ["3f2b8c1e-5d4a-4b6e-9c7d-1a2b3c4d5e6f", "not-a-uuid", "0" * 32, "trace-abc/123"],
)
def test_incoming_request_id_is_echoed_unchanged(incoming, caplog):
    with caplog.at_level("DEBUG"):
        assert resolve_request_id(incoming) == incoming

    assert caplog.records == []


@pytest.mark.parametrize("incoming", [None, ""])
def test_missing_request_id_is_replaced_with_uuid4_hex(incoming):
    generated = resolve_request_id(incoming)

    assert len(generated) == 32
    assert UUID(generated).version == 4


def _app_with_middleware() -> FastAPI:
    app = FastAPI()

    @app.get("/uses-model")
    async def _uses_model():
        record_model_used("ds/1")
        return {"ok": True}

    @app.get("/plain")
    async def _plain():
        return {"ok": True}

    app.add_middleware(ResponseHeadersMiddleware)
    return app


def test_middleware_sets_engine_request_id_and_model_headers():
    client = TestClient(_app_with_middleware())

    response = client.get("/uses-model")

    assert response.headers["x-inference-engine"] == "inference-models"
    assert response.headers["x-model-id"] == "ds/1"
    assert response.headers["x-model-cold-start"] == "false"
    assert response.headers["x-model-cold-start-count"] == "0"
    assert "x-model-load-time" not in response.headers
    assert "x-model-load-details" not in response.headers
    assert UUID(response.headers["x-request-id"]).version == 4


def test_middleware_on_route_without_models_omits_model_id():
    client = TestClient(_app_with_middleware())

    response = client.get("/plain")

    assert "x-model-id" not in response.headers
    assert response.headers["x-model-cold-start"] == "false"
    assert response.headers["x-model-cold-start-count"] == "0"
    assert response.headers["x-inference-engine"] == "inference-models"


def test_middleware_echoes_incoming_request_id_and_exposes_it_to_route(caplog):
    app = FastAPI()

    @app.get("/echo")
    async def _echo(request: Request):
        return {"request_id": request.headers.get("x-request-id")}

    app.add_middleware(ResponseHeadersMiddleware)
    client = TestClient(app)
    incoming = str(uuid4())

    with caplog.at_level("WARNING"):
        echoed = client.get("/echo", headers={"X-Request-ID": incoming})
        custom = client.get("/echo", headers={"X-Request-ID": "bogus"})
    generated = client.get("/echo")

    assert echoed.headers["x-request-id"] == incoming
    assert echoed.json()["request_id"] == incoming
    assert custom.headers["x-request-id"] == "bogus"
    assert custom.json()["request_id"] == "bogus"
    assert UUID(generated.headers["x-request-id"]).version == 4
    assert generated.json()["request_id"] == generated.headers["x-request-id"]
    assert caplog.records == []


@pytest.mark.asyncio
async def test_gateway_records_load_only_for_the_request_that_triggered_it():
    manager = _LoadingManager()
    gateway = ModelManagerGateway(manager)

    with track_model_usage() as first:
        assert await gateway.ensure_loaded("ds/1") == ("model_ready",)
    with track_model_usage() as second:
        assert await gateway.ensure_loaded("ds/1") == ("model_ready",)

    _, first_loads = first.snapshot()
    _, second_loads = second.snapshot()
    assert [model_id for model_id, _ in first_loads] == ["ds/1"]
    assert first_loads[0][1] > 0
    assert second_loads == []
    assert manager.load_calls == ["ds/1"]


@pytest.mark.asyncio
async def test_gateway_load_outside_a_request_records_nothing():
    manager = _LoadingManager()
    gateway = ModelManagerGateway(manager)

    await gateway.ensure_loaded("ds/1")

    assert manager.load_calls == ["ds/1"]


@pytest.mark.asyncio
async def test_concurrent_joiner_of_a_pending_load_is_not_a_cold_start():
    manager = _LoadingManager()
    gateway = ModelManagerGateway(manager)

    with track_model_usage() as trigger:
        first = asyncio.ensure_future(gateway.ensure_loaded("ds/1"))
    await asyncio.sleep(0)
    with track_model_usage() as joiner:
        second = asyncio.ensure_future(gateway.ensure_loaded("ds/1"))
    await asyncio.gather(first, second)

    assert len(trigger.snapshot()[1]) == 1
    assert joiner.snapshot()[1] == []
    assert manager.load_calls == ["ds/1"]


@pytest.mark.asyncio
async def test_v2_dispatch_records_requested_model_id():
    proxy = MagicMock()
    proxy.ensure_loaded = AsyncMock(return_value=("error", 5))

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v2/models/infer",
        "query_string": b"model_id=ds/1",
        "headers": [(b"authorization", b"Bearer k1")],
    }

    async def _receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    interface = ModelInterfaceDescription(task="t", params={}, output_schema={})
    _HANDLERS[("fake-task", "infer")] = ModelHandlerDescription(
        input_parser=AsyncMock(return_value={"images": [b"x"], "params": {}}),
        handler=AsyncMock(),
        output_serializer=MagicMock(),
        interface_provider=lambda: interface,
    )
    try:
        with patch(
            "inference_server.framework.dispatch.stat_model_while_checking_auth",
            new=AsyncMock(return_value=("fake-task", "infer")),
        ), track_model_usage() as usage:
            await handle_model_inference_request(Request(scope, _receive), proxy)
    finally:
        del _HANDLERS[("fake-task", "infer")]

    assert usage.snapshot()[0] == {"ds/1"}


def test_workflow_thread_calls_record_into_the_request_collector():
    """Workflows call the bridge from worker threads through LoopBridge."""
    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()
    try:
        manager = _LoadingManager()
        bridge = LegacyModelBridge(ModelManagerGateway(manager))
        sync_bridge = SyncLegacyBridge(bridge, LoopBridge(loop))
        route = Route(
            model_id="ds/1",
            registry_id="ds/1",
            task_type="object-detection",
            action="infer",
        )
        with track_model_usage() as usage:
            # Workflows steps run on pool threads inside a copy of the
            # request context (run_in_threadpool / wrap_with_context_snapshot).
            context = contextvars.copy_context()
            worker = threading.Thread(
                target=lambda: context.run(sync_bridge.ensure_loaded, route, "k")
            )
            worker.start()
            worker.join(timeout=10)

        model_ids, loads = usage.snapshot()
        assert model_ids == {"ds/1"}
        assert [model_id for model_id, _ in loads] == ["ds/1"]
    finally:
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=5)
        loop.close()
