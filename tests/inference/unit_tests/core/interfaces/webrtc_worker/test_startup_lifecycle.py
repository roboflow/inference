"""Startup errors and Modal transport/cleanup without deploying a Modal app."""

import __future__

import ast
import asyncio
import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from inference.core.interfaces.webrtc_worker import webrtc
from inference.core.interfaces.webrtc_worker.entities import (
    WebRTCWorkerRequest,
    WebRTCWorkerResult,
)


def _request():
    return WebRTCWorkerRequest(
        api_key="test-key",
        workflow_configuration={
            "type": "WorkflowConfiguration",
            "workflow_id": "example",
            "workspace_name": "workspace",
        },
        webrtc_offer={"type": "offer", "sdp": "v=0"},
        stream_output=[],
        processing_timeout=60,
    )


@pytest.mark.parametrize("async_transport", [False, True])
def test_initialization_error_is_delivered_logged_and_reraised(
    monkeypatch, async_transport
):
    error = AttributeError(
        "partially initialized module 'pandas' has no attribute 'core'"
    )
    monkeypatch.setattr(webrtc, "VideoFrameProcessor", MagicMock(side_effect=error))
    peer = MagicMock()
    monkeypatch.setattr(webrtc, "RTCPeerConnectionWithLoop", peer)
    logged = []

    def record_exception(message):
        import sys

        logged.append(sys.exc_info())

    monkeypatch.setattr(webrtc.logger, "exception", record_exception)
    send = AsyncMock() if async_transport else MagicMock()
    with pytest.raises(AttributeError) as caught:
        asyncio.run(
            webrtc.init_rtc_peer_connection_with_loop(_request(), send_answer=send)
        )

    assert caught.value is error
    assert logged[0][1] is error
    assert logged[0][2] is not None
    assert send.call_args.args[0].exception_type == "AttributeError"
    assert send.call_args.args[0].error_message == str(error)
    if async_transport:
        send.assert_awaited_once()
    else:
        send.assert_called_once()
    peer.assert_not_called()


def _modal_functions(watchdog):
    # Execute the real functions without Modal decorators/image registration.
    path = Path(webrtc.__file__).with_name("modal.py")
    tree = ast.parse(path.read_text())
    names = {"run_rtc_peer_connection_with_watchdog", "rtc_peer_connection_modal"}
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in names
    ]
    assert len(functions) == 2
    for node in functions:
        node.decorator_list = []
    namespace = {
        node.id: None
        for function in functions
        for node in ast.walk(function)
        if isinstance(node, ast.Name) and node.id.isupper()
    }
    namespace.update(
        asyncio=asyncio,
        datetime=datetime,
        logger=MagicMock(),
        modal=SimpleNamespace(
            exception=SimpleNamespace(
                InputCancellation=type("InputCancellation", (Exception,), {})
            )
        ),
        WebRTCWorkerResult=WebRTCWorkerResult,
        reuse_resolved_workspace_id_for_webrtc_request=lambda request: "workspace",
        sanitize_source_reference=lambda reference: reference,
        usage_collector=MagicMock(),
        PRELOADED_HF_MODELS={},
        docker_tag="test-image",
        Watchdog=MagicMock(return_value=watchdog),
    )
    exec(
        compile(
            ast.Module(body=functions, type_ignores=[]),
            str(path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    return namespace


@pytest.mark.parametrize("fails", [False, True])
def test_modal_awaits_queue_and_stops_once_outside_event_loop(monkeypatch, fails):
    watchdog = MagicMock(total_heartbeats=1, connection_established=True)

    def stop():
        with pytest.raises(RuntimeError, match="no running event loop"):
            asyncio.get_running_loop()

    watchdog.stop.side_effect = stop
    namespace = _modal_functions(watchdog)
    error = AttributeError("startup failed")
    answer = WebRTCWorkerResult(error_message=str(error) if fails else None)

    async def initialize(*, send_answer, **kwargs):
        await send_answer(answer)
        if fails:
            raise error

    monkeypatch.setattr(webrtc, "init_rtc_peer_connection_with_loop", initialize)
    queue = SimpleNamespace(
        put=MagicMock(side_effect=AssertionError("blocking queue call"))
    )
    queue.put.aio = AsyncMock()
    instance = SimpleNamespace(
        _model_manager=None,
        _function_call_number_on_container=0,
        _cold_start=False,
        _gpu=None,
        _container_startup_time_seconds=0,
    )
    call = namespace["rtc_peer_connection_modal"]
    if fails:
        with pytest.raises(AttributeError) as caught:
            call(instance, _request(), queue)
        assert caught.value is error
    else:
        call(instance, _request(), queue)
    queue.put.assert_not_called()
    queue.put.aio.assert_awaited_once_with(answer)
    watchdog.start.assert_called_once()
    watchdog.stop.assert_called_once()
    namespace["usage_collector"].record_usage.assert_called_once()
    namespace["usage_collector"].push_usage_payloads.assert_called_once()


@pytest.mark.parametrize("invalid", ["timeout", "offer"])
def test_modal_early_validation_awaits_queue_without_starting_watchdog(invalid):
    namespace = _modal_functions(MagicMock())
    request = _request()
    if invalid == "timeout":
        request.processing_timeout = 0
    else:
        request.webrtc_offer.sdp = ""
    queue = SimpleNamespace(
        put=MagicMock(side_effect=AssertionError("blocking queue call"))
    )
    queue.put.aio = AsyncMock()
    instance = SimpleNamespace(
        _model_manager=None,
        _function_call_number_on_container=0,
        _cold_start=False,
        _gpu=None,
        _container_startup_time_seconds=0,
    )
    namespace["rtc_peer_connection_modal"](instance, request, queue)
    queue.put.assert_not_called()
    queue.put.aio.assert_awaited_once()
    assert queue.put.aio.call_args.args[0].error_message
    namespace["Watchdog"].assert_not_called()
