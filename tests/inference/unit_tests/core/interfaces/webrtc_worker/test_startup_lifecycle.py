"""Startup errors and Modal transport/cleanup without deploying a Modal app."""

import __future__

import ast
import asyncio
import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call

import pytest

import inference.core.interfaces.webrtc_worker as webrtc_worker_package
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
    path = Path(webrtc_worker_package.__file__).with_name("modal.py")
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

    def resolve_workspace(request):
        request.workspace_id = "workspace-1"
        request.workflow_configuration.workspace_name = "workspace-1"
        return "workspace-1"

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
        reuse_resolved_workspace_id_for_webrtc_request=resolve_workspace,
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
    namespace["rtc_peer_connection_modal"](instance, _request(), queue)
    assert namespace["logger"].exception.called is fails
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


_STARTED = datetime.datetime(2026, 1, 1, 12, 0, 0)


def _prepare_modal_call(monkeypatch, watchdog, **attributes):
    namespace = _modal_functions(watchdog)

    class _Clock(datetime.datetime):
        ticks = [_STARTED, _STARTED + datetime.timedelta(seconds=90)]

        @classmethod
        def now(cls, tz=None):
            return cls.ticks.pop(0)

    namespace["datetime"] = SimpleNamespace(
        datetime=_Clock, timedelta=datetime.timedelta
    )

    async def initialize(*, send_answer, **kwargs):
        await send_answer(WebRTCWorkerResult())

    monkeypatch.setattr(webrtc, "init_rtc_peer_connection_with_loop", initialize)
    queue = SimpleNamespace(put=MagicMock())
    queue.put.aio = AsyncMock()
    instance = SimpleNamespace(
        **{
            "_model_manager": None,
            "_function_call_number_on_container": 0,
            "_cold_start": False,
            "_gpu": None,
            "_container_startup_time_seconds": 0,
            **attributes,
        }
    )

    def run(request):
        namespace["rtc_peer_connection_modal"](instance, request, queue)

    return namespace, run


def _collector_calls(namespace):
    return [name for name, _, _ in namespace["usage_collector"].mock_calls]


@pytest.mark.parametrize("established", [True, False])
def test_modal_usage_record_payload_is_pinned(monkeypatch, established):
    watchdog = MagicMock(total_heartbeats=1, connection_established=established)
    namespace, run = _prepare_modal_call(monkeypatch, watchdog)
    request = _request()
    request.rtsp_url = "rtsp://camera/stream"
    request.is_preview = True
    request.requested_plan = "webrtc-gpu-large"
    request.requested_gpu = "a100"

    run(request)

    record_usage = namespace["usage_collector"].record_usage
    assert record_usage.call_args.args == ()
    kwargs = dict(record_usage.call_args.kwargs)
    duration = kwargs.pop("execution_duration")
    assert kwargs == {
        "source": "example",
        "category": "modal",
        "api_key": "test-key",
        "resource_id": "example",
        "resource_details": {
            "plan": "webrtc-gpu-large",
            "billable": True,
            "video_source": "rtsp",
            "is_preview": True,
        },
    }
    if established:
        assert duration == 90.0 and type(duration) is float
    else:
        assert duration == 0 and type(duration) is int
    assert _collector_calls(namespace) == ["record_usage", "push_usage_payloads"]
    namespace["usage_collector"].push_usage_payloads.assert_called_once_with()


@pytest.mark.parametrize(
    "attributes, offset",
    [
        ({}, datetime.timedelta(0)),
        (
            {"_cold_start": True, "_container_startup_time_seconds": 12},
            datetime.timedelta(seconds=-12),
        ),
    ],
)
def test_modal_cold_start_backdates_session_start_but_not_billed_duration(
    monkeypatch, attributes, offset
):
    watchdog = MagicMock(total_heartbeats=1, connection_established=True)
    namespace, run = _prepare_modal_call(monkeypatch, watchdog, **attributes)
    request = _request()

    run(request)

    assert request.processing_session_started == _STARTED + offset
    kwargs = namespace["usage_collector"].record_usage.call_args.kwargs
    assert kwargs["execution_duration"] == 90.0


@pytest.mark.parametrize(
    "realtime, expected",
    [(True, "realtime browser stream"), (False, "buffered browser stream")],
)
def test_modal_usage_video_source_for_browser_streams(monkeypatch, realtime, expected):
    watchdog = MagicMock(total_heartbeats=1, connection_established=True)
    namespace, run = _prepare_modal_call(monkeypatch, watchdog)
    request = _request()
    request.webrtc_realtime_processing = realtime

    run(request)

    details = namespace["usage_collector"].record_usage.call_args.kwargs[
        "resource_details"
    ]
    assert details["video_source"] == expected
    assert details["is_preview"] is False


@pytest.mark.parametrize(
    "established, expected_calls, duration, message",
    [
        (
            True,
            ["record_usage", "push_usage_payloads"],
            90.0,
            "WebRTC connection was established but no frames were processed. "
            "This typically indicates an invalid RTSP stream URL or corrupted "
            "video file.",
        ),
        (
            False,
            ["record_usage"],
            0,
            "WebRTC connection could not be established. No frames were processed.",
        ),
    ],
)
def test_modal_without_frames_raises_and_pushes_only_when_established(
    monkeypatch, established, expected_calls, duration, message
):
    watchdog = MagicMock(total_heartbeats=0, connection_established=established)
    namespace, run = _prepare_modal_call(monkeypatch, watchdog)

    with pytest.raises(Exception) as caught:
        run(_request())

    assert type(caught.value) is Exception
    assert str(caught.value) == message
    assert _collector_calls(namespace) == expected_calls
    kwargs = namespace["usage_collector"].record_usage.call_args.kwargs
    assert kwargs["execution_duration"] == duration


def test_modal_inline_specification_is_billed_under_its_resource_hash(monkeypatch):
    watchdog = MagicMock(total_heartbeats=1, connection_established=True)
    namespace, run = _prepare_modal_call(monkeypatch, watchdog)
    namespace["usage_collector"]._calculate_resource_hash.return_value = "hash-1"
    request = _request()
    request.workflow_configuration.workflow_id = None
    request.workflow_configuration.workflow_specification = {"version": "1.0"}

    run(request)

    kwargs = namespace["usage_collector"].record_usage.call_args.kwargs
    assert kwargs["source"] == "hash-1"
    assert kwargs["resource_id"] == "hash-1"
    namespace["usage_collector"]._calculate_resource_hash.assert_called_once_with(
        resource_details={"version": "1.0"}
    )


@pytest.mark.parametrize(
    "workflow_id, specification, expected",
    [("example", {"version": "1.0"}, "example"), (None, None, "unknown")],
)
def test_modal_workflow_id_takes_precedence_over_specification_hash(
    monkeypatch, workflow_id, specification, expected
):
    watchdog = MagicMock(total_heartbeats=1, connection_established=True)
    namespace, run = _prepare_modal_call(monkeypatch, watchdog)
    request = _request()
    request.workflow_configuration.workflow_id = workflow_id
    request.workflow_configuration.workflow_specification = specification

    run(request)

    kwargs = namespace["usage_collector"].record_usage.call_args.kwargs
    assert kwargs["source"] == kwargs["resource_id"] == expected
    namespace["usage_collector"]._calculate_resource_hash.assert_not_called()


def test_modal_method_logs_the_session_identity_before_reading_preloaded_models():
    namespace = _modal_functions(MagicMock())
    manager = MagicMock()
    manager.models.side_effect = RuntimeError("model lookup failed")
    instance = SimpleNamespace(
        _model_manager=manager,
        _function_call_number_on_container=0,
        _cold_start=False,
        _gpu="T4",
        _container_startup_time_seconds=0,
    )
    queue = SimpleNamespace(put=MagicMock())
    queue.put.aio = AsyncMock()

    with pytest.raises(RuntimeError, match="model lookup failed"):
        namespace["rtc_peer_connection_modal"](instance, _request(), queue)

    assert namespace["logger"].info.call_args_list == [
        call("*** Spawning %s:", "SimpleNamespace"),
        call("Running on %s", "T4"),
        call("Inference tag: %s", "test-image"),
        call("Workspace ID: %s", "workspace-1"),
        call("Workflow ID: %s", "example"),
    ]
    namespace["Watchdog"].assert_not_called()


def test_modal_session_wires_request_watchdog_and_model_manager(monkeypatch):
    watchdog = MagicMock(total_heartbeats=1, connection_established=True)
    namespace = _modal_functions(watchdog)
    namespace["WEBRTC_MODAL_WATCHDOG_TIMEMOUT"] = 7
    namespace["WEBRTC_SESSION_HEARTBEAT_URL"] = "https://heartbeat.example"
    captured = {}

    async def initialize(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(webrtc, "init_rtc_peer_connection_with_loop", initialize)
    manager = MagicMock()
    manager.models.return_value = {}
    instance = SimpleNamespace(
        _model_manager=manager,
        _function_call_number_on_container=0,
        _cold_start=False,
        _gpu=None,
        _container_startup_time_seconds=0,
    )
    queue = SimpleNamespace(put=MagicMock())
    queue.put.aio = AsyncMock()
    request = _request()
    request.session_id = "session-1"

    namespace["rtc_peer_connection_modal"](instance, request, queue)

    assert captured["webrtc_request"] is request
    assert captured["model_manager"] is manager
    assert captured["heartbeat_callback"] is watchdog.heartbeat
    assert (
        captured["connection_established_callback"]
        is watchdog.mark_connection_established
    )
    namespace["Watchdog"].assert_called_once_with(
        api_key="test-key",
        timeout_seconds=7,
        workspace_id="workspace-1",
        session_id="session-1",
        heartbeat_url="https://heartbeat.example",
    )
