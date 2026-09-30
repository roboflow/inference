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


def _session_harness(monkeypatch, watchdog):
    # Imported here: a module-level import would run before `inference.core` is set up.
    from streamvision.webrtc_worker import host as worker_host
    from streamvision.webrtc_worker import modal_session

    host = MagicMock()
    monkeypatch.setattr(worker_host, "_HOST", host)
    logger = MagicMock()
    monkeypatch.setattr(modal_session, "logger", logger)
    monkeypatch.setattr(
        modal_session,
        "modal",
        SimpleNamespace(
            exception=SimpleNamespace(
                InputCancellation=type("InputCancellation", (Exception,), {})
            )
        ),
    )
    watchdog_class = MagicMock(return_value=watchdog)
    monkeypatch.setattr(modal_session, "Watchdog", watchdog_class)

    return SimpleNamespace(host=host, logger=logger, watchdog_class=watchdog_class)


def _run_session(request, queue, **overrides):
    from streamvision.webrtc_worker import modal_session

    modal_session.run_modal_session(
        request,
        queue,
        **{
            "workflow_id": "example",
            "model_manager": None,
            "cold_start": False,
            "function_call_number_on_container": 1,
            "container_startup_time_seconds": 0,
            **overrides,
        },
    )


def _modal_method(run_modal_session):
    # Execute the real method without Modal decorators/image registration.
    path = Path(webrtc_worker_package.__file__).with_name("modal.py")
    tree = ast.parse(path.read_text())
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "rtc_peer_connection_modal"
    ]
    assert len(functions) == 1
    functions[0].decorator_list = []
    namespace = {
        node.id: None
        for node in ast.walk(functions[0])
        if isinstance(node, ast.Name) and node.id.isupper()
    }

    def resolve_workspace(request):
        request.workspace_id = "workspace-1"
        request.workflow_configuration.workspace_name = "workspace-1"
        return "workspace-1"

    namespace.update(
        logger=MagicMock(),
        reuse_resolved_workspace_id_for_webrtc_request=resolve_workspace,
        usage_collector=MagicMock(),
        PRELOADED_HF_MODELS={"owl": object()},
        docker_tag="test-image",
        run_modal_session=run_modal_session,
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


def _instance(**attributes):
    return SimpleNamespace(
        **{
            "_model_manager": None,
            "_function_call_number_on_container": 0,
            "_cold_start": False,
            "_gpu": None,
            "_container_startup_time_seconds": 0,
            **attributes,
        }
    )


@pytest.mark.parametrize("fails", [False, True])
def test_modal_awaits_queue_and_stops_once_outside_event_loop(monkeypatch, fails):
    watchdog = MagicMock(total_heartbeats=1, connection_established=True)

    def stop():
        with pytest.raises(RuntimeError, match="no running event loop"):
            asyncio.get_running_loop()

    watchdog.stop.side_effect = stop
    harness = _session_harness(monkeypatch, watchdog)
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
    _run_session(_request(), queue)
    assert harness.logger.exception.called is fails
    queue.put.assert_not_called()
    queue.put.aio.assert_awaited_once_with(answer)
    watchdog.start.assert_called_once()
    watchdog.stop.assert_called_once()
    harness.host.record_session_usage.assert_called_once()
    harness.host.push_usage_payloads.assert_called_once()


@pytest.mark.parametrize("invalid", ["timeout", "offer"])
def test_modal_early_validation_awaits_queue_without_starting_watchdog(
    monkeypatch, invalid
):
    harness = _session_harness(monkeypatch, MagicMock())
    request = _request()
    if invalid == "timeout":
        request.processing_timeout = 0
    else:
        request.webrtc_offer.sdp = ""
    queue = SimpleNamespace(
        put=MagicMock(side_effect=AssertionError("blocking queue call"))
    )
    queue.put.aio = AsyncMock()
    _run_session(request, queue)
    queue.put.assert_not_called()
    queue.put.aio.assert_awaited_once()
    assert queue.put.aio.call_args.args[0].error_message
    harness.watchdog_class.assert_not_called()


_STARTED = datetime.datetime(2026, 1, 1, 12, 0, 0)


def _prepare_session(monkeypatch, watchdog):
    harness = _session_harness(monkeypatch, watchdog)
    from streamvision.webrtc_worker import modal_session

    class _Clock(datetime.datetime):
        ticks = [_STARTED, _STARTED + datetime.timedelta(seconds=90)]

        @classmethod
        def now(cls, tz=None):
            return cls.ticks.pop(0)

    monkeypatch.setattr(
        modal_session,
        "datetime",
        SimpleNamespace(datetime=_Clock, timedelta=datetime.timedelta),
    )

    async def initialize(*, send_answer, **kwargs):
        await send_answer(WebRTCWorkerResult())

    monkeypatch.setattr(webrtc, "init_rtc_peer_connection_with_loop", initialize)
    queue = SimpleNamespace(put=MagicMock())
    queue.put.aio = AsyncMock()

    def run(request, **overrides):
        _run_session(request, queue, **overrides)

    return harness, run


def _host_calls(harness):
    return [name for name, _, _ in harness.host.mock_calls]


@pytest.mark.parametrize("established", [True, False])
def test_modal_session_reports_the_facts_of_the_session(monkeypatch, established):
    watchdog = MagicMock(total_heartbeats=1, connection_established=established)
    harness, run = _prepare_session(monkeypatch, watchdog)
    request = _request()
    request.rtsp_url = "rtsp://camera/stream"
    request.is_preview = True
    request.requested_plan = "webrtc-gpu-large"
    request.requested_gpu = "a100"

    run(request)

    record_session_usage = harness.host.record_session_usage
    assert record_session_usage.call_args.args == ()
    assert record_session_usage.call_args.kwargs == {
        "webrtc_request": request,
        "workflow_id": "example",
        "video_source": "rtsp",
        "session_started": _STARTED,
        "session_stopped": _STARTED + datetime.timedelta(seconds=90),
        "connection_established": established,
    }
    assert record_session_usage.call_args.kwargs["webrtc_request"] is request
    assert _host_calls(harness) == ["record_session_usage", "push_usage_payloads"]
    harness.host.push_usage_payloads.assert_called_once_with()


@pytest.mark.parametrize(
    "overrides, offset",
    [
        ({}, datetime.timedelta(0)),
        (
            {"cold_start": True, "container_startup_time_seconds": 12},
            datetime.timedelta(seconds=-12),
        ),
    ],
)
def test_modal_cold_start_backdates_session_start_but_not_billed_duration(
    monkeypatch, overrides, offset
):
    watchdog = MagicMock(total_heartbeats=1, connection_established=True)
    harness, run = _prepare_session(monkeypatch, watchdog)
    request = _request()

    run(request, **overrides)

    assert request.processing_session_started == _STARTED + offset
    kwargs = harness.host.record_session_usage.call_args.kwargs
    assert kwargs["session_started"] == _STARTED
    assert kwargs["session_stopped"] == _STARTED + datetime.timedelta(seconds=90)


@pytest.mark.parametrize(
    "realtime, expected",
    [(True, "realtime browser stream"), (False, "buffered browser stream")],
)
def test_modal_usage_video_source_for_browser_streams(monkeypatch, realtime, expected):
    watchdog = MagicMock(total_heartbeats=1, connection_established=True)
    harness, run = _prepare_session(monkeypatch, watchdog)
    request = _request()
    request.webrtc_realtime_processing = realtime

    run(request)

    kwargs = harness.host.record_session_usage.call_args.kwargs
    assert kwargs["video_source"] == expected


@pytest.mark.parametrize(
    "established, expected_calls, message",
    [
        (
            True,
            ["record_session_usage", "push_usage_payloads"],
            "WebRTC connection was established but no frames were processed. "
            "This typically indicates an invalid RTSP stream URL or corrupted "
            "video file.",
        ),
        (
            False,
            ["record_session_usage"],
            "WebRTC connection could not be established. No frames were processed.",
        ),
    ],
)
def test_modal_without_frames_raises_and_pushes_only_when_established(
    monkeypatch, established, expected_calls, message
):
    watchdog = MagicMock(total_heartbeats=0, connection_established=established)
    harness, run = _prepare_session(monkeypatch, watchdog)

    with pytest.raises(Exception) as caught:
        run(_request())

    assert type(caught.value) is Exception
    assert str(caught.value) == message
    assert _host_calls(harness) == expected_calls
    kwargs = harness.host.record_session_usage.call_args.kwargs
    assert kwargs["connection_established"] is established
    assert kwargs["session_started"] == _STARTED
    assert kwargs["session_stopped"] == _STARTED + datetime.timedelta(seconds=90)


def test_modal_session_bills_the_resource_identifier_it_was_given(monkeypatch):
    watchdog = MagicMock(total_heartbeats=1, connection_established=True)
    harness, run = _prepare_session(monkeypatch, watchdog)

    run(_request(), workflow_id="hash-1")

    kwargs = harness.host.record_session_usage.call_args.kwargs
    assert kwargs["workflow_id"] == "hash-1"


def test_modal_session_wires_request_watchdog_and_model_manager(monkeypatch):
    from streamvision.webrtc_worker import modal_session

    watchdog = MagicMock(total_heartbeats=1, connection_established=True)
    harness = _session_harness(monkeypatch, watchdog)
    monkeypatch.setattr(modal_session, "WEBRTC_MODAL_WATCHDOG_TIMEMOUT", 7)
    monkeypatch.setattr(
        modal_session, "WEBRTC_SESSION_HEARTBEAT_URL", "https://heartbeat.example"
    )
    captured = {}

    async def initialize(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(webrtc, "init_rtc_peer_connection_with_loop", initialize)
    manager = MagicMock()
    queue = SimpleNamespace(put=MagicMock())
    queue.put.aio = AsyncMock()
    request = _request()
    request.workspace_id = "workspace-1"
    request.session_id = "session-1"

    _run_session(request, queue, model_manager=manager)

    assert captured["webrtc_request"] is request
    assert captured["model_manager"] is manager
    assert captured["heartbeat_callback"] is watchdog.heartbeat
    assert (
        captured["connection_established_callback"]
        is watchdog.mark_connection_established
    )
    harness.watchdog_class.assert_called_once_with(
        api_key="test-key",
        timeout_seconds=7,
        workspace_id="workspace-1",
        session_id="session-1",
        heartbeat_url="https://heartbeat.example",
    )


def test_modal_method_resolves_the_session_identity_then_delegates():
    run = MagicMock()
    namespace = _modal_method(run)
    manager = MagicMock()
    manager.models.return_value = {}
    instance = _instance(
        _model_manager=manager,
        _cold_start=True,
        _gpu="T4",
        _container_startup_time_seconds=1.5,
    )
    request = _request()
    queue = SimpleNamespace(put=MagicMock())

    namespace["rtc_peer_connection_modal"](instance, request, queue)

    assert instance._function_call_number_on_container == 1
    assert len(namespace["logger"].info.call_args_list) == 10
    run.assert_called_once_with(
        request,
        queue,
        workflow_id="example",
        model_manager=manager,
        cold_start=True,
        function_call_number_on_container=1,
        container_startup_time_seconds=1.5,
    )


def test_modal_method_logs_the_session_identity_before_reading_preloaded_models():
    run = MagicMock()
    namespace = _modal_method(run)
    manager = MagicMock()
    manager.models.side_effect = RuntimeError("model lookup failed")
    instance = _instance(_model_manager=manager, _gpu="T4")
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
    run.assert_not_called()


def test_modal_inline_specification_is_billed_under_its_resource_hash():
    run = MagicMock()
    namespace = _modal_method(run)
    namespace["usage_collector"]._calculate_resource_hash.return_value = "hash-1"
    request = _request()
    request.workflow_configuration.workflow_id = None
    request.workflow_configuration.workflow_specification = {"version": "1.0"}

    namespace["rtc_peer_connection_modal"](_instance(), request, MagicMock())

    assert run.call_args.kwargs["workflow_id"] == "hash-1"
    namespace["usage_collector"]._calculate_resource_hash.assert_called_once_with(
        resource_details={"version": "1.0"}
    )


@pytest.mark.parametrize(
    "workflow_id, specification, expected",
    [("example", {"version": "1.0"}, "example"), (None, None, "unknown")],
)
def test_modal_workflow_id_takes_precedence_over_specification_hash(
    workflow_id, specification, expected
):
    run = MagicMock()
    namespace = _modal_method(run)
    request = _request()
    request.workflow_configuration.workflow_id = workflow_id
    request.workflow_configuration.workflow_specification = specification

    namespace["rtc_peer_connection_modal"](_instance(), request, MagicMock())

    assert run.call_args.kwargs["workflow_id"] == expected
    namespace["usage_collector"]._calculate_resource_hash.assert_not_called()
