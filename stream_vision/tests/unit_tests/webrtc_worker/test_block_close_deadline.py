"""The WebRTC worker bounds the workflow blocks' close by its shutdown reserve."""

import asyncio
import inspect
from types import SimpleNamespace
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from streamvision.webrtc_worker import webrtc
from streamvision.webrtc_worker.entities import WebRTCWorkerRequest


@pytest.mark.parametrize(
    "shutdown_reserve, expected",
    [
        (0, 1.0),
        (1, 1.0),
        (2, 1.0),
        (5, 4.0),
        (31, 30.0),
        (120, 30.0),
    ],
)
def test_deadline_keeps_a_second_of_the_reserve_back_within_bounds(
    shutdown_reserve: int, expected: float
) -> None:
    result = webrtc.derive_block_close_deadline_seconds(
        shutdown_reserve, default_deadline_seconds=30.0
    )

    assert result == expected


def test_deadline_never_exceeds_a_smaller_configured_default() -> None:
    result = webrtc.derive_block_close_deadline_seconds(
        10, default_deadline_seconds=3.0
    )

    assert result == 3.0


def _build_processor(monkeypatch, pipeline, **kwargs) -> webrtc.VideoFrameProcessor:
    host = SimpleNamespace(init_workflow_pipeline=lambda **kw: pipeline)
    monkeypatch.setattr(webrtc, "get_webrtc_worker_host", lambda: host)
    monkeypatch.setattr(
        webrtc.VideoFrameProcessor, "_validate_output_fields", lambda *a: None
    )

    return webrtc.VideoFrameProcessor(
        asyncio_loop=MagicMock(),
        workflow_configuration=MagicMock(),
        api_key="test-key",
        **kwargs,
    )


def test_processor_sets_the_deadline_on_the_pipeline(monkeypatch) -> None:
    pipeline = MagicMock()

    _build_processor(monkeypatch, pipeline, block_close_deadline_seconds=4.0)

    pipeline.set_block_close_deadline.assert_called_once_with(4.0)


def test_processor_without_a_deadline_leaves_the_pipeline_default(
    monkeypatch,
) -> None:
    pipeline = MagicMock()

    _build_processor(monkeypatch, pipeline)

    pipeline.set_block_close_deadline.assert_not_called()


def test_processor_tolerates_a_host_pipeline_without_the_setter(monkeypatch) -> None:
    pipeline = SimpleNamespace(join=MagicMock())

    processor = _build_processor(
        monkeypatch, pipeline, block_close_deadline_seconds=4.0
    )

    assert processor._inference_pipeline is pipeline


def _run_session(
    monkeypatch, processing_timeout: Optional[int], shutdown_reserve: int
) -> dict:
    """Run a session to its end and return the processor's constructor kwargs."""
    captured = {}

    async def run() -> None:
        callbacks = {}
        peer = MagicMock(
            connectionState="connected",
            iceConnectionState="completed",
            localDescription=SimpleNamespace(type="answer", sdp="v=0"),
        )
        peer.on.side_effect = lambda event: lambda callback: callbacks.setdefault(
            event, callback
        )
        peer.createAnswer = AsyncMock()
        peer.setLocalDescription = AsyncMock()
        peer.setRemoteDescription = AsyncMock()
        peer.close = AsyncMock()
        processor = MagicMock(_received_frames=0, data_channel=None, track=None)
        processor.close = AsyncMock()

        def build_processor(**kwargs):
            captured.update(kwargs)
            return processor

        monkeypatch.setattr(webrtc, "RTCPeerConnectionWithLoop", lambda **kw: peer)
        monkeypatch.setattr(webrtc, "VideoFrameProcessor", build_processor)
        monkeypatch.setattr(webrtc, "MediaRelay", MagicMock())
        monkeypatch.setattr(webrtc, "_wait_ice_complete", AsyncMock())
        host = SimpleNamespace(async_push_usage_payloads=AsyncMock())
        monkeypatch.setattr(webrtc, "get_webrtc_worker_host", lambda: host)

        async def send_answer(answer) -> None:
            peer.connectionState = "closed"
            result = callbacks["connectionstatechange"]()
            if inspect.isawaitable(result):
                await result

        request = WebRTCWorkerRequest(
            api_key="test-key",
            workflow_configuration={
                "type": "WorkflowConfiguration",
                "workflow_specification": {
                    "version": "1.0",
                    "steps": [],
                    "outputs": [],
                },
            },
            webrtc_offer={"type": "offer", "sdp": "v=0"},
            stream_output=[],
            processing_timeout=processing_timeout,
        )
        await asyncio.wait_for(
            webrtc.init_rtc_peer_connection_with_loop(
                request,
                send_answer=send_answer,
                shutdown_reserve=shutdown_reserve,
            ),
            timeout=5,
        )

    asyncio.run(run())

    return captured


def test_session_with_a_time_limit_derives_the_deadline_from_the_reserve(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        webrtc, "derive_block_close_deadline_seconds", lambda reserve: reserve - 1.5
    )

    captured = _run_session(monkeypatch, processing_timeout=600, shutdown_reserve=6)

    assert captured["termination_date"] is not None
    assert captured["block_close_deadline_seconds"] == 4.5


def test_session_without_a_time_limit_keeps_the_default_deadline(
    monkeypatch,
) -> None:
    captured = _run_session(monkeypatch, processing_timeout=None, shutdown_reserve=6)

    assert captured["termination_date"] is None
    assert captured["block_close_deadline_seconds"] is None
