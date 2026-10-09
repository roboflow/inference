import asyncio
import inspect
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from streamvision.webrtc_worker import webrtc
from streamvision.webrtc_worker.entities import WebRTCWorkerRequest


@pytest.mark.parametrize(
    "ending", ["cancel", "closed", "failed", "setup_error", "close_error"]
)
def test_session_finishes_cleanup_before_returning(monkeypatch, caplog, ending):
    async def run():
        callbacks = {}
        processing_started = asyncio.Event()
        processing_stopped = asyncio.Event()
        transports_closed = asyncio.Event()
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

        async def close():
            peer.connectionState = "closed"
            result = callbacks["connectionstatechange"]()
            if inspect.isawaitable(result):
                await result
            # Simulate the transport work still pending after the closed event.
            await asyncio.sleep(0)
            transports_closed.set()
            if ending == "close_error":
                raise RuntimeError("close failed")

        peer.close = AsyncMock(side_effect=close)
        processor = MagicMock(_received_frames=52, data_channel=None)
        processor.close = AsyncMock()

        async def process():
            processing_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0)
                processing_stopped.set()

        processor.process_frames_data_only = process

        async def set_remote(*args):
            callbacks["track"](MagicMock())
            await processing_started.wait()
            if ending == "setup_error":
                raise ValueError("bad offer")

        peer.setRemoteDescription = AsyncMock(side_effect=set_remote)
        monkeypatch.setattr(webrtc, "RTCPeerConnectionWithLoop", lambda **kw: peer)
        monkeypatch.setattr(webrtc, "VideoFrameProcessor", lambda **kw: processor)
        monkeypatch.setattr(webrtc, "MediaRelay", MagicMock())
        monkeypatch.setattr(webrtc, "_wait_ice_complete", AsyncMock())
        host = SimpleNamespace(async_push_usage_payloads=AsyncMock())
        monkeypatch.setattr(webrtc, "get_webrtc_worker_host", lambda: host)

        async def send_answer(answer):
            if ending == "cancel":
                asyncio.current_task().cancel("watchdog timeout")
            else:
                peer.connectionState = "closed" if ending == "close_error" else ending
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
        )
        task = asyncio.create_task(
            webrtc.init_rtc_peer_connection_with_loop(request, send_answer=send_answer)
        )
        if ending == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=2)
        elif ending == "setup_error":
            with pytest.raises(ValueError, match="bad offer"):
                await asyncio.wait_for(task, timeout=2)
        elif ending == "close_error":
            with pytest.raises(RuntimeError, match="close failed"):
                await asyncio.wait_for(task, timeout=2)
        else:
            await asyncio.wait_for(task, timeout=2)

        assert processing_stopped.is_set()
        assert transports_closed.is_set()
        peer.close.assert_awaited_once()
        processor.track.stop.assert_called_once()
        processor.close.assert_awaited_once()
        host.async_push_usage_payloads.assert_awaited_once()

    with caplog.at_level(logging.INFO):
        asyncio.run(run())

    assert not any("FATAL" in record.message for record in caplog.records)
    assert any(record.levelno == logging.ERROR for record in caplog.records) == (
        ending == "failed"
    )


def test_cancelled_frame_finishes_before_workflow_executors_are_joined(monkeypatch):
    async def run():
        started = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()
        executor = ThreadPoolExecutor(max_workers=1)
        pipeline = SimpleNamespace(join=executor.shutdown)
        host = SimpleNamespace(init_workflow_pipeline=lambda **kw: pipeline)
        monkeypatch.setattr(webrtc, "get_webrtc_worker_host", lambda: host)
        monkeypatch.setattr(
            webrtc.VideoFrameProcessor, "_validate_output_fields", lambda *a: None
        )
        processor = webrtc.VideoFrameProcessor(
            asyncio_loop=loop,
            workflow_configuration=MagicMock(),
            api_key="test-key",
        )
        processor.video_upload_handler = SimpleNamespace(cleanup=AsyncMock())
        finished = []

        def process_frame(*args):
            loop.call_soon_threadsafe(started.set)
            assert release.wait(timeout=2)
            # Teardown must not shut down this pool before the frame finishes.
            executor.submit(lambda: finished.append(True)).result()
            return {}, None, []

        monkeypatch.setattr(webrtc, "process_frame", process_frame)
        task = asyncio.create_task(
            processor._process_frame_async(MagicMock(), frame_id=1)
        )
        try:
            await asyncio.wait_for(started.wait(), timeout=2)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            await processor.close()

        assert finished == [True]
        processor.video_upload_handler.cleanup.assert_awaited_once()
        assert all(not thread.is_alive() for thread in executor._threads)

    asyncio.run(run())


def test_connected_peer_can_be_cancelled_without_pending_transport_tasks(monkeypatch):
    async def run():
        client = webrtc.RTCPeerConnection(
            configuration=webrtc.RTCConfiguration(iceServers=[])
        )
        server = webrtc.RTCPeerConnectionWithLoop(
            asyncio_loop=asyncio.get_running_loop(),
            configuration=webrtc.RTCConfiguration(iceServers=[]),
        )
        processor = MagicMock(_received_frames=0, data_channel=None, track=None)
        processor.close = AsyncMock()
        monkeypatch.setattr(webrtc, "VideoFrameProcessor", lambda **kw: processor)
        monkeypatch.setattr(webrtc, "RTCPeerConnectionWithLoop", lambda **kw: server)
        monkeypatch.setattr(
            webrtc,
            "get_webrtc_worker_host",
            lambda: SimpleNamespace(async_push_usage_payloads=AsyncMock()),
        )
        connected = asyncio.Event()
        client.createDataChannel("inference")
        await client.setLocalDescription(await client.createOffer())
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
            webrtc_offer={"type": "offer", "sdp": client.localDescription.sdp},
            stream_output=[],
        )

        async def send_answer(answer):
            await client.setRemoteDescription(
                webrtc.RTCSessionDescription(
                    sdp=answer.answer.sdp, type=answer.answer.type
                )
            )

        task = asyncio.create_task(
            webrtc.init_rtc_peer_connection_with_loop(
                request,
                send_answer=send_answer,
                connection_established_callback=connected.set,
            )
        )
        try:
            await asyncio.wait_for(connected.wait(), timeout=5)
            task.cancel("watchdog timeout")
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=5)
            assert server.connectionState == "closed"
            processor.close.assert_awaited_once()
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await client.close()
            await server.close()
        await asyncio.sleep(0)
        assert asyncio.all_tasks() == {asyncio.current_task()}

    asyncio.run(run(), debug=True)
