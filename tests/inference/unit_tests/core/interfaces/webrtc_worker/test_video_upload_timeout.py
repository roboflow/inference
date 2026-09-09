import asyncio
import io
import struct
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import av
import pytest

from inference.core.interfaces.webrtc_worker import webrtc
from inference.core.interfaces.webrtc_worker.entities import WebRTCWorkerRequest
from inference.core.interfaces.webrtc_worker.sources import file as video_file


class UploadClock:
    """Drive only the upload timers; asyncio and executor callbacks run normally."""

    def __init__(self):
        self.now = 0
        self.timers = []
        self.loop = asyncio.get_running_loop()
        self.run_in_executor = self.loop.run_in_executor

    def time(self):
        return self.now

    def call_later(self, delay, callback):
        handle = asyncio.TimerHandle(self.now + delay, callback, (), self.loop)
        self.timers.append((handle, callback))
        return handle

    def advance(self, seconds, fire=True):
        self.now += seconds
        if fire:
            for handle, callback in self.timers:
                if not handle.cancelled() and handle.when() <= self.now:
                    callback()


@asynccontextmanager
async def upload_session(
    monkeypatch, tmp_path, timeout=180, video=False, realtime=False
):
    monkeypatch.setattr(video_file.tempfile, "tempdir", str(tmp_path))
    clock = UploadClock()
    monkeypatch.setattr(
        webrtc,
        "asyncio",
        SimpleNamespace(**{**vars(asyncio), "get_running_loop": lambda: clock}),
    )
    events, messages = {}, {}
    peer = MagicMock(connectionState="connected")
    peer.on.side_effect = lambda event: lambda callback: events.setdefault(
        event, callback
    )
    peer.setRemoteDescription = AsyncMock()
    peer.createAnswer = AsyncMock()
    peer.setLocalDescription = AsyncMock()
    peer.close = AsyncMock()
    peer.localDescription.type = "answer"
    peer.localDescription.sdp = ""
    processor = MagicMock(video_upload_handler=None, track=None, _file_processing=False)
    processor.process_frames_data_only = AsyncMock()

    async def close():
        if processor.video_upload_handler:
            await processor.video_upload_handler.cleanup()

    processor.close = close
    for name in ("VideoFrameProcessor", "VideoTransformTrackWithLoop"):
        monkeypatch.setattr(webrtc, name, MagicMock(return_value=processor))
    monkeypatch.setattr(
        webrtc, "RTCPeerConnectionWithLoop", MagicMock(return_value=peer)
    )
    monkeypatch.setattr(webrtc, "_wait_ice_complete", AsyncMock())
    monkeypatch.setattr(
        webrtc.usage_collector, "async_push_usage_payloads", AsyncMock()
    )
    monkeypatch.setattr(webrtc, "get_video_rotation", lambda _: 0)
    monkeypatch.setattr(webrtc, "get_video_fps", lambda _: 1)
    monkeypatch.setattr(webrtc, "MediaPlayer", MagicMock())
    monkeypatch.setattr(webrtc, "ThreadedVideoFileTrack", MagicMock())
    ready = asyncio.Event()
    request = WebRTCWorkerRequest(
        workflow_configuration={
            "type": "WorkflowConfiguration",
            "workflow_specification": {},
        },
        webrtc_offer={"sdp": "", "type": "offer"},
        stream_output=["image"] if video else [],
        processing_timeout=timeout,
        webrtc_realtime_processing=realtime,
    )
    session = asyncio.create_task(
        webrtc.init_rtc_peer_connection_with_loop(
            request, lambda _: ready.set(), shutdown_reserve=0
        )
    )
    try:
        await asyncio.wait_for(ready.wait(), timeout=5)
        channel = MagicMock(label="video_upload")
        channel.on.side_effect = lambda event: lambda callback: messages.setdefault(
            event, callback
        )
        events["datachannel"](channel)
        yield SimpleNamespace(
            clock=clock,
            session=session,
            channel=channel,
            processor=processor,
            peer=peer,
            send=messages["message"],
        )
    finally:
        if not session.done():
            session.cancel()
        await asyncio.gather(session, return_exceptions=True)
        assert all(handle.cancelled() for handle, _ in clock.timers)
        assert list(tmp_path.iterdir()) == []


def chunk(index, total=10, payload=b"x"):
    return struct.pack("<II", index, total) + payload


@pytest.mark.asyncio
async def test_only_new_validated_chunks_refresh_heartbeat(monkeypatch, tmp_path):
    async with upload_session(monkeypatch, tmp_path) as upload:
        await upload.send(b"\0")
        upload.channel.send.assert_called_once_with(b"\0")
        upload.processor.heartbeat_callback.assert_not_called()
        await upload.send(chunk(0))
        upload.processor.heartbeat_callback.assert_called_once()
        timer_count = len(upload.clock.timers)
        await upload.send(chunk(0))
        assert len(upload.clock.timers) == timer_count
        await upload.send(b"bad")
        await asyncio.wait_for(upload.session, timeout=5)
        upload.processor.heartbeat_callback.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("message", [b"\0", chunk(0)])
async def test_idle_upload_expires_despite_connectivity_traffic(
    monkeypatch, tmp_path, message
):
    async with upload_session(monkeypatch, tmp_path) as upload:
        await upload.send(chunk(0))
        idle_timeout = webrtc.WEBRTC_MODAL_WATCHDOG_TIMEMOUT
        upload.clock.advance(idle_timeout - 1)
        await upload.send(message)
        upload.clock.advance(1)
        await asyncio.wait_for(upload.session, timeout=5)
        upload.peer.close.assert_awaited_once()
        upload.processor.heartbeat_callback.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [120, None])
async def test_progress_never_extends_absolute_upload_budget(
    monkeypatch, tmp_path, timeout
):
    monkeypatch.setattr(webrtc, "WEBRTC_MODAL_FUNCTION_TIME_LIMIT", 120)
    async with upload_session(monkeypatch, tmp_path, timeout=timeout) as upload:
        await upload.send(chunk(0))
        for index in (1, 2):
            upload.clock.advance(50)
            await upload.send(chunk(index))
            assert not upload.session.done()
        upload.clock.advance(20)
        await asyncio.wait_for(upload.session, timeout=5)
        upload.processor.set_track.assert_not_called()
        assert upload.processor.heartbeat_callback.call_count == 3


@pytest.fixture
def tiny_mp4():
    output = io.BytesIO()
    with av.open(output, mode="w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=1)
        stream.width = stream.height = 16
        frame = av.VideoFrame(16, 16, "yuv420p")
        for plane in frame.planes:
            plane.update(bytes(plane.buffer_size))
        for packet in [*stream.encode(frame), *stream.encode()]:
            container.mux(packet)
    return output.getvalue()


@pytest.mark.asyncio
@pytest.mark.parametrize("video,realtime", [(False, False), (True, True)])
async def test_slow_valid_upload_survives_idle_windows_and_cancels_timer_on_completion(
    monkeypatch, tmp_path, tiny_mp4, video, realtime
):
    async with upload_session(
        monkeypatch, tmp_path, video=video, realtime=realtime
    ) as upload:
        size = len(tiny_mp4) // 4
        for index in range(4):
            if index:
                upload.clock.advance(50)
            payload = (
                tiny_mp4[index * size : (index + 1) * size]
                if index < 3
                else tiny_mp4[index * size :]
            )
            await upload.send(chunk(index, 4, payload))
        path = upload.processor.video_upload_handler.temp_file_path
        assert Path(path).read_bytes() == tiny_mp4
        assert upload.processor._file_processing
        upload.processor.set_track.assert_called_once()
        assert all(handle.cancelled() for handle, _ in upload.clock.timers)
        upload.clock.advance(60)
        await upload.send(b"\0")
        assert not upload.session.done()
        upload.peer.close.assert_not_awaited()
        assert upload.processor.heartbeat_callback.call_count == 4


@pytest.mark.asyncio
async def test_cancelled_session_cleans_up_incomplete_upload(monkeypatch, tmp_path):
    async with upload_session(monkeypatch, tmp_path) as upload:
        await upload.send(chunk(0))
        upload.session.cancel()
        with pytest.raises(asyncio.CancelledError):
            await upload.session
        assert upload.processor.video_upload_handler._chunks == {}
        upload.peer.close.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout,elapsed", [(5, 6), (180, 61)])
async def test_late_executor_completion_cannot_start_processing(
    monkeypatch, tmp_path, timeout, elapsed
):
    async with upload_session(monkeypatch, tmp_path, timeout=timeout) as upload:
        handler = upload.processor.video_upload_handler
        handle_chunk = handler.handle_chunk
        working, release = threading.Event(), threading.Event()

        def delayed_chunk(*args):
            working.set()
            assert release.wait(timeout=5)
            return handle_chunk(*args)

        monkeypatch.setattr(handler, "handle_chunk", delayed_chunk)
        pending = asyncio.create_task(upload.send(chunk(0, 1)))
        try:
            assert await asyncio.to_thread(working.wait, 5)
            # Even if the event loop delivers the executor callback before the timer.
            upload.clock.advance(elapsed, fire=False)
        finally:
            release.set()
            await pending
        await asyncio.wait_for(upload.session, timeout=5)
        upload.processor.heartbeat_callback.assert_not_called()
        upload.processor.set_track.assert_not_called()
