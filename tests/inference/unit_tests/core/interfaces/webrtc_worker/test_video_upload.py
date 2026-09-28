import asyncio
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from inference.core.interfaces.webrtc_worker import webrtc
from inference.core.interfaces.webrtc_worker.entities import (
    VideoFileUploadState,
    WebRTCWorkerRequest,
)
from inference.core.interfaces.webrtc_worker.sources import file as video_file


@pytest.fixture
def handler(monkeypatch, tmp_path):
    monkeypatch.setattr(video_file.tempfile, "tempdir", str(tmp_path))
    instance = video_file.VideoFileUploadHandler(chunk_size=4)
    instance.max_bytes = 10
    instance.max_chunks = 3
    yield instance
    instance._cleanup()
    assert list(tmp_path.iterdir()) == []


def test_reuses_video_input_budget(monkeypatch):
    monkeypatch.setattr(video_file, "MAX_VIDEO_DOWNLOAD_SIZE_MB", 1)
    instance = video_file.VideoFileUploadHandler(chunk_size=webrtc.CHUNK_SIZE)
    assert instance.max_bytes == 1024 * 1024
    assert instance.max_chunks == 22

    monkeypatch.setattr(video_file, "MAX_VIDEO_DOWNLOAD_SIZE_MB", -1)
    instance = video_file.VideoFileUploadHandler(chunk_size=webrtc.CHUNK_SIZE)
    assert instance.max_bytes is None
    assert instance.max_chunks is None


@pytest.mark.parametrize(
    "index,total,data",
    [
        (0, 0, b"x"),
        (0, 4, b"x"),
        (-1, 3, b"x"),
        (3, 3, b"x"),
        (0, 3, b""),
        (0, 3, b"12345"),
    ],
)
def test_rejects_invalid_chunks_before_retaining_data(handler, index, total, data):
    with pytest.raises(ValueError):
        handler.handle_chunk(index, total, data)
    assert handler._chunks == {}
    assert handler.temp_file_path is None


def test_checks_total_duplicates_and_cumulative_bytes(handler):
    assert handler.handle_chunk(0, 3, b"1234")
    assert not handler.handle_chunk(0, 3, b"1234")
    with pytest.raises(ValueError, match="Conflicting"):
        handler.handle_chunk(0, 3, b"5678")
    with pytest.raises(ValueError, match="count changed"):
        handler.handle_chunk(1, 2, b"x")
    handler.handle_chunk(1, 3, b"5678")
    with pytest.raises(ValueError, match="size limit"):
        handler.handle_chunk(2, 3, b"901")
    assert handler._received_bytes == 8
    assert len(handler._chunks) == 2


def test_out_of_order_upload_at_limit_only_processes_once(handler):
    handler.handle_chunk(2, 3, b"90")
    handler.handle_chunk(0, 3, b"1234")
    handler.handle_chunk(1, 3, b"5678")
    assert handler._chunks == {}
    assert handler._state == VideoFileUploadState.COMPLETE
    with pytest.raises(ValueError, match="no longer accepting"):
        handler.handle_chunk(0, 3, b"xxxx")
    path = handler.try_start_processing()
    assert Path(path).read_bytes() == b"1234567890"
    assert handler.try_start_processing() is None
    with pytest.raises(ValueError, match="no longer accepting"):
        handler.handle_chunk(0, 1, b"new")


@pytest.mark.asyncio
async def test_cleanup_releases_partial_upload_and_is_terminal(handler):
    handler.handle_chunk(0, 3, b"1234")
    await handler.cleanup()
    await handler.cleanup()
    assert handler._received_bytes == 0
    assert handler._chunks == {}
    with pytest.raises(ValueError, match="no longer accepting"):
        handler.handle_chunk(1, 3, b"x")


@pytest.mark.asyncio
async def test_write_error_retains_file_for_cleanup(handler, monkeypatch, tmp_path):
    real_temporary_file = video_file.tempfile.NamedTemporaryFile

    def failing_file(*args, **kwargs):
        file = real_temporary_file(*args, **kwargs)
        file.write = MagicMock(side_effect=OSError("disk full"))
        return file

    monkeypatch.setattr(video_file.tempfile, "NamedTemporaryFile", failing_file)
    with pytest.raises(OSError, match="disk full"):
        handler.handle_chunk(0, 1, b"x")
    assert Path(handler.temp_file_path).exists()
    await handler.cleanup()
    assert handler.temp_file_path is None
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_unlink_failure_can_be_retried(handler, monkeypatch):
    handler.handle_chunk(0, 1, b"x")
    path = handler.temp_file_path
    with monkeypatch.context() as patch:
        patch.setattr(video_file.os, "unlink", MagicMock(side_effect=PermissionError()))
        await handler.cleanup()
    assert handler.temp_file_path == path
    await handler.cleanup()
    assert not Path(path).exists()


def test_concurrent_duplicate_completion_creates_one_file(handler, tmp_path):
    def complete():
        try:
            handler.handle_chunk(0, 1, b"x")
        except ValueError:
            return None
        return handler.try_start_processing()

    with ThreadPoolExecutor(max_workers=2) as pool:
        paths = list(pool.map(lambda _: complete(), range(2)))
    assert sum(path is not None for path in paths) == 1
    assert len(list(tmp_path.iterdir())) == 1


@pytest.mark.asyncio
async def test_cleanup_waits_for_inflight_write(handler, monkeypatch, tmp_path):
    writing = threading.Event()
    release = threading.Event()
    write = handler._write_to_temp_file

    def delayed_write():
        writing.set()
        assert release.wait(timeout=5)
        write()

    monkeypatch.setattr(handler, "_write_to_temp_file", delayed_write)
    upload = asyncio.create_task(asyncio.to_thread(handler.handle_chunk, 0, 1, b"x"))
    try:
        assert await asyncio.to_thread(writing.wait, 5)
        cleanup = asyncio.create_task(handler.cleanup())
        await asyncio.sleep(0)
    finally:
        release.set()
    await upload
    await cleanup
    assert handler._state == VideoFileUploadState.ERROR
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        b"bad",
        "not binary",
        struct.pack("<II", 2, 1) + b"x",
        b"x" * (webrtc.CHUNK_SIZE + 9),
        "queued_bytes",
        "queued_chunks",
    ],
)
async def test_upload_channel_rejects_invalid_messages_and_extra_channel(
    monkeypatch, tmp_path, message
):
    monkeypatch.setattr(video_file.tempfile, "tempdir", str(tmp_path))
    events = {}
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
    processor = MagicMock(
        video_upload_handler=None, track=None, heartbeat_callback=None
    )

    async def close():
        if processor.video_upload_handler:
            await processor.video_upload_handler.cleanup()

    processor.close = close
    monkeypatch.setattr(
        webrtc, "RTCPeerConnectionWithLoop", MagicMock(return_value=peer)
    )
    monkeypatch.setattr(
        webrtc, "VideoFrameProcessor", MagicMock(return_value=processor)
    )
    monkeypatch.setattr(webrtc, "_wait_ice_complete", AsyncMock())
    monkeypatch.setattr(
        webrtc.usage_collector, "async_push_usage_payloads", AsyncMock()
    )
    ready = asyncio.Event()
    request = WebRTCWorkerRequest(
        workflow_configuration={
            "type": "WorkflowConfiguration",
            "workflow_specification": {},
        },
        webrtc_offer={"sdp": "", "type": "offer"},
        stream_output=[],
        processing_timeout=None,
    )
    session = asyncio.create_task(
        webrtc.init_rtc_peer_connection_with_loop(request, lambda _: ready.set())
    )
    try:
        await asyncio.wait_for(ready.wait(), timeout=5)
        messages = {}
        channel = MagicMock(label="video_upload")
        channel.on.side_effect = lambda event: lambda callback: messages.setdefault(
            event, callback
        )
        events["datachannel"](channel)
        first_handler = processor.video_upload_handler
        extra = MagicMock(label="video_upload")
        events["datachannel"](extra)
        extra.close.assert_called_once()
        assert processor.video_upload_handler is first_handler
        if message in ("queued_bytes", "queued_chunks"):
            working = threading.Event()
            release = threading.Event()

            def delayed_chunk(*args):
                working.set()
                assert release.wait(timeout=5)
                return False

            monkeypatch.setattr(first_handler, "handle_chunk", delayed_chunk)
            if message == "queued_bytes":
                first_handler.max_bytes = 1
            else:
                first_handler.max_chunks = 1
            chunk = struct.pack("<II", 0, 1) + b"x"
            pending = asyncio.create_task(messages["message"](chunk))
            try:
                assert await asyncio.to_thread(working.wait, 5)
                await messages["message"](chunk)
            finally:
                release.set()
                await pending
        else:
            await messages["message"](message)
        await asyncio.wait_for(session, timeout=5)
        channel.close.assert_called_once()
        assert first_handler._state == VideoFileUploadState.ERROR
        assert first_handler._chunks == {}
        assert list(tmp_path.iterdir()) == []
    finally:
        if not session.done():
            session.cancel()
            await asyncio.gather(session, return_exceptions=True)
