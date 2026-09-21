import asyncio
import threading
from fractions import Fraction
from unittest.mock import MagicMock

import av
import pytest
from aiortc.mediastreams import MediaStreamError

from inference.core.interfaces.webrtc_worker.sources.file import VIDEO_FRAME_QUEUE_SIZE
from inference.core.interfaces.webrtc_worker.sources.rtsp import ThreadedRTSPTrack


class VideoContainer:
    def __init__(self, frames):
        self.frames = frames
        self.decoded = 0
        self.blocked_frame = threading.Event()
        self.closed = threading.Event()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.closed.set()

    def decode(self, **kwargs):
        assert kwargs == {"video": 0}
        for pts in range(self.frames):
            frame = av.VideoFrame(16, 16, "yuv420p")
            frame.pts = 1000 + pts * 3
            frame.time_base = Fraction(1, 90)
            self.decoded += 1
            if self.decoded == VIDEO_FRAME_QUEUE_SIZE + 1:
                self.blocked_frame.set()
            yield frame


def test_slow_consumer_bounds_decoder_and_preserves_every_frame(monkeypatch):
    container = VideoContainer(VIDEO_FRAME_QUEUE_SIZE + 10)
    open_source = MagicMock(return_value=container)
    monkeypatch.setattr(av, "open", open_source)
    track = ThreadedRTSPTrack("rtsp://camera.example/live")
    try:
        assert container.blocked_frame.wait(2)
        assert track._queue.qsize() == VIDEO_FRAME_QUEUE_SIZE
        # The producer must stay blocked even while the consumer is not called.
        assert not container.closed.wait(0.2)
        assert container.decoded == VIDEO_FRAME_QUEUE_SIZE + 1

        async def consume():
            frames = []
            while True:
                try:
                    frames.append(await asyncio.wait_for(track.recv(), 2))
                except MediaStreamError as exc:
                    assert str(exc) == "End of RTSP stream"
                    return frames

        frames = asyncio.run(consume())
        assert [f.pts for f in frames] == [i * 3 for i in range(container.frames)]
        assert all(f.time_base == Fraction(1, 90) for f in frames)
        assert container.closed.wait(2)
        open_source.assert_called_once_with(
            "rtsp://camera.example/live",
            format="rtsp",
            options={"rtsp_transport": "tcp", "rtsp_flags": "prefer_tcp"},
            timeout=2.0,
        )
    finally:
        track.stop()
        track._decode_thread.join(2)
    assert not track._decode_thread.is_alive()


def test_stop_unblocks_full_queue_and_closes_source(monkeypatch):
    container = VideoContainer(VIDEO_FRAME_QUEUE_SIZE + 10)
    monkeypatch.setattr(av, "open", lambda *args, **kwargs: container)
    track = ThreadedRTSPTrack("rtsp://camera.example/live")
    try:
        assert container.blocked_frame.wait(2)
    finally:
        track.stop()
        track._decode_thread.join(2)
    assert not track._decode_thread.is_alive()
    assert container.closed.is_set()
    assert container.decoded == VIDEO_FRAME_QUEUE_SIZE + 1
    with pytest.raises(MediaStreamError, match="RTSP track stopped"):
        asyncio.run(track.recv())


def test_decoder_failure_does_not_expose_source_credentials(monkeypatch):
    monkeypatch.setattr(
        av, "open", MagicMock(side_effect=RuntimeError("rtsp://user:secret@camera"))
    )
    track = ThreadedRTSPTrack("rtsp://user:secret@camera")
    try:
        assert track._finished.wait(2)
        with pytest.raises(MediaStreamError, match="^Failed to decode RTSP stream$"):
            asyncio.run(track.recv())
    finally:
        track.stop()
        track._decode_thread.join(2)


def test_real_decoder_preserves_frames_after_backpressure(monkeypatch, tmp_path):
    path = str(tmp_path / "tiny.mp4")
    with av.open(path, "w") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width = stream.height = 16
        stream.pix_fmt = "yuv420p"
        for index in range(VIDEO_FRAME_QUEUE_SIZE + 10):
            frame = av.VideoFrame(16, 16, "yuv420p")
            for plane in frame.planes:
                plane.update(bytes(plane.buffer_size))
            frame.pts = 10 + index * 3
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    with av.open(path) as container:
        expected = [frame.pts for frame in container.decode(video=0)]
    real_open = av.open
    monkeypatch.setattr(av, "open", lambda *args, **kwargs: real_open(path))
    track = ThreadedRTSPTrack("rtsp://local-test/live")

    async def consume():
        for _ in range(200):
            if track._queue.full():
                break
            await asyncio.sleep(0.01)
        assert track._queue.full()
        frames = [await asyncio.wait_for(track.recv(), 2) for _ in expected]
        with pytest.raises(MediaStreamError, match="End of RTSP stream"):
            await asyncio.wait_for(track.recv(), 2)
        return frames

    try:
        frames = asyncio.run(consume())
        assert [f.pts for f in frames] == [pts - expected[0] for pts in expected]
    finally:
        track.stop()
        track._decode_thread.join(2)
    assert not track._decode_thread.is_alive()
