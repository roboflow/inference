"""RTSP decoding with producer backpressure for non-realtime processing."""

import asyncio
import queue
import threading
from typing import Optional

import av
from aiortc.mediastreams import MediaStreamError, MediaStreamTrack
from av import VideoFrame

from inference.core.interfaces.webrtc_worker.sources.file import VIDEO_FRAME_QUEUE_SIZE


class ThreadedRTSPTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, url: str):
        super().__init__()
        self._queue = queue.Queue(maxsize=VIDEO_FRAME_QUEUE_SIZE)
        self._stop_event = threading.Event()
        self._finished = threading.Event()
        self._error: Optional[str] = None
        self._decode_thread = threading.Thread(
            target=self._decode, args=(url,), daemon=True
        )
        self._decode_thread.start()

    def _decode(self, url: str) -> None:
        try:
            with av.open(
                url,
                format="rtsp",
                options={"rtsp_transport": "tcp", "rtsp_flags": "prefer_tcp"},
                timeout=2.0,  # Same two-second RTSP open/read budget as MediaPlayer.
            ) as container:
                first_pts = None
                for frame in container.decode(video=0):
                    if self._stop_event.is_set():
                        break
                    if frame.pts is None:
                        raise ValueError("RTSP frame has no timestamp")
                    if first_pts is None:
                        first_pts = frame.pts
                    frame.pts -= first_pts
                    # Block the decoder itself; pending async puts would retain frames.
                    while not self._stop_event.is_set():
                        try:
                            self._queue.put(frame, timeout=0.1)
                            break
                        except queue.Full:
                            continue
                    if self._stop_event.is_set():
                        break
        except Exception:
            # FFmpeg errors can contain credentials from the source URL.
            self._error = "Failed to decode RTSP stream"
        finally:
            self._finished.set()

    async def recv(self) -> VideoFrame:
        while self.readyState == "live":
            try:
                return self._queue.get_nowait()
            except queue.Empty:
                if self._finished.is_set() and self._queue.empty():
                    self.stop()
                    raise MediaStreamError(self._error or "End of RTSP stream")
                await asyncio.sleep(0.001)
        raise MediaStreamError("RTSP track stopped")

    def stop(self) -> None:
        super().stop()
        self._stop_event.set()
