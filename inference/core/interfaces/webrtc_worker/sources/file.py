"""Video file source for WebRTC - handles uploaded video files."""

import asyncio
import os
import queue
import tempfile
import threading
from typing import Dict, Optional

import av
from aiortc.mediastreams import MediaStreamError, MediaStreamTrack
from av import VideoFrame

from inference.core import logger
from inference.core.env import MAX_VIDEO_DOWNLOAD_SIZE_MB
from inference.core.interfaces.webrtc_worker.entities import VideoFileUploadState


def _decode_worker(filepath: str, frame_queue, stop_event):
    """Decode video frames in a separate thread and put them on a queue.

    We decode in a background thread to avoid deadlocks. PyAV (the video decoder)
    uses C code that can block while holding locks. If we decode directly in an
    async method using run_in_executor, PyAV's internal locks can conflict with
    Python's GIL and the asyncio event loop, causing the application to hang at
    random points during video processing.

    By running the decoder in its own dedicated thread with a queue, we completely
    isolate it from the async event loop and we decouple it from the logic; so
    we can create some backpressure
    """
    frame_count = 0
    try:
        container = av.open(filepath)
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        for frame in container.decode(stream):
            if stop_event.is_set():
                break
            try:
                frame_queue.put(frame, timeout=300)
                frame_count += 1
            except queue.Full:
                logger.error(
                    "[DECODE_WORKER] Queue full timeout at frame %d", frame_count
                )
                frame_queue.get_nowait()
                frame_queue.put_nowait(
                    {"error": f"Queue full timeout at frame {frame_count}"}
                )
                return

        container.close()
    except Exception as e:
        logger.error("[DECODE_WORKER] Error at frame %d: %s", frame_count, e)
        try:
            frame_queue.put_nowait({"error": str(e)})
        except queue.Full:
            frame_queue.get_nowait()
            frame_queue.put_nowait({"error": str(e)})
    finally:
        try:
            frame_queue.put(None, timeout=300)
        except queue.Full:
            frame_queue.get_nowait()
            frame_queue.put_nowait(None)


class ThreadedVideoFileTrack(MediaStreamTrack):
    """Video track that decodes frames from a file in a background thread.

    Uses a dedicated thread with a queue to avoid deadlocks with the event loop.
    """

    kind = "video"

    def __init__(self, filepath: str, queue_size: int = 60):
        # TODO: add parameter queue size in settings
        super().__init__()
        self._queue = queue.Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._decode_thread = threading.Thread(
            target=_decode_worker,
            args=(filepath, self._queue, self._stop_event),
            daemon=True,
        )
        self._decode_thread.start()

    async def recv(self) -> VideoFrame:
        while True:
            try:
                data = self._queue.get_nowait()
                break
            except queue.Empty:
                await asyncio.sleep(0.001)

        if data is None:
            self.stop()
            raise MediaStreamError("End of video file")
        if isinstance(data, dict):
            logger.error("[ThreadedVideoTrack] Decode error: %s", data)
            self.stop()
            raise MediaStreamError(data.get("error", "Unknown decode error"))

        return data

    def stop(self):
        super().stop()
        self._stop_event.set()


class VideoFileUploadHandler:
    """Handles video file uploads via data channel.

    Protocol: [chunk_index:u32][total_chunks:u32][payload]
    Auto-completes when all chunks received.
    """

    def __init__(self, chunk_size: int):
        self.chunk_size = chunk_size
        # Use the same clip budget as URL/base64 video input, including its opt-out.
        self.max_bytes = (
            MAX_VIDEO_DOWNLOAD_SIZE_MB * 1024 * 1024
            if MAX_VIDEO_DOWNLOAD_SIZE_MB >= 0
            else None
        )
        self.max_chunks = (
            (self.max_bytes + chunk_size - 1) // chunk_size
            if self.max_bytes is not None
            else None
        )
        self._lock = threading.Lock()
        self._chunks: Dict[int, bytes] = {}
        self._received_bytes = 0
        self._total_chunks: Optional[int] = None
        self._temp_file_path: Optional[str] = None
        self._state = VideoFileUploadState.IDLE
        self.upload_complete_event = asyncio.Event()

    @property
    def temp_file_path(self) -> Optional[str]:
        return self._temp_file_path

    def handle_chunk(self, chunk_index: int, total_chunks: int, data: bytes) -> bool:
        """Accept one chunk; return whether it adds new upload data."""
        with self._lock:
            if self._state not in (
                VideoFileUploadState.IDLE,
                VideoFileUploadState.UPLOADING,
            ):
                raise ValueError("Video upload is no longer accepting chunks")
            if total_chunks <= 0 or (
                self.max_chunks is not None and total_chunks > self.max_chunks
            ):
                raise ValueError("Invalid video upload chunk count")
            if not 0 <= chunk_index < total_chunks:
                raise ValueError("Invalid video upload chunk index")
            if not data or len(data) > self.chunk_size:
                raise ValueError("Invalid video upload chunk size")
            if self._total_chunks is not None and total_chunks != self._total_chunks:
                raise ValueError("Video upload chunk count changed")
            if chunk_index in self._chunks:
                if self._chunks[chunk_index] != data:
                    raise ValueError("Conflicting video upload chunk")
                return False
            if (
                self.max_bytes is not None
                and self._received_bytes + len(data) > self.max_bytes
            ):
                raise ValueError("Video upload exceeds the server video size limit")

            self._total_chunks = total_chunks
            self._state = VideoFileUploadState.UPLOADING
            self._chunks[chunk_index] = data
            self._received_bytes += len(data)
            if len(self._chunks) == self._total_chunks:
                self._write_to_temp_file()
                self._state = VideoFileUploadState.COMPLETE
                self.upload_complete_event.set()
            return True

    def _write_to_temp_file(self) -> None:
        """Reassemble chunks and write to temp file."""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".mp4", delete=False) as f:
            # Retain ownership even if writing or closing the file fails.
            self._temp_file_path = f.name
            for i in range(self._total_chunks):
                f.write(self._chunks[i])

        self._chunks.clear()
        self._received_bytes = 0

    def try_start_processing(self) -> Optional[str]:
        """Check if upload complete and transition to PROCESSING. Returns path or None."""
        with self._lock:
            if self._state == VideoFileUploadState.COMPLETE:
                self._state = VideoFileUploadState.PROCESSING
                return self._temp_file_path
        return None

    async def cleanup(self) -> None:
        """Wait for pending writes, release chunks and remove the owned file."""
        await asyncio.to_thread(self._cleanup)

    def _cleanup(self) -> None:
        with self._lock:
            self._state = VideoFileUploadState.ERROR
            self._chunks.clear()
            self._received_bytes = 0
            if self._temp_file_path:
                try:
                    os.unlink(self._temp_file_path)
                except FileNotFoundError:
                    pass
                except OSError:
                    logger.warning("Could not remove uploaded video", exc_info=True)
                    return
                self._temp_file_path = None
