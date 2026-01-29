"""Video file source for WebRTC - handles uploaded video files."""
import asyncio
import queue
import threading
from typing import Dict, Optional

import av
from aiortc.mediastreams import MediaStreamError, MediaStreamTrack
from av import VideoFrame

from inference.core import logger
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
                logger.error("[DECODE_WORKER] Queue full timeout at frame %d", frame_count)
                frame_queue.put({"error": f"Queue full timeout at frame {frame_count}"})
                return

        container.close()
    except Exception as e:
        logger.error("[DECODE_WORKER] Error at frame %d: %s", frame_count, e)
        frame_queue.put({"error": str(e)})
    finally:
        frame_queue.put(None)


class ThreadedVideoFileTrack(MediaStreamTrack):
    """Video track that decodes frames from a file in a background thread.

    Uses a dedicated thread with a queue to avoid deadlocks with the event loop.
    """

    kind = "video"

    def __init__(self, filepath: str, queue_size: int = 60):
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

    def __init__(self):
        self._chunks: Dict[int, bytes] = {}
        self._total_chunks: Optional[int] = None
        self._temp_file_path: Optional[str] = None
        self._state = VideoFileUploadState.IDLE
        self.upload_complete_event = asyncio.Event()

    @property
    def temp_file_path(self) -> Optional[str]:
        return self._temp_file_path

    def handle_chunk(self, chunk_index: int, total_chunks: int, data: bytes) -> None:
        """Handle a chunk. Auto-completes when all chunks received."""
         # TODO: we need to refactor this...
        if self._total_chunks is None:
            self._total_chunks = total_chunks
            self._state = VideoFileUploadState.UPLOADING

        self._chunks[chunk_index] = data

        if len(self._chunks) == total_chunks:
            self._write_to_temp_file()
            self._state = VideoFileUploadState.COMPLETE
            self.upload_complete_event.set()

    def _write_to_temp_file(self) -> None:
        """Reassemble chunks and write to temp file."""
        import tempfile
        # TODO: we need to refactor this...
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".mp4", delete=False) as f:
            for i in range(self._total_chunks):
                f.write(self._chunks[i])
            self._temp_file_path = f.name

        self._chunks.clear()

    def try_start_processing(self) -> Optional[str]:
        """Check if upload complete and transition to PROCESSING. Returns path or None."""
        if self._state == VideoFileUploadState.COMPLETE:
            self._state = VideoFileUploadState.PROCESSING
            return self._temp_file_path
        return None

    async def cleanup(self) -> None:
        """Clean up temp file."""
         # TODO: we need to refactor this...
        if self._temp_file_path:
            import os
            path = self._temp_file_path
            self._temp_file_path = None
            try:
                await asyncio.to_thread(os.unlink, path)
            except Exception:
                pass
