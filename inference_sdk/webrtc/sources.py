"""Stream source abstractions for WebRTC sessions."""

from __future__ import annotations

import asyncio
import base64
import json
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import cv2
import numpy as np
from aiortc import RTCDataChannel, RTCPeerConnection, VideoStreamTrack
from aiortc.mediastreams import MediaStreamError
from aiortc.rtcrtpreceiver import RemoteStreamTrack
from av import VideoFrame


class StreamSource(ABC):
    """Base abstraction for video sources."""

    @abstractmethod
    async def configure_peer_connection(self, pc: RTCPeerConnection) -> None:
        """Configure the peer connection (tracks, transceivers, etc.)."""

    @abstractmethod
    def get_initialization_params(self) -> Dict[str, Any]:
        """Return initialization payload parameters specific to the source."""

    async def cleanup(self) -> None:
        """Release any resources held by the source."""

    def requires_data_channel(self) -> bool:
        """Whether the source streams frames via a data channel."""

        return False

    def start_data_channel(
        self, channel: RTCDataChannel, stop_event: threading.Event, loop: asyncio.AbstractEventLoop
    ) -> None:
        """Begin streaming frames over data channel (if required)."""

        raise NotImplementedError("Source does not support data channel streaming")


class _WebcamVideoTrack(VideoStreamTrack):
    def __init__(self, device_id: int, resolution: Optional[tuple[int, int]]):
        super().__init__()
        self._cap = cv2.VideoCapture(device_id)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open webcam device {device_id}")
        if resolution:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

    async def recv(self) -> VideoFrame:  # type: ignore[override]
        ret, frame = self._cap.read()
        if not ret or frame is None:
            raise RuntimeError("Failed to read from webcam")
        video_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        video_frame.pts, video_frame.time_base = await self.next_timestamp()
        return video_frame

    def get_declared_fps(self) -> Optional[float]:
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        return float(fps) if fps and fps > 0 else None

    def release(self) -> None:
        self._cap.release()


class WebcamSource(StreamSource):
    def __init__(self, device_id: int = 0, resolution: Optional[tuple[int, int]] = None):
        self._device_id = device_id
        self._resolution = resolution
        self._track: Optional[_WebcamVideoTrack] = None

    async def configure_peer_connection(self, pc: RTCPeerConnection) -> None:
        self._track = _WebcamVideoTrack(self._device_id, self._resolution)
        pc.addTrack(self._track)

    def get_initialization_params(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if self._track:
            fps = self._track.get_declared_fps()
            if fps:
                params["declared_fps"] = fps
        return params

    async def cleanup(self) -> None:
        if self._track:
            self._track.release()


class RTSPSource(StreamSource):
    def __init__(self, url: str):
        if not url.startswith(("rtsp://", "rtsps://")):
            raise ValueError("Invalid RTSP URL")
        self._url = url

    async def configure_peer_connection(self, pc: RTCPeerConnection) -> None:
        pc.addTransceiver("video", direction="recvonly")

    def get_initialization_params(self) -> Dict[str, Any]:
        return {"rtsp_url": self._url}


class _VideoFileTrack(VideoStreamTrack):
    def __init__(self, path: str):
        super().__init__()
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video file: {path}")
        self._fps = self._cap.get(cv2.CAP_PROP_FPS)

    async def recv(self) -> VideoFrame:  # type: ignore[override]
        ret, frame = self._cap.read()
        if not ret or frame is None:
            raise MediaStreamError("End of video file")
        video_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        video_frame.pts, video_frame.time_base = await self.next_timestamp()
        return video_frame

    def get_declared_fps(self) -> Optional[float]:
        if self._fps and self._fps > 0:
            return float(self._fps)
        return None

    def release(self) -> None:
        self._cap.release()


class VideoFileSource(StreamSource):
    def __init__(self, path: str):
        self._path = path
        self._track: Optional[_VideoFileTrack] = None

    async def configure_peer_connection(self, pc: RTCPeerConnection) -> None:
        self._track = _VideoFileTrack(self._path)
        pc.addTrack(self._track)

    def get_initialization_params(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {"video_source": "file"}
        if self._track:
            fps = self._track.get_declared_fps()
            if fps:
                params["declared_fps"] = fps
        return params

    async def cleanup(self) -> None:
        if self._track:
            self._track.release()


class _ManualTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self._queue: "asyncio.Queue[Optional[np.ndarray]]" = asyncio.Queue(maxsize=10)

    async def recv(self) -> VideoFrame:  # type: ignore[override]
        frame = await self._queue.get()
        if frame is None:
            raise MediaStreamError("Manual track stopped")
        video_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        video_frame.pts, video_frame.time_base = await self.next_timestamp()
        return video_frame

    def queue_frame(self, frame: np.ndarray) -> None:
        try:
            self._queue.put_nowait(frame)
        except asyncio.QueueFull:
            # Drop oldest frame to keep latency bounded
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self._queue.put_nowait(frame)

    async def stop(self) -> None:
        await self._queue.put(None)


class ManualSource(StreamSource):
    def __init__(self):
        self._track = _ManualTrack()

    async def configure_peer_connection(self, pc: RTCPeerConnection) -> None:
        pc.addTrack(self._track)

    def get_initialization_params(self) -> Dict[str, Any]:
        return {"manual_mode": True}

    def send(self, frame: np.ndarray) -> None:
        self._track.queue_frame(frame)

    async def cleanup(self) -> None:
        await self._track.stop()


class DataChannelVideoSource(StreamSource):
    """Source that streams frames over a dedicated data channel."""

    def __init__(self, path: str, chunk_size: int = 16_000_000, image_format: str = "jpg", quality: int = 85):
        self._path = path
        self._chunk_size = chunk_size
        self._image_format = image_format
        self._quality = quality
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._frames_sent = 0
        self._chunks_sent = 0
        self._channel: Optional[RTCDataChannel] = None

    async def configure_peer_connection(self, pc: RTCPeerConnection) -> None:
        # No local media track is added. Server is notified via initialization params.
        return None

    def requires_data_channel(self) -> bool:
        return True

    def start_data_channel(
        self, channel: RTCDataChannel, stop_event: threading.Event, loop: asyncio.AbstractEventLoop
    ) -> None:
        self._stop_event = stop_event
        self._loop = loop
        self._channel = channel
        self._thread = threading.Thread(
            target=self._stream_frames,
            args=(channel,),
            name="DataChannelVideoSource",
            daemon=True,
        )
        self._thread.start()
    
    def send_eof_when_ready(self, frames_processed: int) -> None:
        """Send EOF after all frames have been processed."""
        if self._channel and self._channel.readyState == "open":
            if frames_processed >= self._frames_sent:
                print(f"All {self._frames_sent} frames processed, sending EOF...")
                self._send_via_loop(self._channel, json.dumps({"type": "frame_eof"}))
                print("EOF signal sent")

    def _send_via_loop(self, channel: RTCDataChannel, message: str) -> None:
        """Schedule data channel send on the event loop."""
        if self._loop and not self._loop.is_closed():
            asyncio.run_coroutine_threadsafe(
                self._async_send(channel, message), self._loop
            )

    async def _async_send(self, channel: RTCDataChannel, message: str) -> None:
        """Async wrapper for channel send."""
        try:
            channel.send(message)
        except Exception:
            pass

    def _stream_frames(self, channel: RTCDataChannel) -> None:
        cap = cv2.VideoCapture(self._path)
        if not cap.isOpened():
            self._send_via_loop(channel, json.dumps({"type": "frame_error", "error": "failed_to_open"}))
            return

        frame_id = 0
        try:
            sent_eof = False
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret or frame is None:
                    # Don't send EOF immediately - let all frames get processed first
                    # We'll send EOF after a delay to ensure all frames are queued
                    break

                frame_id += 1
                
                # Encode with quality parameter for JPEG
                if self._image_format == "jpg":
                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, self._quality]
                    success, buffer = cv2.imencode(f".{self._image_format}", frame, encode_params)
                else:
                    success, buffer = cv2.imencode(f".{self._image_format}", frame)
                
                if not success:
                    continue

                payload = base64.b64encode(buffer).decode("ascii")
                total_chunks = max(1, (len(payload) + self._chunk_size - 1) // self._chunk_size)

                for idx in range(total_chunks):
                    if self._stop_event.is_set() or channel.readyState != "open":
                        break
                    chunk = payload[idx * self._chunk_size : (idx + 1) * self._chunk_size]
                    message = {
                        "type": "frame_chunk",
                        "frame_id": frame_id,
                        "chunk_id": idx,
                        "chunks": total_chunks,
                        "encoding": self._image_format,
                        "width": frame.shape[1],
                        "height": frame.shape[0],
                        "data": chunk,
                    }
                    self._send_via_loop(channel, json.dumps(message))
                    self._chunks_sent += 1
                
                self._frames_sent += 1

                if self._stop_event.is_set() or channel.readyState != "open":
                    break

            # Don't send EOF here - let the client trigger it after receiving all responses
            print(f"Finished sending all {frame_id} frames. Waiting for server to process...")

        finally:
            cap.release()

    def get_initialization_params(self) -> Dict[str, Any]:
        return {"video_source": "data_channel"}

    def get_stats(self) -> dict:
        """Get sending statistics."""
        return {
            "frames_sent": self._frames_sent,
            "chunks_sent": self._chunks_sent,
        }

    async def cleanup(self) -> None:
        if self._stop_event:
            self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
