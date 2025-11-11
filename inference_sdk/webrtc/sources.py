"""Stream source abstractions for WebRTC SDK.

This module defines the StreamSource interface and concrete implementations
for different video streaming sources (webcam, RTSP, video files, manual frames).
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import cv2
import numpy as np
from aiortc import RTCPeerConnection, VideoStreamTrack
from av import VideoFrame


class StreamSource(ABC):
    """Base interface for all stream sources.

    A StreamSource is responsible for:
    1. Configuring the RTCPeerConnection (adding tracks or transceivers)
    2. Providing initialization parameters for the server
    3. Cleaning up resources when done
    """

    @abstractmethod
    async def configure_peer_connection(self, pc: RTCPeerConnection) -> None:
        """Configure the peer connection for this source type.

        This is where the source decides:
        - Whether to add a local track (webcam, video file, manual)
        - Whether to add a receive-only transceiver (RTSP)
        - Any other peer connection configuration

        Args:
            pc: The RTCPeerConnection to configure
        """
        pass

    @abstractmethod
    def get_initialization_params(self) -> Dict[str, Any]:
        """Get parameters to send to server in /initialise_webrtc_worker payload.

        Returns:
            Dictionary of parameters specific to this source type.
            Examples:
            - RTSP: {"rtsp_url": "rtsp://..."}
            - Video file: {"video_path": "/path/to/file"}
            - Webcam/Manual: {} (empty, no server-side source)
        """
        pass

    async def cleanup(self) -> None:
        """Cleanup resources when session ends.

        Default implementation does nothing. Override if cleanup is needed.
        """
        pass


class WebcamSource(StreamSource):
    """Stream source for local webcam/USB camera.

    This source creates a local video track that captures frames from
    a webcam device using OpenCV and sends them to the server.
    """

    def __init__(
        self, device_id: int = 0, resolution: Optional[tuple[int, int]] = None
    ):
        """Initialize webcam source.

        Args:
            device_id: Camera device index (0 for default camera)
            resolution: Optional (width, height) tuple to set camera resolution
        """
        self.device_id = device_id
        self.resolution = resolution
        self._track: Optional[_WebcamVideoTrack] = None
        self._declared_fps: Optional[float] = None

    async def configure_peer_connection(self, pc: RTCPeerConnection) -> None:
        """Create webcam video track and add it to the peer connection."""
        # Create local video track that reads from OpenCV
        self._track = _WebcamVideoTrack(self.device_id, self.resolution)

        # Capture FPS for server
        self._declared_fps = self._track.get_declared_fps()

        # Add track to send video
        pc.addTrack(self._track)

    def get_initialization_params(self) -> Dict[str, Any]:
        """Return FPS if available."""
        params = {}
        if self._declared_fps:
            params["declared_fps"] = self._declared_fps
        return params

    async def cleanup(self) -> None:
        """Release webcam resources."""
        if self._track:
            self._track.release()


class _WebcamVideoTrack(VideoStreamTrack):
    """aiortc VideoStreamTrack that reads frames from OpenCV webcam."""

    def __init__(self, device_id: int, resolution: Optional[tuple[int, int]]):
        super().__init__()
        self._cap = cv2.VideoCapture(device_id)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open webcam device {device_id}")

        if resolution:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

    async def recv(self) -> VideoFrame:  # type: ignore[override]
        """Read next frame from webcam."""
        # Read frame from OpenCV in async context
        ret, frame = self._cap.read()
        if not ret or frame is None:
            raise RuntimeError("Failed to read from webcam")

        # Convert to VideoFrame
        vf = VideoFrame.from_ndarray(frame, format="bgr24")
        vf.pts, vf.time_base = await self.next_timestamp()
        return vf

    def get_declared_fps(self) -> Optional[float]:
        """Get the camera's declared FPS."""
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        return float(fps) if fps and fps > 0 else None

    def release(self) -> None:
        """Release the camera."""
        try:
            self._cap.release()
        except Exception:
            pass


class RTSPSource(StreamSource):
    """Stream source for RTSP camera streams.

    This source doesn't create a local track - instead, the server
    captures the RTSP stream and sends processed video back to the client.
    """

    def __init__(self, url: str):
        """Initialize RTSP source.

        Args:
            url: RTSP URL (e.g., "rtsp://camera.local/stream")
                Credentials can be included: "rtsp://user:pass@host/stream"
        """
        self.url = url
        self._validate_url()

    def _validate_url(self) -> None:
        """Validate that the URL is a valid RTSP URL."""
        if not self.url.startswith(("rtsp://", "rtsps://")):
            raise ValueError(
                f"Invalid RTSP URL: {self.url}. Must start with rtsp:// or rtsps://"
            )

    async def configure_peer_connection(self, pc: RTCPeerConnection) -> None:
        """Add receive-only video transceiver (server sends video to us)."""
        # Don't create a local track - we're receiving video from server
        # Add receive-only transceiver
        pc.addTransceiver("video", direction="recvonly")

    def get_initialization_params(self) -> Dict[str, Any]:
        """Return RTSP URL for server to capture."""
        # Server needs to know the RTSP URL to capture
        return {"rtsp_url": self.url}


class VideoFileSource(StreamSource):
    """Stream source for video files.

    This source creates a local video track that reads frames from
    a video file and sends them to the server.
    """

    def __init__(self, path: str):
        """Initialize video file source.

        Args:
            path: Path to video file (any format supported by OpenCV)
        """
        self.path = path
        self._track: Optional[_VideoFileTrack] = None

    async def configure_peer_connection(self, pc: RTCPeerConnection) -> None:
        """Create video file track and add it to the peer connection."""
        # Create track that reads from video file
        self._track = _VideoFileTrack(self.path)

        # Add track to send video
        pc.addTrack(self._track)

    def get_initialization_params(self) -> Dict[str, Any]:
        """Return metadata about video source."""
        # Optionally pass source type for logging/metadata
        return {"video_source": "file"}

    async def cleanup(self) -> None:
        """Release video file resources."""
        if self._track:
            self._track.release()


class _VideoFileTrack(VideoStreamTrack):
    """aiortc VideoStreamTrack that reads frames from a video file."""

    def __init__(self, path: str):
        super().__init__()
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video file: {path}")

        # Get video properties
        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._frame_time = 1.0 / self._fps if self._fps > 0 else 1.0 / 30.0

    async def recv(self) -> VideoFrame:  # type: ignore[override]
        """Read next frame from video file."""
        ret, frame = self._cap.read()
        if not ret or frame is None:
            # End of file
            raise Exception("End of video file")

        vf = VideoFrame.from_ndarray(frame, format="bgr24")
        vf.pts, vf.time_base = await self.next_timestamp()
        return vf

    def release(self) -> None:
        """Release the video file."""
        try:
            self._cap.release()
        except Exception:
            pass


class ManualSource(StreamSource):
    """Stream source for manually sent frames.

    This source allows the user to programmatically send frames
    to be processed by the workflow using the send() method.
    """

    def __init__(self):
        """Initialize manual source."""
        self._track: Optional[_ManualTrack] = None

    async def configure_peer_connection(self, pc: RTCPeerConnection) -> None:
        """Create manual track and add it to the peer connection."""
        # Create special track that accepts programmatic frames
        self._track = _ManualTrack()
        pc.addTrack(self._track)

    def get_initialization_params(self) -> Dict[str, Any]:
        """Return manual mode flag."""
        return {"manual_mode": True}

    def send(self, frame: np.ndarray) -> None:
        """Send a frame to be processed by the workflow.

        Args:
            frame: BGR numpy array (H, W, 3) uint8

        Raises:
            RuntimeError: If session not started
        """
        if not self._track:
            raise RuntimeError("Session not started. Use within 'with' context.")
        self._track.queue_frame(frame)


class _ManualTrack(VideoStreamTrack):
    """aiortc VideoStreamTrack that accepts programmatically queued frames."""

    def __init__(self):
        super().__init__()
        self._queue: asyncio.Queue[Optional[np.ndarray]] = asyncio.Queue(maxsize=10)

    async def recv(self) -> VideoFrame:  # type: ignore[override]
        """Wait for next frame to be queued."""
        # Wait for next frame to be queued
        frame = await self._queue.get()
        if frame is None:
            raise Exception("Manual track stopped")

        vf = VideoFrame.from_ndarray(frame, format="bgr24")
        vf.pts, vf.time_base = await self.next_timestamp()
        return vf

    def queue_frame(self, frame: np.ndarray) -> None:
        """Queue a frame to be sent (called from main thread).

        If the queue is full, the oldest frame is dropped.
        """
        try:
            self._queue.put_nowait(frame)
        except asyncio.QueueFull:
            # Drop oldest frame
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(frame)
            except Exception:
                pass
