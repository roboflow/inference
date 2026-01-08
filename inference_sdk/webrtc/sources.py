"""Stream source abstractions for WebRTC SDK.

This module defines the StreamSource interface and concrete implementations
for different video streaming sources (webcam, RTSP, video files, manual frames).
"""

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

import cv2
import numpy as np
from aiortc import RTCPeerConnection, VideoStreamTrack
from av import VideoFrame

from inference_sdk.http.errors import InvalidParameterError
from inference_sdk.webrtc.datachannel import VideoFileUploader

if TYPE_CHECKING:
    from aiortc import RTCDataChannel

    from inference_sdk.webrtc.config import StreamConfig

# Type alias for upload progress callback
UploadProgressCallback = Callable[[int, int], None]  # (uploaded_chunks, total_chunks)


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
    def get_initialization_params(self, config: "StreamConfig") -> Dict[str, Any]:
        """Get parameters to send to server in /initialise_webrtc_worker payload.

        Args:
            config: Stream configuration with stream_output, data_output, etc.

        Returns:
            Dictionary of parameters specific to this source type.
            Examples:
            - RTSP: {"rtsp_url": "rtsp://..."}
            - Video file: {"stream_output": [], "data_output": [...]}
            - Webcam/Manual: {} (empty, no server-side source)
        """
        pass

    async def cleanup(self) -> None:
        """Cleanup resources when session ends.

        Default implementation does nothing. Override if cleanup is needed.
        """
        pass


class _OpenCVVideoTrack(VideoStreamTrack):
    """Base class for video tracks that use OpenCV capture.

    This consolidates common logic for webcam tracks.
    """

    def __init__(self, source: Any, error_name: str):
        """Initialize OpenCV video track.

        Args:
            source: OpenCV VideoCapture source (int for webcam, str for file)
            error_name: Human-readable name for error messages
        """
        super().__init__()
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open {error_name}: {source}")
        self._error_name = error_name

    async def recv(self) -> VideoFrame:  # type: ignore[override]
        """Read next frame from OpenCV capture."""
        ret, frame = self._cap.read()
        if not ret or frame is None:
            raise RuntimeError(f"Failed to read from {self._error_name}")

        return await self._frame_to_video(frame)

    async def _frame_to_video(self, frame: np.ndarray) -> VideoFrame:
        """Convert numpy frame to VideoFrame with timestamp.

        Args:
            frame: BGR numpy array (H, W, 3) uint8

        Returns:
            VideoFrame with proper timestamp
        """
        vf = VideoFrame.from_ndarray(frame, format="bgr24")
        vf.pts, vf.time_base = await self.next_timestamp()
        return vf

    def get_declared_fps(self) -> Optional[float]:
        """Get the declared FPS from the OpenCV capture."""
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        return float(fps) if fps and fps > 0 else None

    def release(self) -> None:
        """Release the OpenCV capture."""
        try:
            self._cap.release()
        except Exception:
            pass


class _WebcamVideoTrack(_OpenCVVideoTrack):
    """aiortc VideoStreamTrack that reads frames from OpenCV webcam."""

    def __init__(self, device_id: int, resolution: Optional[Tuple[int, int]]):
        super().__init__(device_id, "webcam device")

        if resolution:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])


class WebcamSource(StreamSource):
    """Stream source for local webcam/USB camera.

    This source creates a local video track that captures frames from
    a webcam device using OpenCV and sends them to the server.
    """

    def __init__(
        self, device_id: int = 0, resolution: Optional[Tuple[int, int]] = None
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

    def get_initialization_params(self, config: "StreamConfig") -> Dict[str, Any]:
        """Return FPS if available."""
        params: Dict[str, Any] = {}
        if self._declared_fps:
            params["declared_fps"] = self._declared_fps
        return params

    async def cleanup(self) -> None:
        """Release webcam resources."""
        if self._track:
            self._track.release()


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
        if not url.startswith(("rtsp://", "rtsps://")):
            raise InvalidParameterError(
                f"Invalid RTSP URL: {url}. Must start with rtsp:// or rtsps://"
            )
        self.url = url

    async def configure_peer_connection(self, pc: RTCPeerConnection) -> None:
        """Add receive-only video transceiver (server sends video to us)."""
        # Don't create a local track - we're receiving video from server
        # Add receive-only transceiver
        pc.addTransceiver("video", direction="recvonly")

    def get_initialization_params(self, config: "StreamConfig") -> Dict[str, Any]:
        """Return RTSP URL for server to capture."""
        # Server needs to know the RTSP URL to capture
        return {"rtsp_url": self.url}


class MJPEGSource(StreamSource):
    """Stream source for MJPEG streams."""

    def __init__(self, url: str):
        if not url.startswith(("http://", "https://")):
            raise InvalidParameterError(
                f"Invalid MJPEG URL: {url}. Must start with http:// or https://"
            )
        self.url = url

    async def configure_peer_connection(self, pc: RTCPeerConnection) -> None:
        pc.addTransceiver("video", direction="recvonly")

    def get_initialization_params(self, config: "StreamConfig") -> Dict[str, Any]:
        return {"mjpeg_url": self.url}


class VideoFileSource(StreamSource):
    """Stream source for video files.

    Uploads video file via datachannel to the server, which processes it
    and streams results back. This is more efficient than frame-by-frame
    streaming for pre-recorded video files.

    Supports two output modes:
    - Datachannel mode (default): Frames received as base64 JSON via datachannel.
      Higher bandwidth but includes all workflow output data inline.
    - Video track mode: Frames received via WebRTC video track with hardware-
      accelerated codec (H.264/VP8). Lower bandwidth, workflow data sent separately.
    """

    def __init__(
        self,
        path: str,
        on_upload_progress: Optional[UploadProgressCallback] = None,
        use_datachannel_frames: bool = True,
        realtime_processing: bool = False,
    ):
        """Initialize video file source.

        Args:
            path: Path to video file (any format supported by FFmpeg)
            on_upload_progress: Optional callback called during upload with
                (uploaded_chunks, total_chunks). Use to track upload progress.
            use_datachannel_frames: If enabled, frames are received through the
                datachannel. It consumes much more network bandwidth, but it
                provides guaranteed in-order and high quality delivery of the
                frames. If False, frames are received via WebRTC video track
                with hardware-accelerated codec (lower bandwidth).
            realtime_processing: If True, process frames at original video FPS
                (throttled playback for live preview). If False (default),
                process all frames as fast as possible (batch mode).
        """
        self.path = path
        self.on_upload_progress = on_upload_progress
        self.use_datachannel_frames = use_datachannel_frames
        self.realtime_processing = realtime_processing
        self._upload_channel: Optional["RTCDataChannel"] = None
        self._uploader: Optional[VideoFileUploader] = None
        # Note: _upload_started is created lazily in configure_peer_connection()
        # to avoid Python 3.9 issue where asyncio.Event binds to wrong event loop
        self._upload_started: Optional[asyncio.Event] = None

    async def configure_peer_connection(self, pc: RTCPeerConnection) -> None:
        """Configure peer connection for video file upload.

        Creates video_upload datachannel for file transfer. In video track mode,
        also adds a receive-only transceiver for processed video output.
        """
        # Create event in the async context to bind to correct event loop (Python 3.9 compat)
        self._upload_started = asyncio.Event()

        # Create upload channel - server will create VideoFileUploadHandler
        self._upload_channel = pc.createDataChannel("video_upload")

        # Add receive-only transceiver for video track output mode (when not using datachannel)
        if not self.use_datachannel_frames:
            pc.addTransceiver("video", direction="recvonly")

        # Setup channel open handler to signal upload can start
        @self._upload_channel.on("open")
        def on_open() -> None:
            self._upload_started.set()

    def get_initialization_params(self, config: "StreamConfig") -> Dict[str, Any]:
        """Return params for video file processing mode.

        In datachannel mode (default), merges stream_output into data_output
        so frames are received as base64 via the inference datachannel.
        In video track mode, preserves stream_output for video track rendering.
        """
        params: Dict[str, Any] = {
            "webrtc_realtime_processing": self.realtime_processing,
            "video_file_upload": True,  # Signal to server that video will be uploaded
        }

        if not self.use_datachannel_frames:
            # Video track mode: keep stream_output for video track rendering
            return params

        # Datachannel mode (default): merge stream_output into data_output
        data_output = list(config.data_output or [])
        if config.stream_output:
            for field in config.stream_output:
                if field and field not in data_output:
                    data_output.append(field)

        params["stream_output"] = []  # No video track
        params["data_output"] = data_output  # Receive frames via data channel
        return params

    async def start_upload(self) -> None:
        """Start uploading the video file.

        Called by session after connection is established.
        Uses self.on_upload_progress if provided.
        """
        # Wait for channel to open
        await self._upload_started.wait()

        if not self._upload_channel:
            raise RuntimeError("Upload channel not configured")

        self._uploader = VideoFileUploader(self.path, self._upload_channel)
        await self._uploader.upload(on_progress=self.on_upload_progress)
        # self._upload_complete.set()

    async def cleanup(self) -> None:
        """No cleanup needed - upload channel is managed by peer connection."""
        pass


# Configuration constants for manual source
MANUAL_SOURCE_QUEUE_MAX_SIZE = 10  # maximum number of queued frames for manual source


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

    def get_initialization_params(self, config: "StreamConfig") -> Dict[str, Any]:
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
        self._queue: asyncio.Queue[Optional[np.ndarray]] = asyncio.Queue(
            maxsize=MANUAL_SOURCE_QUEUE_MAX_SIZE
        )

    async def recv(self) -> VideoFrame:  # type: ignore[override]
        """Wait for next frame to be queued."""
        # Wait for next frame to be queued
        frame = await self._queue.get()
        if frame is None:
            raise Exception("Manual track stopped")

        return await self._frame_to_video(frame)

    async def _frame_to_video(self, frame: np.ndarray) -> VideoFrame:
        """Convert numpy frame to VideoFrame with timestamp.

        Args:
            frame: BGR numpy array (H, W, 3) uint8

        Returns:
            VideoFrame with proper timestamp
        """
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
