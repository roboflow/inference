"""WebRTC SDK for Inference - Unified streaming API."""

from .client import WebRTCClient  # noqa: F401
from .config import StreamConfig  # noqa: F401
from .session import WebRTCSession  # noqa: F401
from .sources import (  # noqa: F401
    ManualSource,
    RTSPSource,
    StreamSource,
    VideoFileSource,
    WebcamSource,
)

# Convenience factory functions


def webcam(device_id: int = 0, resolution: tuple[int, int] | None = None) -> WebcamSource:
    """Create a webcam source.

    Args:
        device_id: Camera device index (0 for default camera)
        resolution: Optional (width, height) tuple

    Returns:
        WebcamSource instance

    Example:
        from inference_sdk import webrtc
        source = webrtc.webcam(resolution=(1920, 1080))
    """
    return WebcamSource(device_id, resolution)


def rtsp(url: str) -> RTSPSource:
    """Create an RTSP source.

    Args:
        url: RTSP URL (e.g., "rtsp://camera.local/stream")
            Credentials can be included: "rtsp://user:pass@host/stream"

    Returns:
        RTSPSource instance

    Example:
        from inference_sdk import webrtc
        source = webrtc.rtsp("rtsp://camera.local/stream")
    """
    return RTSPSource(url)


def video_file(path: str) -> VideoFileSource:
    """Create a video file source.

    Args:
        path: Path to video file (any format supported by OpenCV)

    Returns:
        VideoFileSource instance

    Example:
        from inference_sdk import webrtc
        source = webrtc.video_file("/path/to/video.mp4")
    """
    return VideoFileSource(path)


def manual() -> ManualSource:
    """Create a manual frame source.

    Returns:
        ManualSource instance with send() method

    Example:
        from inference_sdk import webrtc
        source = webrtc.manual()
        # Later: source.send(frame)
    """
    return ManualSource()


__all__ = [
    # Core classes
    "WebRTCClient",
    "WebRTCSession",
    "StreamConfig",
    # Source classes
    "StreamSource",
    "WebcamSource",
    "RTSPSource",
    "VideoFileSource",
    "ManualSource",
    # Factory functions
    "webcam",
    "rtsp",
    "video_file",
    "manual",
]
