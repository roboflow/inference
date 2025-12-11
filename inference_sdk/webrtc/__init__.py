"""WebRTC SDK for Inference - Unified streaming API."""

from .client import WebRTCClient  # noqa: F401
from .config import StreamConfig  # noqa: F401
from .session import VideoMetadata, WebRTCSession  # noqa: F401
from .sources import (  # noqa: F401
    ManualSource,
    RTSPSource,
    StreamSource,
    UploadProgressCallback,
    VideoFileSource,
    WebcamSource,
)

__all__ = [
    # Core classes
    "WebRTCClient",
    "WebRTCSession",
    "StreamConfig",
    "VideoMetadata",
    # Source classes
    "StreamSource",
    "WebcamSource",
    "RTSPSource",
    "VideoFileSource",
    "ManualSource",
    # Type aliases
    "UploadProgressCallback",
]
