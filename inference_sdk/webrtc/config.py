"""Configuration for WebRTC streaming sessions."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Video file upload via datachannel
VIDEO_UPLOAD_CHUNK_SIZE = 48 * 1024  # 48KB - safe for WebRTC
VIDEO_UPLOAD_BUFFER_LIMIT = 256 * 1024  # 256KB max buffered before backpressure


@dataclass
class StreamConfig:
    """Unified configuration for all WebRTC stream types.

    This configuration applies to all stream sources (webcam, RTSP, video file, manual)
    and controls output routing, processing behavior, and network settings.
    """

    # Output configuration
    stream_output: List[str] = field(default_factory=list)
    """List of workflow output names to stream as video"""

    data_output: List[str] = field(default_factory=list)
    """List of workflow output names to receive via data channel"""

    # Processing configuration
    realtime_processing: bool = True
    """Whether to process frames in realtime (drop if can't keep up) or queue all frames"""

    declared_fps: Optional[float] = None
    """Optional FPS declaration for the stream.

    Note: Some sources (like WebcamSource) auto-detect FPS from the video device and will
    override this value. The source's detected FPS takes precedence over this configuration.
    For sources without auto-detection (like ManualSource), this value will be used if provided.
    """

    # Network configuration
    turn_server: Optional[Dict[str, str]] = None
    """TURN server configuration: {"urls": "turn:...", "username": "...", "credential": "..."}

    Provide this configuration when your network requires a TURN server for WebRTC connectivity.
    TURN is automatically skipped for localhost connections. If not provided, the connection
    will attempt to establish directly without TURN relay.
    """

    # Workflow parameters
    workflow_parameters: Dict[str, Any] = field(default_factory=dict)
    """Parameters to pass to the workflow execution"""
