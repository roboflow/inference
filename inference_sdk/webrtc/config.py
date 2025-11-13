"""Configuration for WebRTC streaming sessions."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class OutputMode(str, Enum):
    """Output mode for WebRTC sessions.

    Determines what data is sent back from the server during processing:

    - DATA_ONLY: Only send JSON data via data channel (no video track sent back).
                 Use this when you only need inference results/metrics and want to
                 save bandwidth. The server won't send processed video frames back.

    - VIDEO_ONLY: Only send processed video via video track (no data channel messages).
                  Use this when you only need to display the processed video and don't
                  need programmatic access to results.

    - BOTH: Send both processed video and JSON data (default behavior).
            Use this when you need both visual output and programmatic access to results.

    Examples:
        # Data-only mode for analytics/logging (saves bandwidth)
        config = StreamConfig(output_mode=OutputMode.DATA_ONLY)

        # Video-only mode for display-only applications
        config = StreamConfig(output_mode=OutputMode.VIDEO_ONLY)

        # Both (default) for full-featured applications
        config = StreamConfig(output_mode=OutputMode.BOTH)
    """

    DATA_ONLY = "data_only"
    VIDEO_ONLY = "video_only"
    BOTH = "both"


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

    output_mode: OutputMode = OutputMode.BOTH
    """Output mode: DATA_ONLY (data channel only), VIDEO_ONLY (video only), or BOTH (default)"""

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

    If not provided, the SDK will automatically attempt to fetch TURN configuration
    from the server endpoint. TURN is automatically skipped for localhost connections.
    """

    # Workflow parameters
    workflow_parameters: Dict[str, Any] = field(default_factory=dict)
    """Parameters to pass to the workflow execution"""
