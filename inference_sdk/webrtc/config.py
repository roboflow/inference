"""Configuration for WebRTC streaming sessions."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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
    """Optional FPS declaration (auto-detected for webcam if not provided)"""

    # Network configuration
    turn_server: Optional[Dict[str, str]] = None
    """TURN server configuration: {"urls": "turn:...", "username": "...", "credential": "..."}"""

    # Workflow parameters
    workflow_parameters: Dict[str, Any] = field(default_factory=dict)
    """Parameters to pass to the workflow execution"""
