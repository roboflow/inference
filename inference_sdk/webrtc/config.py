from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class WebRTCBaseConfig:
    """Subset of fields matching server initialise APIs.

    Focused on worker init compatibility.
    """

    webrtc_realtime_processing: bool = True
    webrtc_turn_config: Optional[Dict[str, str]] = None  # {urls, username, credential}
    stream_output: List[Optional[str]] = field(default_factory=list)
    data_output: List[Optional[str]] = field(default_factory=list)
    declared_fps: Optional[float] = None
    workflows_parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WebcamConfig(WebRTCBaseConfig):
    """Local webcam configuration.

    Resolution is local-only and used to reduce network usage.
    """

    resolution: Optional[tuple[int, int]] = None  # (width, height)

