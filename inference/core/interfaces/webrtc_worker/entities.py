from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from inference.core.env import WEBRTC_REALTIME_PROCESSING
from inference.core.interfaces.stream_manager.manager_app.entities import (
    WebRTCOffer,
    WebRTCTURNConfig,
    WorkflowConfiguration,
)


class WebRTCOutputMode(str, Enum):
    """Defines the output mode for WebRTC worker processing.

    - DATA_ONLY: Only send JSON data via data channel (no video track sent back)
    - VIDEO_ONLY: Only send processed video via video track (no data channel messages)
    - BOTH: Send both video and data (default behavior)
    - OFF: Disable both outputs (useful for pausing processing)
    """
    DATA_ONLY = "data_only"
    VIDEO_ONLY = "video_only"
    BOTH = "both"
    OFF = "off"


class WebRTCWorkerRequest(BaseModel):
    api_key: Optional[str] = None
    workflow_configuration: WorkflowConfiguration
    webrtc_offer: WebRTCOffer
    webrtc_turn_config: Optional[WebRTCTURNConfig] = None
    webrtc_realtime_processing: bool = (
        WEBRTC_REALTIME_PROCESSING  # when set to True, MediaRelay.subscribe will be called with buffered=False
    )
    output_mode: WebRTCOutputMode = WebRTCOutputMode.BOTH
    stream_output: Optional[List[Optional[str]]] = Field(default_factory=list)
    data_output: Optional[List[Optional[str]]] = Field(default_factory=list)
    declared_fps: Optional[float] = None
    rtsp_url: Optional[str] = None


class WebRTCVideoMetadata(BaseModel):
    frame_id: int
    received_at: str  # datetime.datetime.isoformat()
    pts: Optional[int] = None
    time_base: Optional[float] = None
    declared_fps: Optional[float] = None
    measured_fps: Optional[float] = None


class WebRTCOutput(BaseModel):
    """Output sent via WebRTC data channel.

    serialized_output_data contains a dictionary with workflow outputs:
    - If data_output is None: all workflow outputs
    - If data_output is []: None (no data sent)
    - If data_output is ["field1", "field2"]: only those fields
    """
    serialized_output_data: Optional[Dict[str, Any]] = None
    video_metadata: Optional[WebRTCVideoMetadata] = None
    errors: List[str] = Field(default_factory=list)


class WebRTCWorkerResult(BaseModel):
    answer: Optional[WebRTCOffer] = None
    process_id: Optional[Union[int, str]] = None
    exception_type: Optional[str] = None
    error_message: Optional[str] = None
    error_context: Optional[str] = None
    inner_error: Optional[str] = None
