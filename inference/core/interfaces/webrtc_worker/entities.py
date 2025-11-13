from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from inference.core.env import (
    WEBRTC_MODAL_FUNCTION_TIME_LIMIT,
    WEBRTC_REALTIME_PROCESSING,
)
from inference.core.interfaces.stream_manager.manager_app.entities import (
    WebRTCOffer,
    WebRTCTURNConfig,
    WorkflowConfiguration,
)


class WebRTCWorkerRequest(BaseModel):
    api_key: Optional[str] = None
    workflow_configuration: WorkflowConfiguration
    webrtc_offer: WebRTCOffer
    webrtc_turn_config: Optional[WebRTCTURNConfig] = None
    webrtc_realtime_processing: bool = (
        WEBRTC_REALTIME_PROCESSING  # when set to True, MediaRelay.subscribe will be called with buffered=False
    )
    stream_output: Optional[List[str]] = Field(default=None)
    data_output: Optional[List[str]] = Field(default=None)
    declared_fps: Optional[float] = None
    rtsp_url: Optional[str] = None
    processing_timeout: Optional[int] = WEBRTC_MODAL_FUNCTION_TIME_LIMIT
    # https://modal.com/docs/guide/gpu#specifying-gpu-type
    requested_gpu: Optional[
        Literal[
            "T4",
            "L4",
            "A10",
            "A100",
            "A100-40GB",
            "A100-80GB",
            "L40S",
            "H100/H100!",
            "H200",
            "B200",
        ]
    ] = "T4"


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
    - If data_output is None or []: no data sent (only metadata)
    - If data_output is ["*"]: all workflow outputs (excluding images, unless explicitly named)
    - If data_output is ["field1", "field2"]: only those fields (including images if explicitly named)
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


class StreamOutputMode(str, Enum):
    AUTO_DETECT = "auto_detect"  # None -> auto-detect first image
    NO_VIDEO = "no_video"  # [] -> no video track
    SPECIFIC_FIELD = "specific"  # ["field"] -> use specific field


class DataOutputMode(str, Enum):
    NONE = "none"  # None or [] -> no data sent
    ALL = "all"  # ["*"] -> send all (skip images)
    SPECIFIC = "specific"  # ["field1", "field2"] -> send only these
