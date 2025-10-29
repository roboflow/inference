from typing import List, Optional

from pydantic import BaseModel, Field


class WebRTCVideoMetadata(BaseModel):
    frame_id: int
    frame_timestamp: str
    pts: Optional[int] = None
    time_base: Optional[float] = None
    declared_fps: Optional[float] = None


class WebRTCOutput(BaseModel):
    output_name: Optional[str] = None
    serialized_output_data: Optional[str] = None
    video_metadata: Optional[WebRTCVideoMetadata] = None
    errors: List[str] = Field(default_factory=list)
