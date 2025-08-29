from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class CommandContext(BaseModel):
    request_id: Optional[str] = Field(
        description="Server-side request ID", default=None
    )
    pipeline_id: Optional[str] = Field(
        description="Identifier of pipeline connected to operation", default=None
    )


class CommandResponse(BaseModel):
    status: str = Field(description="Operation status")
    context: CommandContext = Field(description="Context of the command.")


class InferencePipelineStatusResponse(CommandResponse):
    report: dict


class ListPipelinesResponse(CommandResponse):
    pipelines: List[str] = Field(description="List IDs of active pipelines")


class FrameMetadata(BaseModel):
    frame_timestamp: datetime
    frame_id: int
    source_id: Optional[int]


class ConsumePipelineResponse(CommandResponse):
    outputs: List[dict]
    frames_metadata: List[FrameMetadata]


class InitializeWebRTCPipelineResponse(CommandResponse):
    sdp: str
    type: str


class LatestFrameResponse(BaseModel):
    success: bool = Field(description="Whether frame was successfully retrieved")
    frame_id: Optional[int] = Field(description="Unique identifier of the frame", default=None)
    data: Optional[str] = Field(description="Base64 encoded JPEG image", default=None)
    camera_fps: Optional[float] = Field(description="Camera FPS", default=None)
    pipeline_fps: Optional[float] = Field(description="Pipeline processing FPS", default=None)
    message: Optional[str] = Field(description="Error or status message", default=None)
