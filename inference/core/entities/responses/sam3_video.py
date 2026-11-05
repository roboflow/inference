"""Responses of the SAM3 offline video tracking routes (``/sam3/video/*``)."""

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class Sam3VideoEmbedResponse(BaseModel):
    """A clip kept for offline tracking."""

    video_id: str = Field(description="The caller's id of the clip.")
    n_frames: int = Field(description="Number of frames decoded from the clip.")
    height: int = Field(description="Frame height in pixels.")
    width: int = Field(description="Frame width in pixels.")
    time: float = Field(description="Seconds the request took.")


class Sam3VideoFrameMasks(BaseModel):
    """The masks of one frame."""

    frame_idx: int = Field(description="The frame.")
    masks: Dict[str, Dict[str, Any]] = Field(
        description=(
            "COCO RLE (size [height, width], string counts) by object id. An "
            "object the frame has no pixels of has an all-background mask."
        ),
    )


class Sam3VideoSegmentResponse(Sam3VideoFrameMasks):
    """The masks on one frame of the objects clicked there."""

    time: float = Field(description="Seconds the request took.")


class Sam3VideoPropagateResponse(BaseModel):
    """Every clicked object tracked over the clip."""

    frames: List[Sam3VideoFrameMasks] = Field(
        description="The returned frames, in frame order."
    )
    time: float = Field(description="Seconds the request took.")
