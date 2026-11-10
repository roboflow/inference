"""Requests of the SAM3 offline video tracking routes (``/sam3/video/*``)."""

from typing import List, Optional

from pydantic import BaseModel, Field

from inference.core.entities.requests.action_recognition import InferenceRequestVideo
from inference.core.entities.requests.inference import BaseRequest
from inference.core.env import SAM3_VIDEO_MODEL_ID


class Sam3VideoRequest(BaseRequest):
    """Base request of the SAM3 offline video tracking routes.

    Attributes:
        model_id (str): The SAM3 video tracker to use.
        video_id (str): The caller's id of the embedded clip.
    """

    model_id: str = Field(
        default=SAM3_VIDEO_MODEL_ID,
        examples=["sam3trackervideo"],
        description="The SAM3 video tracker to use.",
    )
    video_id: str = Field(
        examples=["clip-0001"],
        description=(
            "The caller's id of the clip. Ids are scoped to the caller's API key; "
            "embedding again under an id replaces the clip."
        ),
    )


class Sam3VideoEmbedRequest(Sam3VideoRequest):
    """Request to decode a clip and keep it for offline tracking.

    Attributes:
        video (InferenceRequestVideo): The clip; every decoded frame is a frame.
        encode_frames (bool): Start encoding every frame in the background.
    """

    video: InferenceRequestVideo = Field(
        description=(
            "The clip. Every decoded frame becomes a frame, numbered from 0 in "
            "decode order; nothing is resampled. To control exactly which "
            "pixels are tracked, send the frames as an MJPEG video."
        ),
    )
    encode_frames: bool = Field(
        default=False,
        description=(
            "Start running the image encoder on every frame in the background, "
            "so later segment and propagate calls mostly run only the tracker. "
            "The response does not wait for it, and calls in the meantime are "
            "served between frames. Otherwise each frame is encoded the first "
            "time a call needs it."
        ),
    )


class Sam3VideoPointPrompt(BaseModel):
    """One click on one frame of an embedded clip.

    Attributes:
        frame_idx (int): Frame the click is on.
        obj_id (int): Object the click belongs to.
        x (float): Horizontal position in pixels of the frame.
        y (float): Vertical position in pixels of the frame.
        label (int): 1 to include the point, 0 to exclude it.
    """

    frame_idx: int = Field(ge=0, description="Frame the click is on.")
    obj_id: int = Field(examples=[1], description="Object the click belongs to.")
    x: float = Field(description="Horizontal position in pixels of the frame.")
    y: float = Field(description="Vertical position in pixels of the frame.")
    label: int = Field(
        ge=0,
        le=1,
        description=(
            "1 to include the point in the object, 0 to exclude it. Every "
            "object needs an include point on at least one frame; the first "
            "such frame is its anchor. On its other frames exclude points "
            "alone are valid."
        ),
    )


class Sam3VideoSegmentRequest(Sam3VideoRequest):
    """Request for the masks on one frame, as a preview of its clicks.

    Attributes:
        prompts (List[Sam3VideoPointPrompt]): All clicks on the clip, in click order.
        frame_idx (int): Frame to return masks for.
    """

    prompts: List[Sam3VideoPointPrompt] = Field(
        description=(
            "All clicks on the clip, in click order. Only objects with a click "
            "on frame_idx are segmented, each from all its clicks: an object "
            "is tracked from its anchor to frame_idx when that is another frame."
        ),
    )
    frame_idx: int = Field(ge=0, description="Frame to return masks for.")


class Sam3VideoPropagateRequest(Sam3VideoRequest):
    """Request to track every clicked object over the whole clip.

    Attributes:
        prompts (List[Sam3VideoPointPrompt]): All clicks on the clip, in click order.
        frame_indices (Optional[List[int]]): Frames to return masks for.
    """

    prompts: List[Sam3VideoPointPrompt] = Field(
        description=(
            "All clicks on the clip, in click order. Every object needs an "
            "include point on at least one frame."
        ),
    )
    frame_indices: Optional[List[int]] = Field(
        default=None,
        description=(
            "Frames to return masks for; all frames when omitted. The others "
            "are still tracked through, e.g. in-between frames sent to make "
            "the steps between returned frames smaller."
        ),
    )
