from typing import Any, List, Optional

from pydantic import BaseModel, Field

from inference.core.entities.requests.inference import BaseRequest


class InferenceRequestVideo(BaseModel):
    """Video data for an inference request.

    Prefer a URL. Base64 grows the clip by a third, and the whole request is
    held in memory before the clip reaches disk, so gateways reject a large
    one. Base64 suits a short clip and a quick test.

    Attributes:
        type (str): The type of video data provided, one of 'url' or 'base64'.
        value (Optional[Any]): Video data corresponding to the video type.
    """

    type: str = Field(
        examples=["url"],
        description=(
            "The type of video data provided, one of 'url' or 'base64'. Prefer "
            "'url': base64 grows the clip by a third and holds the whole "
            "request in memory, which gateways reject above a few megabytes."
        ),
    )
    value: Optional[Any] = Field(
        None,
        examples=["https://example.com/clip.mp4"],
        description="Video data corresponding to the video type",
    )


class ActionRecognitionInferenceRequest(BaseRequest):
    """Request for action recognition over a video clip.

    Attributes:
        model_id (str): The model to classify with.
        video (InferenceRequestVideo): The clip to classify.
        class_filter (Optional[List[str]]): The subset of a fine-tuned
            model's classes to report. A zero-shot model answers in its own
            words and ignores this.
    """

    model_id: str = Field(
        examples=["workspace/action-recognition-1"],
        description="The model to classify with",
    )
    video: InferenceRequestVideo
    class_filter: Optional[List[str]] = Field(
        None,
        examples=[["entering", "leaving"]],
        description=(
            "The subset of a fine-tuned model's classes to report. A "
            "zero-shot model answers in its own words and ignores this."
        ),
    )
