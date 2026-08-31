from typing import List

from pydantic import BaseModel, ConfigDict, Field


class ActionRecognitionSegment(BaseModel):
    """One classified frame range of the submitted clip."""

    model_config = ConfigDict(populate_by_name=True)

    start_frame_idx: int = Field(description="First frame of the range")
    end_frame_idx: int = Field(description="Last frame of the range")
    class_name: str = Field(alias="class", description="The class name")
    class_id: int = Field(
        description=(
            "The class position in the model's own class list. A model without "
            "a class list reports -1."
        )
    )


class ActionRecognitionInferenceResponse(BaseModel):
    """Classified ranges covering one clip.

    Frame indices count from the first frame of the submitted clip, so a
    caller converts them to seconds with ``source_fps``.
    """

    timeline: List[ActionRecognitionSegment] = Field(
        description="Classified frame ranges, which can overlap"
    )
    source_fps: float = Field(description="Frames per second of the clip")
    frame_count: int = Field(description="Frames the clip holds")
    windows_classified: int = Field(description="Model calls the clip was cut into")
