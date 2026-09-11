from typing import List

from pydantic import BaseModel, Field

from inference.core.workflows.core_steps.models.roboflow.action_recognition.entities import (  # noqa: F401
    ActionRecognitionPrediction,
)


class ActionRecognitionInferenceResponse(BaseModel):
    """Classified ranges covering one clip.

    Frame indices count from the first frame of the submitted clip, so a
    caller converts them to seconds with ``source_fps``.
    """

    timeline: List[ActionRecognitionPrediction] = Field(
        description="Classified frame ranges, which can overlap"
    )
    source_fps: float = Field(description="Frames per second of the clip")
    frame_count: int = Field(description="Frames the clip holds")
    windows_classified: int = Field(description="Model calls the clip was cut into")
