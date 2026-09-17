"""Action-recognition payload type, owned by Workflows.

Moved here from `inference.core.entities.responses.action_recognition`, which
now re-exports it: the workflow kind, the serializers' `isinstance` dispatch and
the HTTP `ActionRecognitionInferenceResponse` must all see ONE class object.
"""

from typing import Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
)


class ActionRecognitionPrediction(BaseModel):
    """One classified frame range of a video.

    The HTTP response and the workflow kind carry the same shape, so it is
    declared once here and imported by both. A field added on one transport
    would otherwise be silently missing from the other.
    """

    model_config = ConfigDict(populate_by_name=True)

    start_frame_idx: int = Field(description="First frame of the range")
    end_frame_idx: int = Field(description="Last frame of the range")
    class_name: str = Field(alias="class", description="The class name")
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    class_id: int = Field(
        description=(
            "The class position in the model's own class list. A model without "
            "a class list reports -1."
        )
    )

    @model_serializer(mode="wrap")
    def serialize_prediction(self, handler: SerializerFunctionWrapHandler) -> dict:
        result = handler(self)
        # Unscored models retain their existing response shape.
        if self.confidence is None:
            result.pop("confidence", None)
        return result
