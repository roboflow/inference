from typing import Optional

from pydantic import Field

from inference.core.entities.responses.inference import (
    ObjectDetectionInferenceResponse,
)


class RFDETRFewShotInferenceResponse(ObjectDetectionInferenceResponse):
    """Response for RF-DETR few-shot object detection.

    Extends the standard object detection response with the model hash
    so the client can reuse it for subsequent requests without re-sending
    training data.
    """

    model_hash: str = Field(
        description="Hash identifying the cached LoRA adapter for this training data"
    )
