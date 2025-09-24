from typing import List, Optional

from pydantic import Field

from inference.core.entities.requests.inference import CVInferenceRequest


class DynamicClassBaseInferenceRequest(CVInferenceRequest):
    """Request for zero-shot object detection models (with dynamic class lists).

    Attributes:
        text (List[str]): A list of strings.
    """

    model_id: Optional[str] = Field(None)
    text: List[str] = Field(
        examples=[["person", "dog", "cat"]],
        description="A list of strings",
    )
