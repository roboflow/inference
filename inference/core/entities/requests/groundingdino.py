from typing import List

from pydantic import Field

from inference.core.entities.requests.inference import InferenceRequest


class GroundingDINOInferenceRequest(InferenceRequest):
    """Request for Grounding DINO zero-shot predictions.

    Attributes:
        text (List[str]): A list of strings.
    """

    text: List[str] = Field(
        example=["person", "dog", "cat"],
        description="A list of strings",
    )
