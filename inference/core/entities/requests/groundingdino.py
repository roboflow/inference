from typing import List, Optional

from pydantic import Field

from inference.core.entities.requests.inference import CVInferenceRequest


class GroundingDINOInferenceRequest(CVInferenceRequest):
    """Request for Grounding DINO zero-shot predictions.

    Attributes:
        text (List[str]): A list of strings.
    """

    text: List[str] = Field(
        example=["person", "dog", "cat"],
        description="A list of strings",
    )
    grounding_dino_version_id: Optional[str] = "default"
