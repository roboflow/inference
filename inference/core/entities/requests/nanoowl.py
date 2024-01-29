from typing import List, Optional

from pydantic import Field

from inference.core.entities.requests.inference import CVInferenceRequest


class NanoOwlInferenceRequest(CVInferenceRequest):
    """Request for NanoOwl zero-shot predictions.

    Attributes:
        text (List[str]): A list of strings.
    """

    text: List[str] = Field(
        example=["person", "dog", "cat"],
        description="A list of strings",
    )
    nanoowl_version_id: Optional[str] = "default"
