from typing import List, Optional

from inference.core.entities.requests.dynamic_class_base import (
    DynamicClassBaseInferenceRequest,
)


class GroundingDINOInferenceRequest(DynamicClassBaseInferenceRequest):
    """Request for Grounding DINO zero-shot predictions.

    Attributes:
        text (List[str]): A list of strings.
    """

    grounding_dino_version_id: Optional[str] = "default"
