from typing import List, Optional

from pydantic import Field, validator

from inference.core.entities.requests.inference import (
    BaseRequest,
    InferenceRequestImage,
)
from inference.core.env import SAM3_MAX_PROMPT_BATCH_SIZE
from inference.core.workflows.core_steps.models.foundation.segment_anything_common.prompts import (  # noqa: F401
    Sam3Prompt,
)


class Sam3InferenceRequest(BaseRequest):
    """SAM3 inference request.

    Attributes:
        model_id (Optional[str]): The model ID to be used, typically `sam3`.
    """

    model_id: Optional[str] = Field(
        default="sam3/sam3_final",
        description="The model ID of SAM3. Use 'sam3/sam3_final' to target the generic base model.",
    )


class Sam3SegmentationRequest(Sam3InferenceRequest):
    format: Optional[str] = Field(
        default="polygon",
        description="One of 'polygon', 'rle'",
    )
    image: InferenceRequestImage = Field(description="The image to be segmented.")
    image_id: Optional[str] = Field(
        default=None, description="Optional ID for caching embeddings."
    )
    output_prob_thresh: Optional[float] = Field(
        default=0.5, description="Score threshold for outputs."
    )

    # Unified prompts list (required)
    prompts: List[Sam3Prompt] = Field(
        description="List of prompts (text and/or visual)", min_items=1
    )

    nms_iou_threshold: Optional[float] = Field(
        default=None,
        description="IoU threshold for cross-prompt NMS. If None, NMS is disabled. Must be in [0.0, 1.0] when set.",
    )

    @validator("nms_iou_threshold")
    def _validate_nms_iou_threshold(cls, v):
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError("nms_iou_threshold must be between 0.0 and 1.0")
        return v

    @validator("prompts")
    def _validate_prompts(cls, prompts: List[Sam3Prompt]):
        if not prompts or len(prompts) == 0:
            raise ValueError("At least one prompt is required")
        if len(prompts) > SAM3_MAX_PROMPT_BATCH_SIZE:
            raise ValueError(
                f"Exceeded SAM3_MAX_PROMPT_BATCH_SIZE={SAM3_MAX_PROMPT_BATCH_SIZE}: got {len(prompts)}"
            )
        return prompts
