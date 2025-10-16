from typing import List, Optional, Union

from pydantic import BaseModel, Field, validator

from inference.core.entities.requests.inference import (
    BaseRequest,
    InferenceRequestImage,
)

from inference.core.env import SAM3_MAX_PROMPT_BATCH_SIZE


class Sam3Prompt(BaseModel):
    """Unified prompt that can contain text and/or geometry.

    Geometry is normalized to [0,1] in request: boxes in XYWH, points in XY.
    Labels accept 0/1 or booleans.
    """

    type: Optional[str] = Field(
        default=None, description="Optional hint: 'text' or 'visual'"
    )
    text: Optional[str] = Field(default=None)
    boxes: Optional[List[List[float]]] = Field(
        default=None, description="List of [x, y, w, h] normalized to 0-1"
    )
    box_labels: Optional[List[Union[int, bool]]] = Field(
        default=None, description="List of 0/1 or booleans for boxes"
    )


class Sam3InferenceRequest(BaseRequest):
    """SAM3 inference request.

    Attributes:
        model_id (Optional[str]): The model ID to be used, typically `sam3`.
    """

    model_id: Optional[str] = Field(
        default="sam3/sam3_image_model_only",
        description="The model ID of SAM3. Use 'sam3/sam3_image_model_only' to target the generic base model.",
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

    @validator("prompts")
    def _validate_prompts(cls, prompts: List[Sam3Prompt]):
        if not prompts or len(prompts) == 0:
            raise ValueError("At least one prompt is required")
        if len(prompts) > SAM3_MAX_PROMPT_BATCH_SIZE:
            raise ValueError(
                f"Exceeded SAM3_MAX_PROMPT_BATCH_SIZE={SAM3_MAX_PROMPT_BATCH_SIZE}: got {len(prompts)}"
            )
        return prompts
