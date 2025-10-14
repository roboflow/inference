from typing import List, Optional, Union

from pydantic import BaseModel, Field, validator, root_validator

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
    points: Optional[List[List[float]]] = Field(
        default=None,
        description="List of [x, y] normalized to 0-1 (PCS ignores points)",
    )
    point_labels: Optional[List[Union[int, bool]]] = Field(
        default=None, description="List of 0/1 or booleans for points"
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


class Sam3EmbeddingRequest(Sam3InferenceRequest):
    image: Optional[InferenceRequestImage] = Field(
        default=None, description="The image to be embedded. Optional for caching."
    )
    image_id: Optional[str] = Field(
        default=None,
        description="Optional ID for caching embeddings.",
    )


class Sam3SegmentationRequest(Sam3InferenceRequest):
    format: Optional[str] = Field(
        default="polygon",
        description="One of 'polygon', 'rle', or 'binary'.",
    )
    image: InferenceRequestImage = Field(description="The image to be segmented.")
    image_id: Optional[str] = Field(
        default=None, description="Optional ID for caching embeddings."
    )
    # Legacy single-prompt fields (backward-compatible)
    text: Optional[str] = Field(
        default=None, description="Text query for open-vocabulary segmentation."
    )
    points: Optional[List[List[float]]] = Field(
        default=None,
        description="List of [x, y] points normalized to 0-1. Used only with instance_prompt=True.",
    )
    point_labels: Optional[List[int]] = Field(
        default=None, description="List of 0/1 labels for points."
    )
    boxes: Optional[List[List[float]]] = Field(
        default=None,
        description="List of [x, y, w, h] boxes normalized to 0-1.",
    )
    box_labels: Optional[List[int]] = Field(
        default=None, description="List of 0/1 labels for boxes."
    )
    instance_prompt: Optional[bool] = Field(
        default=False, description="Enable instance tracking style point prompts."
    )
    output_prob_thresh: Optional[float] = Field(
        default=0.5, description="Score threshold for outputs."
    )

    # New unified prompts list for batched prompts (preferred)
    prompts: Optional[List[Sam3Prompt]] = Field(
        default=None, description="List of unified prompts (text and/or visual)."
    )

    @root_validator(skip_on_failure=True)
    def _normalize_prompts(cls, values):
        prompts = values.get("prompts")
        # If prompts not provided, wrap legacy fields into a single prompt when present
        if not prompts:
            legacy_any = any(
                values.get(k) is not None
                for k in ["text", "boxes", "box_labels", "points", "point_labels"]
            )
            if legacy_any:
                prompts = [
                    Sam3Prompt(
                        type="visual" if values.get("boxes") else "text",
                        text=values.get("text"),
                        boxes=values.get("boxes"),
                        box_labels=values.get("box_labels"),
                        points=values.get("points"),
                        point_labels=values.get("point_labels"),
                    )
                ]
        # Enforce max prompts limit
        if prompts:
            if len(prompts) > SAM3_MAX_PROMPT_BATCH_SIZE:
                raise ValueError(
                    f"Exceeded SAM3_MAX_PROMPT_BATCH_SIZE={SAM3_MAX_PROMPT_BATCH_SIZE}: got {len(prompts)}"
                )
            values["prompts"] = prompts
        return values
