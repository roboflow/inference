from typing import List, Optional, Union

from pydantic import BaseModel, Field, validator

from inference.core.entities.requests.inference import (
    BaseRequest,
    InferenceRequestImage,
)
from inference.core.env import SAM3_MAX_PROMPT_BATCH_SIZE


class Sam3Prompt(BaseModel):
    """Unified prompt that can contain text and/or geometry.

    Absolute pixel coordinates are used for boxes. Labels accept 0/1 or booleans.
    """

    type: Optional[str] = Field(
        default=None, description="Optional hint: 'text' or 'visual'"
    )
    text: Optional[str] = Field(default=None)

    output_prob_thresh: Optional[float] = Field(
        default=None,
        description="Score threshold for this prompt's outputs. Overrides request-level threshold if set.",
    )

    # Absolute-coordinate boxes (preferred) in pixels.
    # XYWH absolute pixels
    class Box(BaseModel):
        x: float
        y: float
        width: float
        height: float

    # XYXY absolute pixels
    class BoxXYXY(BaseModel):
        x0: float
        y0: float
        x1: float
        y1: float

    # Single unified boxes field; each entry can be XYWH or XYXY
    boxes: Optional[List[Union[Box, BoxXYXY]]] = Field(
        default=None,
        description="Absolute pixel boxes as either XYWH or XYXY entries",
    )
    box_labels: Optional[List[Union[int, bool]]] = Field(
        default=None, description="List of 0/1 or booleans for boxes"
    )

    @validator("boxes", always=True)
    def _validate_visual_boxes(cls, boxes, values):
        prompt_type = values.get("type")
        if prompt_type == "visual":
            if not boxes or len(boxes) == 0:
                raise ValueError("Visual prompt requires at least one box")
        return boxes

    @validator("box_labels", always=True)
    def _validate_box_labels(cls, labels, values):
        boxes = values.get("boxes")
        if labels is None:
            return labels
        if boxes is None or len(labels) != len(boxes):
            raise ValueError("box_labels must match boxes length when provided")
        return labels

    @validator("output_prob_thresh")
    def _validate_output_prob_thresh(cls, v):
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError("output_prob_thresh must be between 0.0 and 1.0")
        return v


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
