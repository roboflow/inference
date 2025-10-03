from typing import List, Optional

from pydantic import BaseModel, Field, validator

from inference.core.entities.requests.inference import (
    BaseRequest,
    InferenceRequestImage,
)


class Sam3InferenceRequest(BaseRequest):
    """SAM3 inference request.

    Attributes:
        model_id (Optional[str]): The model ID to be used, typically `sam3`.
    """

    model_id: Optional[str] = Field(
        default="sam3/paper_image_only_checkpoint_presence_0.35_completed_model_only",
        description="The model ID of SAM3. Use 'sam3/paper_image_only_checkpoint_presence_0.35_completed_model_only' to target the generic base model.",
    )
    # sam3_version_id: Optional[str] = Field(
    #     default="paper_image_only_checkpoint_presence_0.35_completed_model_only",
    #     description="Placeholder version field required by core model loader.",
    # )

    # @validator("model_id", always=True)
    # def validate_model_id(cls, value, values):
    #     if value is not None:
    #         return value
    #     if values.get("sam_version_id") is None:
    #         return None
    #     return f"sam3/{values['sam_version_id']}"


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
    # Prompts: support text, boxes, and points similar to SAM3 demo API
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
