from typing import Literal

from pydantic import Field, field_validator

from inference.core.entities.requests.inference import CVInferenceRequest


class ImageEmbeddingRequest(CVInferenceRequest):
    """Raw features or logits from an existing ResNet, ViT or DINOv3 classifier."""

    model_id: str = Field(
        min_length=1, description="Classification model ID or pretrained alias"
    )
    model_type: Literal["classification"] = "classification"
    output_type: Literal["feature_vector", "logits"] = Field(
        default="feature_vector",
        description="Feature Vector: input to the final linear layer. Logits: its output before Softmax or Sigmoid.",
    )

    @field_validator("image")
    @classmethod
    def require_images(cls, image):
        if isinstance(image, list) and not image:
            raise ValueError("At least one image is required.")
        return image
