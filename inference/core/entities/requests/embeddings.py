from typing import Literal

from pydantic import Field, field_validator

from inference.core.entities.requests.inference import CVInferenceRequest


class ImageEmbeddingRequest(CVInferenceRequest):
    """Raw features or logits from a ResNet, ViT or DINOv3 classifier, or raw
    features from a pretrained embedding model."""

    model_id: str = Field(
        min_length=1,
        description="Classification model ID, pretrained classifier alias, or "
        "pretrained embedding model ID such as `dinov3/vits16`",
    )
    model_type: Literal["classification"] = "classification"
    output_type: Literal["feature_vector", "logits"] = Field(
        default="feature_vector",
        description="Feature Vector: input to a classifier's final linear layer, or the "
        "image embedding of a pretrained embedding model. Logits: a classifier's final "
        "layer output before Softmax or Sigmoid; pretrained embedding models have none.",
    )

    @field_validator("image")
    @classmethod
    def require_images(cls, image):
        if isinstance(image, list) and not image:
            raise ValueError("At least one image is required.")
        return image
