from typing import List, Optional, Union

from pydantic import BaseModel, Field, model_validator

from inference.core.entities.requests.inference import (
    BaseRequest,
    InferenceRequestImage,
)
from inference.core.entities.requests.owlv2 import TrainingImage


class RFDETRFewShotInferenceRequest(BaseRequest):
    """Request for RF-DETR few-shot object detection inference.

    Either ``training_data`` or ``model_hash`` must be provided.
    If ``training_data`` is provided, the server will hash it, check for
    a cached LoRA adapter, and train one if needed.  If ``model_hash``
    is provided, the server will look up the cached adapter directly.

    Attributes:
        image: Query image(s) to run detection on.
        training_data: Training images with bounding box annotations.
        model_hash: Hash of a previously-trained adapter (skip training_data).
        confidence: Confidence threshold for predictions.
        iou_threshold: IoU threshold for NMS.
        model_variant: RF-DETR variant (rfdetr-nano/small/medium/base/large).
        lora_rank: Rank of LoRA adapters.
        learning_rate: Learning rate for LoRA fine-tuning.
        num_epochs: Number of training epochs.
        visualize_predictions: If true, return visualised predictions.
    """

    image: Union[List[InferenceRequestImage], InferenceRequestImage] = Field(
        description="Image(s) to run detection on"
    )
    training_data: Optional[List[TrainingImage]] = Field(
        default=None,
        description="Training images with bounding box annotations for few-shot learning",
    )
    model_hash: Optional[str] = Field(
        default=None,
        description="Hash of a previously cached adapter; skips training_data",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for predictions",
    )
    iou_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="IoU threshold for NMS",
    )
    model_variant: str = Field(
        default="rfdetr-base",
        description="RF-DETR model variant: rfdetr-nano, rfdetr-small, rfdetr-medium, rfdetr-base, rfdetr-large",
    )
    lora_rank: int = Field(default=8, description="LoRA rank")
    learning_rate: float = Field(default=2e-3, description="Learning rate")
    num_epochs: int = Field(default=15, ge=1, description="Number of training epochs")
    visualize_predictions: bool = Field(
        default=False,
        description="If true, return visualised predictions as base64",
    )

    @model_validator(mode="after")
    def validate_training_or_hash(self):
        if self.training_data is None and self.model_hash is None:
            raise ValueError(
                "Either 'training_data' or 'model_hash' must be provided."
            )
        return self
