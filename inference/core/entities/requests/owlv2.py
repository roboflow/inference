from typing import List, Optional, Union

from pydantic import BaseModel, Field, validator

from inference.core.entities.common import ApiKey
from inference.core.entities.requests.inference import (
    BaseRequest,
    InferenceRequestImage,
)
from inference.core.env import OWLV2_VERSION_ID


class TrainBox(BaseModel):
    x: int = Field(description="Center x coordinate in pixels of train box")
    y: int = Field(description="Center y coordinate in pixels of train box")
    w: int = Field(description="Width in pixels of train box")
    h: int = Field(description="Height in pixels of train box")
    cls: str = Field(description="Class name of object this box encloses")
    negative: bool = Field(
        default=False,
        description="Whether this object is a positive or negative example for this class",
    )


class TrainingImage(BaseModel):
    boxes: List[TrainBox] = Field(
        description="List of boxes and corresponding classes of examples for the model to learn from"
    )
    image: InferenceRequestImage = Field(
        description="Image data that `boxes` describes"
    )


class OwlV2InferenceRequest(BaseRequest):
    """Request for gaze detection inference.

    Attributes:
        api_key (Optional[str]): Roboflow API Key.
        owlv2_version_id (Optional[str]): The version ID of Gaze to be used for this request.
        image (Union[List[InferenceRequestImage], InferenceRequestImage]): Image(s) for inference.
        training_data (List[TrainingImage]): Training data to ground the model on
        confidence (float): Confidence threshold to filter predictions by
    """

    owlv2_version_id: Optional[str] = Field(
        default=OWLV2_VERSION_ID,
        examples=["owlv2-base-patch16-ensemble"],
        description="The version ID of owlv2 to be used for this request.",
    )
    model_id: Optional[str] = Field(
        default=None, description="Model id to be used in the request."
    )

    image: Union[List[InferenceRequestImage], InferenceRequestImage] = Field(
        description="Images to run the model on"
    )
    training_data: List[TrainingImage] = Field(
        description="Training images for the owlvit model to learn form"
    )
    confidence: Optional[float] = Field(
        default=0.99,
        examples=[0.99],
        description="Default confidence threshold for owlvit predictions. "
        "Needs to be much higher than you're used to, probably 0.99 - 0.9999",
    )
    visualize_predictions: Optional[bool] = Field(
        default=False,
        examples=[False],
        description="If true, return visualized predictions as a base64 string",
    )
    visualization_labels: Optional[bool] = Field(
        default=False,
        examples=[False],
        description="If true, labels will be rendered on prediction visualizations",
    )
    visualization_stroke_width: Optional[int] = Field(
        default=1,
        examples=[1],
        description="The stroke width used when visualizing predictions",
    )
    visualize_predictions: Optional[bool] = Field(
        default=False,
        examples=[False],
        description="If true, the predictions will be drawn on the original image and returned as a base64 string",
    )

    # TODO[pydantic]: We couldn't refactor the `validator`, please replace it by `field_validator` manually.
    # Check https://docs.pydantic.dev/dev-v2/migration/#changes-to-validators for more information.
    @validator("model_id", always=True, allow_reuse=True)
    def validate_model_id(cls, value, values):
        if value is not None:
            return value
        if values.get("owl2_version_id") is None:
            return None
        return f"google/{values['owl2_version_id']}"
