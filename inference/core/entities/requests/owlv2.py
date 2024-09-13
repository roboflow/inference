from typing import List, Optional, Union

from pydantic import BaseModel, Field, validator

from inference.core.entities.common import ApiKey
from inference.core.entities.requests.inference import (
    BaseRequest,
    InferenceRequestImage,
)
from inference.core.env import OWLV2_VERSION_ID


class TrainBox(BaseModel):
    x: int
    y: int
    w: int
    h: int
    cls: str


class TrainingImage(BaseModel):
    boxes: List[TrainBox]
    image: InferenceRequestImage


class OwlV2InferenceRequest(BaseRequest):
    """Request for gaze detection inference.

    Attributes:
        api_key (Optional[str]): Roboflow API Key.
        owlv2_version_id (Optional[str]): The version ID of Gaze to be used for this request.
        image (Union[List[InferenceRequestImage], InferenceRequestImage]): Image(s) for inference.
        training_data (List[TrainingImage])
        confidence (float)
    """

    owlv2_version_id: Optional[str] = Field(
        default=OWLV2_VERSION_ID,
        examples=["owlv2-base-patch16-ensemble"],
        description="The version ID of owlv2 to be used for this request.",
    )

    image: Union[List[InferenceRequestImage], InferenceRequestImage]
    training_data: List[TrainingImage]
    model_id: Optional[str] = Field(None)
    confidence: Optional[float] = 0.99
    visualize_predictions: bool = False
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
