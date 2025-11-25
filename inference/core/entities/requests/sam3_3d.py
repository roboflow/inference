from typing import List, Optional

from pydantic import Field, validator

from inference.core.entities.requests.inference import (
    BaseRequest,
    InferenceRequestImage,
)


class Sam3_3D_Objects_InferenceRequest(BaseRequest):
    """SAM3_3D inference request for 3D object generation.

    Attributes:
        api_key (Optional[str]): Roboflow API Key.
        image (InferenceRequestImage): The input image to be used for 3D generation.
        mask_input (List[float]): The mask input defining the object region as a flat list of coordinates.
            Format: [x1, y1, x2, y2, x3, y3, ...] (COCO polygon format)
    """

    image: InferenceRequestImage = Field(
        description="The input image to be used for 3D generation.",
    )

    mask_input: List[float] = Field(
        description="The mask input defining the object region as a flat list of polygon coordinates "
        "in the format [x1, y1, x2, y2, x3, y3, ...] (COCO polygon format).",
        examples=[[100.0, 100.0, 200.0, 100.0, 200.0, 200.0, 100.0, 200.0]],
    )

    model_id: Optional[str] = Field(
        default="sam3-3d-objects", description="The model ID for SAM3_3D."
    )

    @validator("model_id", always=True)
    def validate_model_id(cls, value):
        if value is not None:
            return value
        return "sam3-3d-objects"
