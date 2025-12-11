from typing import Any, Dict, List, Optional, Union

from pydantic import Field, validator

from inference.core.entities.requests.inference import (
    BaseRequest,
    InferenceRequestImage,
)


class Sam3_3D_Objects_InferenceRequest(BaseRequest):
    """SAM3D inference request for 3D object generation.

    Attributes:
        api_key (Optional[str]): Roboflow API Key.
        image (InferenceRequestImage): The input image to be used for 3D generation.
        mask_input: Mask(s) in any supported format - polygon, binary mask, or RLE.
    """

    image: InferenceRequestImage = Field(
        description="The input image to be used for 3D generation.",
    )

    mask_input: Any = Field(
        description="Mask input in any supported format: "
        "polygon [x1,y1,x2,y2,...], binary mask (base64), RLE dict, or list of these.",
    )

    model_id: Optional[str] = Field(
        default="sam3-3d-objects", description="The model ID for SAM3_3D."
    )

    @validator("model_id", always=True)
    def validate_model_id(cls, value):
        if value is not None:
            return value
        return "sam3-3d-objects"
