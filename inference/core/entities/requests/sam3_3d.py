from typing import List, Optional, Union

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
        mask_input: The mask input defining object region(s). Can be either:
            - A single mask as flat list: [x1, y1, x2, y2, x3, y3, ...] (COCO polygon format)
            - Multiple masks as list of flat lists: [[x1, y1, ...], [x1, y1, ...], ...]
    """

    image: InferenceRequestImage = Field(
        description="The input image to be used for 3D generation.",
    )

    mask_input: Union[List[float], List[List[float]]] = Field(
        description="The mask input defining object region(s). Can be either a single mask "
        "as a flat list of polygon coordinates [x1, y1, x2, y2, ...] (COCO polygon format), "
        "or multiple masks as a list of flat lists [[x1, y1, ...], [x1, y1, ...], ...].",
        examples=[
            [100.0, 100.0, 200.0, 100.0, 200.0, 200.0, 100.0, 200.0],
            [[100.0, 100.0, 200.0, 100.0, 200.0, 200.0], [300.0, 300.0, 400.0, 300.0, 400.0, 400.0]],
        ],
    )

    model_id: Optional[str] = Field(
        default="sam3-3d-objects", description="The model ID for SAM3_3D."
    )

    @validator("model_id", always=True)
    def validate_model_id(cls, value):
        if value is not None:
            return value
        return "sam3-3d-objects"
