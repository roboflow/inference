from typing import List, Optional, Union

from pydantic import Field, validator

from inference.core.entities.requests.inference import (
    BaseRequest,
    InferenceRequestImage,
)
from inference.core.env import EASYOCR_VERSION_ID


class EasyOCRInferenceRequest(BaseRequest):
    """
    EasyOCR inference request.

    Attributes:
        api_key (Optional[str]): Roboflow API Key.
    """

    image: Union[List[InferenceRequestImage], InferenceRequestImage]
    easy_ocr_version_id: Optional[str] = EASYOCR_VERSION_ID
    model_id: Optional[str] = Field(None)
    language_codes: Optional[List[str]] = Field(default=["en"])
    quantize: Optional[bool] = Field(
        default=False,
        description="Quantized models are smaller and faster, but may be less accurate and won't work correctly on all hardware.",
    )

    # TODO[pydantic]: We couldn't refactor the `validator`, please replace it by `field_validator` manually.
    # Check https://docs.pydantic.dev/dev-v2/migration/#changes-to-validators for more information.
    @validator("model_id", always=True, allow_reuse=True)
    def validate_model_id(cls, value, values):
        if value is not None:
            return value
        if values.get("easy_ocr_version_id") is None:
            return None
        return f"easy_ocr/{values['easy_ocr_version_id']}"
