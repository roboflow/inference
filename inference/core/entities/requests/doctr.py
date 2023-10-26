from typing import List, Optional, Union

from pydantic import Field, validator

from inference.core.entities.requests.inference import (
    BaseRequest,
    InferenceRequestImage,
)


class DoctrOCRInferenceRequest(BaseRequest):
    """
    DocTR inference request.

    Attributes:
        api_key (Optional[str]): Roboflow API Key.
    """

    image: Union[List[InferenceRequestImage], InferenceRequestImage]
    doctr_version_id: Optional[str] = "default"
    model_id: Optional[str] = Field()

    @validator("model_id", always=True, allow_reuse=True)
    def validate_model_id(cls, value, values):
        if value is not None:
            return value
        if values.get("doctr_version_id") is None:
            return None
        return f"doctr/{values['doctr_version_id']}"
