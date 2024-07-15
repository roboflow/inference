from typing import List, Optional, Union

from pydantic import Field, ValidationInfo, field_validator

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
    model_id: Optional[str] = Field(None)

    @field_validator("model_id", validate_default=True)
    @classmethod
    def validate_model_id(cls, value, info: ValidationInfo):
        if value is not None:
            return value

        doctr_version_id = info.data.get("doctr_version_id")
        if doctr_version_id is None:
            return None
        return f"doctr/{doctr_version_id}"
