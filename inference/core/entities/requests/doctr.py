from typing import List, Optional, Union

from pydantic import BaseModel, validator, Field

from inference.core.entities.common import ApiKey
from inference.core.entities.requests.inference import InferenceRequestImage, BaseRequest


class DoctrOCRInferenceRequest(BaseRequest):
    """
    DocTR inference request.

    Attributes:
        api_key (Optional[str]): Roboflow API Key.
    """


    image: Union[List[InferenceRequestImage], InferenceRequestImage]
    doctr_version_id: Optional[str] = "default"
    model_id: Optional[str] = Field()
    
    @validator("model_id", always=True)
    def validate_model_id(cls, value, values):
        if values.get("doctr_version_id") is None:
            return None
        return f"doctr/{values['doctr_version_id']}"


