from typing import List, Optional, Union

from pydantic import BaseModel

from inference.core.entities.common import ApiKey
from inference.core.entities.requests.inference import InferenceRequestImage


class DoctrOCRInferenceRequest(BaseModel):
    """
    DocTR inference request.

    Attributes:
        api_key (Optional[str]): Roboflow API Key.
    """

    api_key: Optional[str] = ApiKey
    image: Union[List[InferenceRequestImage], InferenceRequestImage]
    doctr_version_id: Optional[str] = "default"
