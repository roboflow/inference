from typing import Optional

from pydantic import ConfigDict

from inference.core.entities.common import ApiKey, ModelID, ModelType
from inference.core.entities.requests.inference import InferenceRequest


class AddModelRequest(InferenceRequest):
    """Request to add a model to the inference server.

    Attributes:
        model_id (str): A unique model identifier.
        model_type (Optional[str]): The type of the model, usually referring to what task the model performs.
        api_key (Optional[str]): Roboflow API Key that will be passed to the model during initialization for artifact retrieval.
    """

    model_config = ConfigDict(protected_namespaces=())
    model_id: str = ModelID
    model_type: Optional[str] = ModelType
    api_key: Optional[str] = ApiKey


class ClearModelRequest(InferenceRequest):
    """Request to clear a model from the inference server.

    Attributes:
        model_id (str): A unique model identifier.
    """

    model_config = ConfigDict(protected_namespaces=())
    model_id: str = ModelID
    api_key: Optional[str] = ApiKey
