from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from inference.core.entities.common import ApiKey, ModelID, ModelType
from inference.core.entities.requests.model_selection import validate_model_selection


class AddModelRequest(BaseModel):
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
    model_package_id: Optional[str] = Field(
        default=None, min_length=1, description="Exact model package ID."
    )
    backend: Optional[str] = Field(
        default=None,
        min_length=1,
        description="Required package backend, such as trt or onnx.",
    )
    quantization: Optional[str] = Field(
        default=None,
        min_length=1,
        description="Required package quantization, such as fp16 or fp32.",
    )

    validate_model_selection = model_validator(mode="after")(validate_model_selection)


class ClearModelRequest(BaseModel):
    """Request to clear a model from the inference server.

    Attributes:
        model_id (str): A unique model identifier.
    """

    model_config = ConfigDict(protected_namespaces=())
    model_id: str = ModelID
    api_key: Optional[str] = ApiKey
    model_package_id: Optional[str] = Field(
        default=None, min_length=1, description="Exact model package ID."
    )
    backend: Optional[str] = Field(
        default=None,
        min_length=1,
        description="Required package backend, such as trt or onnx.",
    )
    quantization: Optional[str] = Field(
        default=None,
        min_length=1,
        description="Required package quantization, such as fp16 or fp32.",
    )

    validate_model_selection = model_validator(mode="after")(validate_model_selection)
