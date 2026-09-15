from typing import Optional

from pydantic import BaseModel, Field, model_validator

from inference.core import env
from inference_sdk.http.entities import (
    model_selection_cache_key as model_selection_cache_key,
)


class ModelSelectionRequest(BaseModel):
    model_package_id: Optional[str] = Field(
        default=None,
        min_length=1,
        description="Exact model package ID. Cannot be combined with backend or quantization.",
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

    @model_validator(mode="after")
    def validate_model_selection(self):
        if self.model_package_id is not None and (
            self.backend is not None or self.quantization is not None
        ):
            raise ValueError(
                "model_package_id cannot be combined with backend or quantization."
            )
        if model_selection_kwargs(self) and not env.USE_INFERENCE_MODELS:
            raise ValueError(
                "Model package selection requires USE_INFERENCE_MODELS=true."
            )
        return self


def model_selection_kwargs(request) -> dict:
    return {
        name: value
        for name in ("model_package_id", "backend", "quantization")
        if (value := getattr(request, name, None)) is not None
    }
