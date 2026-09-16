import hmac
import json
import secrets
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from inference.core import env
from inference.models.aliases import resolve_roboflow_model_alias

_MODEL_SELECTION_SECRET = secrets.token_bytes(32)


class ModelSelectionRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

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


def model_selection_cache_key(
    model_id: str, selectors: dict, api_key: Optional[str] = None
) -> str:
    if not selectors:
        return model_id
    model_id = resolve_roboflow_model_alias(model_id)
    payload = json.dumps(
        {"model_id": model_id, "selectors": selectors, "api_key": api_key},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    digest = hmac.digest(_MODEL_SELECTION_SECRET, payload, "sha256").hex()
    return f"{model_id}:package:{digest}"
