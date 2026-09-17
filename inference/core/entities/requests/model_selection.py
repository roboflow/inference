import hmac
import json
import secrets
from typing import Optional

from inference.core import env
from inference.models.aliases import resolve_roboflow_model_alias

_MODEL_SELECTION_SECRET = secrets.token_bytes(32)


def validate_model_selection(request):
    selectors = model_selection_kwargs(request)
    if "model_package_id" in selectors and (
        "backend" in selectors or "quantization" in selectors
    ):
        raise ValueError(
            "model_package_id cannot be combined with backend or quantization."
        )
    if selectors and not env.USE_INFERENCE_MODELS:
        raise ValueError("Model package selection requires USE_INFERENCE_MODELS=true.")
    return request


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
