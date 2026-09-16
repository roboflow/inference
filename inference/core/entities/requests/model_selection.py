import hmac
import json
import secrets
from typing import Optional

from inference.models.aliases import resolve_roboflow_model_alias

_MODEL_SELECTION_SECRET = secrets.token_bytes(32)


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
