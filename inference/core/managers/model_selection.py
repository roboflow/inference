import hmac
import json
import secrets

from inference.models.aliases import resolve_roboflow_model_alias

_MODEL_SELECTION_SECRET = secrets.token_bytes(32)


def model_selection_cache_key(model_id: str, model_package_id: str) -> str:
    model_id = resolve_roboflow_model_alias(model_id)
    payload = json.dumps([model_id, model_package_id], separators=(",", ":")).encode()
    digest = hmac.digest(_MODEL_SELECTION_SECRET, payload, "sha256").hex()
    return f"{model_id}:package:{digest}"


def public_model_id(model_id: str) -> str:
    return model_id.partition(":package:")[0]
