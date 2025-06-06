from typing import Optional

from inference.v1.errors import ModelRetrievalError
from inference.v1.weights_providers.entities import ModelMetadata
from inference.v1.weights_providers.roboflow import get_roboflow_model

WEIGHTS_PROVIDERS = {
    "roboflow": get_roboflow_model,
}


def get_model_from_provider(
    model_id: str, provider: str, api_key: Optional[str] = None
) -> ModelMetadata:
    if provider not in WEIGHTS_PROVIDERS:
        raise ModelRetrievalError(
            f"Requested model to be retrieved using '{provider}' provider which is not implemented."
        )
    return WEIGHTS_PROVIDERS[provider](model_id=model_id, api_key=api_key)
