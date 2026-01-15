from typing import Callable, Dict, Optional

from inference_models.errors import ModelRetrievalError
from inference_models.weights_providers.entities import ModelMetadata
from inference_models.weights_providers.roboflow import get_roboflow_model

ModelId = str
ApiKey = Optional[str]
WeightsProvider = Callable[[ModelId, ApiKey, ...], ModelMetadata]

WEIGHTS_PROVIDERS: Dict[str, WeightsProvider] = {  # type: ignore
    "roboflow": get_roboflow_model,
}


def get_model_from_provider(
    model_id: ModelId, provider: str, api_key: ApiKey = None, **kwargs
) -> ModelMetadata:
    if provider not in WEIGHTS_PROVIDERS:
        raise ModelRetrievalError(
            message=f"Requested model to be retrieved using '{provider}' provider which is not implemented.",
            help_url="https://todo",
        )
    return WEIGHTS_PROVIDERS[provider](model_id, api_key, **kwargs)


def register_model_provider(
    provider_name: str, provider_handler: WeightsProvider
) -> None:
    WEIGHTS_PROVIDERS[provider_name] = provider_handler
