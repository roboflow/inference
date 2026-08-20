"""`roboflow-offline-weights` provider — serves the offline-weights registry.

In ``OFFLINE_MODE`` the auto-loader transparently swaps the ``roboflow``
provider for this one. It returns the provider response recorded during an
``OFFLINE_MODE_WARM_UP`` run (or a local TRT compiler install), with every
package pointing at the local cache. Auto-negotiation then filters and ranks
those packages against the current environment exactly as it would online.
"""

from typing import Optional

from inference_models.errors import ModelRetrievalError
from inference_models.weights_providers import offline_registry
from inference_models.weights_providers.entities import ModelMetadata

ROBOFLOW_OFFLINE_WEIGHTS_PROVIDER = "roboflow-offline-weights"


def get_roboflow_offline_weights(
    model_id: str,
    api_key: Optional[str] = None,
    **kwargs,
) -> ModelMetadata:
    model_metadata = offline_registry.load_model_metadata(model_id=model_id)
    if model_metadata is None:
        raise ModelRetrievalError(
            message=(
                f"No offline-weights record found for model '{model_id}'. "
                f"Build the offline cache by running once with network access "
                f"and OFFLINE_MODE_WARM_UP=True, then restart with "
                f"OFFLINE_MODE=True."
            ),
            help_url="https://inference-models.roboflow.com/errors/model-retrieval/#modelretrievalerror",
        )
    return model_metadata
