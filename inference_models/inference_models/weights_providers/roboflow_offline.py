"""`roboflow-offline-weights` provider — serves the offline-weights registry.

In ``OFFLINE_MODE`` the auto-loader transparently swaps the ``roboflow``
provider for this one. It returns the provider response recorded during an
``OFFLINE_MODE_WARM_UP`` run (or a local TRT compiler install), with every
package pointing at the local cache. Auto-negotiation then filters and ranks
those packages against the current environment exactly as it would online.
"""

import os
from typing import Optional

from inference_models.errors import ModelRetrievalError
from inference_models.logger import LOGGER
from inference_models.models.auto_loaders.model_cache_paths import (
    resolve_existing_model_package_cache_path,
)
from inference_models.weights_providers import offline_registry
from inference_models.weights_providers.entities import (
    ModelMetadata,
    ModelPackageMetadata,
)

ROBOFLOW_OFFLINE_WEIGHTS_PROVIDER = "roboflow-offline-weights"


def get_roboflow_offline_weights(
    model_id: str,
    api_key: Optional[str] = None,
    **kwargs,
) -> ModelMetadata:
    model_metadata = offline_registry.load_model_metadata(model_id=model_id)
    if model_metadata is not None:
        model_metadata = _with_materialized_packages_only(
            model_metadata=model_metadata
        )
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


def _package_is_materialized(
    model_metadata: ModelMetadata,
    package: ModelPackageMetadata,
) -> bool:
    try:
        package_dir = resolve_existing_model_package_cache_path(
            model_id=package.cache_model_id or model_metadata.model_id,
            package_id=package.package_id,
        )
    except Exception:
        return False
    if package_dir is None:
        return False
    return all(
        os.path.isfile(os.path.join(package_dir, artefact.file_handle))
        for artefact in package.package_artefacts
    )


def _with_materialized_packages_only(
    model_metadata: ModelMetadata,
) -> Optional[ModelMetadata]:
    """Drop recorded packages whose artefacts are not on disk.

    The registry records the full provider response; only packages that were
    actually loaded have materialized files. Filtering here keeps negotiation
    from attempting - and loudly failing on - packages that cannot serve.
    """
    materialized = [
        package
        for package in model_metadata.model_packages
        if _package_is_materialized(model_metadata=model_metadata, package=package)
    ]
    if not materialized:
        LOGGER.warning(
            "Offline-weights record for %s lists %s package(s) but none have "
            "materialized artefacts on disk.",
            model_metadata.model_id,
            len(model_metadata.model_packages),
        )
        return None
    if len(materialized) == len(model_metadata.model_packages):
        return model_metadata
    return ModelMetadata(
        model_id=model_metadata.model_id,
        model_architecture=model_metadata.model_architecture,
        model_packages=materialized,
        task_type=model_metadata.task_type,
        model_variant=model_metadata.model_variant,
        model_dependencies=model_metadata.model_dependencies,
        recommended_parameters=model_metadata.recommended_parameters,
    )
