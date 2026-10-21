"""Internal pre-load selection and exact-candidate handoff for process managers."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AdmissionCandidate:
    """Ordinary canonical identifiers of a candidate admitted by the manager.

    Args:
        model_id: Canonical provider model ID.
        package_id: Single selected package ID; not proof of compatibility.
    """

    model_id: str
    package_id: str


@dataclass(frozen=True)
class PreparedPackages:
    """Authorized metadata and ranked candidates obtained without loading weights.

    Args:
        metadata: Provider model metadata used for negotiation.
        candidates: Ordered compatible packages, including a usable cached preference.
        runtime: Same runtime observations used by ordinary negotiation.
        model_init_kwargs: Effective constructor arguments before initialization.
    """

    metadata: Any
    candidates: tuple
    runtime: Any
    model_init_kwargs: dict


def prepare_model_packages(
    model_id: str, *, api_key: str, **kwargs
) -> PreparedPackages:
    """Run the ordinary online provider and negotiation path without model loading.

    Args:
        model_id: Requested alias or canonical model ID.
        api_key: Credential forwarded to the ordinary provider.
        **kwargs: The same loader configuration that the dedicated worker will use.

    Returns:
        Authorized ranked metadata; no admission is granted by preparation.

    Raises:
        NotImplementedError: For offline, local, shared/dependency or proxy paths.
        Exception: Ordinary authorization, provider and negotiation failures propagate.
    """
    from inference_models.models.auto_loaders.core import AutoModel

    result = AutoModel.from_pretrained(
        model_id, api_key=api_key, _admission="prepare", **kwargs
    )
    return result


def load_admitted_candidate(
    model_id: str, *, candidate: AdmissionCandidate, api_key: str, **kwargs
) -> Any:
    """Revalidate and load one admitted package without another-package fallback.

    Args:
        model_id: Original requested alias or canonical ID.
        candidate: Manager-selected ordinary canonical model/package identifiers.
        api_key: Credential rechecked through the existing provider.
        **kwargs: Original effective loader configuration, unchanged from preparation.

    Returns:
        Initialized single-candidate model.

    Raises:
        ValueError: If alias resolution or an explicit package constraint changed.
        Exception: Original compatibility, integrity and initialization failures propagate.
    """
    from inference_models.models.auto_loaders.core import AutoModel

    requested = kwargs.pop("model_package_id", None)
    if requested is not None and requested != candidate.package_id:
        raise ValueError("Explicit requested package differs from admitted candidate")
    result = AutoModel.from_pretrained(
        model_id,
        api_key=api_key,
        model_package_id=requested,
        _admission=candidate,
        **kwargs
    )
    return result
