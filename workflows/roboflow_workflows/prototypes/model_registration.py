"""Shared identities for capability-specific model registrations."""

from typing import Iterable, Optional


def model_registration_key(
    model_id: str,
    *,
    required_capabilities: Optional[Iterable[str]] = None,
    output_type: str = "feature_vector",
) -> str:
    """Identify the model instance required by a registration.

    Args:
        model_id: Model version or alias being registered.
        required_capabilities: Operations the loaded instance must support.
        output_type: Feature or logit representation for image embeddings.

    Returns:
        Cache identity shared by dependency preloading and the server manager.
        Registrations without capabilities retain the bare model ID.
    """
    if not required_capabilities:
        return model_id

    capabilities = sorted(set(required_capabilities))
    key = f"{model_id}:capabilities={','.join(capabilities)}"
    if "image_embeddings" in capabilities and output_type != "feature_vector":
        key += f":output_type={output_type}"

    return key
