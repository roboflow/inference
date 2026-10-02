import hashlib
import json

from roboflow_workflows.prototypes.model_registration import model_registration_key

from inference.core.entities.responses.embeddings import EmbeddingInfo

IMAGE_EMBEDDINGS = "image_embeddings"


def model_cache_key(model_id, required_capabilities=None, output_type="feature_vector"):
    """Return the shared model-registration identity using the legacy signature.

    Args:
        model_id: Model version or alias being registered.
        required_capabilities: Required model operations, if any.
        output_type: Selected image-embedding representation.

    Returns:
        Model-manager cache key, unchanged for registrations without capabilities.
    """
    key = model_registration_key(
        model_id,
        required_capabilities=required_capabilities,
        output_type=output_type,
    )

    return key


def make_embedding_info(
    model_id, feature_info, preprocessing, backend, precision, dimension
):
    identity = {
        "model_id": model_id,
        "feature_definition": feature_info["feature_definition"],
        "dimension": dimension,
        "normalization": feature_info["normalization"],
        "preprocessing": preprocessing,
    }
    space_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return EmbeddingInfo(
        **feature_info,
        **{key: value for key, value in identity.items() if key not in feature_info},
        space_id=space_id,
        backend=backend,
        precision=precision,
    )
