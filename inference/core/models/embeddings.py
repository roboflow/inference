import hashlib
import json
from typing import List

import numpy as np
from roboflow_workflows.prototypes.model_registration import model_registration_key

from inference.core.entities.responses.embeddings import EmbeddingInfo
from inference.core.utils.image_utils import load_image_bgr
from inference_models.errors import ModelInputError

IMAGE_EMBEDDINGS = "image_embeddings"
FEATURE_VECTOR_OUTPUT_TYPE = "feature_vector"


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


def require_feature_vector_output(output_type):
    """Reject representations other than feature vectors as a user input error.

    Pretrained embedding models have no classification head, so they have no
    logits. Adapters check before loading weights, so the request fails as bad
    input rather than as a model package that could not be loaded.

    Args:
        output_type: Requested image-embedding representation.

    Raises:
        ModelInputError: If the representation is not a feature vector.
    """
    if output_type != FEATURE_VECTOR_OUTPUT_TYPE:
        raise ModelInputError(
            message=f"Pretrained embedding models return only "
            f"`{FEATURE_VECTOR_OUTPUT_TYPE}` embeddings, requested `{output_type}`. "
            f"Logits are available for ResNet, ViT and DINOv3 classifiers.",
            help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
        )


def load_embedding_request_images(request) -> List[np.ndarray]:
    """Decode the image or images of an embedding request as BGR arrays.

    Args:
        request: Image embedding request with one image or a list.

    Returns:
        Decoded images in request order.
    """
    images = request.image if isinstance(request.image, list) else [request.image]
    decoded = [
        load_image_bgr(
            image,
            disable_preproc_auto_orient=bool(request.disable_preproc_auto_orient),
        )
        for image in images
    ]

    return decoded


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
