"""Single source of truth for on-disk model cache paths.

Kept dependency-light (no weights-provider or auto-loader imports) so it can be
imported by both the auto-loader and weights-provider layers without creating
circular imports.
"""

import hashlib
import os
import re

from inference_models.configuration import INFERENCE_HOME
from inference_models.errors import InsecureModelIdentifierError


def slugify_model_id_to_os_safe_format(model_id: str) -> str:
    model_id_slug = re.sub(r"[^A-Za-z0-9_-]+", "-", model_id)
    model_id_slug = re.sub(r"[_-]{2,}", "-", model_id_slug)
    if not model_id_slug:
        model_id_slug = "special-char-only-model-id"
    if len(model_id_slug) > 48:
        model_id_slug = model_id_slug[:48]
    digest = hashlib.blake2s(model_id.encode("utf-8"), digest_size=4).hexdigest()
    return f"{model_id_slug}-{digest}"


def ensure_package_id_is_os_safe(model_id: str, package_id: str) -> None:
    if re.search(r"[^A-Za-z0-9]", package_id):
        raise InsecureModelIdentifierError(
            message=f"Attempted to load model: {model_id} using package ID: {package_id} which "
            f"has invalid format. ID is expected to contain only ASCII characters and numbers to "
            f"ensure safety of local cache. If you see this error running your model on Roboflow platform, "
            f"raise the issue: https://github.com/roboflow/inference/issues. If you are running `inference` "
            f"outside of the platform, verify that your weights provider keeps the model packages identifiers "
            f"in the expected format.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#insecuremodelidentifiererror",
        )


def generate_models_cache_dir() -> str:
    return os.path.abspath(os.path.join(INFERENCE_HOME, "models-cache"))


def generate_model_cache_root_for_model_id(model_id: str) -> str:
    model_id_slug = slugify_model_id_to_os_safe_format(model_id=model_id)
    return os.path.join(generate_models_cache_dir(), model_id_slug)


def generate_model_package_cache_path(model_id: str, package_id: str) -> str:
    ensure_package_id_is_os_safe(model_id=model_id, package_id=package_id)
    return os.path.join(
        generate_model_cache_root_for_model_id(model_id=model_id), package_id
    )


def generate_shared_blobs_path() -> str:
    return os.path.abspath(os.path.join(INFERENCE_HOME, "shared-blobs"))
