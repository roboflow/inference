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


def _ensure_cache_path_has_no_child_symlinks(
    cache_root: str,
    target_path: str,
    model_id: str,
) -> None:
    cache_root = os.path.abspath(cache_root)
    target_path = os.path.abspath(target_path)
    try:
        relative_path = os.path.relpath(target_path, cache_root)
        if relative_path == os.pardir or relative_path.startswith(
            os.pardir + os.sep
        ):
            raise ValueError
        expected_resolved_path = os.path.normpath(
            os.path.join(os.path.realpath(cache_root), relative_path)
        )
        if os.path.realpath(target_path) != expected_resolved_path:
            raise ValueError
    except ValueError as error:
        raise InsecureModelIdentifierError(
            message=(
                f"Refusing model cache path for {model_id} because it escapes "
                "its cache root or traverses a symbolic link."
            ),
            help_url="https://inference-models.roboflow.com/errors/model-loading/#insecuremodelidentifiererror",
        ) from error


def generate_model_cache_root_for_model_id(model_id: str) -> str:
    model_id_slug = slugify_model_id_to_os_safe_format(model_id=model_id)
    models_cache_dir = generate_models_cache_dir()
    result = os.path.join(models_cache_dir, model_id_slug)
    _ensure_cache_path_has_no_child_symlinks(
        cache_root=models_cache_dir,
        target_path=result,
        model_id=model_id,
    )
    return result


def generate_model_package_cache_path(model_id: str, package_id: str) -> str:
    ensure_package_id_is_os_safe(model_id=model_id, package_id=package_id)
    model_cache_root = generate_model_cache_root_for_model_id(model_id=model_id)
    result = os.path.join(model_cache_root, package_id)
    _ensure_cache_path_has_no_child_symlinks(
        cache_root=model_cache_root,
        target_path=result,
        model_id=model_id,
    )
    return result


def generate_shared_blobs_path() -> str:
    return os.path.abspath(os.path.join(INFERENCE_HOME, "shared-blobs"))
