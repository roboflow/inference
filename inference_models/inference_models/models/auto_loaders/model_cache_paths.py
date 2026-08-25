"""Single source of truth for on-disk model cache paths.

Kept dependency-light (no weights-provider or auto-loader imports) so it can be
imported by both the auto-loader and weights-provider layers without creating
circular imports.

Paths are deduced, not discovered: a package lives at
``models-cache/<slug(model_id)>/<package_id>`` and nowhere else. The slug is
the collision-resistant ``v2-<prefix>-<128-bit digest>`` format introduced in
`0.32.0` (kept for on-disk compatibility with every cache written since).
Legacy 32-bit-digest cache directories from releases before `0.32.0` are no
longer read; they need one online run (or `OFFLINE_MODE_WARM_UP`) to
re-materialize under current paths.
"""

import hashlib
import os
import re
from typing import Optional

from inference_models.configuration import INFERENCE_HOME
from inference_models.errors import InsecureModelIdentifierError

MODEL_CONFIG_FILE_NAME = "model_config.json"
MODEL_CACHE_SLUG_VERSION = "v2"
MODEL_PACKAGE_ID_MAX_LENGTH = 255
_WINDOWS_RESERVED_PACKAGE_IDS = {
    "AUX",
    "CON",
    "NUL",
    "PRN",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


def _slugify_model_id_prefix(model_id: str) -> str:
    model_id_slug = re.sub(r"[^A-Za-z0-9_-]+", "-", model_id)
    model_id_slug = re.sub(r"[_-]{2,}", "-", model_id_slug)
    if not model_id_slug:
        model_id_slug = "special-char-only-model-id"
    if len(model_id_slug) > 48:
        model_id_slug = model_id_slug[:48]
    return model_id_slug


def slugify_model_id_to_os_safe_format(model_id: str) -> str:
    """Return the model-cache slug: readable prefix + 128-bit digest.

    The digest makes distinct model ids collision-resistant on shared,
    long-lived cache volumes; the regex strips path separators, so slugs can
    never escape the cache directory.
    """

    model_id_slug = _slugify_model_id_prefix(model_id=model_id)
    digest = hashlib.blake2s(model_id.encode("utf-8"), digest_size=16).hexdigest()
    return f"{MODEL_CACHE_SLUG_VERSION}-{model_id_slug}-{digest}"


def ensure_package_id_is_os_safe(model_id: str, package_id: str) -> None:
    valid_format = (
        isinstance(package_id, str)
        and re.fullmatch(r"[A-Za-z0-9]+", package_id) is not None
        and len(package_id) <= MODEL_PACKAGE_ID_MAX_LENGTH
        and package_id.upper() not in _WINDOWS_RESERVED_PACKAGE_IDS
    )
    if not valid_format:
        raise InsecureModelIdentifierError(
            message=f"Attempted to load model: {model_id} using package ID: {package_id} which "
            f"has invalid format. ID must be non-empty, contain only ASCII letters and numbers, "
            f"fit within {MODEL_PACKAGE_ID_MAX_LENGTH} characters, and not be a reserved device name "
            f"to ensure safety of local cache. If you see this error running your model on Roboflow platform, "
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


def resolve_existing_model_package_cache_path(
    model_id: str,
    package_id: str,
) -> Optional[str]:
    """Return the package directory when it exists on disk, else ``None``.

    Pure deduction: identities are validated, the path is joined, existence is
    checked. Ownership/attribution of directory contents is the concern of the
    layers that read them (auto-loader manifests, local-TRT discovery, the
    offline-weights registry) — not of path resolution.
    """

    package_path = generate_model_package_cache_path(
        model_id=model_id, package_id=package_id
    )
    if not os.path.isdir(package_path):
        return None
    return package_path


def generate_shared_blobs_path() -> str:
    return os.path.abspath(os.path.join(INFERENCE_HOME, "shared-blobs"))
