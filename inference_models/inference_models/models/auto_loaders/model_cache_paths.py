"""Single source of truth for on-disk model cache paths.

Kept dependency-light (no weights-provider or auto-loader imports) so it can be
imported by both the auto-loader and weights-provider layers without creating
circular imports.
"""

import hashlib
import json
import os
import re
import stat
from typing import Optional, Tuple

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
_UNATTRIBUTED_MODEL_CACHE = object()
_INVALID_MODEL_CACHE_ATTRIBUTION = object()


def _slugify_model_id_prefix(model_id: str) -> str:
    model_id_slug = re.sub(r"[^A-Za-z0-9_-]+", "-", model_id)
    model_id_slug = re.sub(r"[_-]{2,}", "-", model_id_slug)
    if not model_id_slug:
        model_id_slug = "special-char-only-model-id"
    if len(model_id_slug) > 48:
        model_id_slug = model_id_slug[:48]
    return model_id_slug


def slugify_model_id_to_os_safe_format_v1(model_id: str) -> str:
    """Return the exact legacy model-cache slug.

    V1 used a 32-bit digest and is retained strictly for reading caches written
    by older releases. New cache entries must use the V2 helper below.
    """

    model_id_slug = _slugify_model_id_prefix(model_id=model_id)
    digest = hashlib.blake2s(model_id.encode("utf-8"), digest_size=4).hexdigest()
    return f"{model_id_slug}-{digest}"


def slugify_model_id_to_os_safe_format_v2(model_id: str) -> str:
    """Return the versioned model-cache slug used for new writes."""

    model_id_slug = _slugify_model_id_prefix(model_id=model_id)
    digest = hashlib.blake2s(model_id.encode("utf-8"), digest_size=16).hexdigest()
    return f"{MODEL_CACHE_SLUG_VERSION}-{model_id_slug}-{digest}"


def slugify_model_id_to_os_safe_format(model_id: str) -> str:
    """Return the current model-cache slug used for new writes."""

    return slugify_model_id_to_os_safe_format_v2(model_id=model_id)


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


def _package_id_has_exact_directory_entry(
    model_cache_root: str,
    package_id: str,
) -> bool:
    """Reject case-fold aliases that collide on case-insensitive filesystems."""

    try:
        entries = os.listdir(model_cache_root)
    except OSError:
        return False
    matching_entries = [
        entry for entry in entries if entry.casefold() == package_id.casefold()
    ]
    return matching_entries == [package_id]


def _ensure_package_id_has_no_case_alias(
    model_cache_root: str,
    package_id: str,
    model_id: str,
) -> None:
    try:
        entries = os.listdir(model_cache_root)
    except FileNotFoundError:
        return
    except OSError as error:
        raise InsecureModelIdentifierError(
            message=(
                f"Refusing model cache package {package_id} for {model_id} "
                "because its model cache root could not be inspected safely."
            ),
            help_url="https://inference-models.roboflow.com/errors/model-loading/#insecuremodelidentifiererror",
        ) from error
    aliases = [
        entry
        for entry in entries
        if entry.casefold() == package_id.casefold() and entry != package_id
    ]
    if aliases:
        raise InsecureModelIdentifierError(
            message=(
                f"Refusing model cache package {package_id} for {model_id} "
                "because a case-insensitive package ID alias already exists."
            ),
            help_url="https://inference-models.roboflow.com/errors/model-loading/#insecuremodelidentifiererror",
        )


def generate_models_cache_dir() -> str:
    return os.path.abspath(os.path.join(INFERENCE_HOME, "models-cache"))


def _cache_path_has_no_child_symlinks(cache_root: str, target_path: str) -> bool:
    """Return whether *target_path* is contained without symlinks below root."""

    cache_root = os.path.abspath(cache_root)
    target_path = os.path.abspath(target_path)
    try:
        relative_path = os.path.relpath(target_path, cache_root)
        if (
            relative_path in ("", os.curdir)
            or relative_path == os.pardir
            or relative_path.startswith(os.pardir + os.sep)
        ):
            return False
        if os.path.commonpath(
            [cache_root, target_path]
        ) != cache_root or os.path.realpath(target_path) != os.path.normpath(
            os.path.join(os.path.realpath(cache_root), relative_path)
        ):
            return False
    except ValueError:
        return False
    return True


def _ensure_cache_path_has_no_child_symlinks(
    cache_root: str,
    target_path: str,
    model_id: str,
) -> None:
    if not _cache_path_has_no_child_symlinks(
        cache_root=cache_root,
        target_path=target_path,
    ):
        raise InsecureModelIdentifierError(
            message=(
                f"Refusing model cache path for {model_id} because it escapes "
                "its cache root or traverses a symbolic link."
            ),
            help_url="https://inference-models.roboflow.com/errors/model-loading/#insecuremodelidentifiererror",
        )


def generate_model_cache_root_candidates_for_model_id(
    model_id: str,
) -> Tuple[str, str]:
    """Return V2 then legacy V1 model cache roots.

    The candidates are lexical paths. Callers that consume existing filesystem
    entries must use :func:`resolve_existing_model_package_cache_path`, which
    validates containment, symlinks and cache attribution before returning a
    package path.
    """

    models_cache_dir = generate_models_cache_dir()
    return (
        os.path.join(
            models_cache_dir,
            slugify_model_id_to_os_safe_format_v2(model_id=model_id),
        ),
        os.path.join(
            models_cache_dir,
            slugify_model_id_to_os_safe_format_v1(model_id=model_id),
        ),
    )


def generate_model_cache_root_for_model_id(model_id: str) -> str:
    models_cache_dir = generate_models_cache_dir()
    result = generate_model_cache_root_candidates_for_model_id(model_id=model_id)[0]
    _ensure_cache_path_has_no_child_symlinks(
        cache_root=models_cache_dir,
        target_path=result,
        model_id=model_id,
    )
    return result


def generate_legacy_model_cache_root_for_model_id(model_id: str) -> str:
    """Return the validated legacy V1 root for read/migration tooling."""

    models_cache_dir = generate_models_cache_dir()
    result = generate_model_cache_root_candidates_for_model_id(model_id=model_id)[1]
    _ensure_cache_path_has_no_child_symlinks(
        cache_root=models_cache_dir,
        target_path=result,
        model_id=model_id,
    )
    return result


def generate_model_package_cache_path_candidates(
    model_id: str,
    package_id: str,
) -> Tuple[str, str]:
    """Return V2 then legacy V1 package paths without treating either as a hit."""

    ensure_package_id_is_os_safe(model_id=model_id, package_id=package_id)
    roots = generate_model_cache_root_candidates_for_model_id(model_id=model_id)
    return (
        os.path.join(roots[0], package_id),
        os.path.join(roots[1], package_id),
    )


def generate_model_package_cache_path(model_id: str, package_id: str) -> str:
    ensure_package_id_is_os_safe(model_id=model_id, package_id=package_id)
    model_cache_root = generate_model_cache_root_for_model_id(model_id=model_id)
    _ensure_package_id_has_no_case_alias(
        model_cache_root=model_cache_root,
        package_id=package_id,
        model_id=model_id,
    )
    result = os.path.join(model_cache_root, package_id)
    _ensure_cache_path_has_no_child_symlinks(
        cache_root=model_cache_root,
        target_path=result,
        model_id=model_id,
    )
    return result


def generate_legacy_model_package_cache_path(
    model_id: str,
    package_id: str,
) -> str:
    """Return the validated legacy V1 package path for compatibility tooling."""

    ensure_package_id_is_os_safe(model_id=model_id, package_id=package_id)
    model_cache_root = generate_legacy_model_cache_root_for_model_id(model_id=model_id)
    _ensure_package_id_has_no_case_alias(
        model_cache_root=model_cache_root,
        package_id=package_id,
        model_id=model_id,
    )
    result = os.path.join(model_cache_root, package_id)
    _ensure_cache_path_has_no_child_symlinks(
        cache_root=model_cache_root,
        target_path=result,
        model_id=model_id,
    )
    return result


def _read_model_cache_attribution(package_path: str) -> object:
    """Read a regular, non-symlinked model manifest from *package_path*."""

    config_path = os.path.join(package_path, MODEL_CONFIG_FILE_NAME)
    try:
        config_stat = os.lstat(config_path)
    except FileNotFoundError:
        return _UNATTRIBUTED_MODEL_CACHE
    except OSError:
        return _INVALID_MODEL_CACHE_ATTRIBUTION
    if not stat.S_ISREG(config_stat.st_mode):
        return _INVALID_MODEL_CACHE_ATTRIBUTION

    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    file_descriptor = -1
    try:
        file_descriptor = os.open(config_path, flags)
        opened_stat = os.fstat(file_descriptor)
        if not stat.S_ISREG(opened_stat.st_mode) or (
            opened_stat.st_dev,
            opened_stat.st_ino,
        ) != (config_stat.st_dev, config_stat.st_ino):
            return _INVALID_MODEL_CACHE_ATTRIBUTION
        config_file = os.fdopen(file_descriptor, encoding="utf-8")
        file_descriptor = -1
        with config_file:
            config = json.load(config_file)
    except (json.JSONDecodeError, OSError, UnicodeError, ValueError):
        return _INVALID_MODEL_CACHE_ATTRIBUTION
    finally:
        if file_descriptor >= 0:
            os.close(file_descriptor)
    if not isinstance(config, dict):
        return _INVALID_MODEL_CACHE_ATTRIBUTION
    if "model_id" not in config:
        return _UNATTRIBUTED_MODEL_CACHE
    attributed_model_id = config["model_id"]
    if not isinstance(attributed_model_id, str) or not attributed_model_id:
        return _INVALID_MODEL_CACHE_ATTRIBUTION
    return attributed_model_id


def resolve_existing_model_package_cache_path(
    model_id: str,
    package_id: str,
    allow_unattributed_local_cache: bool = False,
) -> Optional[str]:
    """Resolve a safely attributed existing V2 or legacy V1 package.

    V2 is always preferred. V1 is a read-only compatibility path: this helper
    never creates directories. Ordinarily, a regular ``model_config.json`` must
    contain the exact non-empty requested ``model_id``. The explicit
    ``allow_unattributed_local_cache`` escape hatch is reserved for locally
    compiled packages in the current collision-resistant V2 namespace. Legacy
    V1 paths use a 32-bit digest and therefore always require exact ownership
    metadata. A present malformed, non-regular, empty or conflicting
    attribution is never accepted.
    """

    ensure_package_id_is_os_safe(model_id=model_id, package_id=package_id)
    models_cache_dir = generate_models_cache_dir()
    unattributed_fallback = None
    package_path_candidates = generate_model_package_cache_path_candidates(
        model_id=model_id,
        package_id=package_id,
    )
    for candidate_index, package_path in enumerate(package_path_candidates):
        if not _cache_path_has_no_child_symlinks(
            cache_root=models_cache_dir,
            target_path=package_path,
        ):
            continue
        if not _package_id_has_exact_directory_entry(
            model_cache_root=os.path.dirname(package_path),
            package_id=package_id,
        ):
            continue
        try:
            package_stat = os.lstat(package_path)
        except OSError:
            continue
        if not stat.S_ISDIR(package_stat.st_mode):
            continue
        attribution = _read_model_cache_attribution(package_path=package_path)
        try:
            final_package_stat = os.lstat(package_path)
        except OSError:
            continue
        if (
            not stat.S_ISDIR(final_package_stat.st_mode)
            or (package_stat.st_dev, package_stat.st_ino)
            != (final_package_stat.st_dev, final_package_stat.st_ino)
            or not _cache_path_has_no_child_symlinks(
                cache_root=models_cache_dir,
                target_path=package_path,
            )
        ):
            continue
        if attribution == model_id:
            return package_path
        if (
            attribution is _UNATTRIBUTED_MODEL_CACHE
            and allow_unattributed_local_cache
            and candidate_index == 0
            and unattributed_fallback is None
        ):
            unattributed_fallback = package_path
    return unattributed_fallback


def generate_shared_blobs_path() -> str:
    return os.path.abspath(os.path.join(INFERENCE_HOME, "shared-blobs"))
