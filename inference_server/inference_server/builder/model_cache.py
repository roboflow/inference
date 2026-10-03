"""Discovery of models cached on disk, in both cache roots and both layouts.

The roots are `MODEL_CACHE_DIR` and `INFERENCE_HOME`. Each root may hold the
traditional layout (`<model path>/model_type.json`) and the inference-models
layout (`models-cache/<slug>/<package>/model_config.json`).
"""

import hashlib
import json
import logging
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from inference_models.models.auto_loaders.model_cache_paths import (
    MODEL_CONFIG_FILE_NAME,
    slugify_model_id_to_os_safe_format,
)

from inference_server import configuration

logger = logging.getLogger(__name__)

MODEL_ID_CACHE_SLUG_PREFIX_LENGTH = 48
MODEL_ID_CACHE_SLUG_HASH_BYTES = 16
LEGACY_MODEL_ID_CACHE_SLUG_HASH_BYTES = 4
MODEL_ID_CACHE_SLUG_NAMESPACE_PREFIX = "~"
SPECIAL_CHAR_ONLY_MODEL_ID_SLUG = "special-char-only-model-id"
MAX_PATH_BYTES = 4096
MAX_PATH_SEGMENT_BYTES = 255
MAX_SCAN_ENTRIES = 100_000
MAX_SCAN_DEPTH = 8
MAX_METADATA_FILE_BYTES = 1024 * 1024
MODELS_CACHE_DIR_NAME = "models-cache"
TRADITIONAL_METADATA_FILE_NAME = "model_type.json"
_TASK_TYPE_KEY = "project_task_type"
_MODEL_TYPE_KEY = "model_type"
_PORTABLE_RAW_MODEL_ID_SEGMENT = re.compile(r"[a-z0-9._ -]+")
_LEGACY_MODEL_ID_CACHE_SLUG = re.compile(r"[A-Za-z0-9_-]{1,48}-[0-9a-f]{8}")
_LEGACY_TRADITIONAL_MODEL_SLUG = re.compile(r"[A-Za-z0-9_-]+-[0-9a-f]{8}")
_PACKAGE_ID = re.compile(r"[A-Za-z0-9]+")
_WINDOWS_RESERVED_PATH_SEGMENTS = {
    "aux",
    "con",
    "nul",
    "prn",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}
RESERVED_CACHE_ROOT_NAMESPACES = {
    "_file_locks",
    "auto-resolution-cache",
    "hf_home",
    "huggingface",
    "lora-bases",
    MODELS_CACHE_DIR_NAME,
    "owl-v2-serialized-data",
    "shared-blobs",
    "usage.db",
    "workflow",
}
_SKIP_TOP_LEVEL = RESERVED_CACHE_ROOT_NAMESPACES - {MODELS_CACHE_DIR_NAME}
_INVALID_CACHE_METADATA = object()
_OVERSIZED_CACHE_METADATA = object()


class _ScanLimitReached(Exception):
    pass


class _OversizedMetadata(Exception):
    pass


@dataclass
class CacheScan:
    """Models found by a cache scan and whether a bound cut the scan short."""

    models: List[Dict[str, Any]]
    truncated: bool = False


def configured_cache_roots() -> List[str]:
    """Return the de-duplicated absolute cache roots to search.

    Returns:
        `MODEL_CACHE_DIR` first, then `INFERENCE_HOME` when it names a different
        directory.
    """
    roots = [os.path.abspath(configuration.MODEL_CACHE_DIR)]
    inference_home = os.path.abspath(configuration.INFERENCE_HOME)
    if os.path.realpath(inference_home) != os.path.realpath(roots[0]):
        roots.append(inference_home)

    return roots


def list_cached_models() -> List[Dict[str, Any]]:
    """Return the models found in every configured root, in both layouts.

    Models whose cached metadata conflicts between packages or roots are
    omitted.
    """
    return list_cached_models_with_status().models


def list_cached_models_with_status() -> CacheScan:
    """Like `list_cached_models`, also reporting whether any scan was truncated."""
    cache_roots = configured_cache_roots()
    user_models: List[Dict[str, Any]] = []
    truncated = False
    for cache_root in cache_roots:
        resolved_cache_root = os.path.realpath(cache_root)
        nested_cache_roots = [
            other_root
            for other_root in cache_roots
            if other_root != cache_root
            and _path_is_strict_descendant(
                path=os.path.realpath(other_root),
                parent=resolved_cache_root,
            )
        ]
        scan = scan_cached_models(cache_root, excluded_cache_roots=nested_cache_roots)
        user_models.extend(scan.models)
        truncated = truncated or scan.truncated

    unambiguous_models = list(
        collect_unambiguous_user_models(user_models=user_models).values()
    )

    return CacheScan(models=unambiguous_models, truncated=truncated)


def collect_unambiguous_user_models(
    user_models: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """De-duplicate identical entries and omit IDs with conflicting metadata."""

    models_by_id: Dict[str, Dict[str, Any]] = {}
    conflicting_model_ids = set()
    for model in user_models:
        model_id = model.get("model_id")
        if not isinstance(model_id, str) or not model_id:
            logger.warning("Skipping cached model metadata without a valid model_id")
            continue

        if model_id in conflicting_model_ids:
            continue

        existing_model = models_by_id.get(model_id)
        if existing_model is None:
            models_by_id[model_id] = model
            continue

        if existing_model == model:
            continue

        models_by_id.pop(model_id)
        conflicting_model_ids.add(model_id)
        logger.warning(
            "Excluding cached model %s because configured cache roots contain "
            "conflicting metadata for that model ID",
            model_id,
        )

    return models_by_id


def has_cached_model_variant(model_variants: Optional[List[str]]) -> bool:
    """Return True if any of the given model variant IDs has cached artifacts.

    Args:
        model_variants: Model IDs as returned by a block manifest's
            `get_supported_model_variants()`. `None` or empty gives `False`.

    Returns:
        Whether at least one variant looks cached.
    """
    if not model_variants:
        return False

    has_variant = any(is_model_cached(model_id) for model_id in model_variants)

    return has_variant


def is_model_cached(model_id: str) -> bool:
    """Best-effort check whether `model_id` has cached artifacts.

    The traditional layout is looked up under `MODEL_CACHE_DIR` and the
    inference-models layout under `INFERENCE_HOME`. A directory with non-hidden
    files counts as a usable model; integrity is verified when the model loads.

    Args:
        model_id: Model identifier.

    Returns:
        `True` when there is a chance the model is cached.
    """
    try:
        traditional_path = _traditional_cache_dir_for_read(
            model_id=model_id,
            cache_dir_root=configuration.MODEL_CACHE_DIR,
        )
    except (TypeError, ValueError):
        return False

    if traditional_path is None:
        return False

    if (
        _is_safe_model_cache_directory(
            cache_root=configuration.MODEL_CACHE_DIR,
            model_path=traditional_path,
        )
        and os.path.isdir(traditional_path)
        and _has_non_hidden_children(path=traditional_path)
    ):
        return True

    in_inference_models_layout = _is_cached_in_inference_models_layout(
        model_id=model_id
    )

    return in_inference_models_layout


def scan_cached_models(
    cache_dir: str,
    excluded_cache_roots: Optional[List[str]] = None,
) -> CacheScan:
    """Walk `cache_dir` for cached model metadata in both layouts.

    Symlinked directories are never followed. The walk stops when
    `MAX_SCAN_ENTRIES` is exceeded and does not descend below
    `MAX_SCAN_DEPTH`; a metadata file above `MAX_METADATA_FILE_BYTES` is
    skipped. Each of those marks the scan as truncated and logs one warning.

    Args:
        cache_dir: Cache root to walk.
        excluded_cache_roots: Other cache roots nested under `cache_dir` that
            are scanned on their own.

    Returns:
        The scan result: one model entry per model with `model_id`, `name`,
        `task_type`, `model_architecture` and `is_foundation`, and whether the
        scan was cut short by a bound.
    """
    results_by_id: Dict[str, Dict[str, Any]] = {}
    conflicting_ids: Set[str] = set()
    if not os.path.isdir(cache_dir):
        return CacheScan(models=[], truncated=False)

    cache_dir = os.path.abspath(cache_dir)
    excluded_cache_roots = [
        (os.path.abspath(root), os.path.realpath(root))
        for root in (excluded_cache_roots or [])
        if os.path.abspath(root) != cache_dir
    ]
    truncated = False
    visited_entries = [0]
    pending_directories = [cache_dir]
    while pending_directories:
        root = pending_directories.pop()
        try:
            dirs, files = _list_directory(path=root, visited_entries=visited_entries)
        except _ScanLimitReached:
            truncated = True
            break

        dirs = [
            directory
            for directory in dirs
            if not _is_excluded_directory(
                directory_path=os.path.abspath(os.path.join(root, directory)),
                excluded_cache_roots=excluded_cache_roots,
            )
        ]
        rel = os.path.relpath(root, cache_dir)
        if rel == ".":
            dirs = [d for d in dirs if d not in _SKIP_TOP_LEVEL]
            pending_directories.extend(
                os.path.join(root, directory) for directory in reversed(dirs)
            )
            continue

        relative_parts = rel.split(os.sep)
        if dirs and len(relative_parts) >= MAX_SCAN_DEPTH:
            truncated = True
        else:
            pending_directories.extend(
                os.path.join(root, directory) for directory in reversed(dirs)
            )

        try:
            result = _read_model_entry(
                cache_dir=cache_dir,
                model_root=root,
                relative_parts=relative_parts,
                files=files,
            )
        except _OversizedMetadata:
            truncated = True
            continue

        if result is None:
            continue

        model_id = result["model_id"]
        if model_id in conflicting_ids:
            continue

        existing_result = results_by_id.get(model_id)
        if existing_result is not None and existing_result != result:
            logger.warning(
                "Skipping cached model %s because packages expose conflicting "
                "metadata.",
                model_id,
            )
            results_by_id.pop(model_id, None)
            conflicting_ids.add(model_id)
            continue

        results_by_id[model_id] = result

    if truncated:
        logger.warning(
            "Cached model scan is incomplete: a scan bound was reached "
            "(entries, depth or metadata file size), so some models may be "
            "missing from the listing."
        )

    return CacheScan(models=list(results_by_id.values()), truncated=truncated)


def _list_directory(
    path: str, visited_entries: List[int]
) -> Tuple[List[str], List[str]]:
    directories: List[str] = []
    files: List[str] = []
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                visited_entries[0] += 1
                if visited_entries[0] > MAX_SCAN_ENTRIES:
                    raise _ScanLimitReached
                try:
                    is_directory = entry.is_dir()
                    is_symlink = entry.is_symlink()
                except OSError:
                    is_directory = False
                    is_symlink = False
                if is_directory:
                    if not is_symlink:
                        directories.append(entry.name)
                else:
                    files.append(entry.name)
    except OSError:
        return [], []

    return sorted(directories), sorted(files)


def _is_excluded_directory(
    directory_path: str, excluded_cache_roots: List[Tuple[str, str]]
) -> bool:
    resolved_directory_path = os.path.realpath(directory_path)

    return any(
        directory_path == excluded_root
        or directory_path.startswith(excluded_root + os.sep)
        or resolved_directory_path == resolved_excluded_root
        or resolved_directory_path.startswith(resolved_excluded_root + os.sep)
        for excluded_root, resolved_excluded_root in excluded_cache_roots
    )


def _read_model_entry(
    cache_dir: str,
    model_root: str,
    relative_parts: List[str],
    files: List[str],
) -> Optional[Dict[str, Any]]:
    """Read and validate the model metadata stored in one directory.

    Raises:
        _OversizedMetadata: When a metadata file exceeds
            `MAX_METADATA_FILE_BYTES`.
    """
    has_model_type = TRADITIONAL_METADATA_FILE_NAME in files
    has_model_config = MODEL_CONFIG_FILE_NAME in files
    if not has_model_type and not has_model_config:
        return None

    metadata: Optional[dict] = None
    stored_model_id: Optional[str] = None
    if has_model_config:
        valid_inference_models_location = (
            len(relative_parts) == 3
            and relative_parts[0] == MODELS_CACHE_DIR_NAME
            and _PACKAGE_ID.fullmatch(relative_parts[2]) is not None
        )
        if not valid_inference_models_location:
            has_model_config = False

    if has_model_config:
        config = _read_regular_json(
            path=os.path.join(model_root, MODEL_CONFIG_FILE_NAME)
        )
        if config is _OVERSIZED_CACHE_METADATA:
            raise _OversizedMetadata
        if isinstance(config, dict) and config.get("task_type"):
            metadata = config
            manifest_model_id = config.get("model_id")
            if isinstance(manifest_model_id, str) and manifest_model_id:
                stored_model_id = manifest_model_id
                try:
                    expected_slugs = {
                        slugify_model_id_to_os_safe_format(model_id=stored_model_id)
                    }
                except (ImportError, TypeError):
                    expected_slugs = set()
                if relative_parts[1] not in expected_slugs:
                    metadata = None
                    stored_model_id = None
            else:
                metadata = None
                stored_model_id = None

    if (
        metadata is None
        and has_model_type
        and relative_parts[0] != MODELS_CACHE_DIR_NAME
    ):
        metadata = _read_regular_json(
            path=os.path.join(model_root, TRADITIONAL_METADATA_FILE_NAME)
        )
        if metadata is _OVERSIZED_CACHE_METADATA:
            raise _OversizedMetadata
        if metadata is _INVALID_CACHE_METADATA:
            return None

        if isinstance(metadata, dict):
            stored_model_id = _resolve_traditional_cache_model_id(
                cache_dir=cache_dir,
                model_root=model_root,
                metadata=metadata,
            )
            if stored_model_id is None:
                metadata = None

    if not isinstance(metadata, dict):
        return None

    task_type = (
        metadata.get("task_type")
        or metadata.get(_TASK_TYPE_KEY)
        or metadata.get("taskType", "")
    )
    model_architecture = (
        metadata.get("model_architecture")
        or metadata.get(_MODEL_TYPE_KEY)
        or metadata.get("modelArchitecture", "")
    )
    if not isinstance(task_type, str) or not task_type:
        return None

    if not isinstance(model_architecture, str):
        return None

    if stored_model_id is not None:
        model_id = stored_model_id
    elif has_model_config:
        return None
    else:
        model_id = os.path.relpath(model_root, cache_dir).replace(os.sep, "/")

    if not isinstance(model_id, str) or not model_id:
        return None

    return {
        "model_id": model_id,
        "name": model_id,
        "task_type": task_type,
        "model_architecture": model_architecture,
        "is_foundation": False,
    }


def get_model_id_cache_path(model_id: str, cache_dir_root: str) -> str:
    """Return the path of a model's traditional cache directory under a root.

    Args:
        model_id: Model identifier.
        cache_dir_root: Cache root.

    Returns:
        The model id itself when it is a portable path, otherwise a slug.

    Raises:
        ValueError: When the model id has an unsafe or ambiguous path segment.
    """
    validate_model_id_for_cache(model_id=model_id)
    raw_cache_path = os.path.join(cache_dir_root, model_id)
    if _model_id_can_use_raw_cache_path(
        model_id=model_id,
        path=raw_cache_path,
        cache_dir_root=cache_dir_root,
    ):
        return model_id

    cache_key = _slugify_model_id_to_cache_key(
        model_id=model_id,
        digest_size=MODEL_ID_CACHE_SLUG_HASH_BYTES,
        namespace_prefix=MODEL_ID_CACHE_SLUG_NAMESPACE_PREFIX,
    )

    return cache_key


def get_legacy_model_id_cache_path(model_id: str, cache_dir_root: str) -> Optional[str]:
    """Return the pre-v2 cache path when the current path changed.

    Those paths are only candidates: ownership metadata must confirm them.
    """
    validate_model_id_for_cache(model_id=model_id)
    raw_cache_path = os.path.join(cache_dir_root, model_id)
    current_cache_path = get_model_id_cache_path(
        model_id=model_id,
        cache_dir_root=cache_dir_root,
    )
    if _raw_cache_path_fits(path=raw_cache_path, cache_dir_root=cache_dir_root):
        return None if current_cache_path == model_id else model_id

    legacy_cache_key = _slugify_model_id_to_cache_key(
        model_id=model_id,
        digest_size=LEGACY_MODEL_ID_CACHE_SLUG_HASH_BYTES,
        namespace_prefix="",
    )

    return legacy_cache_key


def validate_model_id_for_cache(model_id: str) -> None:
    if not isinstance(model_id, str):
        raise ValueError("Model ID used for cache access must be a string.")

    path_segments = re.split(r"[\\/]", model_id)
    if any(segment in {"", ".", ".."} for segment in path_segments):
        raise ValueError(
            f"Model ID {model_id!r} contains an unsafe or ambiguous path segment."
        )


def _traditional_cache_dir_for_read(
    model_id: str, cache_dir_root: str
) -> Optional[str]:
    current_cache_dir = os.path.join(
        cache_dir_root,
        get_model_id_cache_path(model_id=model_id, cache_dir_root=cache_dir_root),
    )
    if not configuration.LEGACY_OFFLINE_MODE:
        return current_cache_dir

    if os.path.lexists(current_cache_dir) and not _cache_directory_tree_is_safe(
        cache_dir=current_cache_dir,
        cache_dir_root=cache_dir_root,
    ):
        return None

    legacy_cache_path = get_legacy_model_id_cache_path(
        model_id=model_id,
        cache_dir_root=cache_dir_root,
    )
    if legacy_cache_path is None or _cache_directory_has_exact_owner(
        cache_dir=current_cache_dir,
        cache_dir_root=cache_dir_root,
        model_id=model_id,
    ):
        return current_cache_dir

    legacy_cache_dir = os.path.join(cache_dir_root, legacy_cache_path)
    if _cache_directory_has_exact_owner(
        cache_dir=legacy_cache_dir,
        cache_dir_root=cache_dir_root,
        model_id=model_id,
    ):
        return legacy_cache_dir

    return current_cache_dir


def _is_cached_in_inference_models_layout(model_id: str) -> bool:
    try:
        model_slug = slugify_model_id_to_os_safe_format(model_id=model_id)
    except (TypeError, ValueError):
        return False

    model_root = (
        Path(os.path.abspath(configuration.INFERENCE_HOME))
        / MODELS_CACHE_DIR_NAME
        / model_slug
    )
    if model_root.is_symlink() or not model_root.is_dir():
        return False

    for package_dir in _sorted_children(path=model_root):
        if package_dir.is_symlink() or not package_dir.is_dir():
            continue

        if (package_dir / MODEL_CONFIG_FILE_NAME).is_file():
            return True

    return False


def _sorted_children(path: Path) -> List[Path]:
    try:
        children = sorted(path.iterdir())
    except OSError:
        return []

    return children


def _path_is_strict_descendant(path: str, parent: str) -> bool:
    resolved_path = os.path.realpath(path)
    resolved_parent = os.path.realpath(parent)
    try:
        return (
            resolved_path != resolved_parent
            and os.path.commonpath([resolved_parent, resolved_path]) == resolved_parent
        )
    except ValueError:
        return False


def _has_non_hidden_children(path: str) -> bool:
    try:
        return any(not name.startswith(".") for name in os.listdir(path))
    except OSError:
        return False


def _is_safe_model_cache_directory(cache_root: str, model_path: str) -> bool:
    absolute_cache_root = os.path.abspath(cache_root)
    absolute_model_path = os.path.abspath(model_path)
    try:
        if (
            os.path.commonpath([absolute_cache_root, absolute_model_path])
            != absolute_cache_root
        ):
            return False

        relative_model_path = os.path.relpath(absolute_model_path, absolute_cache_root)
    except ValueError:
        return False

    if relative_model_path in ("", os.curdir) or relative_model_path.startswith(
        os.pardir + os.sep
    ):
        return False

    current_path = absolute_cache_root
    for path_part in relative_model_path.split(os.sep):
        current_path = os.path.join(current_path, path_part)
        if os.path.islink(current_path):
            return False

    expected_resolved_path = os.path.normpath(
        os.path.join(os.path.realpath(absolute_cache_root), relative_model_path)
    )

    return os.path.realpath(absolute_model_path) == expected_resolved_path


def _resolve_traditional_cache_model_id(
    cache_dir: str,
    model_root: str,
    metadata: dict,
) -> Optional[str]:
    relative_root = os.path.relpath(model_root, cache_dir).replace(os.sep, "/")
    if "model_id" in metadata:
        stored_model_id = metadata["model_id"]
        if not isinstance(stored_model_id, str) or not stored_model_id:
            return None

        try:
            expected_paths = {
                get_model_id_cache_path(
                    model_id=stored_model_id,
                    cache_dir_root=cache_dir,
                ).replace(os.sep, "/")
            }
            legacy_path = get_legacy_model_id_cache_path(
                model_id=stored_model_id,
                cache_dir_root=cache_dir,
            )
            if legacy_path is not None:
                expected_paths.add(legacy_path.replace(os.sep, "/"))
            else:
                expected_paths.add(stored_model_id.replace(os.sep, "/"))
        except ValueError:
            return None

        if relative_root not in expected_paths:
            return None

        return stored_model_id

    if "/" not in relative_root and (
        relative_root.startswith("~")
        or _LEGACY_TRADITIONAL_MODEL_SLUG.fullmatch(relative_root) is not None
    ):
        return None

    try:
        expected_raw_path = get_model_id_cache_path(
            model_id=relative_root,
            cache_dir_root=cache_dir,
        ).replace(os.sep, "/")
    except ValueError:
        return None

    if expected_raw_path != relative_root:
        return None

    return relative_root


def _model_id_can_use_raw_cache_path(
    model_id: str, path: str, cache_dir_root: str
) -> bool:
    path_segments = re.split(r"[\\/]", model_id)
    first_path_segment = path_segments[0]

    return (
        "\\" not in model_id
        and not first_path_segment.startswith(MODEL_ID_CACHE_SLUG_NAMESPACE_PREFIX)
        and first_path_segment not in RESERVED_CACHE_ROOT_NAMESPACES
        and _LEGACY_MODEL_ID_CACHE_SLUG.fullmatch(first_path_segment) is None
        and all(_path_segment_is_portable(segment) for segment in path_segments)
        and _raw_cache_path_fits(path=path, cache_dir_root=cache_dir_root)
    )


def _path_segment_is_portable(path_segment: str) -> bool:
    windows_device_name = path_segment.split(".", maxsplit=1)[0].rstrip(" ").lower()

    return (
        _PORTABLE_RAW_MODEL_ID_SEGMENT.fullmatch(path_segment) is not None
        and not path_segment.endswith((" ", "."))
        and windows_device_name not in _WINDOWS_RESERVED_PATH_SEGMENTS
    )


def _raw_cache_path_fits(path: str, cache_dir_root: str) -> bool:
    return _cache_path_is_within_root(
        path=path, cache_dir_root=cache_dir_root
    ) and _path_fits_os_limits(path=path)


def _cache_path_is_within_root(path: str, cache_dir_root: str) -> bool:
    try:
        root = os.path.abspath(cache_dir_root)
        candidate = os.path.abspath(path)
        return os.path.commonpath([root, candidate]) == root
    except ValueError:
        return False


def _path_fits_os_limits(path: str) -> bool:
    if len(os.fsencode(os.path.abspath(path))) >= MAX_PATH_BYTES:
        return False

    _, path_without_drive = os.path.splitdrive(path)
    if os.altsep is not None:
        path_without_drive = path_without_drive.replace(os.altsep, os.sep)

    return all(
        len(os.fsencode(path_segment)) <= MAX_PATH_SEGMENT_BYTES
        for path_segment in path_without_drive.split(os.sep)
        if path_segment
    )


def _slugify_model_id_to_cache_key(
    model_id: str, digest_size: int, namespace_prefix: str
) -> str:
    model_id_slug = re.sub(r"[^A-Za-z0-9_-]+", "-", model_id)
    model_id_slug = re.sub(r"[_-]{2,}", "-", model_id_slug)
    if not model_id_slug:
        model_id_slug = SPECIAL_CHAR_ONLY_MODEL_ID_SLUG

    if len(model_id_slug) > MODEL_ID_CACHE_SLUG_PREFIX_LENGTH:
        model_id_slug = model_id_slug[:MODEL_ID_CACHE_SLUG_PREFIX_LENGTH]

    digest = hashlib.blake2s(
        model_id.encode("utf-8"), digest_size=digest_size
    ).hexdigest()

    return f"{namespace_prefix}{model_id_slug}-{digest}"


def _cache_directory_has_exact_owner(
    cache_dir: str,
    cache_dir_root: str,
    model_id: str,
) -> bool:
    if not _cache_directory_tree_is_safe(
        cache_dir=cache_dir,
        cache_dir_root=cache_dir_root,
    ):
        return False

    metadata = _read_regular_json(
        path=os.path.join(os.path.abspath(cache_dir), TRADITIONAL_METADATA_FILE_NAME)
    )

    return isinstance(metadata, dict) and metadata.get("model_id") == model_id


def _cache_directory_tree_is_safe(cache_dir: str, cache_dir_root: str) -> bool:
    absolute_root = os.path.abspath(cache_dir_root)
    absolute_cache_dir = os.path.abspath(cache_dir)
    if not _cache_path_is_within_root(
        path=absolute_cache_dir,
        cache_dir_root=absolute_root,
    ):
        return False

    try:
        relative_cache_dir = os.path.relpath(absolute_cache_dir, absolute_root)
    except ValueError:
        return False

    if relative_cache_dir in {"", os.curdir}:
        return False

    is_junction = getattr(os.path, "isjunction", lambda _path: False)
    current_path = absolute_root
    for path_part in relative_cache_dir.split(os.sep):
        current_path = os.path.join(current_path, path_part)
        if os.path.islink(current_path) or is_junction(current_path):
            return False

    expected_resolved_cache_dir = os.path.normpath(
        os.path.join(os.path.realpath(absolute_root), relative_cache_dir)
    )
    if os.path.normcase(os.path.realpath(absolute_cache_dir)) != os.path.normcase(
        expected_resolved_cache_dir
    ):
        return False

    pending_directories = [absolute_cache_dir]
    while pending_directories:
        directory = pending_directories.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if entry.is_symlink() or is_junction(entry.path):
                        return False

                    entry_status = entry.stat(follow_symlinks=False)
                    if stat.S_ISDIR(entry_status.st_mode):
                        pending_directories.append(entry.path)
                    elif not stat.S_ISREG(entry_status.st_mode):
                        return False
        except OSError:
            return False

    return True


def _read_regular_json(path: str) -> object:
    """Read JSON from a stable regular file without following a final symlink."""

    try:
        path_status = os.lstat(path)
    except OSError:
        return _INVALID_CACHE_METADATA

    if not stat.S_ISREG(path_status.st_mode):
        return _INVALID_CACHE_METADATA

    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
        )
        descriptor_status = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_status.st_mode) or (
            path_status.st_dev,
            path_status.st_ino,
        ) != (descriptor_status.st_dev, descriptor_status.st_ino):
            return _INVALID_CACHE_METADATA

        file_handle = os.fdopen(descriptor, "rb")
        descriptor = -1
        with file_handle:
            content = file_handle.read(MAX_METADATA_FILE_BYTES + 1)

        if len(content) > MAX_METADATA_FILE_BYTES:
            return _OVERSIZED_CACHE_METADATA

        return json.loads(content.decode("utf-8"))
    except (
        json.JSONDecodeError,
        OSError,
        RecursionError,
        TypeError,
        UnicodeError,
        ValueError,
    ):
        return _INVALID_CACHE_METADATA
    finally:
        if descriptor >= 0:
            os.close(descriptor)
