"""Utilities for discovering models and foundation-model weights in the local cache.

Used by the air-gapped workflow builder to enumerate what is available for
offline workflow construction.
"""

import json
import logging
import os
import re
import stat
from typing import Any, Dict, List, Optional, Set

from inference.core.cache.model_artifacts import (
    RESERVED_CACHE_ROOT_NAMESPACES,
    get_cache_dir,
    get_cache_dir_for_read,
    get_legacy_model_id_cache_path,
    get_model_id_cache_path,
)
from inference.core.env import MODEL_CACHE_DIR, USE_INFERENCE_MODELS
from inference.core.exceptions import ModelArtefactError
from inference.core.roboflow_api import MODEL_TYPE_KEY, PROJECT_TASK_TYPE_KEY

logger = logging.getLogger(__name__)

# Directories directly under MODEL_CACHE_DIR that are not model trees.
# ``models-cache`` is the one reserved namespace this scanner intentionally
# traverses because it contains the inference-models layout.
_SKIP_TOP_LEVEL = RESERVED_CACHE_ROOT_NAMESPACES - {"models-cache"}
_LEGACY_TRADITIONAL_MODEL_SLUG = re.compile(r"[A-Za-z0-9_-]+-[0-9a-f]{8}")
_INVALID_CACHE_METADATA = object()


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
        file_handle = os.fdopen(descriptor, encoding="utf-8")
        descriptor = -1
        with file_handle:
            return json.load(file_handle)
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


def _has_non_hidden_children(path: str) -> bool:
    """Return True if *path* contains at least one non-hidden entry.

    Hidden files (names starting with ``"."``) are excluded because they may be
    stale lock-file leftovers that do not represent usable model artifacts.
    """
    try:
        return any(not f.startswith(".") for f in os.listdir(path))
    except OSError:
        return False


def _is_safe_model_cache_directory(cache_root: str, model_path: str) -> bool:
    """Reject cache-root aliases and symlinks below a configured cache root."""

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

    # The configured cache root itself may be a mounted symlink. Any symlink
    # below that boundary can make one model alias another model or an outside
    # directory and must not count as a cache hit.
    current_path = absolute_cache_root
    for path_part in relative_model_path.split(os.sep):
        current_path = os.path.join(current_path, path_part)
        if os.path.islink(current_path):
            return False

    expected_resolved_path = os.path.normpath(
        os.path.join(
            os.path.realpath(absolute_cache_root),
            relative_model_path,
        )
    )
    return os.path.realpath(absolute_model_path) == expected_resolved_path


def _resolve_traditional_cache_model_id(
    cache_dir: str,
    model_root: str,
    metadata: dict,
) -> Optional[str]:
    """Resolve raw and slugged traditional cache roots without trusting slugs."""

    relative_root = os.path.relpath(model_root, cache_dir)
    relative_root = relative_root.replace(os.sep, "/")
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

    # Historical raw-path entries predate model_id attribution. A namespaced V2
    # slug or ambiguous single-component V1-looking path cannot prove ownership
    # without metadata and must not be reported as the slug itself.
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
def is_model_cached(model_id: str) -> bool:
    """Best-effort check whether *model_id* has cached artifacts.

    Checks both the traditional and ``inference-models`` cache layouts,
    respecting the ``USE_INFERENCE_MODELS`` flag to avoid false positives
    from one layout when the runtime uses the other.

    .. note::

       This is intentionally optimistic — a directory with non-hidden files
       is assumed to contain a usable model.  Full integrity verification
       (hash checks, registry validation) happens at model-load time inside
       ``inference-models``.  Treat the result as *"there is a chance the
       model is cached"* rather than a guarantee.
    """
    try:
        traditional_path = get_cache_dir_for_read(
            model_id=model_id,
            cache_dir_root=MODEL_CACHE_DIR,
        )
    except (ModelArtefactError, TypeError, ValueError):
        return False

    if not USE_INFERENCE_MODELS:
        # Only check the traditional layout when inference-models is disabled.
        return (
            _is_safe_model_cache_directory(
                cache_root=MODEL_CACHE_DIR,
                model_path=traditional_path,
            )
            and os.path.isdir(traditional_path)
            and _has_non_hidden_children(traditional_path)
        )

    # When inference-models is enabled, check both layouts — models cached
    # before the migration still sit in the traditional tree.
    if (
        _is_safe_model_cache_directory(
            cache_root=MODEL_CACHE_DIR,
            model_path=traditional_path,
        )
        and os.path.isdir(traditional_path)
        and _has_non_hidden_children(traditional_path)
    ):
        return True

    try:
        from inference_models.models.auto_loaders import core as auto_loaders

        finder = getattr(auto_loaders, "find_cached_model_package_dir", None)
        if finder is not None:
            return finder(model_id=model_id) is not None
        return _find_cached_model_package_dir_compat(model_id=model_id) is not None
    except ImportError:
        return False


def _find_cached_model_package_dir_compat(model_id: str) -> Optional[str]:
    """Find a package using the cache API available before the public helper."""

    try:
        from inference_models.models.auto_loaders.model_cache_paths import (
            generate_model_cache_root_candidates_for_model_id,
            generate_models_cache_dir,
            resolve_existing_model_package_cache_path,
        )
    except ImportError:
        return None
    models_cache_root = os.path.realpath(generate_models_cache_dir())
    seen_package_ids = set()
    for lexical_model_root in generate_model_cache_root_candidates_for_model_id(
        model_id=model_id
    ):
        if os.path.islink(lexical_model_root):
            continue
        model_root = os.path.realpath(lexical_model_root)
        if not model_root.startswith(models_cache_root + os.sep):
            continue
        try:
            entries = sorted(os.listdir(lexical_model_root))
        except OSError:
            continue
        for package_id in entries:
            if (
                package_id in seen_package_ids
                or package_id.startswith(".")
                or re.fullmatch(r"[A-Za-z0-9]+", package_id) is None
            ):
                continue
            seen_package_ids.add(package_id)
            package_dir = resolve_existing_model_package_cache_path(
                model_id=model_id,
                package_id=package_id,
            )
            if package_dir is None:
                continue
            config_path = os.path.join(package_dir, "model_config.json")
            config = _read_regular_json(path=config_path)
            if not isinstance(config, dict):
                continue
            if not isinstance(config.get("task_type"), str) or not config.get(
                "task_type"
            ):
                continue
            if not (
                isinstance(config.get("model_architecture"), str)
                and config.get("model_architecture")
            ) and not (
                isinstance(config.get("model_module"), str)
                and config.get("model_module")
                and isinstance(config.get("model_class"), str)
                and config.get("model_class")
            ):
                continue
            return package_dir
    return None


def get_configured_model_cache_roots() -> List[str]:
    """Return de-duplicated server and inference-models cache roots."""

    roots = [os.path.abspath(MODEL_CACHE_DIR)]
    if USE_INFERENCE_MODELS:
        try:
            from inference_models.configuration import INFERENCE_HOME

            inference_models_root = os.path.abspath(INFERENCE_HOME)
            if all(
                os.path.realpath(inference_models_root) != os.path.realpath(root)
                for root in roots
            ):
                roots.append(inference_models_root)
        except ImportError:
            pass
    return roots


def has_cached_model_variant(model_variants: Optional[List[str]]) -> bool:
    """Return True if **any** of the given model variant IDs has cached artifacts.

    Args:
        model_variants: List of model IDs as returned by
            ``WorkflowBlockManifest.get_supported_model_variants()``.
            Returns ``False`` when *None* or empty.
    """
    if not model_variants:
        return False
    return any(is_model_cached(mid) for mid in model_variants)


def _load_blocks() -> list:
    """Load workflow blocks, isolating the heavy import for testability."""
    from inference.core.workflows.execution_engine.introspection.blocks_loader import (
        load_workflow_blocks,
    )

    return load_workflow_blocks()


def scan_cached_models(
    cache_dir: str,
    excluded_cache_roots: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Walk *cache_dir* looking for cached model metadata files.

    Scans two cache layouts:

    1. **Traditional** — ``model_type.json`` marker files written by the model
       registry. Current and legacy slugged roots require exact ``model_id``
       attribution; compatible historical raw roots may derive it from the path.
    2. **inference-models** — ``model_config.json`` files written by
       ``dump_model_config_for_offline_use``.  The cache-owning ``model_id`` is
       read from the file (the directory name is an opaque slug in this
       layout).

    Returns a list of dicts with the following shape::

        {
            "model_id": "workspace/project/3",
            "name": "workspace/project/3",
            "task_type": "object-detection",
            "model_architecture": "yolov8n",
            "is_foundation": False,
        }
    """
    results_by_id: Dict[str, Dict[str, Any]] = {}
    conflicting_ids: Set[str] = set()
    if not os.path.isdir(cache_dir):
        return []
    cache_dir = os.path.abspath(cache_dir)
    resolved_cache_dir = os.path.realpath(cache_dir)
    excluded_cache_roots = [
        (os.path.abspath(root), os.path.realpath(root))
        for root in (excluded_cache_roots or [])
        if os.path.abspath(root) != cache_dir
    ]
    for root, dirs, files in os.walk(cache_dir, followlinks=True):
        root_is_cache_dir = os.path.abspath(root) == cache_dir
        filtered_dirs = []
        for directory in sorted(dirs):
            directory_path = os.path.abspath(os.path.join(root, directory))
            resolved_directory_path = os.path.realpath(directory_path)
            if any(
                directory_path == excluded_root
                or directory_path.startswith(excluded_root + os.sep)
                or resolved_directory_path == resolved_excluded_root
                or resolved_directory_path.startswith(resolved_excluded_root + os.sep)
                for excluded_root, resolved_excluded_root in excluded_cache_roots
            ):
                continue
            if os.path.islink(directory_path) and not (
                root_is_cache_dir and directory == "models-cache"
            ):
                continue
            if os.path.islink(directory_path):
                if (
                    resolved_directory_path == resolved_cache_dir
                    or resolved_directory_path.startswith(resolved_cache_dir + os.sep)
                    or resolved_cache_dir.startswith(resolved_directory_path + os.sep)
                ):
                    continue
            filtered_dirs.append(directory)
        dirs[:] = filtered_dirs
        files = sorted(files)
        # Prune top-level directories we know are not model trees.
        rel = os.path.relpath(root, cache_dir)
        if rel == ".":
            dirs[:] = [d for d in dirs if d not in _SKIP_TOP_LEVEL]
            continue
        relative_parts = rel.split(os.sep)

        has_model_type = "model_type.json" in files
        has_model_config = "model_config.json" in files

        if not has_model_type and not has_model_config:
            continue

        metadata: Optional[dict] = None
        stored_model_id: Optional[str] = None

        # Prefer model_config.json when present — it contains the cache-owning
        # model_id, while the directory name is an opaque slug.
        if has_model_config:
            config_path = os.path.join(root, "model_config.json")
            valid_inference_models_location = (
                len(relative_parts) == 3
                and relative_parts[0] == "models-cache"
                and re.fullmatch(r"[A-Za-z0-9]+", relative_parts[2]) is not None
            )
            if not valid_inference_models_location:
                has_model_config = False
        if has_model_config:
            cfg = _read_regular_json(path=config_path)
            if isinstance(cfg, dict) and cfg.get("task_type"):
                metadata = cfg
                manifest_model_id = cfg.get("model_id")
                if isinstance(manifest_model_id, str) and manifest_model_id:
                    stored_model_id = manifest_model_id
                    try:
                        from inference_models.models.auto_loaders.model_cache_paths import (
                            slugify_model_id_to_os_safe_format,
                        )

                        expected_slugs = {
                            slugify_model_id_to_os_safe_format(
                                model_id=stored_model_id
                            )
                        }
                    except (ImportError, TypeError):
                        expected_slugs = set()
                    if relative_parts[1] not in expected_slugs:
                        metadata = None
                        stored_model_id = None
                else:
                    metadata = None
                    stored_model_id = None

        # Fall back to model_type.json for the traditional layout.
        if metadata is None and has_model_type and relative_parts[0] != "models-cache":
            model_type_path = os.path.join(root, "model_type.json")
            metadata = _read_regular_json(path=model_type_path)
            if metadata is _INVALID_CACHE_METADATA:
                continue
            if isinstance(metadata, dict):
                stored_model_id = _resolve_traditional_cache_model_id(
                    cache_dir=cache_dir,
                    model_root=root,
                    metadata=metadata,
                )
                if stored_model_id is None:
                    metadata = None

        if not isinstance(metadata, dict):
            continue

        # Support traditional keys, inference-models config keys and
        # inference-models API metadata keys.
        task_type = (
            metadata.get("task_type")
            or metadata.get(PROJECT_TASK_TYPE_KEY)
            or metadata.get("taskType", "")
        )
        model_architecture = (
            metadata.get("model_architecture")
            or metadata.get(MODEL_TYPE_KEY)
            or metadata.get("modelArchitecture", "")
        )

        if not isinstance(task_type, str) or not task_type:
            continue
        if not isinstance(model_architecture, str):
            continue

        if stored_model_id is not None:
            model_id = stored_model_id
        elif has_model_config:
            continue
        else:
            model_id = os.path.relpath(root, cache_dir)
            # Normalise path separators on Windows.
            model_id = model_id.replace(os.sep, "/")

        if not isinstance(model_id, str) or not model_id:
            continue
        if model_id in conflicting_ids:
            continue
        result = {
            "model_id": model_id,
            "name": model_id,
            "task_type": task_type,
            "model_architecture": model_architecture,
            "is_foundation": False,
        }
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

    return list(results_by_id.values())


def get_cached_foundation_models(
    blocks: Optional[list] = None,
) -> List[Dict[str, Any]]:
    """Return metadata for workflow blocks whose required weights are cached.

    Each block whose manifest class exposes ``get_supported_model_variants``
    is inspected.  If any variant it declares is present in the local cache
    the block is included in the result list.

    Args:
        blocks: Optional pre-loaded list of block specifications.  When
            *None* (the default) the blocks are loaded via the engine's
            block loader.
    """
    results: List[Dict[str, Any]] = []
    if blocks is None:
        try:
            blocks = _load_blocks()
        except Exception:
            logger.debug(
                "Could not load workflow blocks for foundation model scan",
                exc_info=True,
            )
            return results

    for block in blocks:
        manifest_cls = block.manifest_class
        model_variants = manifest_cls.get_supported_model_variants()
        if model_variants is None:
            continue

        cached_model_id = next(
            (
                model_variant
                for model_variant in model_variants
                if is_model_cached(model_variant)
            ),
            None,
        )
        if cached_model_id is None:
            continue

        model_id = cached_model_id

        block_name = model_id
        try:
            schema = manifest_cls.model_json_schema()
            block_name = schema.get("name", model_id)
        except Exception:
            pass

        block_type_id = _get_block_type_identifier(block)

        results.append(
            {
                "model_id": model_id,
                "name": block_name,
                "task_type": "",
                "model_architecture": "",
                "is_foundation": True,
                "block_type": block_type_id,
            }
        )

    return results


def get_task_type_to_block_mapping(
    blocks: Optional[list] = None,
) -> Dict[str, List[str]]:
    """Build a reverse mapping from task_type to compatible block type identifiers.

    Uses ``get_compatible_task_types()`` on block manifests.  Blocks whose
    method returns *None* (the base-class default) are skipped.

    Args:
        blocks: Optional pre-loaded list of block specifications.  When
            *None* (the default) the blocks are loaded via the engine's
            block loader.
    """
    mapping: Dict[str, List[str]] = {}
    if blocks is None:
        try:
            blocks = _load_blocks()
        except Exception:
            logger.debug(
                "Could not load workflow blocks for task-type mapping",
                exc_info=True,
            )
            return mapping

    for block in blocks:
        manifest_cls = block.manifest_class
        task_types = manifest_cls.get_compatible_task_types()
        if task_types is None:
            continue

        block_type_id = _get_block_type_identifier(block)
        for tt in task_types:
            mapping.setdefault(tt, []).append(block_type_id)

    return mapping


def _get_block_type_identifier(block) -> str:
    """Extract the canonical ``type`` identifier from a block specification."""
    try:
        schema = block.manifest_class.model_json_schema()
        type_prop = schema.get("properties", {}).get("type", {})
        # The type field is typically a const or enum with one value.
        if "const" in type_prop:
            return type_prop["const"]
        if "enum" in type_prop and type_prop["enum"]:
            return type_prop["enum"][0]
    except Exception:
        pass
    return block.identifier
