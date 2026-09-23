import json
import logging
import os
import stat
from typing import Any, Dict, List, Optional, Set

from inference_models.models.auto_loaders.model_cache_paths import (
    MODEL_CONFIG_FILE_NAME,
    generate_models_cache_dir,
    slugify_model_id_to_os_safe_format,
)
from inference_sdk.http.utils.aliases import REGISTERED_ALIASES

from inference_server import configuration

logger = logging.getLogger(__name__)

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


def _collect_unambiguous_user_models(
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


def _offline_loadable_model_ids() -> Optional[Set[str]]:
    """Ids (canonical + recorded aliases) the offline-weights registry can serve.

    A model is offline-loadable only when its registry record exists AND at
    least one recorded package has all its files on disk - the same bar the
    roboflow-offline-weights provider applies before negotiation.

    Returns None when the registry cannot be consulted, so the caller keeps
    the listing unfiltered instead of showing an empty picker on an internal
    error.
    """
    try:
        from inference_models.weights_providers.offline_registry import (
            OfflinePackagePresence,
            list_records_status,
        )

        loadable: Set[str] = set()
        for record_status in list_records_status():
            if not any(
                package.presence is OfflinePackagePresence.OK
                for package in record_status.packages
            ):
                continue
            loadable.add(record_status.canonical_model_id)
            loadable.update(record_status.requested_aliases)
        return loadable
    except Exception:
        logger.warning(
            "Could not consult the offline-weights registry - the model "
            "listing will not be filtered to offline-loadable models.",
            exc_info=True,
        )
        return None


def _listdir(path: str) -> List[str]:
    try:
        return sorted(os.listdir(path))
    except OSError:
        return []


def _is_cached(model_id: str) -> bool:
    """True when at least one non-symlinked package of *model_id* is on disk."""

    model_root = os.path.join(
        generate_models_cache_dir(),
        slugify_model_id_to_os_safe_format(model_id=model_id),
    )
    if os.path.islink(model_root) or not os.path.isdir(model_root):
        return False
    for package_id in _listdir(model_root):
        package_dir = os.path.join(model_root, package_id)
        if os.path.islink(package_dir) or not os.path.isdir(package_dir):
            continue
        if os.path.isfile(os.path.join(package_dir, MODEL_CONFIG_FILE_NAME)):
            return True
    return False


def _scan_inference_models_cache() -> List[Dict[str, Any]]:
    """Read every ``<slug>/<package_id>/model_config.json`` in the models cache."""

    results: List[Dict[str, Any]] = []
    models_cache_dir = generate_models_cache_dir()
    if not os.path.isdir(models_cache_dir):
        return results
    for model_slug in _listdir(models_cache_dir):
        model_root = os.path.join(models_cache_dir, model_slug)
        if os.path.islink(model_root) or not os.path.isdir(model_root):
            continue
        for package_id in _listdir(model_root):
            package_dir = os.path.join(model_root, package_id)
            if os.path.islink(package_dir) or not os.path.isdir(package_dir):
                continue
            config_path = os.path.join(package_dir, MODEL_CONFIG_FILE_NAME)
            config = _read_regular_json(path=config_path)
            if not isinstance(config, dict):
                continue
            model_id = config.get("model_id") or config.get("canonical_model_id")
            if not isinstance(model_id, str) or not model_id:
                continue
            task_type = config.get("task_type") or ""
            model_architecture = config.get("model_architecture") or ""
            if not isinstance(task_type, str) or not isinstance(
                model_architecture, str
            ):
                continue
            results.append(
                {
                    "model_id": model_id,
                    "name": model_id,
                    "task_type": task_type,
                    "model_architecture": model_architecture,
                    "is_foundation": False,
                }
            )
    return results


def _get_block_type_identifier(block) -> str:
    """Extract the canonical ``type`` identifier from a block specification."""
    try:
        schema = block.manifest_class.model_json_schema()
        type_prop = schema.get("properties", {}).get("type", {})
        if "const" in type_prop:
            return type_prop["const"]
        if "enum" in type_prop and type_prop["enum"]:
            return type_prop["enum"][0]
    except Exception:
        pass
    return block.identifier


def get_cached_foundation_models(blocks: list) -> List[Dict[str, Any]]:
    """Return metadata for workflow blocks whose required weights are cached."""

    results: List[Dict[str, Any]] = []
    for block in blocks:
        manifest_cls = block.manifest_class
        model_variants = manifest_cls.get_supported_model_variants()
        if model_variants is None:
            continue

        cached_model_id = next(
            (
                model_variant
                for model_variant in model_variants
                if _is_cached(model_variant)
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


def get_task_type_to_block_mapping(blocks: list) -> Dict[str, List[str]]:
    """Build a reverse mapping from task_type to compatible block type identifiers."""

    mapping: Dict[str, List[str]] = {}
    for block in blocks:
        manifest_cls = block.manifest_class
        task_types = manifest_cls.get_compatible_task_types()
        if task_types is None:
            continue

        block_type_id = _get_block_type_identifier(block)
        for tt in task_types:
            mapping.setdefault(tt, []).append(block_type_id)

    return mapping


async def list_models(bridge) -> List[Dict[str, Any]]:
    """Models the Workflow Builder picker can offer for this server."""

    try:
        from roboflow_workflows.execution_engine.introspection.blocks_loader import (
            load_workflow_blocks,
        )

        blocks = load_workflow_blocks()
    except Exception:
        logger.warning(
            "Failed to load workflow blocks — foundation model data will "
            "be unavailable. This may indicate a broken build or missing "
            "dependencies.",
            exc_info=True,
        )
        blocks = []

    reverse_aliases: Dict[str, List[str]] = {}
    for alias, canonical in REGISTERED_ALIASES.items():
        reverse_aliases.setdefault(canonical, []).append(alias)

    user_models: List[Dict[str, Any]] = [
        {
            "model_id": route.model_id,
            "name": route.model_id,
            "task_type": route.task_type or "",
            "model_architecture": "",
            "is_foundation": False,
        }
        for route in await bridge.describe()
    ]
    user_models.extend(_scan_inference_models_cache())

    seen = _collect_unambiguous_user_models(user_models=user_models)
    for m in get_cached_foundation_models(blocks=blocks):
        seen[m["model_id"]] = m

    if configuration.OFFLINE_MODE:
        offline_loadable = _offline_loadable_model_ids()
        if offline_loadable is not None:
            dropped = [
                model_id
                for model_id in seen
                if model_id not in offline_loadable
                and not set(reverse_aliases.get(model_id, [])) & offline_loadable
            ]
            for model_id in dropped:
                seen.pop(model_id)
            if dropped:
                logger.info(
                    "OFFLINE_MODE: hiding %d cached model(s) absent from "
                    "the offline-weights registry: %s. Run once with "
                    "OFFLINE_MODE_WARM_UP=True on a connected machine to "
                    "register them.",
                    len(dropped),
                    ", ".join(sorted(dropped)),
                )

    task_to_blocks = get_task_type_to_block_mapping(blocks=blocks)
    models: List[Dict[str, Any]] = []
    for m in seen.values():
        entry = dict(m)
        block_type = entry.get("block_type")
        if block_type:
            entry.setdefault("compatible_block_types", [block_type])
        else:
            entry.setdefault(
                "compatible_block_types",
                task_to_blocks.get(m.get("task_type", ""), []),
            )
        model_id = m.get("model_id", "")
        aliases = reverse_aliases.get(model_id, [])
        entry["aliases"] = aliases
        if aliases and (entry.get("name") == model_id or not entry.get("name")):
            entry["name"] = min(aliases, key=len)
        entry.pop("block_type", None)
        models.append(entry)

    return models
