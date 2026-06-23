"""Utilities for discovering models and foundation-model weights in the local cache.

Used by the air-gapped workflow builder to enumerate what is available for
offline workflow construction.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from inference.core.cache.model_artifacts import (
    get_cache_dir,
    slugify_model_id_to_cache_key,
)
from inference.core.env import MODEL_CACHE_DIR, USE_INFERENCE_MODELS
from inference.core.roboflow_api import MODEL_TYPE_KEY, PROJECT_TASK_TYPE_KEY

logger = logging.getLogger(__name__)

# Directories directly under MODEL_CACHE_DIR that are not model trees.
_SKIP_TOP_LEVEL = {"workflow", "_file_locks"}


def _slugify_model_id(model_id: str) -> str:
    """Reproduce the slug used by inference-models for cache directory names.

    Must stay in sync with
    ``inference_models.models.auto_loaders.core.slugify_model_id_to_os_safe_format``.
    """
    return slugify_model_id_to_cache_key(model_id=model_id)


def _has_non_hidden_children(path: str) -> bool:
    """Return True if *path* contains at least one non-hidden entry.

    Hidden files (names starting with ``"."``) are excluded because they may be
    stale lock-file leftovers that do not represent usable model artifacts.
    """
    try:
        return any(not f.startswith(".") for f in os.listdir(path))
    except OSError:
        return False


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
    if not USE_INFERENCE_MODELS:
        # Only check the traditional layout when inference-models is disabled.
        traditional_path = get_cache_dir(
            model_id=model_id, cache_dir_root=MODEL_CACHE_DIR
        )
        return os.path.isdir(traditional_path) and _has_non_hidden_children(
            traditional_path
        )

    # When inference-models is enabled, check both layouts — models cached
    # before the migration still sit in the traditional tree.
    traditional_path = get_cache_dir(model_id=model_id, cache_dir_root=MODEL_CACHE_DIR)
    if os.path.isdir(traditional_path) and _has_non_hidden_children(traditional_path):
        return True

    slug = _slugify_model_id(model_id)
    models_cache_path = os.path.join(MODEL_CACHE_DIR, "models-cache", slug)
    if os.path.isdir(models_cache_path) and _has_non_hidden_children(models_cache_path):
        return True

    return False


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


def scan_cached_models(cache_dir: str) -> List[Dict[str, Any]]:
    """Walk *cache_dir* looking for ``model_type.json`` marker files.

    Each marker is written by the model registry when a model is first
    downloaded.  The file contains at least ``project_task_type`` and
    ``model_type`` keys.

    Returns a list of dicts with the following shape::

        {
            "model_id": "workspace/project/3",
            "name": "workspace/project/3",
            "task_type": "object-detection",
            "model_architecture": "yolov8n",
            "is_foundation": False,
        }
    """
    results: List[Dict[str, Any]] = []
    if not os.path.isdir(cache_dir):
        return results

    for root, dirs, files in os.walk(cache_dir):
        # Prune top-level directories we know are not model trees.
        rel = os.path.relpath(root, cache_dir)
        if rel == ".":
            dirs[:] = [d for d in dirs if d not in _SKIP_TOP_LEVEL]
            continue

        if "model_type.json" not in files:
            continue

        model_type_path = os.path.join(root, "model_type.json")
        try:
            with open(model_type_path, "r") as fh:
                metadata = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Skipping unreadable model_type.json at %s: %s",
                model_type_path,
                exc,
            )
            continue

        if not isinstance(metadata, dict):
            continue

        # Support both traditional keys and inference-models metadata keys.
        task_type = metadata.get(PROJECT_TASK_TYPE_KEY) or metadata.get("taskType", "")
        model_architecture = metadata.get(MODEL_TYPE_KEY) or metadata.get(
            "modelArchitecture", ""
        )

        if not task_type:
            continue

        model_id = os.path.relpath(root, cache_dir)
        # Normalise path separators on Windows.
        model_id = model_id.replace(os.sep, "/")

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

        if not has_cached_model_variant(model_variants):
            continue

        model_id = model_variants[0] if model_variants else ""

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
