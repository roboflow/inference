"""Utilities for discovering models and foundation-model weights in the local cache.

Used by the air-gapped workflow builder to enumerate what is available for
offline workflow construction.
"""

import hashlib
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from inference.core.cache.model_artifacts import are_all_files_cached, get_cache_dir
from inference.core.env import MODEL_CACHE_DIR
from inference.core.roboflow_api import MODEL_TYPE_KEY, PROJECT_TASK_TYPE_KEY

logger = logging.getLogger(__name__)

# Directories directly under MODEL_CACHE_DIR that are not model trees.
_SKIP_TOP_LEVEL = {"workflow", "_file_locks"}


def _slugify_model_id(model_id: str) -> str:
    """Reproduce the slug used by inference-models for cache directory names.

    Must stay in sync with
    ``inference_models.models.auto_loaders.core.slugify_model_id_to_os_safe_format``.
    """
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", model_id)
    slug = re.sub(r"[_-]{2,}", "-", slug)
    if not slug:
        slug = "special-char-only-model-id"
    if len(slug) > 48:
        slug = slug[:48]
    digest = hashlib.blake2s(model_id.encode("utf-8"), digest_size=4).hexdigest()
    return f"{slug}-{digest}"


def _is_model_cached(model_id: str) -> bool:
    """Check if *model_id* has cached artifacts in either cache layout.

    Layout 1 (traditional): ``MODEL_CACHE_DIR/{model_id}/`` with files inside.
    Layout 2 (inference-models): ``MODEL_CACHE_DIR/models-cache/{slug}/`` with
    sub-directories containing model files.
    """
    # Traditional layout
    traditional_path = os.path.join(MODEL_CACHE_DIR, model_id)
    if os.path.isdir(traditional_path) and os.listdir(traditional_path):
        return True

    # inference-models layout
    slug = _slugify_model_id(model_id)
    models_cache_path = os.path.join(MODEL_CACHE_DIR, "models-cache", slug)
    if os.path.isdir(models_cache_path) and os.listdir(models_cache_path):
        return True

    return False


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
        task_type = metadata.get(PROJECT_TASK_TYPE_KEY) or metadata.get(
            "taskType", ""
        )
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

    Each block whose manifest class exposes a ``get_required_cache_artifacts``
    classmethod is inspected.  If every artifact it declares is already present
    in the local cache the block is included in the result list.

    Blocks that do not expose the classmethod are silently skipped.

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
        if not hasattr(manifest_cls, "get_required_cache_artifacts"):
            continue

        try:
            artifacts_spec = manifest_cls.get_required_cache_artifacts()
        except Exception:
            logger.debug(
                "Error calling get_required_cache_artifacts on %s",
                block.identifier,
                exc_info=True,
            )
            continue

        # artifacts_spec can be:
        #   - list of model_id strings (new format): block is cached if ANY
        #     variant directory exists and contains files
        #   - dict with {"model_id": ..., "files": [...]} (legacy format)
        if isinstance(artifacts_spec, list):
            cached = any(_is_model_cached(mid) for mid in artifacts_spec)
            if not cached:
                continue
            model_id = artifacts_spec[0] if artifacts_spec else ""
        elif isinstance(artifacts_spec, dict):
            model_id = artifacts_spec.get("model_id")
            required_files = artifacts_spec.get("files", [])
            if not model_id or not required_files:
                continue
            if not are_all_files_cached(files=required_files, model_id=model_id):
                continue
        else:
            continue

        # Derive name from the block's manifest schema (json_schema_extra)
        # rather than requiring it in the artifacts dict.
        block_name = model_id
        try:
            schema = manifest_cls.model_json_schema()
            block_name = schema.get("name", model_id)
        except Exception:
            pass

        # Use the block type identifier from the manifest's type field.
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

    Uses ``get_compatible_task_types()`` classmethod on block manifests when
    available.  Blocks that do not expose the classmethod are skipped.

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
        if not hasattr(manifest_cls, "get_compatible_task_types"):
            continue

        try:
            task_types = manifest_cls.get_compatible_task_types()
        except Exception:
            logger.debug(
                "Error calling get_compatible_task_types on %s",
                block.identifier,
                exc_info=True,
            )
            continue

        if not isinstance(task_types, (list, tuple, set)):
            continue

        # Derive the manifest type identifier
        # (e.g. "roboflow_core/roboflow_object_detection_model@v2")
        # from the block schema.
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
