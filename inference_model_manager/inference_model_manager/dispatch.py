"""Action dispatch and discovery — delegates to model registry.

- **Dispatch**: resolve action name → model method via registry, call it.
- **Discovery**: resolve model_id → model class → registered actions (no loading).
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Dict, Optional

from inference_model_manager.image_embeddings import describe_image_embeddings
from inference_model_manager.registry import ActionEntry
from inference_model_manager.registry_defaults import (
    _ACTION_CONFIGS,
    IMAGE_EMBEDDINGS_RESPONSE_TYPE,
    PRE_PROCESSING_OVERRIDE_FIELDS,
    SAM_IMAGE_EMBEDDINGS_TYPE,
    _unpack_config,
    lazy_register,
    registry,
)

logger = logging.getLogger(__name__)


def _get_registry():
    """Return the default action registry."""
    return registry


def discover_actions_by_mro(mro_names: list[str]) -> Dict[str, Dict[str, Any]]:
    """Discover supported actions from MRO class name strings.

    Used for backends whose model does not live in-process.
    Matches names against registry configs.
    """
    result: Dict[str, Dict[str, Any]] = {}
    for name in mro_names:
        config = _ACTION_CONFIGS.get(name)
        if config is None:
            continue
        for cfg in config:
            action_name, method, default, params, _v, _s, resp_type, _a = _unpack_config(cfg)
            if action_name not in result:
                result[action_name] = {
                    "method": method,
                    "default": default,
                    "params": params,
                    "response_type": resp_type,
                }
    return result


def resolve_action(model: Any, action: Optional[str] = None) -> tuple[str, ActionEntry]:
    """Resolve action name to (action_name, ActionEntry) via registry.

    Args:
        model: Model instance.
        action: Action name. None → default action for this model's class.

    Returns:
        Tuple of (resolved_action_name, ActionEntry).

    Raises:
        ValueError: If action not found or no default action registered.
    """
    lazy_register(type(model))

    registry = _get_registry()
    actions = _entries_for_model(model, registry)

    if action is None:
        defaults = [(n, e) for n, e in actions.items() if e.default]
        if not defaults:
            raise ValueError(
                f"No default action registered for {type(model).__name__}. "
                f"Available actions: {list(actions.keys())}"
            )
        return defaults[0]

    if action not in actions:
        raise ValueError(
            f"Action '{action}' not registered for {type(model).__name__}. "
            f"Available actions: {list(actions.keys())}"
        )
    return action, actions[action]


def invoke_action(
    model: Any,
    action: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Resolve action via registry and call the model method.

    Returns whatever the model method returns.
    """
    action_name, entry = resolve_action(model, action)
    method = getattr(model, entry.method, None)
    if method is None:
        raise ValueError(
            f"Model {type(model).__name__} has no method '{entry.method}' "
            f"(registered for action '{action_name}')"
        )
    if entry.param_aliases:
        kwargs = {entry.param_aliases.get(k, k): v for k, v in kwargs.items()}
    kwargs = _build_pre_processing_overrides(kwargs, entry)
    kwargs = _build_sam_image_embeddings(kwargs, entry)
    result = method(**kwargs)
    if entry.response_type == IMAGE_EMBEDDINGS_RESPONSE_TYPE:
        result = describe_image_embeddings(model, result, kwargs)
    return result


def _build_pre_processing_overrides(kwargs: dict, entry: ActionEntry) -> dict:
    if not any(flag in entry.params for flag in PRE_PROCESSING_OVERRIDE_FIELDS):
        for flag in PRE_PROCESSING_OVERRIDE_FIELDS:
            kwargs.pop(flag, None)
        return kwargs
    if not any(flag in kwargs for flag in PRE_PROCESSING_OVERRIDE_FIELDS):
        return kwargs

    from inference_models.models.auto_loaders.entities import PreProcessingOverrides

    overrides = kwargs.get("pre_processing_overrides") or PreProcessingOverrides()
    enabled = {
        field: True
        for flag, field in PRE_PROCESSING_OVERRIDE_FIELDS.items()
        if kwargs.pop(flag, False)
    }
    kwargs["pre_processing_overrides"] = dataclasses.replace(overrides, **enabled)
    return kwargs


def _build_sam_image_embeddings(kwargs: dict, entry: ActionEntry) -> dict:
    if not isinstance(entry.params, dict):
        return kwargs
    declared = entry.params.get("embeddings") or {}
    if declared.get("type") != SAM_IMAGE_EMBEDDINGS_TYPE:
        return kwargs
    embeddings = kwargs.get("embeddings")
    if isinstance(embeddings, dict):
        kwargs["embeddings"] = _sam_image_embeddings_from_wire(
            embeddings, _single_image(kwargs.get("images"))
        )
    elif isinstance(embeddings, list) and any(isinstance(e, dict) for e in embeddings):
        images = kwargs.get("images")
        kwargs["embeddings"] = [
            (
                _sam_image_embeddings_from_wire(e, _image_at(images, index))
                if isinstance(e, dict)
                else e
            )
            for index, e in enumerate(embeddings)
        ]
    return kwargs


def _single_image(images: Any) -> Any:
    if isinstance(images, list):
        return images[0] if images else None
    return images


def _image_at(images: Any, index: int) -> Any:
    if isinstance(images, list):
        return images[index] if index < len(images) else None
    return images if index == 0 else None


def _validate_sam_embeddings_wire(wire: Any) -> None:
    import numpy as np

    if not isinstance(wire, dict) or not {
        "embeddings",
        "image_hash",
        "image_size_hw",
    } <= set(wire):
        raise ValueError(
            "embeddings must be a dict with embeddings, image_hash and image_size_hw"
        )
    array = wire["embeddings"]
    if not isinstance(array, np.ndarray) or array.dtype.kind not in "fiu":
        raise ValueError("embeddings must be a numeric array")
    if wire["image_hash"] is not None and not isinstance(wire["image_hash"], str):
        raise ValueError("image_hash must be a string")
    size = wire["image_size_hw"]
    if size is not None and not (
        isinstance(size, (list, tuple))
        and len(size) == 2
        and all(
            isinstance(side, int) and not isinstance(side, bool) and side > 0
            for side in size
        )
    ):
        raise ValueError("image_size_hw must be two positive integers")


def _sam_image_embeddings_from_wire(wire: dict, image: Any) -> Any:
    import torch
    from inference_models.models.sam.entities import SAMImageEmbeddings
    from inference_models.models.sam.sam_torch import compute_image_hash

    _validate_sam_embeddings_wire(wire)
    image_hash = wire.get("image_hash")
    if image_hash is None:
        if image is None:
            raise ValueError("image_id is required when image not provided")
        image_hash = compute_image_hash(image=image)
    image_size_hw = wire.get("image_size_hw")
    if image_size_hw is None:
        if image is None:
            raise ValueError(
                "orig_im_size is required when image not provided and embeddings "
                "are injected by client."
            )
        image_size_hw = (image.shape[0], image.shape[1])

    entity = SAMImageEmbeddings(
        image_hash=image_hash,
        image_size_hw=(image_size_hw[0], image_size_hw[1]),
        embeddings=torch.from_numpy(wire["embeddings"]),
    )
    return entity


def list_actions(model: Any) -> Dict[str, Dict[str, Any]]:
    """Return human-readable action info for a model instance."""
    registry = _get_registry()
    actions = _entries_for_model(model, registry)
    return _entries_to_dict(actions)


def list_actions_for_class(model_class: type) -> Dict[str, Dict[str, Any]]:
    """Return human-readable action info for a model class (no instance needed)."""
    registry = _get_registry()
    actions = _entries_for_class(model_class, registry)
    return _entries_to_dict(actions)


def _entries_for_model(model: Any, registry) -> Dict[str, ActionEntry]:
    """Collect all registered actions for a model instance, following MRO."""
    return _entries_for_class(type(model), registry)


def _entries_for_class(model_class: type, registry) -> Dict[str, ActionEntry]:
    """Collect all registered actions for a model class, following MRO."""
    result: Dict[str, ActionEntry] = {}
    for cls in model_class.__mro__:
        class_entries = registry._entries.get(cls, {})
        for name, entry in class_entries.items():
            if name not in result:
                result[name] = entry
    return result


def list_actions_by_mro_names(mro_names: list[str]) -> Dict[str, Dict[str, Any]]:
    """Return action info by MRO class name strings."""
    registry = _get_registry()
    result: Dict[str, "ActionEntry"] = {}
    with registry._lock:
        for name in mro_names:
            for cls, class_entries in registry._entries.items():
                if cls.__name__ == name:
                    for action_name, entry in class_entries.items():
                        if action_name not in result:
                            result[action_name] = entry
    return _entries_to_dict(result)


def _entries_to_dict(actions: Dict[str, ActionEntry]) -> Dict[str, Dict[str, Any]]:
    return {
        name: {
            "method": entry.method,
            "default": entry.default,
            "params": entry.params,
            "response_type": entry.response_type,
        }
        for name, entry in actions.items()
    }
