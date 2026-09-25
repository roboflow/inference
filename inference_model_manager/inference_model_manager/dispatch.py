"""Action dispatch and discovery — delegates to model registry.

- **Dispatch**: resolve action name → model method via registry, call it.
- **Discovery**: resolve model_id → model class → registered actions (no loading).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from inference_model_manager.registry import ActionEntry
from inference_model_manager.registry_defaults import (
    _ACTION_CONFIGS,
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
    return method(**kwargs)


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
