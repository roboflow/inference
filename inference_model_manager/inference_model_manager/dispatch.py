"""Task dispatch and discovery — delegates to model registry.

- **Dispatch**: resolve task name → model method via registry, call it.
- **Discovery**: resolve model_id → model class → registered tasks (no loading).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from inference_model_manager.registry import TaskEntry

logger = logging.getLogger(__name__)


def _get_registry():
    """Lazy import to avoid heavy imports at module level."""
    from inference_model_manager.registry_defaults import registry

    return registry


def discover_tasks_by_mro(mro_names: list[str]) -> Dict[str, Dict[str, Any]]:
    """Discover supported tasks from MRO class name strings.

    Used for subprocess backends where model lives in worker process.
    Matches names against registry configs.
    """
    from inference_model_manager.registry_defaults import _TASK_CONFIGS

    result: Dict[str, Dict[str, Any]] = {}
    for name in mro_names:
        config = _TASK_CONFIGS.get(name)
        if config is None:
            continue
        for task_name, method, default, params, _v, _s, resp_type in config:
            if task_name not in result:
                result[task_name] = {
                    "method": method,
                    "default": default,
                    "params": params,
                    "response_type": resp_type,
                }
    return result


def resolve_task(model: Any, task: Optional[str] = None) -> tuple[str, TaskEntry]:
    """Resolve task name to (task_name, TaskEntry) via registry.

    Args:
        model: Model instance.
        task: Task name. None → default task for this model's class.

    Returns:
        Tuple of (resolved_task_name, TaskEntry).

    Raises:
        ValueError: If task not found or no default task registered.
    """
    from inference_model_manager.registry_defaults import lazy_register

    lazy_register(type(model))

    registry = _get_registry()
    tasks = _entries_for_model(model, registry)

    if task is None:
        defaults = [(n, e) for n, e in tasks.items() if e.default]
        if not defaults:
            raise ValueError(
                f"No default task registered for {type(model).__name__}. "
                f"Available tasks: {list(tasks.keys())}"
            )
        return defaults[0]

    if task not in tasks:
        raise ValueError(
            f"Task '{task}' not registered for {type(model).__name__}. "
            f"Available tasks: {list(tasks.keys())}"
        )
    return task, tasks[task]


def invoke_task(
    model: Any,
    task: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Resolve task via registry and call the model method.

    Returns whatever the model method returns.
    """
    task_name, entry = resolve_task(model, task)
    method = getattr(model, entry.method, None)
    if method is None:
        raise ValueError(
            f"Model {type(model).__name__} has no method '{entry.method}' "
            f"(registered for task '{task_name}')"
        )
    return method(**kwargs)


def list_tasks(model: Any) -> Dict[str, Dict[str, Any]]:
    """Return human-readable task info for a model instance."""
    registry = _get_registry()
    tasks = _entries_for_model(model, registry)
    return _entries_to_dict(tasks)


def list_tasks_for_class(model_class: type) -> Dict[str, Dict[str, Any]]:
    """Return human-readable task info for a model class (no instance needed)."""
    registry = _get_registry()
    tasks = _entries_for_class(model_class, registry)
    return _entries_to_dict(tasks)


def _entries_for_model(model: Any, registry) -> Dict[str, TaskEntry]:
    """Collect all registered tasks for a model instance, following MRO."""
    return _entries_for_class(type(model), registry)


def _entries_for_class(model_class: type, registry) -> Dict[str, TaskEntry]:
    """Collect all registered tasks for a model class, following MRO."""
    result: Dict[str, TaskEntry] = {}
    for cls in model_class.__mro__:
        class_entries = registry._entries.get(cls, {})
        for name, entry in class_entries.items():
            if name not in result:
                result[name] = entry
    return result


def list_tasks_by_mro_names(mro_names: list[str]) -> Dict[str, Dict[str, Any]]:
    """Return task info by MRO class name strings (subprocess path)."""
    registry = _get_registry()
    result: Dict[str, "TaskEntry"] = {}
    for name in mro_names:
        for cls, class_entries in registry._entries.items():
            if cls.__name__ == name:
                for task_name, entry in class_entries.items():
                    if task_name not in result:
                        result[task_name] = entry
    return _entries_to_dict(result)


def _entries_to_dict(tasks: Dict[str, TaskEntry]) -> Dict[str, Dict[str, Any]]:
    return {
        name: {
            "method": entry.method,
            "default": entry.default,
            "params": entry.params,
            "response_type": entry.response_type,
        }
        for name, entry in tasks.items()
    }
