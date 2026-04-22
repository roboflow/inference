"""Task dispatch for ManagedModel instances.

Resolves a task name to a model method and calls it with the provided kwargs.
Used by ModelManager.invoke() and SubprocessBackend worker.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from inference_models.models.base.task_dispatch import ManagedModel, TaskSpec

logger = logging.getLogger(__name__)


def resolve_task(model: ManagedModel, task: Optional[str] = None) -> TaskSpec:
    """Resolve task name to TaskSpec.

    Args:
        model: Model instance (must have ``supported_tasks`` property).
        task: Task name. ``None`` → default task.

    Returns:
        TaskSpec for the resolved task.

    Raises:
        ValueError: If task not found or no default task defined.
    """
    tasks = model.supported_tasks

    if task is None:
        defaults = [t for t in tasks.values() if t.default]
        if not defaults:
            raise ValueError(
                f"Model {type(model).__name__} has no default task. "
                f"Available tasks: {list(tasks.keys())}"
            )
        return defaults[0]

    if task not in tasks:
        raise ValueError(
            f"Task '{task}' not supported by {type(model).__name__}. "
            f"Available tasks: {list(tasks.keys())}"
        )
    return tasks[task]


def invoke_task(
    model: ManagedModel,
    task: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Resolve task and call the corresponding method on the model.

    Args:
        model: Model instance.
        task: Task name. ``None`` → default task.
        **kwargs: Passed to the model method.

    Returns:
        Whatever the model method returns.

    Raises:
        ValueError: If task not found or required params missing.
    """
    spec = resolve_task(model, task)
    method = getattr(model, spec.method, None)
    if method is None:
        raise ValueError(
            f"Model {type(model).__name__} declares task '{task or 'default'}' "
            f"with method '{spec.method}' but method does not exist"
        )
    return method(**kwargs)


def list_tasks(model: ManagedModel) -> Dict[str, Dict[str, Any]]:
    """Return human-readable task info for a model.

    Returns:
        Dict mapping task name → {"method", "default", "params"}.
    """
    return {
        name: {
            "method": spec.method,
            "default": spec.default,
            "params": spec.params,
        }
        for name, spec in model.supported_tasks.items()
    }
