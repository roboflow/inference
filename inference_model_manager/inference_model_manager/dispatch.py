"""Task dispatch and discovery for ManagedModel.

- **Dispatch**: resolve task name → model method, call it.
- **Discovery**: resolve model_id → model class → supported tasks (no loading).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from inference_models.models.base.task_dispatch import ManagedModel, TaskSpec

logger = logging.getLogger(__name__)


def discover_tasks(
    model_id: str,
    api_key: str = "",
    **resolve_kwargs,
) -> Dict[str, Dict[str, Any]]:
    """Discover supported tasks for a model_id WITHOUT loading the model.

    Resolves model_id → model class via AutoModel's resolution chain
    (Roboflow API call, cached 24h), then reads ``get_supported_tasks()``
    classmethod. No download, no GPU, no instantiation.

    Args:
        model_id: Model identifier (e.g. ``"yolov8n-640"``, ``"workspace/model/1"``).
        api_key: Roboflow API key (needed for custom models).
        **resolve_kwargs: Forwarded to AutoModel resolution (device, backend, etc.).

    Returns:
        Dict mapping task name → {"method", "default", "params"}.

    Raises:
        RuntimeError: If model_id cannot be resolved.
    """
    from inference_models.models.auto_loaders.core import AutoModel

    model_class = AutoModel.resolve_class(
        model_id, api_key=api_key, **resolve_kwargs
    )
    return list_tasks(model_class)


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


def list_tasks(model_or_class) -> Dict[str, Dict[str, Any]]:
    """Return human-readable task info for a model instance or class.

    Args:
        model_or_class: ManagedModel instance OR class.

    Returns:
        Dict mapping task name → {"method", "default", "params"}.
    """
    if isinstance(model_or_class, type):
        tasks = model_or_class.get_supported_tasks()
    else:
        tasks = model_or_class.supported_tasks
    return {
        name: {
            "method": spec.method,
            "default": spec.default,
            "params": spec.params,
        }
        for name, spec in tasks.items()
    }
