"""Task dispatch mixin for inference models.

Every model must declare which tasks it supports via ``get_supported_tasks()``
classmethod. ModelManager uses this for:
  - **Dispatch**: ``process(model_id, task, **kwargs)`` → correct model method.
  - **Discovery**: ``get_supported_tasks(model_id)`` → what can this model do?
    Works without loading the model (class-level, no instance needed).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class TaskSpec:
    """Describes one invocable task on a model."""

    method: str
    """Name of the method to call on the model instance."""

    default: bool = False
    """True for exactly one task — used when caller omits ``task`` param."""

    params: List[str] = field(default_factory=list)
    """Parameter names the method accepts (for validation / documentation)."""


class ManagedModel(ABC):
    """Mixin ABC for models managed by ModelManager.

    Any model that wants to work with ModelManager must inherit this
    and implement ``get_supported_tasks()``. ABC enforces it —
    instantiation raises ``TypeError`` if the classmethod is missing.
    """

    @classmethod
    @abstractmethod
    def get_supported_tasks(cls) -> Dict[str, TaskSpec]:
        """Return task dispatch table. Class-level — no instance needed.

        Must return a dict with at least one entry where ``default=True``.

        Example::

            {
                "infer": TaskSpec(method="infer", default=True, params=["images"]),
            }
        """
        ...

    @property
    def supported_tasks(self) -> Dict[str, TaskSpec]:
        """Instance-level convenience — delegates to classmethod."""
        return self.get_supported_tasks()

    @property
    def max_batch_size(self) -> Optional[int]:
        """Max batch size for inference. Override in subclass if model supports batching."""
        return getattr(self, "_max_batch_size", 1)
