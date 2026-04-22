"""Task dispatch mixin for inference models.

Every model must declare which tasks it supports via the ``supported_tasks``
property. ModelManager uses this to dispatch ``invoke(model_id, task, **kwargs)``
to the correct method without hard-coded task→method mappings.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List


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
    and implement ``supported_tasks``. ABC enforces it — instantiation
    raises ``TypeError`` if the property is missing.
    """

    @property
    @abstractmethod
    def supported_tasks(self) -> Dict[str, TaskSpec]:
        """Return task dispatch table.

        Must return a dict with at least one entry where ``default=True``.

        Example::

            {
                "infer": TaskSpec(method="infer", default=True, params=["images"]),
            }
        """
        ...
