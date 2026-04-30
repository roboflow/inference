"""Centralized model registry — validation, serialization, dispatch.

Maps (model_base_class, task_name) → TaskEntry. Lookup follows MRO:
exact class first, then base classes up the hierarchy. One registration
for ObjectDetectionModel covers all YOLO/RFDETR/etc. subclasses.

Models in inference_models are unaware of this registry. Registration
happens once at import time in registry_defaults.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaskEntry:
    """Everything needed to validate, invoke, and serialize one task."""

    method: str
    """Name of the method to call on the model instance."""

    default: bool
    """True if this is the default task for the model class."""

    params: dict
    """Parameter definitions: {name: {type, required, default?}} for docs / validation."""

    validator: Callable[[dict], dict]
    """(kwargs) → validated kwargs. Raises ValueError on bad input."""

    serializer: Callable[[Any, Any], dict]
    """(raw_output, model_instance) → typed dict for JSON response."""

    response_type: str
    """e.g. 'roboflow-object-detection-compact-v1'"""


class ModelRegistry:
    """Maps (base_class, task_name) → TaskEntry.

    Lookup follows Python MRO: checks exact class, then each base class
    up the hierarchy. First match wins.
    """

    def __init__(self) -> None:
        self._entries: Dict[type, Dict[str, TaskEntry]] = {}

    def register(
        self,
        model_class: type,
        task_name: str,
        *,
        method: Optional[str] = None,
        default: bool = False,
        params: Optional[dict] = None,
        validator: Callable[[dict], dict],
        serializer: Callable[[Any, Any], dict],
        response_type: str,
    ) -> None:
        """Register a task entry for a model class.

        Args:
            model_class: Base class (e.g. ObjectDetectionModel). Models
                inheriting from this class get this entry via MRO lookup.
            task_name: Task name (e.g. "infer", "embed_text", "caption").
            method: Model method to call. Defaults to task_name.
            default: True if this is the default task for this class.
            params: Parameter names for docs/validation.
            validator: Validates kwargs before invocation.
            serializer: Converts raw model output to typed dict.
            response_type: Type string for response envelope.
        """
        entry = TaskEntry(
            method=method or task_name,
            default=default,
            params=params or {},
            validator=validator,
            serializer=serializer,
            response_type=response_type,
        )

        if model_class not in self._entries:
            self._entries[model_class] = {}
        self._entries[model_class][task_name] = entry

    def get_entry(self, model: Any, task_name: str) -> Optional[TaskEntry]:
        """Look up TaskEntry for model instance + task, following MRO.

        Returns None if no entry found (caller falls back to raw dispatch).
        """
        for cls in type(model).__mro__:
            class_entries = self._entries.get(cls)
            if class_entries and task_name in class_entries:
                return class_entries[task_name]
        return None

    def validate(self, model: Any, task_name: str, kwargs: dict) -> dict:
        """Validate kwargs for a task. Returns validated kwargs.

        If no registry entry exists, returns kwargs unchanged (no validation).
        Raises ValueError on validation failure.
        """
        entry = self.get_entry(model, task_name)
        if entry is None:
            return kwargs
        return entry.validator(kwargs)

    def serialize(self, model: Any, task_name: str, raw_output: Any) -> Optional[dict]:
        """Serialize model output to typed dict.

        Returns None if no registry entry (caller uses raw output).
        """
        entry = self.get_entry(model, task_name)
        if entry is None:
            return None
        return entry.serializer(raw_output, model)

    def response_type(self, model: Any, task_name: str) -> Optional[str]:
        """Get response type string for a task."""
        entry = self.get_entry(model, task_name)
        return entry.response_type if entry else None

    def get_entry_for_class(
        self, model_class: type, task_name: str
    ) -> Optional[TaskEntry]:
        """Look up TaskEntry by model class (not instance), following MRO."""
        for cls in model_class.__mro__:
            class_entries = self._entries.get(cls)
            if class_entries and task_name in class_entries:
                return class_entries[task_name]
        return None

    def get_entry_by_mro_names(
        self, mro_names: list[str], task_name: str
    ) -> Optional[TaskEntry]:
        """Look up TaskEntry by MRO class name strings (subprocess path).

        Used when model instance is in worker process and parent only has
        class name strings from the READY pipe message.
        """
        for name in mro_names:
            for cls, class_entries in self._entries.items():
                if cls.__name__ == name and task_name in class_entries:
                    return class_entries[task_name]
        return None

    def get_default_task_by_mro_names(self, mro_names: list[str]) -> Optional[str]:
        """Find default task name by MRO class name strings."""
        for name in mro_names:
            for cls, class_entries in self._entries.items():
                if cls.__name__ == name:
                    for task_name, entry in class_entries.items():
                        if entry.default:
                            return task_name
        return None

    def registered_classes(self) -> List[type]:
        """Return all classes with registered entries."""
        return list(self._entries.keys())

    def registered_tasks(self, model_class: type) -> List[str]:
        """Return all task names registered for a class (exact, not MRO)."""
        return list(self._entries.get(model_class, {}).keys())
