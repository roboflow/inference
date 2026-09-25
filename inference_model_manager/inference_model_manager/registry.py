"""Centralized model registry — validation, serialization, dispatch.

Maps (model_base_class, action_name) → ActionEntry. Lookup follows MRO:
exact class first, then base classes up the hierarchy. One registration
for ObjectDetectionModel covers all YOLO/RFDETR/etc. subclasses.

Models in inference_models are unaware of this registry. Registration
happens once at import time in registry_defaults.py.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ActionEntry:
    """Everything needed to validate, invoke, and serialize one action."""

    method: str
    """Name of the method to call on the model instance."""

    default: bool
    """True if this is the default action for the model class."""

    params: dict
    """Parameter definitions: {name: {type, required, default?}} for docs / validation."""

    validator: Callable[[dict], dict]
    """(kwargs) → validated kwargs. Raises ValueError on bad input."""

    serializer: Callable[[Any, Any], dict]
    """(raw_output, model_instance) → typed dict for JSON response."""

    response_type: str
    """e.g. 'roboflow-object-detection-compact-v1'"""

    param_aliases: dict
    """Maps external param name -> method kwarg name. Applied at invocation."""


class ModelRegistry:
    """Maps (base_class, action_name) → ActionEntry.

    Lookup follows Python MRO: checks exact class, then each base class
    up the hierarchy. First match wins.
    """

    def __init__(self) -> None:
        self._entries: Dict[type, Dict[str, ActionEntry]] = {}
        # Guards _entries against concurrent first-registration (lazy_register*
        # runs on load paths) racing the per-request MRO-name scans — iterating
        # an unlocked dict while register inserts raises RuntimeError mid-inference.
        self._lock = threading.RLock()

    def register(
        self,
        model_class: type,
        action_name: str,
        *,
        method: Optional[str] = None,
        default: bool = False,
        params: Optional[dict] = None,
        validator: Callable[[dict], dict],
        serializer: Callable[[Any, Any], dict],
        response_type: str,
        param_aliases: Optional[dict] = None,
    ) -> None:
        """Register an action entry for a model class.

        Args:
            model_class: Base class (e.g. ObjectDetectionModel). Models
                inheriting from this class get this entry via MRO lookup.
            action_name: Action name (e.g. "infer", "embed_text", "caption").
            method: Model method to call. Defaults to action_name.
            default: True if this is the default action for this class.
            params: Parameter names for docs/validation.
            validator: Validates kwargs before invocation.
            serializer: Converts raw model output to typed dict.
            response_type: Type string for response envelope.
            param_aliases: Maps external param names to method kwargs.
        """
        entry = ActionEntry(
            method=method or action_name,
            default=default,
            params=params or {},
            validator=validator,
            serializer=serializer,
            response_type=response_type,
            param_aliases=param_aliases or {},
        )

        with self._lock:
            self._entries.setdefault(model_class, {})[action_name] = entry

    def get_entry(self, model: Any, action_name: str) -> Optional[ActionEntry]:
        """Look up ActionEntry for model instance + action, following MRO.

        Returns None if no entry found (caller falls back to raw dispatch).
        """
        for cls in type(model).__mro__:
            class_entries = self._entries.get(cls)
            if class_entries and action_name in class_entries:
                return class_entries[action_name]
        return None

    def validate(self, model: Any, action_name: str, kwargs: dict) -> dict:
        """Validate kwargs for an action. Returns validated kwargs.

        If no registry entry exists, returns kwargs unchanged (no validation).
        Raises ValueError on validation failure.
        """
        entry = self.get_entry(model, action_name)
        if entry is None:
            return kwargs
        return entry.validator(kwargs)

    def serialize(
        self, model: Any, action_name: str, raw_output: Any
    ) -> Optional[dict]:
        """Serialize model output to typed dict.

        Returns None if no registry entry (caller uses raw output).
        """
        entry = self.get_entry(model, action_name)
        if entry is None:
            return None
        return entry.serializer(raw_output, model)

    def response_type(self, model: Any, action_name: str) -> Optional[str]:
        """Get response type string for an action."""
        entry = self.get_entry(model, action_name)
        return entry.response_type if entry else None

    def get_entry_for_class(
        self, model_class: type, action_name: str
    ) -> Optional[ActionEntry]:
        """Look up ActionEntry by model class (not instance), following MRO."""
        for cls in model_class.__mro__:
            class_entries = self._entries.get(cls)
            if class_entries and action_name in class_entries:
                return class_entries[action_name]
        return None

    def get_entry_by_mro_names(
        self, mro_names: list[str], action_name: str
    ) -> Optional[ActionEntry]:
        """Look up ActionEntry by MRO class name strings.

        Used when the model instance does not live in-process and the
        backend reports class name strings instead.
        """
        with self._lock:
            for name in mro_names:
                for cls, class_entries in self._entries.items():
                    if cls.__name__ == name and action_name in class_entries:
                        return class_entries[action_name]
        return None

    def get_default_action_by_mro_names(self, mro_names: list[str]) -> Optional[str]:
        """Find default action name by MRO class name strings."""
        with self._lock:
            for name in mro_names:
                for cls, class_entries in self._entries.items():
                    if cls.__name__ == name:
                        for action_name, entry in class_entries.items():
                            if entry.default:
                                return action_name
        return None

    def registered_classes(self) -> List[type]:
        """Return all classes with registered entries."""
        with self._lock:
            return list(self._entries.keys())

    def registered_actions(self, model_class: type) -> List[str]:
        """Return all action names registered for a class (exact, not MRO)."""
        with self._lock:
            return list(self._entries.get(model_class, {}).keys())
