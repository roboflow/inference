"""Import model implementations only when their registry entries are requested."""

import importlib
import warnings
from collections.abc import Iterator, MutableMapping
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Optional

from inference.core.warnings import InferenceModelsStackMissing


@dataclass
class _LazyModelClass:
    path: str
    optional: bool = False
    warning_message: Optional[str] = None
    warning_category: type[Warning] = Warning
    _resolved: Any = None
    _lock: Any = field(default_factory=RLock, init=False, repr=False, compare=False)

    def _resolve(self) -> Any:
        with self._lock:
            if self._resolved is None:
                module_path, class_name = self.path.split(":", 1)
                module = importlib.import_module(module_path)
                self._resolved = getattr(module, class_name)

        return self._resolved


@dataclass
class _AdapterModelClass:
    adapter: _LazyModelClass
    fallback: Any


class _LazyModelRegistry(MutableMapping):
    def __init__(self, entries=None):
        self._entries = dict(entries or {})
        self._lock = RLock()

    def __getitem__(self, key):
        entry = self._entries[key]
        if not isinstance(entry, (_LazyModelClass, _AdapterModelClass)):
            return entry

        model_class = self._resolve(entry, key=key)
        with self._lock:
            if self._entries.get(key) is entry:
                self._entries[key] = model_class

        return model_class

    def _resolve(self, entry, *, key):
        if isinstance(entry, _AdapterModelClass):
            try:
                model_class = entry.adapter._resolve()
            except Exception as error:
                task, variant = key
                warnings.warn(
                    f"`inference-models` stack is unavailable for model: {variant} "
                    f"and task: {task}, falling back to regular `inference` "
                    f"stack - error: {error}",
                    category=InferenceModelsStackMissing,
                    stacklevel=3,
                )
                if entry.fallback is None:
                    raise KeyError(key) from error

                model_class = self._resolve(entry.fallback, key=key)

            return model_class

        if not isinstance(entry, _LazyModelClass):
            return entry

        try:
            model_class = entry._resolve()
        except Exception as error:
            if not entry.optional:
                raise

            if entry.warning_message:
                warnings.warn(
                    entry.warning_message,
                    category=entry.warning_category,
                    stacklevel=3,
                )
            raise KeyError(key) from error

        return model_class

    def __setitem__(self, key, value):
        with self._lock:
            self._entries[key] = value

    def __delitem__(self, key):
        with self._lock:
            del self._entries[key]

    def __iter__(self) -> Iterator:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key) -> bool:
        return key in self._entries

    def clear(self) -> None:
        """Remove entries without importing their implementations."""
        with self._lock:
            self._entries.clear()

    def copy(self):
        """Copy registry entries without importing their implementations.

        Returns:
            _LazyModelRegistry: An independently mutable registry.
        """
        with self._lock:
            registry = _LazyModelRegistry(self._entries)

        return registry

    def set_adapter(self, key, adapter: _LazyModelClass) -> None:
        """Install an adapter while retaining a deferred legacy fallback.

        Args:
            key (tuple): Task and model variant to register.
            adapter (_LazyModelClass): Adapter implementation reference.
        """
        with self._lock:
            fallback = self._entries.get(key)
            self._entries[key] = _AdapterModelClass(adapter, fallback)
