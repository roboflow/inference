from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Set

from inference_models.models.auto_loaders.entities import BackendType
from inference_models.models.auto_loaders.models_registry import (
    REGISTERED_MODELS,
    RegistryEntry,
)
from inference_models.utils.imports import LazyClass


@dataclass(frozen=True)
class TorchRegistryRow:
    architecture: str
    task_type: Optional[str]
    module_name: str
    class_name: str
    required_model_features: Optional[Set[str]]
    supported_model_features: Optional[Set[str]]


def iter_torch_registry_rows() -> Iterator[TorchRegistryRow]:
    """Yield every PyTorch (``BackendType.TORCH``) entry from ``REGISTERED_MODELS``."""
    for key, entry in REGISTERED_MODELS.items():
        architecture, task_type, backend = key
        if backend != BackendType.TORCH:
            continue
        lazy = entry.model_class if isinstance(entry, RegistryEntry) else entry
        assert isinstance(lazy, LazyClass)
        yield TorchRegistryRow(
            architecture=architecture,
            task_type=task_type,
            module_name=lazy._module_name,
            class_name=lazy._class_name,
            required_model_features=(
                entry.required_model_features
                if isinstance(entry, RegistryEntry)
                else None
            ),
            supported_model_features=(
                entry.supported_model_features
                if isinstance(entry, RegistryEntry)
                else None
            ),
        )


def list_torch_registry_rows() -> List[TorchRegistryRow]:
    return sorted(iter_torch_registry_rows(), key=lambda r: (r.architecture, r.task_type or ""))
