from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Set

from inference_models.models.auto_loaders.entities import BackendType
from inference_models.models.auto_loaders.models_registry import (
    REGISTERED_MODELS,
    RegistryEntry,
)
from inference_models.utils.imports import LazyClass

# Registry backends profiled via profiling.memory.workers.torch (PyTorch CUDA metrics).
TORCH_MEMORY_PROFILING_BACKENDS = (
    BackendType.TORCH,
    BackendType.HF,
)


@dataclass(frozen=True)
class RegistryBackendRow:
    architecture: str
    task_type: Optional[str]
    backend: BackendType
    module_name: str
    class_name: str
    required_model_features: Optional[Set[str]]
    supported_model_features: Optional[Set[str]]


def iter_backend_registry_rows(
    backend: BackendType,
) -> Iterator[RegistryBackendRow]:
    """Yield every registry entry for the requested backend."""
    for key, entry in REGISTERED_MODELS.items():
        architecture, task_type, entry_backend = key
        if entry_backend != backend:
            continue

        lazy = entry.model_class if isinstance(entry, RegistryEntry) else entry
        assert isinstance(lazy, LazyClass)

        yield RegistryBackendRow(
            architecture=architecture,
            task_type=task_type,
            backend=entry_backend,
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


def list_backend_registry_rows(
    backend: BackendType,
) -> List[RegistryBackendRow]:
    rows = sorted(
        iter_backend_registry_rows(backend),
        key=lambda r: (r.architecture, r.task_type or ""),
    )

    return rows


def iter_torch_memory_profiling_registry_rows() -> Iterator[RegistryBackendRow]:
    """Yield registry rows for backends measured with the Torch memory harness."""
    for backend in TORCH_MEMORY_PROFILING_BACKENDS:
        yield from iter_backend_registry_rows(backend=backend)


def list_torch_registry_rows() -> List[RegistryBackendRow]:
    rows = sorted(
        iter_torch_memory_profiling_registry_rows(),
        key=lambda row: (row.architecture, row.task_type or "", row.backend.value),
    )

    return rows


def list_onnx_registry_rows() -> List[RegistryBackendRow]:
    rows = list_backend_registry_rows(backend=BackendType.ONNX)

    return rows


def list_trt_registry_rows() -> List[RegistryBackendRow]:
    rows = list_backend_registry_rows(backend=BackendType.TRT)

    return rows
