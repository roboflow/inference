from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Set

from inference_models.models.auto_loaders.entities import BackendType
from inference_models.models.auto_loaders.models_registry import (
    REGISTERED_MODELS,
    RegistryEntry,
)
from inference_models.utils.imports import LazyClass

# Registry backends profiled via profiling.memory.workers.torch (PyTorch CUDA metrics).
TORCH_MEMORY_PROFILING_BACKENDS = (
    BackendType.TORCH,
    BackendType.TORCH_SCRIPT,
    BackendType.HF,
)


@dataclass(frozen=True)
class RegistryBackendRow:
    """One row from ``REGISTERED_MODELS`` for CLI listing and profiling."""

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
    """Yield every registry entry for the requested backend.

    Args:
        backend: Registry backend filter (for example ``BackendType.ONNX``).

    Yields:
        One ``RegistryBackendRow`` per matching registry key.
    """
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
    """Return sorted registry rows for a single backend.

    Args:
        backend: Registry backend filter.

    Returns:
        Rows sorted by architecture and task type.
    """
    rows = sorted(
        iter_backend_registry_rows(backend),
        key=lambda r: (r.architecture, r.task_type or ""),
    )

    return rows


def iter_torch_memory_profiling_registry_rows() -> Iterator[RegistryBackendRow]:
    """Yield registry rows for backends measured with the Torch memory harness.

    Includes ``torch``, ``torch-script``, and ``hugging-face`` entries.

    Yields:
        One ``RegistryBackendRow`` per matching registry key.
    """
    for backend in TORCH_MEMORY_PROFILING_BACKENDS:
        yield from iter_backend_registry_rows(backend=backend)


def list_torch_registry_rows() -> List[RegistryBackendRow]:
    """Return sorted rows for the PyTorch CUDA memory harness.

    Returns:
        Rows for ``TORCH_MEMORY_PROFILING_BACKENDS``, sorted by architecture,
        task type, and backend value.
    """
    rows = sorted(
        iter_torch_memory_profiling_registry_rows(),
        key=lambda row: (row.architecture, row.task_type or "", row.backend.value),
    )

    return rows


def list_onnx_registry_rows() -> List[RegistryBackendRow]:
    """Return sorted ONNX backend registry rows.

    Returns:
        Rows for ``BackendType.ONNX``.
    """
    rows = list_backend_registry_rows(backend=BackendType.ONNX)

    return rows


def list_trt_registry_rows() -> List[RegistryBackendRow]:
    """Return sorted TensorRT backend registry rows.

    Returns:
        Rows for ``BackendType.TRT``.
    """
    rows = list_backend_registry_rows(backend=BackendType.TRT)

    return rows


HARNESS_BACKEND_VALUES = (
    BackendType.TORCH.value,
    BackendType.ONNX.value,
    BackendType.TRT.value,
)


def list_registry_rows_for_harness(harness_backend: str) -> List[RegistryBackendRow]:
    """Return registry rows eligible for a profiling harness backend.

    ``torch`` includes ``torch``, ``torch-script``, and ``hugging-face`` rows.
    """
    if harness_backend == BackendType.TORCH.value:
        return list_torch_registry_rows()
    if harness_backend == BackendType.ONNX.value:
        return list_onnx_registry_rows()
    if harness_backend == BackendType.TRT.value:
        return list_trt_registry_rows()

    raise ValueError(
        f"Unsupported harness backend {harness_backend!r}; "
        f"expected one of: {', '.join(HARNESS_BACKEND_VALUES)}"
    )


def registry_features_from_row(row: RegistryBackendRow) -> Dict[str, Any]:
    """Build ``registry_features`` metadata from a resolved registry row."""
    registry_features: Dict[str, Any] = {}
    if row.required_model_features:
        registry_features["required_model_features"] = sorted(
            row.required_model_features
        )
    if row.supported_model_features:
        registry_features["supported_model_features"] = sorted(
            row.supported_model_features
        )

    return registry_features


def resolve_registry_row(
    *,
    architecture: str,
    task_type: str,
    harness_backend: str,
) -> RegistryBackendRow:
    """Resolve ``module_name`` / ``class_name`` from ``REGISTERED_MODELS``.

    Args:
        architecture: Registry architecture key (e.g. ``yolov8``).
        task_type: Registry task type (e.g. ``object-detection``).
        harness_backend: CLI profiling backend (``torch``, ``onnx``, or ``trt``).

    Raises:
        ValueError: When no row matches or multiple rows match.
    """
    rows = list_registry_rows_for_harness(harness_backend)
    matches = [
        row
        for row in rows
        if row.architecture == architecture and row.task_type == task_type
    ]

    if len(matches) == 1:
        return matches[0]

    if not matches:
        raise ValueError(
            f"No REGISTERED_MODELS entry for architecture={architecture!r}, "
            f"task_type={task_type!r}, harness backend={harness_backend!r}. "
            "Use --list-torch-models, --list-onnx-models, or --list-trt-models."
        )

    candidates = ", ".join(
        f"{match.module_name}.{match.class_name} "
        f"(registry backend {match.backend.value})"
        for match in matches
    )
    raise ValueError(
        f"Multiple REGISTERED_MODELS entries for architecture={architecture!r}, "
        f"task_type={task_type!r}, harness backend={harness_backend!r}: "
        f"{candidates}"
    )
