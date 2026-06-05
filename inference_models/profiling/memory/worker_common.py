"""Shared helpers for memory profiling workers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from profiling.memory.metadata import (
    InputMetadata,
    ProfilingRunMetadata,
    build_input_metadata_with_registry,
)
from profiling.memory.worker_config import MemoryProfilingWorkerPayload
from profiling.memory.package_input_profile import (
    PackageProfilingShapeSpec,
    shape_spec_from_model,
    validate_profiling_image_shapes,
)


def ensure_profiling_image_shapes(
    model: Any,
    *,
    batch_size: int,
    height: int,
    width: int,
) -> Tuple[int, int, int]:
    """Validate that profiling shapes match the loaded package; do not silently override."""
    spec = shape_spec_from_model(model)

    return validate_profiling_image_shapes(
        batch_size=batch_size,
        height=height,
        width=width,
        spec=spec,
    )


def build_input_metadata(
    *,
    module_name: Optional[str],
    class_name: Optional[str],
    architecture: Optional[str],
    task_type: Optional[str],
    backend: Optional[str],
    batch_size: int,
    height: int,
    width: int,
    infer_kwargs: Dict[str, Any],
    shape_spec: Optional[PackageProfilingShapeSpec] = None,
) -> InputMetadata:
    input_metadata = build_input_metadata_with_registry(
        module_name=module_name,
        class_name=class_name,
        architecture=architecture,
        task_type=task_type,
        backend=backend,
        batch_size=batch_size,
        height=height,
        width=width,
        infer_kwargs=infer_kwargs,
        shape_spec=shape_spec,
    )

    return input_metadata


def build_profiling_run_metadata(
    config: MemoryProfilingWorkerPayload,
    *,
    trace_files: List[str],
    nvml_sampling_interval_seconds: Optional[float] = None,
) -> ProfilingRunMetadata:
    profiling_run = ProfilingRunMetadata(
        warmup_iterations=config.warmup_iterations,
        measured_iterations=config.measured_iterations,
        concurrency=config.num_execution_contexts,
        in_process=config.in_process,
        torch_profiler_memory_enabled=config.torch_profiler_memory,
        nvml_sampling_interval_seconds=nvml_sampling_interval_seconds,
        trace_files=trace_files,
    )

    return profiling_run
