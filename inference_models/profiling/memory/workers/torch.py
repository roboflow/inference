from __future__ import annotations

import importlib
import tempfile
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch

from profiling.memory.input_factory import build_random_rgb_images
from profiling.memory.profiling_inputs import compute_runtime_axis_values
from profiling.memory.registry_profiles import resolve_registry_input_context
from profiling.memory.metadata import (
    TorchBackendMetadata,
    TorchMetrics,
    build_model_metadata_from_context,
    build_runtime_metadata,
    collect_environment_metadata,
    collect_torch_backend_metadata,
    finalize_profile_record,
)
from profiling.memory.package_input_profile import shape_spec_from_model
from profiling.memory.worker_config import MemoryProfilingWorkerPayload
from profiling.memory.worker_common import (
    build_input_metadata,
    build_profiling_run_metadata,
    ensure_profiling_image_shapes,
)


def _optional_nvml_free_bytes(device_index: int) -> Optional[int]:
    try:
        import pynvml  # type: ignore[import-untyped]

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_bytes = int(meminfo.free)

        return free_bytes
    except Exception:
        return None


def _resolve_class(
    *,
    module_name: str,
    class_name: str,
) -> type:
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)

    return model_class


def _invoke_method(
    model: Any,
    *,
    method_name: str,
    images: Any,
    infer_kwargs: Dict[str, Any],
) -> Any:
    fn: Callable[..., Any] = getattr(model, method_name)
    result = fn(images, **infer_kwargs)

    return result


def _prepare_from_pretrained_kwargs(
    from_pretrained_kwargs: Dict[str, Any],
    *,
    device: torch.device,
) -> Dict[str, Any]:
    kwargs = dict(from_pretrained_kwargs)
    kwargs.setdefault("device", device)

    return kwargs


def _export_torch_profiler_traces(prof: Any, *, trace_dir: Path) -> List[str]:
    trace_path = trace_dir / "torch_profiler_trace.json"

    try:
        prof.export_chrome_trace(str(trace_path))
    except Exception:
        return []

    return [str(trace_path)]


def worker_run(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Execute one PyTorch CUDA memory profiling scenario.

    Intended to run in a fresh process (see ``run_profile_subprocess``). Supports
    native ``torch``, ``torch-script``, and ``hugging-face`` registry classes.

    Args:
        payload: Worker configuration (model path, shape, iterations, device, etc.).

    Returns:
        JSON-serializable ``MemoryProfileRecord`` dict.

    Raises:
        RuntimeError: If CUDA is unavailable or the device is not CUDA.
    """
    config = MemoryProfilingWorkerPayload.from_payload(payload)
    batch_size, height, width = config.profiling_shape

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this PyTorch memory profiler harness.")

    device = torch.device(config.device_str)
    if device.type != "cuda":
        raise RuntimeError(f"Expected a CUDA device, got {config.device_str!r}.")

    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    device_index = device.index if device.index is not None else 0
    baseline_free = _optional_nvml_free_bytes(device_index=device_index)

    model_cls = _resolve_class(
        module_name=config.module_name,
        class_name=config.class_name,
    )
    from_pretrained_kwargs = _prepare_from_pretrained_kwargs(
        config.from_pretrained_kwargs,
        device=device,
    )
    model = model_cls.from_pretrained(
        str(config.package_path),
        **from_pretrained_kwargs,
    )

    if hasattr(model, "eval"):
        model.eval()

    torch.cuda.synchronize(device)

    idle_after_load_allocated = int(torch.cuda.memory_allocated(device))
    idle_after_load_reserved = int(torch.cuda.memory_reserved(device))

    batch_size, height, width = ensure_profiling_image_shapes(
        model,
        batch_size=batch_size,
        height=height,
        width=width,
    )
    shape_spec = shape_spec_from_model(model)

    images = build_random_rgb_images(
        batch_size,
        height=height,
        width=width,
    )
    infer_kwargs = dict(config.infer_kwargs)
    registry_context = resolve_registry_input_context(
        module_name=config.module_name,
        class_name=config.class_name,
        architecture=config.architecture,
        task_type=config.task_type,
        backend=config.backend,
    )
    task_profile_spec = registry_context.get("task_profile_spec")
    runtime_axis_values = compute_runtime_axis_values(
        model,
        infer_kwargs,
        task_profile_spec=(
            task_profile_spec if isinstance(task_profile_spec, dict) else None
        ),
    )

    def run_inference_steps(num_steps: int) -> None:
        for _ in range(num_steps):
            _invoke_method(
                model,
                method_name=config.method_name,
                images=images,
                infer_kwargs=infer_kwargs,
            )

    torch.cuda.reset_peak_memory_stats(device)
    run_inference_steps(config.warmup_iterations)

    torch.cuda.reset_peak_memory_stats(device)
    profiler_extra: Dict[str, Any] = {}
    trace_files: List[str] = []
    if config.torch_profiler_memory:
        trace_dir = Path(tempfile.mkdtemp(prefix="torch-profiler-"))
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        with torch.profiler.profile(
            activities=activities,
            profile_memory=True,
            record_shapes=False,
        ) as prof:
            run_inference_steps(config.measured_iterations)

        trace_files = _export_torch_profiler_traces(
            prof,
            trace_dir=trace_dir,
        )
        profiler_extra["trace_dir"] = str(trace_dir)
        profiler_extra["profiler_key_averages_events"] = len(prof.key_averages())
    else:
        run_inference_steps(config.measured_iterations)

    torch.cuda.synchronize(device)

    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    end_reserved = int(torch.cuda.memory_reserved(device))

    peak_incremental_allocated = max(0, peak_allocated - idle_after_load_allocated)
    peak_incremental_reserved = max(0, peak_reserved - idle_after_load_reserved)

    notes = [
        "idle snapshot uses torch.cuda.memory_{allocated,reserved} after load",
        "peak counters reset immediately before measured iterations",
        "incremental peaks subtract idle-after-load in the same allocator units",
    ]
    if baseline_free is None:
        notes.append(
            "NVML baseline unavailable (install profiling-memory extra or GPU driver)"
        )
    if config.torch_profiler_memory:
        notes.append(
            "Torch profiler Chrome trace written when --torch-profiler-memory is set"
        )

    metrics = TorchMetrics(
        idle_after_load_allocated_bytes=idle_after_load_allocated,
        idle_after_load_reserved_bytes=idle_after_load_reserved,
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
        end_reserved_bytes=end_reserved,
        peak_incremental_allocated_bytes=peak_incremental_allocated,
        peak_incremental_reserved_bytes=peak_incremental_reserved,
        baseline_gpu_free_bytes_nvml=baseline_free,
    )
    model_metadata = build_model_metadata_from_context(config.metadata_context)
    runtime_metadata = build_runtime_metadata(
        module_name=config.module_name,
        class_name=config.class_name,
        method_name=config.method_name,
    )
    environment_metadata = collect_environment_metadata(device=device)
    input_metadata = build_input_metadata(
        module_name=config.module_name,
        class_name=config.class_name,
        architecture=config.architecture,
        task_type=config.task_type,
        backend=config.backend,
        batch_size=batch_size,
        height=height,
        width=width,
        infer_kwargs=infer_kwargs,
        shape_spec=shape_spec,
        runtime_axis_values=runtime_axis_values,
    )
    profiling_run = build_profiling_run_metadata(
        config,
        trace_files=trace_files,
    )
    record = finalize_profile_record(
        profile_tier=config.resolved_profile_tier,
        metrics=metrics,
        model_metadata=model_metadata,
        runtime_metadata=runtime_metadata,
        backend_metadata=collect_torch_backend_metadata(),
        input_metadata=input_metadata,
        environment_metadata=environment_metadata,
        profiling_run=profiling_run,
        notes=notes,
    )
    result_dict = record.as_json_dict()

    return result_dict


def worker_main(conn_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Multiprocessing entry point; wraps ``worker_run`` for parent reporting.

    Args:
        conn_payload: Same dict as ``worker_run``.

    Returns:
        Dict with ``ok`` bool and either ``result`` or ``error`` traceback text.
    """
    try:
        result = worker_run(conn_payload)
        response = {"ok": True, "result": result}
    except Exception:
        error = traceback.format_exc()
        response = {"ok": False, "error": error}

    return response
