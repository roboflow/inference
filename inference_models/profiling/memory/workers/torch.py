from __future__ import annotations

import importlib
import tempfile
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch

from profiling.memory.input_factory import (
    build_random_rgb_images,
    describe_shape_signature,
    merge_infer_kwargs,
)
from profiling.memory.schema import TorchMemoryProfileResult, ShapeProfile


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


def _resolve_class(module_name: str, class_name: str) -> type:
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


def _resolve_roboflow_package_profiling_shape(
    model: Any,
    *,
    batch_size: int,
    height: int,
    width: int,
) -> tuple[int, int, int, List[str]]:
    """Align synthetic inputs with Roboflow package metadata (TorchScript, ONNX export paths).

    TorchScript weights are exported with a fixed batch size and spatial size in
    ``inference_config.json``. Profiling at other CLI dimensions still runs (via padding
    or resize in ``pre_process``) but understates or mislabels the real serving regime.
    """
    notes: List[str] = []
    inference_config = getattr(model, "_inference_config", None)
    if inference_config is None:
        return batch_size, height, width, notes

    static_batch_size = inference_config.forward_pass.static_batch_size
    if static_batch_size is not None and batch_size != static_batch_size:
        notes.append(
            "Package defines static_batch_size="
            f"{static_batch_size}; profiling at that batch size "
            f"(CLI --batch-size was {batch_size})."
        )
        batch_size = static_batch_size

    training_size = inference_config.network_input.training_input_size
    package_height = training_size.height
    package_width = training_size.width
    if (height, width) != (package_height, package_width):
        notes.append(
            f"Package training_input_size is {package_height}x{package_width}; "
            f"profiling synthetic images at that resolution "
            f"(CLI --height/--width were {height}x{width})."
        )
        height = package_height
        width = package_width

    return batch_size, height, width, notes


def _export_torch_profiler_traces(prof: Any, *, trace_dir: Path) -> List[str]:
    trace_path = trace_dir / "torch_profiler_trace.json"

    try:
        prof.export_chrome_trace(str(trace_path))
    except Exception:
        return []

    return [str(trace_path)]


def worker_run(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Execute one profiling scenario. Intended to run in a fresh process (see harness)."""
    module_name = payload["module_name"]
    class_name = payload["class_name"]
    model_name_or_path = payload["model_name_or_path"]
    from_pretrained_kwargs: Dict[str, Any] = payload.get("from_pretrained_kwargs") or {}
    device_str: str = payload["device_str"]
    batch_size = int(payload["batch_size"])
    height = int(payload["height"])
    width = int(payload["width"])
    infer_kwargs_user: Dict[str, Any] = payload.get("infer_kwargs") or {}
    task_type: Optional[str] = payload.get("task_type")
    method_name: str = payload["method_name"]
    warmup_iterations = int(payload["warmup_iterations"])
    measured_iterations = int(payload["measured_iterations"])
    model_id = payload.get("model_id") or model_name_or_path
    architecture = payload.get("architecture")
    quantization = payload.get("quantization")
    torch_profiler_memory = bool(payload.get("torch_profiler_memory"))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this PyTorch memory profiler harness.")

    device = torch.device(device_str)
    if device.type != "cuda":
        raise RuntimeError(f"Expected a CUDA device, got {device_str!r}.")

    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    device_index = device.index if device.index is not None else 0
    baseline_free = _optional_nvml_free_bytes(device_index=device_index)

    model_cls = _resolve_class(
        module_name=module_name,
        class_name=class_name,
    )
    from_pretrained_kwargs = _prepare_from_pretrained_kwargs(
        from_pretrained_kwargs,
        device=device,
    )
    model = model_cls.from_pretrained(
        model_name_or_path,
        **from_pretrained_kwargs,
    )

    if hasattr(model, "eval"):
        model.eval()

    torch.cuda.synchronize(device)

    idle_after_load_allocated = int(torch.cuda.memory_allocated(device))
    idle_after_load_reserved = int(torch.cuda.memory_reserved(device))

    batch_size, height, width, package_shape_notes = (
        _resolve_roboflow_package_profiling_shape(
            model,
            batch_size=batch_size,
            height=height,
            width=width,
        )
    )

    images = build_random_rgb_images(
        batch_size,
        height=height,
        width=width,
    )
    infer_kwargs = merge_infer_kwargs(
        task_type,
        user=infer_kwargs_user,
    )
    shape_signature = describe_shape_signature(
        batch_size,
        height=height,
        width=width,
        infer_kwargs=infer_kwargs,
    )

    def run_inference_steps(num_steps: int) -> None:
        for _ in range(num_steps):
            _invoke_method(
                model,
                method_name=method_name,
                images=images,
                infer_kwargs=infer_kwargs,
            )

    torch.cuda.reset_peak_memory_stats(device)
    run_inference_steps(warmup_iterations)

    torch.cuda.reset_peak_memory_stats(device)
    profiler_extra: Dict[str, Any] = {}
    trace_files: List[str] = []
    if torch_profiler_memory:
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
            run_inference_steps(measured_iterations)

        trace_files = _export_torch_profiler_traces(
            prof,
            trace_dir=trace_dir,
        )
        profiler_extra["trace_dir"] = str(trace_dir)
        profiler_extra["profiler_key_averages_events"] = len(prof.key_averages())
    else:
        run_inference_steps(measured_iterations)

    torch.cuda.synchronize(device)

    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    end_reserved = int(torch.cuda.memory_reserved(device))

    peak_incremental_allocated = max(0, peak_allocated - idle_after_load_allocated)
    peak_incremental_reserved = max(0, peak_reserved - idle_after_load_reserved)

    gpu_name: Optional[str] = None
    try:
        gpu_name = torch.cuda.get_device_name(device)
    except Exception:
        pass

    notes = [
        "idle snapshot uses torch.cuda.memory_{allocated,reserved} after load",
        "peak counters reset immediately before measured iterations",
        "incremental peaks subtract idle-after-load in the same allocator units",
        *package_shape_notes,
    ]
    if baseline_free is None:
        notes.append(
            "NVML baseline unavailable (install profiling-memory extra or GPU driver)"
        )
    if torch_profiler_memory:
        notes.append(
            "Torch profiler Chrome trace written when --torch-profiler-memory is set"
        )

    result = TorchMemoryProfileResult(
        model_id=model_id,
        gpu_name=gpu_name,
        quantization=quantization,
        shape_profile=ShapeProfile(
            batch_size=batch_size,
            height=height,
            width=width,
        ),
        concurrency=1,
        idle_after_load_allocated_bytes=idle_after_load_allocated,
        idle_after_load_reserved_bytes=idle_after_load_reserved,
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
        end_reserved_bytes=end_reserved,
        peak_incremental_allocated_bytes=peak_incremental_allocated,
        peak_incremental_reserved_bytes=peak_incremental_reserved,
        baseline_gpu_free_bytes_nvml=baseline_free,
        warmup_iterations=warmup_iterations,
        measured_iterations=measured_iterations,
        method_name=method_name,
        shape_signature=shape_signature,
        module_name=module_name,
        class_name=class_name,
        architecture=architecture,
        task_type=task_type,
        notes=notes,
        torch_profiler_memory_enabled=torch_profiler_memory,
        trace_files=trace_files,
        extra=profiler_extra,
    )
    result_dict = result.as_json_dict()

    return result_dict


def worker_main(conn_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Top-level entry used by multiprocessing; wraps errors for parent reporting."""
    try:
        result = worker_run(conn_payload)
        response = {"ok": True, "result": result}
    except Exception:
        error = traceback.format_exc()
        response = {"ok": False, "error": error}

    return response
