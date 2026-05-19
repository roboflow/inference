from __future__ import annotations

import importlib
import json
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch

from profiling.memory.input_factory import (
    build_random_rgb_images,
    describe_shape_signature,
    merge_infer_kwargs,
)
from profiling.memory.sampler import NvmlProcessMemorySampler
from profiling.memory.schema import ShapeProfile, TensorRTMemoryProfileResult


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


def _delta_bytes(
    end_value: Optional[int],
    *,
    start_value: Optional[int],
) -> Optional[int]:
    if end_value is None or start_value is None:
        return None

    delta = max(0, end_value - start_value)

    return delta


def _read_model_package_metadata(
    model_name_or_path: str,
) -> tuple[Optional[int], Optional[Dict[str, Any]], Optional[int]]:
    package_dir = Path(model_name_or_path)
    if not package_dir.is_dir():
        return None, None, None

    engine_path = package_dir / "engine.plan"
    engine_size_bytes = (
        int(engine_path.stat().st_size) if engine_path.is_file() else None
    )

    optimization_profile: Optional[Dict[str, Any]] = None
    trt_config_path = package_dir / "trt_config.json"
    if trt_config_path.is_file():
        with trt_config_path.open(encoding="utf-8") as config_file:
            loaded_config = json.load(config_file)

        if isinstance(loaded_config, dict):
            optimization_profile = loaded_config

    max_workspace_setting: Optional[int] = None
    build_config_path = package_dir / "build_config.json"
    if build_config_path.is_file():
        with build_config_path.open(encoding="utf-8") as build_config_file:
            loaded_build_config = json.load(build_config_file)

        if isinstance(loaded_build_config, dict):
            workspace_gb = loaded_build_config.get("workspace_size_gb")
            if workspace_gb is not None:
                max_workspace_setting = int(workspace_gb) * (1024**3)

    return engine_size_bytes, optimization_profile, max_workspace_setting


def worker_run(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Execute one TensorRT runtime profiling scenario in a fresh process."""
    import tensorrt

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
    sampling_interval_seconds = float(
        payload.get("trt_nvml_sampling_interval_seconds")
        or payload.get("onnx_nvml_sampling_interval_seconds")
        or 0.01
    )
    num_contexts_profiled = int(payload.get("num_execution_contexts") or 1)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this TensorRT memory profiler harness.")

    device = torch.device(device_str)
    if device.type != "cuda":
        raise RuntimeError(f"Expected a CUDA device, got {device_str!r}.")

    torch.cuda.set_device(device)
    torch.cuda.empty_cache()

    device_index = device.index if device.index is not None else 0
    sampler = NvmlProcessMemorySampler(
        device_index,
        interval_seconds=sampling_interval_seconds,
    )
    baseline_process_gpu_bytes = sampler.snapshot()

    engine_size_bytes, optimization_profile, max_workspace_setting = (
        _read_model_package_metadata(model_name_or_path)
    )

    from_pretrained_kwargs.setdefault("device", device)

    model_cls = _resolve_class(
        module_name=module_name,
        class_name=class_name,
    )
    model = model_cls.from_pretrained(
        model_name_or_path,
        **from_pretrained_kwargs,
    )

    if hasattr(model, "eval"):
        model.eval()

    torch.cuda.synchronize(device)

    idle_after_deserialize_bytes = sampler.snapshot()

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

    run_inference_steps(warmup_iterations)
    torch.cuda.synchronize(device)

    sampler.start()
    run_inference_steps(measured_iterations)
    torch.cuda.synchronize(device)
    peak_request_bytes = sampler.stop()

    delta_peak_bytes = _delta_bytes(
        peak_request_bytes,
        start_value=idle_after_deserialize_bytes,
    )

    notes = [
        "TensorRT runtime profile (engine deserialize + context via from_pretrained)",
        f"NVML sampling source: {sampler.source}",
    ]
    if peak_request_bytes is None:
        notes.append("NVML sampling unavailable; install profiling-memory extra or GPU driver")
    if num_contexts_profiled != 1:
        notes.append(
            "num_execution_contexts > 1 requested; registry from_pretrained currently "
            "creates one execution context per model instance"
        )

    result = TensorRTMemoryProfileResult(
        model_id=model_id,
        gpu_name=torch.cuda.get_device_name(device),
        quantization=quantization,
        shape_profile=ShapeProfile(
            batch_size=batch_size,
            height=height,
            width=width,
        ),
        concurrency=num_contexts_profiled,
        warmup_iterations=warmup_iterations,
        measured_iterations=measured_iterations,
        method_name=method_name,
        shape_signature=shape_signature,
        module_name=module_name,
        class_name=class_name,
        architecture=architecture,
        task_type=task_type,
        notes=notes,
        baseline_process_gpu_bytes_nvml=baseline_process_gpu_bytes,
        idle_after_deserialize_bytes=idle_after_deserialize_bytes,
        peak_request_bytes=peak_request_bytes,
        delta_peak_bytes=delta_peak_bytes,
        engine_size_bytes=engine_size_bytes,
        optimization_profile=optimization_profile,
        max_workspace_setting=max_workspace_setting,
        num_contexts_profiled=num_contexts_profiled,
        tensorrt_version=tensorrt.__version__,
        nvml_sampling_interval_seconds=sampling_interval_seconds,
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
