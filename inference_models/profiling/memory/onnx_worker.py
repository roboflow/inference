from __future__ import annotations

import importlib
import os
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch

from inference_models.models.auto_loaders.entities import BackendType
from profiling.memory.input_factory import (
    build_random_rgb_images,
    describe_shape_signature,
    merge_infer_kwargs,
)
from profiling.memory.schema import OnnxMemoryProfileResult, ShapeProfile


class NvmlProcessMemorySampler:
    def __init__(
        self,
        device_index: int,
        *,
        interval_seconds: float,
    ) -> None:
        self._device_index = device_index
        self._interval_seconds = interval_seconds
        self._pid = os.getpid()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._peak_bytes: Optional[int] = None
        self._source = "unavailable"

        try:
            import pynvml  # type: ignore[import-untyped]

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self._available = True
        except Exception:
            self._pynvml = None
            self._handle = None
            self._available = False

    @property
    def source(self) -> str:
        return self._source

    def snapshot(self) -> Optional[int]:
        if not self._available:
            return None

        process_bytes = self._get_process_memory_bytes()
        if process_bytes is not None:
            self._source = "process"

            return process_bytes

        device_bytes = self._get_device_used_bytes()
        if device_bytes is not None:
            self._source = "device"

            return device_bytes

        self._source = "unavailable"

        return None

    def start(self) -> None:
        self._peak_bytes = self.snapshot()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._sample_loop,
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> Optional[int]:
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join()

        final_value = self.snapshot()
        if final_value is not None:
            self._record(final_value)

        return self._peak_bytes

    def _sample_loop(self) -> None:
        while not self._stop_event.is_set():
            value = self.snapshot()
            if value is not None:
                self._record(value)

            time.sleep(self._interval_seconds)

    def _record(self, value: int) -> None:
        if self._peak_bytes is None or value > self._peak_bytes:
            self._peak_bytes = value

    def _get_process_memory_bytes(self) -> Optional[int]:
        process_entries = self._get_nvml_process_entries()
        if process_entries is None:
            return None

        for process in process_entries:
            if getattr(process, "pid", None) != self._pid:
                continue

            used_gpu_memory = getattr(process, "usedGpuMemory", None)
            if used_gpu_memory is None:
                continue

            process_bytes = int(used_gpu_memory)

            return process_bytes

        return 0

    def _get_nvml_process_entries(self) -> Optional[List[Any]]:
        entries: List[Any] = []

        for getter_name in (
            "nvmlDeviceGetComputeRunningProcesses_v3",
            "nvmlDeviceGetComputeRunningProcesses",
            "nvmlDeviceGetGraphicsRunningProcesses_v3",
            "nvmlDeviceGetGraphicsRunningProcesses",
        ):
            getter = getattr(self._pynvml, getter_name, None)
            if getter is None:
                continue

            try:
                entries.extend(getter(self._handle))
            except Exception:
                continue

        if not entries:
            return None

        return entries

    def _get_device_used_bytes(self) -> Optional[int]:
        try:
            meminfo = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            used_bytes = int(meminfo.used)
        except Exception:
            return None

        return used_bytes


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


def _patch_onnxruntime_sessions(
    onnxruntime: Any,
    *,
    trace_dir: Path,
    sessions: List[Any],
) -> Any:
    original_inference_session = onnxruntime.InferenceSession

    def create_profiling_session(*args: Any, **kwargs: Any) -> Any:
        session_options = kwargs.get("sess_options")
        if session_options is None:
            session_options = onnxruntime.SessionOptions()
            kwargs["sess_options"] = session_options

        session_options.enable_profiling = True
        session_options.profile_file_prefix = str(trace_dir / "onnxruntime_profile")

        session = original_inference_session(*args, **kwargs)
        sessions.append(session)

        return session

    onnxruntime.InferenceSession = create_profiling_session

    return original_inference_session


def _finish_onnx_profiling(sessions: List[Any]) -> List[str]:
    trace_files = []

    for session in sessions:
        try:
            trace_file = session.end_profiling()
        except Exception:
            continue

        if trace_file:
            trace_files.append(trace_file)

    return trace_files


def _get_execution_providers(sessions: List[Any]) -> List[str]:
    execution_providers = []

    for session in sessions:
        for provider in session.get_providers():
            if provider in execution_providers:
                continue

            execution_providers.append(provider)

    return execution_providers


def _delta_bytes(
    end_value: Optional[int],
    *,
    start_value: Optional[int],
) -> Optional[int]:
    if end_value is None or start_value is None:
        return None

    delta = max(0, end_value - start_value)

    return delta


def worker_run(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Execute one ONNX Runtime profiling scenario in a fresh process."""
    import onnxruntime

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
        payload.get("onnx_nvml_sampling_interval_seconds") or 0.01
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this ONNX memory profiler harness.")

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

    trace_dir = Path(tempfile.mkdtemp(prefix="onnxruntime-profile-"))
    sessions: List[Any] = []
    original_inference_session = _patch_onnxruntime_sessions(
        onnxruntime,
        trace_dir=trace_dir,
        sessions=sessions,
    )

    from_pretrained_kwargs.setdefault(
        "onnx_execution_providers",
        ["CUDAExecutionProvider"],
    )
    from_pretrained_kwargs.setdefault("device", device)

    try:
        model_cls = _resolve_class(
            module_name=module_name,
            class_name=class_name,
        )
        model = model_cls.from_pretrained(
            model_name_or_path,
            **from_pretrained_kwargs,
        )
    finally:
        onnxruntime.InferenceSession = original_inference_session

    torch.cuda.synchronize(device)

    idle_after_session_create_bytes = sampler.snapshot()

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
    peak_process_gpu_bytes = sampler.stop()

    trace_files = _finish_onnx_profiling(sessions=sessions)
    execution_providers = _get_execution_providers(sessions=sessions)
    delta_peak_bytes = _delta_bytes(
        peak_process_gpu_bytes,
        start_value=idle_after_session_create_bytes,
    )

    notes = [
        "ONNX Runtime profiling enabled via SessionOptions injected in worker",
        f"NVML sampling source: {sampler.source}",
    ]
    if peak_process_gpu_bytes is None:
        notes.append("NVML sampling unavailable; install profiling-memory extra or GPU driver")

    result = OnnxMemoryProfileResult(
        model_id=model_id,
        gpu_name=torch.cuda.get_device_name(device),
        quantization=quantization,
        shape_profile=ShapeProfile(
            batch_size=batch_size,
            height=height,
            width=width,
        ),
        concurrency=1,
        warmup_iterations=warmup_iterations,
        measured_iterations=measured_iterations,
        method_name=method_name,
        shape_signature=shape_signature,
        module_name=module_name,
        class_name=class_name,
        architecture=architecture,
        task_type=task_type,
        notes=notes,
        extra={
            "trace_dir": str(trace_dir),
        },
        baseline_process_gpu_bytes_nvml=baseline_process_gpu_bytes,
        idle_after_session_create_bytes=idle_after_session_create_bytes,
        peak_process_gpu_bytes=peak_process_gpu_bytes,
        delta_peak_bytes=delta_peak_bytes,
        execution_providers=execution_providers,
        onnxruntime_version=onnxruntime.__version__,
        trace_files=trace_files,
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
