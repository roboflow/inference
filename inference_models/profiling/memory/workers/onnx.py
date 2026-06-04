from __future__ import annotations

import importlib
import tempfile
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

import torch

from profiling.memory.input_factory import (
    build_random_rgb_images,
    merge_infer_kwargs,
)
from profiling.memory.metadata import (
    OnnxMetrics,
    ProfileTier,
    build_model_metadata_from_context,
    build_runtime_metadata,
    collect_environment_metadata,
    collect_onnx_backend_metadata,
    finalize_profile_record,
    resolve_onnx_opset_from_metadata_context,
)
from profiling.memory.sampler import NvmlProcessMemorySampler
from profiling.memory.package_input_profile import shape_spec_from_model
from profiling.memory.worker_common import (
    build_input_metadata,
    build_profiling_run_metadata,
    ensure_profiling_image_shapes,
    parse_requested_shape,
)


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


@contextmanager
def _onnxruntime_profiling_sessions(
    onnxruntime: Any,
    *,
    trace_dir: Path,
    sessions: List[Any],
) -> Iterator[None]:
    original_inference_session_init_fn = onnxruntime.InferenceSession

    def create_profiling_session(*args: Any, **kwargs: Any) -> Any:
        session_options = kwargs.get("sess_options")
        if session_options is None:
            session_options = onnxruntime.SessionOptions()
            kwargs["sess_options"] = session_options

        session_options.enable_profiling = True
        session_options.profile_file_prefix = str(trace_dir / "onnxruntime_profile")

        session = original_inference_session_init_fn(*args, **kwargs)
        sessions.append(session)

        return session

    onnxruntime.InferenceSession = create_profiling_session

    try:
        yield
    finally:
        onnxruntime.InferenceSession = original_inference_session_init_fn


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
    """Execute one ONNX Runtime memory profiling scenario in a fresh process.

    Args:
        payload: Worker configuration (model path, shape, iterations, device, etc.).

    Returns:
        JSON-serializable ``MemoryProfileRecord`` dict.

    Raises:
        RuntimeError: If CUDA is unavailable or the device is not CUDA.
    """
    import onnxruntime

    module_name = payload["module_name"]
    class_name = payload["class_name"]
    model_name_or_path = payload["model_name_or_path"]
    from_pretrained_kwargs: Dict[str, Any] = payload.get("from_pretrained_kwargs") or {}
    device_str: str = payload["device_str"]
    metadata_context: Dict[str, Any] = payload.get("metadata_context") or {}
    profile_tier = ProfileTier(metadata_context.get("profile_tier"))
    requested_batch, requested_height, requested_width = parse_requested_shape(payload)
    batch_size = requested_batch
    height = requested_height
    width = requested_width
    infer_kwargs_user: Dict[str, Any] = payload.get("infer_kwargs") or {}
    task_type: Optional[str] = payload.get("task_type")
    method_name: str = payload["method_name"]
    warmup_iterations = int(payload["warmup_iterations"])
    measured_iterations = int(payload["measured_iterations"])
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

    from_pretrained_kwargs.setdefault(
        "onnx_execution_providers",
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    from_pretrained_kwargs.setdefault("device", device)

    with _onnxruntime_profiling_sessions(
        onnxruntime,
        trace_dir=trace_dir,
        sessions=sessions,
    ):
        model_cls = _resolve_class(
            module_name=module_name,
            class_name=class_name,
        )
        model = model_cls.from_pretrained(
            model_name_or_path,
            **from_pretrained_kwargs,
        )

    torch.cuda.synchronize(device)

    idle_after_session_create_bytes = sampler.snapshot()

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
    infer_kwargs = merge_infer_kwargs(
        task_type=task_type,
        user=infer_kwargs_user,
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

    metrics = OnnxMetrics(
        baseline_process_gpu_bytes_nvml=baseline_process_gpu_bytes,
        idle_after_session_create_bytes=idle_after_session_create_bytes,
        peak_process_gpu_bytes=peak_process_gpu_bytes,
        delta_peak_bytes=delta_peak_bytes,
    )
    model_metadata = build_model_metadata_from_context(metadata_context)
    runtime_metadata = build_runtime_metadata(
        module_name=module_name,
        class_name=class_name,
        method_name=method_name,
    )
    backend_metadata = collect_onnx_backend_metadata(
        onnxruntime_version=onnxruntime.__version__,
        execution_providers=execution_providers,
        trace_dir=str(trace_dir),
        opset=resolve_onnx_opset_from_metadata_context(metadata_context),
    )
    record = finalize_profile_record(
        profile_tier=profile_tier,
        metrics=metrics,
        model_metadata=model_metadata,
        runtime_metadata=runtime_metadata,
        backend_metadata=backend_metadata,
        input_metadata=build_input_metadata(
            module_name=module_name,
            class_name=class_name,
            architecture=architecture,
            task_type=task_type,
            backend=payload.get("backend"),
            batch_size=batch_size,
            height=height,
            width=width,
            infer_kwargs=infer_kwargs,
            shape_spec=shape_spec,
        ),
        environment_metadata=collect_environment_metadata(device=device),
        profiling_run=build_profiling_run_metadata(payload, trace_files=trace_files),
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
