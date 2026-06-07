from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
from rich.console import Console
from rich.table import Table

from inference_models import BackendType, Quantization
from profiling.memory.backend_registry import (
    list_onnx_registry_rows,
    list_torch_registry_rows,
    list_trt_registry_rows,
    registry_features_from_row,
    resolve_registry_row,
)
from profiling.memory.metadata import ProfileTier, profile_tier_field_description
from profiling.memory.package_input_profile import (
    resolve_profiling_image_shapes_for_package_dir,
)
from profiling.memory.profiling_inputs import (
    build_profiling_infer_kwargs,
    resolve_profiling_method,
)
from profiling.memory.package_resolve import (
    extract_onnx_opset_from_package_dir,
    onnx_opset_from_package,
    resolve_package_directory,
)
from profiling.memory.subprocess_harness import dump_result_json, run_profile_subprocess
from profiling.memory.workers.onnx import worker_run as onnx_worker_run
from profiling.memory.workers.torch import worker_run as torch_worker_run
from profiling.memory.workers.trt import worker_run as trt_worker_run


BYTES_IN_GB = 1024**3
TORCH_MEMORY_FIELDS = (
    ("Idle allocated", "idle_after_load_allocated_bytes"),
    ("Idle reserved", "idle_after_load_reserved_bytes"),
    ("Peak allocated", "peak_allocated_bytes"),
    ("Peak reserved", "peak_reserved_bytes"),
    ("End reserved", "end_reserved_bytes"),
    ("Peak incremental allocated", "peak_incremental_allocated_bytes"),
    ("Peak incremental reserved", "peak_incremental_reserved_bytes"),
    ("Baseline free NVML", "baseline_gpu_free_bytes_nvml"),
)
ONNX_MEMORY_FIELDS = (
    ("Baseline process GPU", "baseline_process_gpu_bytes_nvml"),
    ("Idle after session create", "idle_after_session_create_bytes"),
    ("Peak process GPU", "peak_process_gpu_bytes"),
    ("Delta peak", "delta_peak_bytes"),
)
TRT_MEMORY_FIELDS = (
    ("Baseline process GPU", "baseline_process_gpu_bytes_nvml"),
    ("Idle after deserialize", "idle_after_deserialize_bytes"),
    ("Peak request", "peak_request_bytes"),
    ("Delta peak", "delta_peak_bytes"),
    ("Engine file size", "engine_size_bytes"),
)


def _parse_onnx_execution_providers(raw: str) -> List[str]:
    providers = [part.strip() for part in raw.split(",") if part.strip()]
    if not providers:
        raise click.ClickException(
            "--onnx-execution-providers must list at least one provider "
            "(e.g. CUDAExecutionProvider,CPUExecutionProvider)."
        )

    return providers


def _load_json_dict(raw: Optional[str], path: Optional[str]) -> Dict[str, Any]:
    try:
        if path:
            json_path = Path(path)
            with json_path.open(encoding="utf-8") as f:
                value = json.load(f)
        elif raw:
            value = json.loads(raw)
        else:
            return {}
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"Invalid JSON: {exc}") from exc

    if not isinstance(value, dict):
        raise click.ClickException("JSON option values must decode to an object.")
    return value


def _cmd_list(
    console: Console,
    *,
    backend: BackendType,
) -> None:
    if backend == BackendType.ONNX:
        rows = list_onnx_registry_rows()
    elif backend == BackendType.TRT:
        rows = list_trt_registry_rows()
    else:
        rows = list_torch_registry_rows()
        table_title = (
            "REGISTERED_MODELS — PyTorch memory harness "
            "(torch, torch-script, hugging-face)"
        )

    if backend == BackendType.ONNX:
        table_title = f"REGISTERED_MODELS — BackendType.{backend.name}"
    elif backend == BackendType.TRT:
        table_title = f"REGISTERED_MODELS — BackendType.{backend.name}"

    table = Table(title=table_title)
    table.add_column("architecture")
    table.add_column("task_type")
    table.add_column("backend")
    table.add_column("module")
    table.add_column("class")
    table.add_column("required_features")
    table.add_column("supported_features")
    for r in rows:
        table.add_row(
            r.architecture,
            r.task_type or "",
            r.backend.value,
            r.module_name,
            r.class_name,
            ",".join(sorted(r.required_model_features or [])),
            ",".join(sorted(r.supported_model_features or [])),
        )

    console.print(table)


def _format_bytes_as_gb(value: Optional[int]) -> str:
    if value is None:
        return "n/a"

    gb_value = value / BYTES_IN_GB
    formatted_value = f"{gb_value:.3f}"

    return formatted_value


def _resolve_profiling_package(
    *,
    model_id: str,
    architecture: str,
    task_type: str,
    backend: str,
    quantization: str,
    package_id: Optional[str],
    packages_target_dir: Path,
    force_download: bool,
    provider: str,
    api_key: Optional[str],
) -> tuple[Path, str, str, Optional[str], Optional[int]]:
    backend_type = BackendType(backend)
    quantization_type = Quantization(quantization)
    package_dir, package, model_variant = resolve_package_directory(
        model_id=model_id,
        model_architecture=architecture,
        task_type=task_type,
        backend=backend_type,
        quantization=quantization_type,
        package_id=package_id,
        target_dir=packages_target_dir,
        provider=provider,
        api_key=api_key,
        force_download=force_download,
    )
    resolved_onnx_opset = onnx_opset_from_package(package)

    return (
        package_dir,
        model_id,
        package.package_id,
        model_variant,
        resolved_onnx_opset,
    )


def _build_metadata_context(
    *,
    profile_tier: str,
    model_id: Optional[str],
    package_id: Optional[str],
    package_path: str,
    backend: str,
    architecture: Optional[str],
    task_type: Optional[str],
    quantization: str,
    model_variant: Optional[str],
    registry_features: Dict[str, Any],
    onnx_opset: Optional[int] = None,
) -> Dict[str, Any]:
    metadata_context = {
        "profile_tier": profile_tier,
        "model_id": model_id,
        "package_id": package_id,
        "package_path": package_path,
        "backend": backend,
        "architecture": architecture,
        "task_type": task_type,
        "model_variant": model_variant,
        "quantization": quantization,
        "registry_features": registry_features,
    }
    if onnx_opset is not None:
        metadata_context["onnx_opset"] = onnx_opset

    return metadata_context


def _print_human_readable_result(console: Console, result: Dict[str, Any]) -> None:
    model_meta = result.get("model_metadata") or {}
    registered = model_meta.get("registered_model") or {}
    package = model_meta.get("package") or {}
    runtime_meta = result.get("runtime_metadata") or {}
    backend_meta = result.get("backend_metadata") or {}
    input_meta = result.get("input_metadata") or {}
    env_meta = result.get("environment_metadata") or {}
    profiling_run = result.get("profiling_run") or {}
    metrics = result.get("metrics") or {}

    backend = str(package.get("backend") or "unknown")
    summary = Table(title=f"{backend} Memory Profile")
    summary.add_column("Field")
    summary.add_column("Value")

    summary.add_row("Profile ID", str(result.get("profile_id")))
    summary.add_row("Profile tier", str(result.get("profile_tier")))
    summary.add_row("Model ID", str(registered.get("model_id")))
    summary.add_row("Model variant", str(registered.get("model_variant")))
    summary.add_row("Package ID", str(package.get("package_id")))
    summary.add_row("Package path", str(package.get("package_path")))
    summary.add_row("Backend", backend)
    summary.add_row("Architecture", str(registered.get("architecture")))
    summary.add_row("Task", str(registered.get("task_type")))
    summary.add_row("Quantization", str(package.get("quantization")))
    summary.add_row("Method", str(runtime_meta.get("method")))
    summary.add_row("GPU", str(env_meta.get("gpu")))

    profile_name = input_meta.get("task_inference_profile")
    if profile_name:
        summary.add_row("Task input profile", str(profile_name))

    exercised_inputs = input_meta.get("inputs") or {}
    image_inputs = (
        exercised_inputs.get("images")
        or exercised_inputs.get("image")
        or next(iter(exercised_inputs.values()), {})
    )
    batch_axis = image_inputs.get("batch") or {}
    height_axis = image_inputs.get("height") or {}
    width_axis = image_inputs.get("width") or {}
    shape_text = (
        f"batch={batch_axis.get('value')}, "
        f"{height_axis.get('value')}x{width_axis.get('value')}"
    )
    summary.add_row("Profiled shape", shape_text)

    trace_files = ", ".join(profiling_run.get("trace_files") or [])
    if trace_files:
        summary.add_row("Trace files", trace_files)

    if backend == BackendType.ONNX.value:
        execution_providers = ", ".join(backend_meta.get("execution_providers") or [])
        summary.add_row("ONNX Runtime", str(backend_meta.get("onnxruntime_version")))
        summary.add_row("ONNX opset", str(backend_meta.get("opset")))
        summary.add_row("Execution providers", execution_providers)

    if backend == BackendType.TRT.value:
        summary.add_row("TensorRT", str(backend_meta.get("tensorrt_version")))
        summary.add_row(
            "Contexts profiled",
            str(backend_meta.get("num_contexts_profiled")),
        )
        optimization_profile = backend_meta.get("optimization_profile")
        if optimization_profile:
            summary.add_row(
                "Optimization profile",
                json.dumps(optimization_profile, sort_keys=True),
            )

    memory = Table(title="Memory Metrics")
    memory.add_column("Metric")
    memory.add_column("GB", justify="right")
    memory.add_column("Bytes", justify="right")
    if backend == BackendType.ONNX.value:
        memory_fields = ONNX_MEMORY_FIELDS
    elif backend == BackendType.TRT.value:
        memory_fields = TRT_MEMORY_FIELDS
    else:
        memory_fields = TORCH_MEMORY_FIELDS

    for label, key in memory_fields:
        value = metrics.get(key)
        bytes_text = "n/a" if value is None else str(value)
        gb_text = _format_bytes_as_gb(value=value)

        memory.add_row(
            label,
            gb_text,
            bytes_text,
        )

    console.print(summary)
    console.print(memory)


@click.command(
    context_settings={"help_option_names": ["-h", "--help"]},
    help=(
        "GPU memory profiling for inference_models registry classes "
        "(see profiling/memory/docs/description.md). "
        "Use --backend torch for torch, torch-script, and Hugging Face registry entries."
    ),
)
@click.option(
    "--list-torch-models",
    is_flag=True,
    help=(
        "Print torch, torch-script, and Hugging Face registry rows for the "
        "PyTorch harness, then exit."
    ),
)
@click.option(
    "--list-onnx-models",
    is_flag=True,
    help="Print ONNX backend rows from models_registry and exit.",
)
@click.option(
    "--list-trt-models",
    is_flag=True,
    help="Print TensorRT backend rows from models_registry and exit.",
)
@click.option(
    "--backend",
    type=click.Choice(
        [
            BackendType.TORCH.value,
            BackendType.ONNX.value,
            BackendType.TRT.value,
        ],
    ),
    help="Profiling backend harness to run.",
)
@click.option(
    "--model-id",
    type=str,
    default=None,
    help=(
        "Registered model id (required for profiling). Resolves package metadata "
        "from the provider and uses {packages-target-dir}/{model_id}/{package_id}/ "
        "locally; reuses cached artifacts unless missing or --force-download."
    ),
)
@click.option(
    "--package-id",
    type=str,
    default=None,
    help="Exact package version when using --model-id.",
)
@click.option(
    "--packages-target-dir",
    type=click.Path(
        file_okay=False,
        path_type=Path,
    ),
    default=Path("/tmp/inference_model_packages"),
    show_default=True,
    help="Root directory for downloaded model packages.",
)
@click.option(
    "--force-download",
    is_flag=True,
    help=(
        "Re-download package artifacts even when the local directory already exists."
    ),
)
@click.option(
    "--provider",
    type=str,
    default="roboflow",
    show_default=True,
    help="Package provider when fetching (see fetch_model_package).",
)
@click.option(
    "--profile-tier",
    type=click.Choice(
        [tier.value for tier in ProfileTier],
        case_sensitive=False,
    ),
    default=ProfileTier.CUSTOMER.value,
    show_default=True,
    help=profile_tier_field_description(),
)
@click.option(
    "--model-variant",
    type=str,
    default=None,
    help=(
        "Registered model variant (e.g. yolov8-n). "
        "Filled automatically when resolving via --model-id."
    ),
)
@click.option(
    "--architecture",
    type=str,
    default=None,
    help=(
        "Registry architecture (e.g. yolov8). Required for profiling; "
        "resolves the model class with --task-type and --backend."
    ),
)
@click.option(
    "--task-type",
    type=str,
    default=None,
    help=(
        "Registry task type (e.g. object-detection). Required for profiling; "
        "resolves the model class with --architecture and --backend."
    ),
)
@click.option(
    "--device",
    type=str,
    default="cuda:0",
    show_default=True,
)
@click.option(
    "--warmup",
    "warmup_iterations",
    type=click.IntRange(
        min=0,
    ),
    default=2,
    show_default=True,
)
@click.option(
    "--measured",
    "measured_iterations",
    type=click.IntRange(
        min=1,
    ),
    default=5,
    show_default=True,
)
@click.option(
    "--method",
    type=str,
    default=None,
    help=(
        "Method to call with synthetic images. Defaults to the registry task "
        "profile profiling_method (e.g. infer, prompt, detect)."
    ),
)
@click.option(
    "--infer-kwargs-json",
    type=str,
    default=None,
    help="Inline JSON object.",
)
@click.option(
    "--infer-kwargs-path",
    type=click.Path(
        exists=True,
        dir_okay=False,
        path_type=str,
    ),
    default=None,
    help="JSON file path.",
)
@click.option(
    "--from-pretrained-kwargs-json",
    type=str,
    default=None,
    help="Inline JSON object passed to the model ``from_pretrained`` method.",
)
@click.option(
    "--from-pretrained-kwargs-path",
    type=click.Path(
        exists=True,
        dir_okay=False,
        path_type=str,
    ),
    default=None,
    help="Path to a JSON file passed to the model ``from_pretrained`` method.",
)
@click.option(
    "--quantization",
    type=click.Choice(
        [quantization.value for quantization in Quantization],
    ),
    default="fp32",
    show_default=True,
    help="Package quantization.",
)
@click.option(
    "--torch-profiler-memory",
    is_flag=True,
    help="Wrap measured iterations with torch.profiler (profile_memory=True).",
)
@click.option(
    "--onnx-execution-providers",
    type=str,
    default=None,
    help=(
        "Comma-separated ONNX Runtime execution providers in priority order "
        "(e.g. CUDAExecutionProvider,CPUExecutionProvider). "
        "Defaults to CUDA then CPU when omitted."
    ),
)
@click.option(
    "--onnx-nvml-sampling-interval-seconds",
    type=float,
    default=0.01,
    show_default=True,
    help="NVML polling interval for ONNX process-memory peaks.",
)
@click.option(
    "--trt-nvml-sampling-interval-seconds",
    type=float,
    default=0.01,
    show_default=True,
    help="NVML polling interval for TensorRT process-memory peaks.",
)
@click.option(
    "--num-execution-contexts",
    type=click.IntRange(
        min=1,
    ),
    default=1,
    show_default=True,
    help=(
        "Number of concurrent execution contexts to record in results "
        "(registry models currently load one context per instance)."
    ),
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Resolve package, shapes, and inputs; print worker payload JSON and exit.",
)
@click.option(
    "--in-process",
    is_flag=True,
    help="Run in the current process (debug only; breaks isolation between scenarios).",
)
@click.option(
    "--output-json",
    type=click.Path(
        dir_okay=False,
        path_type=str,
    ),
    default=None,
    help="Write result JSON to this path.",
)
def main(
    list_torch_models: bool,
    list_onnx_models: bool,
    list_trt_models: bool,
    backend: Optional[str],
    model_id: Optional[str],
    package_id: Optional[str],
    packages_target_dir: Path,
    force_download: bool,
    provider: str,
    profile_tier: str,
    model_variant: Optional[str],
    architecture: Optional[str],
    task_type: Optional[str],
    device: str,
    warmup_iterations: int,
    measured_iterations: int,
    method: Optional[str],
    infer_kwargs_json: Optional[str],
    infer_kwargs_path: Optional[str],
    from_pretrained_kwargs_json: Optional[str],
    from_pretrained_kwargs_path: Optional[str],
    quantization: Optional[str],
    torch_profiler_memory: bool,
    onnx_execution_providers: Optional[str],
    onnx_nvml_sampling_interval_seconds: float,
    trt_nvml_sampling_interval_seconds: float,
    num_execution_contexts: int,
    dry_run: bool,
    in_process: bool,
    output_json: Optional[str],
) -> None:
    """Run or list GPU memory profiling for ``inference_models`` registry classes."""
    console = Console()

    if list_torch_models:
        _cmd_list(
            console,
            backend=BackendType.TORCH,
        )
        return

    if list_onnx_models:
        _cmd_list(
            console,
            backend=BackendType.ONNX,
        )
        return

    if list_trt_models:
        _cmd_list(
            console,
            backend=BackendType.TRT,
        )
        return

    missing = []
    if not backend:
        missing.append("--backend")
    if not architecture:
        missing.append("--architecture")
    if not task_type:
        missing.append("--task-type")
    if not model_id:
        missing.append("--model-id")
    if not quantization:
        missing.append("--quantization")
    if missing:
        raise click.UsageError(
            "Missing required arguments: "
            + ", ".join(missing)
            + ". Use --list-torch-models, --list-onnx-models, --list-trt-models, or --help."
        )

    try:
        registry_row = resolve_registry_row(
            architecture=architecture,
            task_type=task_type,
            harness_backend=backend,
        )
    except ValueError as error:
        raise click.ClickException(str(error)) from error

    module_name = registry_row.module_name
    class_name = registry_row.class_name
    registry_features = registry_features_from_row(registry_row)

    infer_extra = _load_json_dict(
        raw=infer_kwargs_json,
        path=infer_kwargs_path,
    )
    fp_extra = _load_json_dict(
        raw=from_pretrained_kwargs_json,
        path=from_pretrained_kwargs_path,
    )

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    try:
        (
            package_dir,
            resolved_model_id,
            resolved_package_id,
            fetched_model_variant,
            fetched_onnx_opset,
        ) = _resolve_profiling_package(
            model_id=model_id,
            architecture=architecture,
            task_type=task_type,
            backend=backend,
            quantization=quantization or "fp32",
            package_id=package_id,
            packages_target_dir=packages_target_dir,
            force_download=force_download,
            provider=provider,
            api_key=api_key,
        )
    except ValueError as error:
        raise click.ClickException(str(error)) from error

    batch_size, height, width = resolve_profiling_image_shapes_for_package_dir(
        package_dir,
    )
    effective_method = resolve_profiling_method(
        module_name=module_name,
        class_name=class_name,
        architecture=architecture,
        task_type=task_type,
        backend=backend,
        user_method=method,
    )
    effective_infer_kwargs = build_profiling_infer_kwargs(
        package_dir=package_dir,
        module_name=module_name,
        class_name=class_name,
        architecture=architecture,
        task_type=task_type,
        backend=backend,
        user=infer_extra,
    )

    resolved_path = str(package_dir)
    effective_model_variant = model_variant or fetched_model_variant
    effective_onnx_opset = fetched_onnx_opset
    if effective_onnx_opset is None and backend == BackendType.ONNX.value:
        effective_onnx_opset = extract_onnx_opset_from_package_dir(package_dir)

    metadata_context = _build_metadata_context(
        profile_tier=profile_tier,
        model_id=resolved_model_id,
        package_id=resolved_package_id,
        package_path=resolved_path,
        backend=backend,
        architecture=architecture,
        task_type=task_type,
        quantization=quantization or "fp32",
        model_variant=effective_model_variant,
        registry_features=registry_features,
        onnx_opset=effective_onnx_opset,
    )

    if onnx_execution_providers is not None:
        if backend != BackendType.ONNX.value:
            raise click.ClickException(
                "--onnx-execution-providers is only valid with --backend onnx."
            )

        fp_extra["onnx_execution_providers"] = _parse_onnx_execution_providers(
            onnx_execution_providers,
        )

    payload: Dict[str, Any] = {
        "module_name": module_name,
        "class_name": class_name,
        "package_path": resolved_path,
        "from_pretrained_kwargs": fp_extra,
        "device_str": device,
        "batch_size": batch_size,
        "height": height,
        "width": width,
        "infer_kwargs": effective_infer_kwargs,
        "task_type": task_type,
        "method_name": effective_method,
        "warmup_iterations": warmup_iterations,
        "measured_iterations": measured_iterations,
        "architecture": architecture,
        "backend": backend,
        "quantization": quantization,
        "torch_profiler_memory": torch_profiler_memory,
        "onnx_nvml_sampling_interval_seconds": onnx_nvml_sampling_interval_seconds,
        "trt_nvml_sampling_interval_seconds": trt_nvml_sampling_interval_seconds,
        "num_execution_contexts": num_execution_contexts,
        "metadata_context": metadata_context,
        "profile_tier": profile_tier,
        "in_process": in_process,
    }

    if dry_run:
        console.print(json.dumps(payload, indent=2))
        return

    if backend == BackendType.ONNX.value:
        if in_process:
            result = onnx_worker_run(payload)
        else:
            result = run_profile_subprocess(
                payload,
                worker_module="profiling.memory.workers.onnx",
                harness_label="ONNX profiling",
            )
    elif backend == BackendType.TRT.value:
        if in_process:
            result = trt_worker_run(payload)
        else:
            result = run_profile_subprocess(
                payload,
                worker_module="profiling.memory.workers.trt",
                harness_label="TensorRT profiling",
            )
    else:
        if in_process:
            result = torch_worker_run(payload)
        else:
            result = run_profile_subprocess(
                payload,
                worker_module="profiling.memory.workers.torch",
                harness_label="Torch profiling",
            )

    _print_human_readable_result(
        console,
        result=result,
    )

    text = json.dumps(result, indent=2)
    console.print(text)

    if output_json:
        dump_result_json(
            result,
            path=output_json,
        )


if __name__ == "__main__":
    main()
