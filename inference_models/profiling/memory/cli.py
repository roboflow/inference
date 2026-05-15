from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import click
from rich.console import Console
from rich.table import Table

from inference_models import BackendType, Quantization
from profiling.memory.onnx_harness import run_onnx_profile_subprocess
from profiling.memory.onnx_worker import worker_run as onnx_worker_run
from profiling.memory.pytorch_harness import (
    dump_result_json,
    run_pytorch_profile_subprocess,
)
from profiling.memory.pytorch_worker import worker_run as pytorch_worker_run
from profiling.memory.torch_registry import (
    list_onnx_registry_rows,
    list_torch_registry_rows,
)

BYTES_IN_GB = 1024**3
PYTORCH_MEMORY_FIELDS = (
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


def _load_json_dict(raw: Optional[str], path: Optional[str]) -> Dict[str, Any]:
    try:
        if path:
            with open(path, encoding="utf-8") as f:
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
    else:
        rows = list_torch_registry_rows()

    table = Table(title=f"REGISTERED_MODELS — BackendType.{backend.name}")
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


def _print_human_readable_result(console: Console, result: Dict[str, Any]) -> None:
    backend = str(result.get("backend"))
    summary = Table(title=f"{backend} Memory Profile")
    summary.add_column("Field")
    summary.add_column("Value")

    shape_profile = result.get("shape_profile") or {}
    shape_text = (
        f"batch={shape_profile.get('batch_size')}, "
        f"height={shape_profile.get('height')}, "
        f"width={shape_profile.get('width')}"
    )
    summary.add_row("Model", str(result.get("model_id")))
    summary.add_row("Backend", str(result.get("backend")))
    summary.add_row("GPU", str(result.get("gpu_name")))
    summary.add_row("Quantization", str(result.get("quantization")))
    summary.add_row("Shape", shape_text)
    summary.add_row("Method", str(result.get("method_name")))
    if backend == BackendType.ONNX.value:
        execution_providers = ", ".join(result.get("execution_providers") or [])
        trace_files = ", ".join(result.get("trace_files") or [])
        summary.add_row("ONNX Runtime", str(result.get("onnxruntime_version")))
        summary.add_row("Execution providers", execution_providers)
        summary.add_row("Trace files", trace_files)

    memory = Table(title="Memory Metrics")
    memory.add_column("Metric")
    memory.add_column("GB", justify="right")
    memory.add_column("Bytes", justify="right")
    memory_fields = (
        ONNX_MEMORY_FIELDS
        if backend == BackendType.ONNX.value
        else PYTORCH_MEMORY_FIELDS
    )

    for label, key in memory_fields:
        value = result.get(key)
        bytes_text = "n/a" if value is None else str(value)
        gb_text = _format_bytes_as_gb(value=value)

        memory.add_row(
            label,
            gb_text,
            bytes_text,
        )

    console.print(summary)
    console.print(memory)


def _infer_default_backend() -> str:
    executable_name = Path(sys.argv[0]).name
    if "onnx" in executable_name:
        return BackendType.ONNX.value

    return BackendType.TORCH.value


@click.command(
    context_settings={"help_option_names": ["-h", "--help"]},
    help=(
        "GPU memory profiling for inference_models registry classes "
        "(see profiling/memory/docs/description.md)."
    ),
)
@click.option(
    "--list-torch-models",
    is_flag=True,
    help="Print Torch backend rows from models_registry and exit.",
)
@click.option(
    "--list-onnx-models",
    is_flag=True,
    help="Print ONNX backend rows from models_registry and exit.",
)
@click.option(
    "--backend",
    type=click.Choice(
        [
            BackendType.TORCH.value,
            BackendType.ONNX.value,
        ],
    ),
    default=None,
    show_default="torch for memory-profile-pytorch, onnx for memory-profile-onnx",
    help="Profiling backend harness to run.",
)
@click.option(
    "--module-name",
    type=str,
    help="Model module (e.g. inference_models....).",
)
@click.option(
    "--class-name",
    type=str,
    help="Model class name.",
)
@click.option(
    "--model-path",
    type=str,
    help=(
        "Path passed to from_pretrained "
        "(local package dir or hub id as supported by the model)."
    ),
)
@click.option(
    "--model-id",
    type=str,
    default=None,
    help="Label stored in the JSON result (defaults to --model-path).",
)
@click.option(
    "--architecture",
    type=str,
    default=None,
)
@click.option(
    "--task-type",
    type=str,
    default=None,
)
@click.option(
    "--device",
    type=str,
    default="cuda:0",
    show_default=True,
)
@click.option(
    "--batch-size",
    type=click.IntRange(
        min=1,
    ),
    default=1,
    show_default=True,
)
@click.option(
    "--height",
    type=click.IntRange(
        min=1,
    ),
    default=640,
    show_default=True,
)
@click.option(
    "--width",
    type=click.IntRange(
        min=1,
    ),
    default=640,
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
    default="infer",
    show_default=True,
    help=(
        "Method to call with synthetic images "
        "(e.g. infer, embed_images, segment_images)."
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
)
@click.option(
    "--from-pretrained-kwargs-path",
    type=click.Path(
        exists=True,
        dir_okay=False,
        path_type=str,
    ),
    default=None,
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
    "--onnx-nvml-sampling-interval-seconds",
    type=float,
    default=0.01,
    show_default=True,
    help="NVML polling interval for ONNX process-memory peaks.",
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
    backend: str,
    module_name: Optional[str],
    class_name: Optional[str],
    model_path: Optional[str],
    model_id: Optional[str],
    architecture: Optional[str],
    task_type: Optional[str],
    device: str,
    batch_size: int,
    height: int,
    width: int,
    warmup_iterations: int,
    measured_iterations: int,
    method: str,
    infer_kwargs_json: Optional[str],
    infer_kwargs_path: Optional[str],
    from_pretrained_kwargs_json: Optional[str],
    from_pretrained_kwargs_path: Optional[str],
    quantization: Optional[str],
    torch_profiler_memory: bool,
    onnx_nvml_sampling_interval_seconds: float,
    in_process: bool,
    output_json: Optional[str],
) -> None:
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

    missing = []
    if not module_name:
        missing.append("--module-name")
    if not class_name:
        missing.append("--class-name")
    if not model_path:
        missing.append("--model-path")
    if not quantization:
        missing.append("--quantization")
    if missing:
        raise click.UsageError(
            "Missing required arguments: "
            + ", ".join(missing)
            + ". Use --list-torch-models or see --help."
        )

    selected_backend = backend or _infer_default_backend()

    infer_extra = _load_json_dict(
        raw=infer_kwargs_json,
        path=infer_kwargs_path,
    )
    fp_extra = _load_json_dict(
        raw=from_pretrained_kwargs_json,
        path=from_pretrained_kwargs_path,
    )

    payload: Dict[str, Any] = {
        "module_name": module_name,
        "class_name": class_name,
        "model_name_or_path": model_path,
        "from_pretrained_kwargs": fp_extra,
        "device_str": device,
        "batch_size": batch_size,
        "height": height,
        "width": width,
        "infer_kwargs": infer_extra,
        "task_type": task_type,
        "method_name": method,
        "warmup_iterations": warmup_iterations,
        "measured_iterations": measured_iterations,
        "model_id": model_id or model_path,
        "architecture": architecture,
        "backend": selected_backend,
        "quantization": quantization,
        "torch_profiler_memory": torch_profiler_memory,
        "onnx_nvml_sampling_interval_seconds": onnx_nvml_sampling_interval_seconds,
    }

    if selected_backend == BackendType.ONNX.value:
        if in_process:
            result = onnx_worker_run(payload)
        else:
            result = run_onnx_profile_subprocess(payload)
    else:
        if in_process:
            result = pytorch_worker_run(payload)
        else:
            result = run_pytorch_profile_subprocess(payload)

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
