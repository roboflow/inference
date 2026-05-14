from __future__ import annotations

import json
from typing import Any, Dict, Optional

import click
from rich.console import Console
from rich.table import Table

from profiling.memory.pytorch_harness import (
    dump_result_json,
    run_pytorch_profile_subprocess,
)
from profiling.memory.pytorch_worker import worker_run
from profiling.memory.torch_registry import list_torch_registry_rows


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


def _cmd_list(console: Console) -> None:
    rows = list_torch_registry_rows()
    table = Table(title="REGISTERED_MODELS — BackendType.TORCH")
    table.add_column("architecture")
    table.add_column("task_type")
    table.add_column("module")
    table.add_column("class")
    table.add_column("required_features")
    table.add_column("supported_features")
    for r in rows:
        table.add_row(
            r.architecture,
            r.task_type or "",
            r.module_name,
            r.class_name,
            ",".join(sorted(r.required_model_features or [])),
            ",".join(sorted(r.supported_model_features or [])),
        )
    console.print(table)


@click.command(
    context_settings={"help_option_names": ["-h", "--help"]},
    help=(
        "PyTorch GPU memory profiling for inference_models registry classes "
        "(see profiling/memory/docs/description.md)."
    ),
)
@click.option(
    "--list-torch-models",
    is_flag=True,
    help="Print Torch backend rows from models_registry and exit.",
)
@click.option(
    "--module-name",
    type=str,
    help="Model module (e.g. inference_models....).",
)
@click.option("--class-name", type=str, help="Model class name.")
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
@click.option("--architecture", type=str, default=None)
@click.option("--task-type", type=str, default=None)
@click.option("--device", type=str, default="cuda:0", show_default=True)
@click.option(
    "--batch-size",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
)
@click.option("--height", type=click.IntRange(min=1), default=640, show_default=True)
@click.option("--width", type=click.IntRange(min=1), default=640, show_default=True)
@click.option(
    "--warmup",
    "warmup_iterations",
    type=click.IntRange(min=0),
    default=2,
    show_default=True,
)
@click.option(
    "--measured",
    "measured_iterations",
    type=click.IntRange(min=1),
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
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    default=None,
    help="JSON file path.",
)
@click.option("--from-pretrained-kwargs-json", type=str, default=None)
@click.option(
    "--from-pretrained-kwargs-path",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    default=None,
)
@click.option("--precision", type=str, default=None)
@click.option(
    "--torch-profiler-memory",
    is_flag=True,
    help="Wrap measured iterations with torch.profiler (profile_memory=True).",
)
@click.option(
    "--in-process",
    is_flag=True,
    help="Run in the current process (debug only; breaks isolation between scenarios).",
)
@click.option(
    "--output-json",
    type=click.Path(dir_okay=False, path_type=str),
    default=None,
    help="Write result JSON to this path.",
)
def main(
    list_torch_models: bool,
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
    precision: Optional[str],
    torch_profiler_memory: bool,
    in_process: bool,
    output_json: Optional[str],
) -> None:
    console = Console()

    if list_torch_models:
        _cmd_list(console)
        return

    missing = []
    if not module_name:
        missing.append("--module-name")
    if not class_name:
        missing.append("--class-name")
    if not model_path:
        missing.append("--model-path")
    if missing:
        raise click.UsageError(
            "Missing required arguments: "
            + ", ".join(missing)
            + ". Use --list-torch-models or see --help."
        )

    infer_extra = _load_json_dict(infer_kwargs_json, infer_kwargs_path)
    fp_extra = _load_json_dict(
        from_pretrained_kwargs_json, from_pretrained_kwargs_path
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
        "precision": precision,
        "torch_profiler_memory": torch_profiler_memory,
    }

    if in_process:
        result = worker_run(payload)
    else:
        result = run_pytorch_profile_subprocess(payload)

    text = json.dumps(result, indent=2)
    console.print(text)
    if output_json:
        dump_result_json(result, output_json)


if __name__ == "__main__":
    main()
