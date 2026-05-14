from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, Optional

from rich.console import Console
from rich.table import Table

from profiling.memory.pytorch_harness import dump_result_json, run_pytorch_profile_subprocess
from profiling.memory.pytorch_worker import worker_run
from profiling.memory.torch_registry import list_torch_registry_rows


def _load_json_dict(raw: Optional[str], path: Optional[str]) -> Dict[str, Any]:
    if path:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    if raw:
        return json.loads(raw)
    return {}


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PyTorch GPU memory profiling for inference_models registry classes "
        "(see profiling/memory/docs/description.md)."
    )
    p.add_argument(
        "--list-torch-models",
        action="store_true",
        help="Print Torch backend rows from models_registry and exit.",
    )
    p.add_argument("--module-name", type=str, help="Model module (e.g. inference_models....)")
    p.add_argument("--class-name", type=str, help="Model class name.")
    p.add_argument(
        "--model-path",
        type=str,
        help="Path passed to from_pretrained (local package dir or hub id as supported by the model).",
    )
    p.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Label stored in the JSON result (defaults to --model-path).",
    )
    p.add_argument("--architecture", type=str, default=None)
    p.add_argument("--task-type", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--height", type=int, default=640)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--warmup", type=int, default=2, dest="warmup_iterations")
    p.add_argument("--measured", type=int, default=5, dest="measured_iterations")
    p.add_argument(
        "--method",
        type=str,
        default="infer",
        help="Method to call with synthetic images (e.g. infer, embed_images, segment_images).",
    )
    p.add_argument("--infer-kwargs-json", type=str, default=None, help="Inline JSON object.")
    p.add_argument("--infer-kwargs-path", type=str, default=None, help="JSON file path.")
    p.add_argument("--from-pretrained-kwargs-json", type=str, default=None)
    p.add_argument("--from-pretrained-kwargs-path", type=str, default=None)
    p.add_argument("--precision", type=str, default=None)
    p.add_argument(
        "--torch-profiler-memory",
        action="store_true",
        help="Wrap measured iterations with torch.profiler (profile_memory=True).",
    )
    p.add_argument(
        "--in-process",
        action="store_true",
        help="Run in the current process (debug only; breaks isolation between scenarios).",
    )
    p.add_argument("--output-json", type=str, default=None, help="Write result JSON to this path.")
    return p.parse_args(argv)


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


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    console = Console()

    if args.list_torch_models:
        _cmd_list(console)
        return

    missing = []
    if not args.module_name:
        missing.append("--module-name")
    if not args.class_name:
        missing.append("--class-name")
    if not args.model_path:
        missing.append("--model-path")
    if missing:
        console.print(
            "[red]Missing required arguments:[/red] "
            + ", ".join(missing)
            + ". Use --list-torch-models or see --help."
        )
        sys.exit(2)

    infer_extra = _load_json_dict(args.infer_kwargs_json, args.infer_kwargs_path)
    fp_extra = _load_json_dict(
        args.from_pretrained_kwargs_json, args.from_pretrained_kwargs_path
    )

    payload: Dict[str, Any] = {
        "module_name": args.module_name,
        "class_name": args.class_name,
        "model_name_or_path": args.model_path,
        "from_pretrained_kwargs": fp_extra,
        "device_str": args.device,
        "batch_size": args.batch_size,
        "height": args.height,
        "width": args.width,
        "infer_kwargs": infer_extra,
        "task_type": args.task_type,
        "method_name": args.method,
        "warmup_iterations": args.warmup_iterations,
        "measured_iterations": args.measured_iterations,
        "model_id": args.model_id or args.model_path,
        "architecture": args.architecture,
        "precision": args.precision,
        "torch_profiler_memory": args.torch_profiler_memory,
    }

    if args.in_process:
        result = worker_run(payload)
    else:
        result = run_pytorch_profile_subprocess(payload)

    text = json.dumps(result, indent=2)
    console.print(text)
    if args.output_json:
        dump_result_json(result, args.output_json)


if __name__ == "__main__":
    main()
