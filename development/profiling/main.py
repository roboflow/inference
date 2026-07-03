import random
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import click
import torch
import yaml

from development.profiling.config import (
    DEFAULT_CAPTURE_RANGE,
    ProfileConfig,
    apply_overrides,
    load_profile_config,
)
from development.profiling.data import build_data_source
from development.profiling.nvtx import profiling_range
from development.profiling.registry import resolve_target


def main(argv: list[str] | None = None) -> int:
    """Run the profiling CLI.

    Args:
        argv (list[str] | None): Optional command-line arguments. When omitted,
            Click reads arguments from ``sys.argv``.

    Returns:
        Process exit code.
    """
    exit_code = cli.main(args=argv, standalone_mode=False)

    return exit_code


@click.command()
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(
        exists=True,
        dir_okay=False,
        path_type=Path,
    ),
    help="Path to profile config YAML.",
)
@click.option(
    "--target",
    type=str,
    help="Override target name from config.",
)
@click.option(
    "--data-source",
    type=str,
    help="Override data source name from config.",
)
@click.option(
    "--device",
    type=str,
    help="Override device from config, e.g. cpu or cuda.",
)
@click.option(
    "--warmup",
    type=int,
    help="Override warmup count.",
)
@click.option(
    "--iterations",
    type=int,
    help="Override measured iteration count.",
)
@click.option(
    "--capture-range",
    type=str,
    help=f"Override top-level NVTX capture range. Default: {DEFAULT_CAPTURE_RANGE}.",
)
@click.option(
    "--record-loading",
    type=click.Choice(
        ["eager", "lazy"],
        case_sensitive=True,
    ),
    help="Override record loading mode from config.",
)
@click.option(
    "--seed",
    type=int,
    help="Override reproducibility seed from config.",
)
@click.option(
    "--run-id",
    type=str,
    help="Stable run id for manifests and trace output.",
)
@click.option(
    "--print-nsys-command",
    is_flag=True,
    help="Print the Nsight Systems command for this profile and exit.",
)
def cli(
    config_path: Path,
    target: str | None,
    data_source: str | None,
    device: str | None,
    warmup: int | None,
    iterations: int | None,
    capture_range: str | None,
    record_loading: str | None,
    seed: int | None,
    run_id: str | None,
    print_nsys_command: bool,
) -> int:
    """Run a local profiling target.

    Args:
        config_path (Path): Path to the profile YAML file.
        target (str | None): Optional target name override.
        data_source (str | None): Optional data source name override.
        device (str | None): Optional torch device override.
        warmup (int | None): Optional warmup count override.
        iterations (int | None): Optional measured iteration count override.
        repetitions (int | None): Optional repetition count override.
        capture_range (str | None): Optional top-level NVTX capture range override.
        record_loading (str | None): Optional record loading mode override.
        seed (int | None): Optional reproducibility seed override.
        run_id (str | None): Optional stable run identifier.
        print_nsys_command (bool): Whether to print the Nsight command and exit.

    Returns:
        Process exit code.
    """
    config = apply_overrides(
        load_profile_config(config_path),
        target=target,
        data_source=data_source,
        device=device,
        warmup=warmup,
        iterations=iterations,
        capture_range=capture_range,
        record_loading=record_loading,
        seed=seed,
    )

    resolved_run_id = run_id or make_run_id()
    run_dir = config.output_dir / config.profile_name / "runs" / resolved_run_id

    if print_nsys_command:
        click.echo(
            build_nsys_command(
                config=config,
                run_dir=run_dir,
                config_path=config_path,
            )
        )
        return 0

    manifest = run_profile(
        config=config,
        run_id=resolved_run_id,
        run_dir=run_dir,
        argv=sys.argv,
    )
    if config.write_manifest:
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = run_dir / "manifest.yaml"
        with manifest_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(manifest, file, sort_keys=True)
        click.echo(f"Wrote manifest: {manifest_path}")

    return 0


def run_profile(
    *,
    config: ProfileConfig,
    run_id: str,
    run_dir: Path,
    argv: list[str] | None = None,
) -> dict[str, Any]:
    """Execute a configured profiling workload and build its run manifest.

    Args:
        config (ProfileConfig): Parsed profile configuration.
        run_id (str): Stable run identifier.
        run_dir (Path): Directory where run outputs are expected.
        argv (list[str] | None): Command line to record in the manifest.

    Returns:
        Manifest data describing the profiled workload.
    """
    if config.record_loading == "lazy" and not config.target.profile_prepare:
        raise ValueError(
            "Lazy record loading requires target.profile_prepare=true because "
            "records are not retained for precomputed preparation."
        )

    seed_everything(config.seed)

    device = torch.device(config.device)
    data_source = build_data_source(
        name=config.data_source.name,
        config=config.data_source.parameters,
    )

    records = None
    if config.record_loading == "eager":
        records = list(data_source.iter_records())
        if not records:
            raise ValueError("Data source yielded no records.")

    target = resolve_target(
        name=config.target.name,
        import_path=config.target.import_path,
    )

    prepared_records = None
    if records is not None and not config.target.profile_prepare:
        prepared_records = [target.prepare(record, device=device) for record in records]

    synchronize_cuda(device, config.cuda.synchronize_before_warmup)
    for _ in range(config.warmup):
        _run_pass(
            target=target,
            records=_records_for_pass(
                records=records,
                data_source=data_source,
            ),
            prepared_records=prepared_records,
            device=device,
            profile_prepare=config.target.profile_prepare,
            validate_output=config.validate_output,
            synchronize_each_iteration=config.cuda.synchronize_each_iteration,
        )
    synchronize_cuda(device, config.cuda.synchronize_after_warmup)

    last_outputs = []
    last_record_ids = []
    synchronize_cuda(device, config.cuda.synchronize_before_capture)
    with profiling_range(config.capture_range):
        for iteration_index in range(config.iterations):
            with profiling_range(f"iteration {iteration_index}"):
                last_outputs, pass_record_ids = _run_pass(
                    target=target,
                    records=_records_for_pass(
                        records=records,
                        data_source=data_source,
                    ),
                    prepared_records=prepared_records,
                    device=device,
                    profile_prepare=config.target.profile_prepare,
                    validate_output=False,
                    synchronize_each_iteration=(
                        config.cuda.synchronize_each_iteration
                    ),
                )
                last_record_ids = pass_record_ids
    synchronize_cuda(device, config.cuda.synchronize_after_capture)

    if not last_outputs:
        raise ValueError("Data source yielded no records.")

    output_summaries = [target.summarize(output) for output in last_outputs]
    manifest = build_manifest(
        config=config,
        run_id=run_id,
        run_dir=run_dir,
        data_source_description=data_source.describe(),
        record_ids=last_record_ids,
        output_summaries=output_summaries,
        argv=argv,
    )

    return manifest


def _run_pass(
    *,
    target: Any,
    records: Iterable[Any],
    prepared_records: list[Any] | None,
    device: torch.device,
    profile_prepare: bool,
    validate_output: bool,
    synchronize_each_iteration: bool,
) -> tuple[list[Any], list[str]]:
    outputs = []
    record_ids = []
    if profile_prepare:
        iterable = (
            (record, target.prepare(record, device=device))
            for record in records
        )
    else:
        if prepared_records is None:
            raise ValueError(
                "prepared_records is required when profile_prepare is false."
            )
        iterable = zip(records, prepared_records)

    for record, prepared in iterable:
        output = target.run(prepared)
        if validate_output:
            target.validate(output)
        outputs.append(output)
        record_ids.append(record.id)
        synchronize_cuda(device, synchronize_each_iteration)

    return outputs, record_ids


def _records_for_pass(*, records: list[Any] | None, data_source: Any) -> Iterable[Any]:
    if records is not None:
        return records

    lazy_records = data_source.iter_records()

    return lazy_records


def synchronize_cuda(device: torch.device, enabled: bool) -> None:
    """Synchronize CUDA work when requested and available.

    Args:
        device (torch.device): Device selected for the profile run.
        enabled (bool): Whether synchronization is enabled for this point.
    """
    if enabled and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def seed_everything(seed: int | None) -> None:
    """Seed supported random number generators for reproducible profiling.

    Args:
        seed (int | None): Seed value, or ``None`` to leave RNG state unchanged.
    """
    if seed is None:
        return

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        import numpy as np
    except ImportError:
        return

    np.random.seed(seed)


def build_manifest(
    *,
    config: ProfileConfig,
    run_id: str,
    run_dir: Path,
    data_source_description: Any,
    record_ids: list[str],
    output_summaries: list[dict[str, Any]],
    argv: list[str] | None,
) -> dict[str, Any]:
    """Build manifest metadata for a profiling run.

    Args:
        config (ProfileConfig): Parsed profile configuration.
        run_id (str): Stable run identifier.
        run_dir (Path): Directory where run outputs are expected.
        data_source_description (Any): Data-source-specific metadata.
        record_ids (list[str]): Selected record identifiers.
        output_summaries (list[dict[str, Any]]): Lightweight target summaries.
        argv (list[str] | None): Command line to record in the manifest.

    Returns:
        Manifest dictionary ready to serialize as YAML.
    """
    manifest = {
        "profile_name": config.profile_name,
        "run_id": run_id,
        "target": {
            "name": config.target.name,
            "import_path": config.target.import_path,
            "profile_prepare": config.target.profile_prepare,
        },
        "data_source": {
            "name": config.data_source.name,
            "description": data_source_description,
        },
        "record_ids": record_ids,
        "workload": {
            "warmup": config.warmup,
            "iterations": config.iterations,
            "record_loading": config.record_loading,
            "seed": config.seed,
            "validate_output": config.validate_output,
        },
        "capture_range": config.capture_range,
        "cuda": config.cuda.__dict__,
        "device": config.device,
        "git_commit": current_git_commit(),
        "command_line": " ".join(sys.argv if argv is None else argv),
        "expected_trace_path": str(run_dir / "trace.nsys-rep"),
        "output_summaries": output_summaries,
    }

    return manifest


def build_nsys_command(
    config: ProfileConfig,
    run_dir: Path,
    config_path: str | Path,
) -> str:
    """Build the Nsight Systems command for a profile run.

    Args:
        config (ProfileConfig): Parsed profile configuration.
        run_dir (Path): Directory where run outputs are expected.
        config_path (str | Path): Path to the YAML config file.

    Returns:
        Copy/paste-ready ``nsys profile`` command.
    """
    trace_output = run_dir / "trace"
    command = [
        "nsys",
        "profile",
        "-o",
        str(trace_output),
        "--trace=cuda,osrt,nvtx",
        "--sample=cpu",
        "--cpuctxsw=process-tree",
        "--capture-range=nvtx",
        "--capture-range-end=stop",
        f"--nvtx-capture={config.capture_range}@*",
        "-e",
        "NSYS_NVTX_PROFILER_REGISTER_ONLY=0",
        "--trace-fork-before-exec=true",
        "uv",
        "run",
        "python",
        "development/profiling/main.py",
        "--target",
        config.target.name,
        "--data-source",
        config.data_source.name,
        "--config",
        str(config_path),
        "--device",
        config.device,
        "--warmup",
        str(config.warmup),
        "--iterations",
        str(config.iterations),
        "--record-loading",
        config.record_loading,
        "--run-id",
        run_dir.name,
    ]
    if config.seed is not None:
        command.extend(["--seed", str(config.seed)])

    nsys_command = " \\\n  ".join(shlex.quote(part) for part in command)

    return nsys_command


def make_run_id() -> str:
    """Make a UTC timestamp run id.

    Returns:
        UTC timestamp formatted for use in local paths.
    """
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    return run_id


def current_git_commit() -> str | None:
    """Read the current git commit when available.

    Returns:
        Current git commit SHA, or ``None`` when git is unavailable.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    git_commit = result.stdout.strip()

    return git_commit


if __name__ == "__main__":
    raise SystemExit(main())
