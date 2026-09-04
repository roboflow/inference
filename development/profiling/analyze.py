from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import click
import yaml

from development.profiling.analysis import ProfileAnalysisError, build_profile_analysis
from development.profiling.nsys_stats import (
    NVTX_GPU_PROJECTION_TRACE_REPORT,
    NVTX_PUSHPOP_TRACE_REPORT,
    NsysStatsError,
    parse_nvtx_gpu_projection_trace,
    parse_nvtx_pushpop_trace,
    run_nsys_stats,
)


def main(argv: list[str] | None = None) -> int:
    """Run the profiling analysis CLI."""
    exit_code = cli.main(args=argv, standalone_mode=False)

    return exit_code


@click.command()
@click.option(
    "--run-dir",
    required=True,
    type=click.Path(
        exists=True,
        file_okay=False,
        path_type=Path,
    ),
    help="Profiling run directory containing manifest.yaml and trace.nsys-rep.",
)
def cli(run_dir: Path) -> int:
    """Export Nsight reports and write a stable analysis JSON file."""
    try:
        output_path = analyze_run(run_dir=run_dir)
    except (NsysStatsError, OSError, ProfileAnalysisError, ValueError) as error:
        raise click.ClickException(str(error)) from error

    click.echo(f"Wrote analysis: {output_path}")

    return 0


def analyze_run(
    *,
    run_dir: Path,
    nsys_executable: str = "nsys",
) -> Path:
    """Analyze one profiling run and write ``analysis.json``."""
    manifest_path = run_dir / "manifest.yaml"
    manifest = _load_manifest(manifest_path)
    trace_path = run_dir / "trace.nsys-rep"
    artifacts = run_nsys_stats(
        trace_path=trace_path,
        output_dir=run_dir / "stats",
        executable=nsys_executable,
    )
    host_ranges = parse_nvtx_pushpop_trace(
        artifacts.report_paths[NVTX_PUSHPOP_TRACE_REPORT]
    )
    gpu_projected_ranges = parse_nvtx_gpu_projection_trace(
        artifacts.report_paths[NVTX_GPU_PROJECTION_TRACE_REPORT]
    )
    analysis = build_profile_analysis(
        manifest=manifest,
        host_ranges=host_ranges,
        gpu_projected_ranges=gpu_projected_ranges,
        nsys_version=artifacts.nsys_version,
        run_dir=run_dir,
        trace_path=trace_path,
        report_paths=artifacts.report_paths,
    )

    output_path = run_dir / "analysis.json"
    output_path.write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return output_path


def _load_manifest(path: Path) -> Mapping[str, Any]:
    try:
        raw_manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise NsysStatsError(f"Profiling manifest does not exist: {path}") from error
    except yaml.YAMLError as error:
        raise NsysStatsError(f"Profiling manifest is invalid YAML: {path}") from error

    if not isinstance(raw_manifest, Mapping):
        raise NsysStatsError(f"Profiling manifest must contain a mapping: {path}")

    return raw_manifest


if __name__ == "__main__":
    raise SystemExit(main())
