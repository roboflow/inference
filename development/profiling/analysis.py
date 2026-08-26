from __future__ import annotations

import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from development.profiling.nsys_stats import GpuProjectedRange, HostRange

ANALYSIS_SCHEMA_VERSION = 1
ITERATION_RANGE_PATTERN = re.compile(r"^iteration (?P<index>\d+)$")


class ProfileAnalysisError(RuntimeError):
    """Raised when parsed profiling data cannot form a valid analysis."""


def build_profile_analysis(
    *,
    manifest: Mapping[str, Any],
    host_ranges: Sequence[HostRange],
    gpu_projected_ranges: Sequence[GpuProjectedRange],
    nsys_version: str,
    run_dir: Path,
    trace_path: Path,
    report_paths: Mapping[str, Path],
) -> dict[str, Any]:
    """Build a compact, manifest-linked analysis from parsed Nsight reports."""
    host_iterations, host_summaries = _summarize_host_ranges(host_ranges)
    gpu_iterations, gpu_summaries = _summarize_gpu_ranges(gpu_projected_ranges)
    iterations = _build_iteration_analysis(host_iterations, gpu_iterations)
    warnings = _build_warnings(
        manifest=manifest,
        iterations=iterations,
        host_range_names={item["name"] for item in host_summaries},
        gpu_range_names={item["name"] for item in gpu_summaries},
    )

    workload = manifest.get("workload")
    if not isinstance(workload, Mapping):
        workload = {}
    cuda = manifest.get("cuda")
    if not isinstance(cuda, Mapping):
        cuda = {}

    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "run": {
            "profile_name": manifest.get("profile_name"),
            "run_id": manifest.get("run_id"),
            "device": manifest.get("device"),
            "capture_range": manifest.get("capture_range"),
            "git_commit": manifest.get("git_commit"),
            "record_count": len(manifest.get("record_ids") or []),
            "workload": dict(workload),
            "cuda": dict(cuda),
        },
        "provenance": {
            "nsys_version": nsys_version,
            "manifest_path": _portable_path(run_dir / "manifest.yaml", run_dir),
            "trace_path": _portable_path(trace_path, run_dir),
            "reports": {
                name: _portable_path(path, run_dir)
                for name, path in sorted(report_paths.items())
            },
        },
        "host_ranges": host_summaries,
        "gpu_projected_ranges": gpu_summaries,
        "iterations": iterations,
        "iteration_summary": _summarize_iterations(iterations),
        "warnings": warnings,
    }


def _summarize_host_ranges(
    ranges: Sequence[HostRange],
) -> tuple[dict[int, HostRange], list[dict[str, Any]]]:
    iterations: dict[int, HostRange] = {}
    groups: dict[str, list[HostRange]] = defaultdict(list)

    for item in ranges:
        iteration_index = _iteration_index(item.name)
        if iteration_index is None:
            groups[item.name].append(item)
            continue
        if iteration_index in iterations:
            raise ProfileAnalysisError(
                f"Duplicate host range for iteration {iteration_index}."
            )
        iterations[iteration_index] = item

    summaries = []
    for name, items in sorted(groups.items()):
        summaries.append(
            {
                "name": name,
                "instances": len(items),
                "inclusive": _duration_summary([item.duration_ns for item in items]),
                "exclusive": _duration_summary(
                    [item.exclusive_duration_ns for item in items]
                ),
                "child_total_ns": sum(item.child_duration_ns for item in items),
            }
        )

    return iterations, summaries


def _summarize_gpu_ranges(
    ranges: Sequence[GpuProjectedRange],
) -> tuple[dict[int, GpuProjectedRange], list[dict[str, Any]]]:
    iterations: dict[int, GpuProjectedRange] = {}
    summaries = []

    for item in sorted(ranges, key=lambda value: value.name):
        iteration_index = _iteration_index(item.name)
        if iteration_index is not None:
            if iteration_index in iterations:
                raise ProfileAnalysisError(
                    f"Duplicate GPU projection for iteration {iteration_index}."
                )
            iterations[iteration_index] = item
            continue

        summaries.append(
            {
                "name": item.name,
                "style": item.style,
                "instances": item.instances,
                "projected": {
                    "total_ns": item.projected_total_ns,
                    "mean_ns": item.projected_average_ns,
                    "median_ns": item.projected_median_ns,
                    "minimum_ns": item.projected_minimum_ns,
                    "maximum_ns": item.projected_maximum_ns,
                    "stddev_ns": item.projected_stddev_ns,
                },
                "host_total_ns": item.host_total_ns,
                "gpu_operation_count": item.gpu_operation_count,
            }
        )

    return iterations, summaries


def _build_iteration_analysis(
    host_iterations: Mapping[int, HostRange],
    gpu_iterations: Mapping[int, GpuProjectedRange],
) -> list[dict[str, Any]]:
    iteration_indexes = sorted(set(host_iterations) | set(gpu_iterations))
    result = []

    for index in iteration_indexes:
        host = host_iterations.get(index)
        gpu = gpu_iterations.get(index)
        result.append(
            {
                "index": index,
                "host": (
                    {
                        "inclusive_ns": host.duration_ns,
                        "exclusive_ns": host.exclusive_duration_ns,
                        "child_total_ns": host.child_duration_ns,
                    }
                    if host is not None
                    else None
                ),
                "gpu_projection": (
                    {
                        "projected_ns": gpu.projected_total_ns,
                        "host_range_total_ns": gpu.host_total_ns,
                        "gpu_operation_count": gpu.gpu_operation_count,
                    }
                    if gpu is not None
                    else None
                ),
            }
        )

    return result


def _build_warnings(
    *,
    manifest: Mapping[str, Any],
    iterations: Sequence[Mapping[str, Any]],
    host_range_names: set[str],
    gpu_range_names: set[str],
) -> list[str]:
    warnings = []
    workload = manifest.get("workload")
    expected_iterations = (
        workload.get("iterations") if isinstance(workload, Mapping) else None
    )
    if isinstance(expected_iterations, int) and expected_iterations != len(iterations):
        warnings.append(
            f"Manifest expected {expected_iterations} iterations, but analysis found "
            f"{len(iterations)}."
        )

    missing_host = [item["index"] for item in iterations if item["host"] is None]
    if missing_host:
        warnings.append(f"Missing host ranges for iterations: {missing_host}.")

    missing_gpu = [
        item["index"] for item in iterations if item["gpu_projection"] is None
    ]
    if missing_gpu:
        warnings.append(f"Missing GPU projections for iterations: {missing_gpu}.")

    capture_range = manifest.get("capture_range")
    if isinstance(capture_range, str) and capture_range not in host_range_names:
        warnings.append(
            f"Host report does not contain capture range {capture_range!r}."
        )
    if isinstance(capture_range, str) and capture_range not in gpu_range_names:
        warnings.append(
            f"GPU projection report does not contain capture range {capture_range!r}."
        )

    return warnings


def _summarize_iterations(
    iterations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    host_iterations = [item["host"] for item in iterations if item["host"] is not None]
    gpu_iterations = [
        item["gpu_projection"]
        for item in iterations
        if item["gpu_projection"] is not None
    ]

    return {
        "iterations": len(iterations),
        "host_iterations": len(host_iterations),
        "gpu_projected_iterations": len(gpu_iterations),
        "host_inclusive": _optional_duration_summary(
            [item["inclusive_ns"] for item in host_iterations]
        ),
        "host_exclusive": _optional_duration_summary(
            [item["exclusive_ns"] for item in host_iterations]
        ),
        "gpu_projected": _optional_duration_summary(
            [item["projected_ns"] for item in gpu_iterations]
        ),
    }


def _duration_summary(values: Sequence[int]) -> dict[str, int | float]:
    return {
        "total_ns": sum(values),
        "mean_ns": statistics.fmean(values),
        "median_ns": statistics.median(values),
        "minimum_ns": min(values),
        "maximum_ns": max(values),
        "stddev_ns": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _optional_duration_summary(
    values: Sequence[int],
) -> dict[str, int | float] | None:
    if not values:
        return None

    return _duration_summary(values)


def _iteration_index(name: str) -> int | None:
    match = ITERATION_RANGE_PATTERN.fullmatch(name)
    if match is None:
        return None

    return int(match.group("index"))


def _portable_path(path: Path, run_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(run_dir.resolve()))
    except ValueError:
        return str(path)
