from __future__ import annotations

import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from development.profiling.nsys_stats import (
    GpuProjectedRange,
    HostRange,
    get_nsys_version_warning,
)

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
    capture_range = manifest.get("capture_range")
    if not isinstance(capture_range, str):
        raise ProfileAnalysisError("Manifest must define a string capture_range.")

    (
        host_iterations,
        host_summaries,
        harness_range_keys,
        host_scope_paths,
    ) = _summarize_host_ranges(
        host_ranges,
        capture_range_name=capture_range,
    )
    gpu_iterations, gpu_summaries = _summarize_gpu_ranges(
        gpu_projected_ranges,
        host_iterations=host_iterations,
        harness_range_keys=harness_range_keys,
        host_scope_paths=host_scope_paths,
    )
    iterations = _build_iteration_analysis(host_iterations, gpu_iterations)
    warnings = _build_warnings(
        manifest=manifest,
        iterations=iterations,
        host_range_names={item["name"] for item in host_summaries},
        gpu_range_names={item["name"] for item in gpu_summaries},
        nsys_version=nsys_version,
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
    *,
    capture_range_name: str,
) -> tuple[
    dict[int, HostRange],
    list[dict[str, Any]],
    set[tuple[int, int, int]],
    dict[tuple[int, int, int], tuple[str, ...]],
]:
    capture_range = _find_capture_range(ranges, capture_range_name)
    iterations: dict[int, HostRange] = {}

    for item in ranges:
        iteration_index = _iteration_index(item.name)
        is_direct_capture_child = (
            item.process_id == capture_range.process_id
            and item.thread_id == capture_range.thread_id
            and item.parent_id == capture_range.range_id
        )
        if iteration_index is None or not is_direct_capture_child:
            continue
        if iteration_index in iterations:
            raise ProfileAnalysisError(
                f"Duplicate host range for iteration {iteration_index}."
            )
        iterations[iteration_index] = item

    capture_range_key = _range_key(capture_range)
    iteration_range_keys = {_range_key(item) for item in iterations.values()}
    harness_range_keys = {capture_range_key} | iteration_range_keys
    scope_paths = _build_scope_paths(
        ranges,
        harness_range_keys=harness_range_keys,
    )
    groups: dict[tuple[str, ...], list[HostRange]] = defaultdict(list)
    for item in ranges:
        if _range_key(item) in iteration_range_keys:
            continue
        groups[scope_paths[_range_key(item)]].append(item)

    summaries = []
    for scope_path, items in sorted(groups.items()):
        summaries.append(
            {
                "name": ".".join(scope_path),
                "instances": len(items),
                "inclusive": _duration_summary([item.duration_ns for item in items]),
                "exclusive": _duration_summary(
                    [item.exclusive_duration_ns for item in items]
                ),
                "child_total_ns": sum(item.child_duration_ns for item in items),
            }
        )

    return iterations, summaries, harness_range_keys, scope_paths


def _summarize_gpu_ranges(
    ranges: Sequence[GpuProjectedRange],
    *,
    host_iterations: Mapping[int, HostRange],
    harness_range_keys: set[tuple[int, int, int]],
    host_scope_paths: Mapping[tuple[int, int, int], tuple[str, ...]],
) -> tuple[dict[int, GpuProjectedRange], list[dict[str, Any]]]:
    iterations: dict[int, GpuProjectedRange] = {}
    iteration_indexes_by_range_key = {
        _range_key(item): index for index, item in host_iterations.items()
    }
    gpu_scope_paths = _build_scope_paths(
        ranges,
        harness_range_keys=harness_range_keys,
    )
    groups: dict[
        tuple[tuple[str, ...], str],
        list[GpuProjectedRange],
    ] = defaultdict(list)

    for item in ranges:
        iteration_index = iteration_indexes_by_range_key.get(_range_key(item))
        if iteration_index is not None:
            if iteration_index in iterations:
                raise ProfileAnalysisError(
                    f"Duplicate GPU projection for iteration {iteration_index}."
                )
            iterations[iteration_index] = item
            continue

        item_key = _range_key(item)
        scope_path = host_scope_paths.get(item_key)
        if scope_path is None:
            scope_path = gpu_scope_paths[item_key]
        groups[(scope_path, item.style)].append(item)

    summaries = []
    for (scope_path, style), items in sorted(groups.items()):
        summaries.append(
            {
                "name": ".".join(scope_path),
                "style": style,
                "instances": len(items),
                "projected": _duration_summary(
                    [item.projected_duration_ns for item in items]
                ),
                "host_total_ns": sum(item.original_duration_ns for item in items),
                "gpu_operation_count": sum(item.gpu_operation_count for item in items),
            }
        )

    return iterations, summaries


def _build_scope_paths(
    ranges: Sequence[HostRange | GpuProjectedRange],
    *,
    harness_range_keys: set[tuple[int, int, int]],
) -> dict[tuple[int, int, int], tuple[str, ...]]:
    ranges_by_key = {_range_key(item): item for item in ranges}
    scope_paths = {}

    for item in ranges:
        item_key = _range_key(item)
        current = item
        scope_path = []
        visited_keys = set()

        while _range_key(current) not in harness_range_keys:
            current_key = _range_key(current)
            if current_key in visited_keys:
                raise ProfileAnalysisError(
                    f"Cycle in NVTX range hierarchy at {current_key}."
                )
            visited_keys.add(current_key)
            scope_path.append(current.name)
            if current.parent_id is None:
                break
            parent_key = (
                current.process_id,
                current.thread_id,
                current.parent_id,
            )
            parent = ranges_by_key.get(parent_key)
            if parent is None:
                break
            current = parent

        scope_paths[item_key] = tuple(reversed(scope_path)) or (item.name,)

    return scope_paths


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
                        "projected_ns": gpu.projected_duration_ns,
                        "host_range_total_ns": gpu.original_duration_ns,
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
    nsys_version: str,
) -> list[str]:
    warnings = []
    version_warning = get_nsys_version_warning(nsys_version)
    if version_warning is not None:
        warnings.append(version_warning)
    workload = manifest.get("workload")
    expected_iterations = (
        workload.get("iterations") if isinstance(workload, Mapping) else None
    )
    if isinstance(expected_iterations, int):
        expected_indexes = set(range(expected_iterations))
        observed_indexes = {item["index"] for item in iterations}
        missing_indexes = sorted(expected_indexes - observed_indexes)
        unexpected_indexes = sorted(observed_indexes - expected_indexes)
        if missing_indexes:
            warnings.append(f"Missing expected iteration indexes: {missing_indexes}.")
        if unexpected_indexes:
            warnings.append(f"Unexpected iteration indexes: {unexpected_indexes}.")

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


def _find_capture_range(
    ranges: Sequence[HostRange],
    capture_range_name: str,
) -> HostRange:
    candidates = [
        item
        for item in ranges
        if item.name == capture_range_name and item.parent_id is None
    ]
    if len(candidates) != 1:
        raise ProfileAnalysisError(
            "Expected exactly one top-level host capture range named "
            f"{capture_range_name!r}, but found {len(candidates)}."
        )

    return candidates[0]


def _range_key(item: HostRange | GpuProjectedRange) -> tuple[int, int, int]:
    return item.process_id, item.thread_id, item.range_id


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
