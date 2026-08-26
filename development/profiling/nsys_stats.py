from __future__ import annotations

import csv
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

NVTX_PUSHPOP_TRACE_REPORT = "nvtx_pushpop_trace"
NVTX_GPU_PROJECTION_REPORT = "nvtx_gpu_proj_sum"
DEFAULT_REPORTS = (
    NVTX_PUSHPOP_TRACE_REPORT,
    NVTX_GPU_PROJECTION_REPORT,
)


class NsysStatsError(RuntimeError):
    """Raised when Nsight statistics cannot be exported or parsed."""


@dataclass(frozen=True)
class HostRange:
    """One NVTX push/pop range recorded on the host timeline."""

    name: str
    raw_name: str
    start_ns: int
    end_ns: int
    duration_ns: int
    child_duration_ns: int
    exclusive_duration_ns: int
    process_id: int
    thread_id: int
    level: int
    child_count: int
    range_id: int
    parent_id: int | None


@dataclass(frozen=True)
class GpuProjectedRange:
    """Aggregated GPU work projected from an NVTX range."""

    name: str
    raw_name: str
    style: str
    projected_total_ns: int
    host_total_ns: int
    instances: int
    projected_average_ns: float
    projected_median_ns: float
    projected_minimum_ns: int
    projected_maximum_ns: int
    projected_stddev_ns: float
    gpu_operation_count: int
    average_gpu_operations: float
    average_level: float
    average_child_count: float


@dataclass(frozen=True)
class NsysStatsArtifacts:
    """Files and tool metadata produced by an ``nsys stats`` export."""

    nsys_version: str
    report_paths: Mapping[str, Path]


def build_nsys_stats_command(
    *,
    trace_path: Path,
    output_base: Path,
    executable: str,
    reports: Sequence[str] = DEFAULT_REPORTS,
) -> list[str]:
    """Build an argument-safe command for exporting Nsight report CSV files."""
    command = [executable, "stats"]
    for report in reports:
        command.extend(["--report", report])
    command.extend(
        [
            "--format",
            "csv",
            "--output",
            str(output_base),
            "--force-overwrite=true",
            str(trace_path),
        ]
    )

    return command


def run_nsys_stats(
    *,
    trace_path: Path,
    output_dir: Path,
    executable: str = "nsys",
    reports: Sequence[str] = DEFAULT_REPORTS,
) -> NsysStatsArtifacts:
    """Export supported CSV reports from an existing Nsight trace."""
    resolved_executable = shutil.which(executable)
    if resolved_executable is None:
        raise NsysStatsError(
            f"Nsight Systems executable not found: {executable!r}. "
            "Install Nsight Systems and ensure `nsys` is on PATH."
        )
    if not trace_path.is_file():
        raise NsysStatsError(f"Nsight trace does not exist: {trace_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_base = output_dir / "nsys"
    command = build_nsys_stats_command(
        trace_path=trace_path,
        output_base=output_base,
        executable=resolved_executable,
        reports=reports,
    )
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        details = (result.stderr or result.stdout).strip()
        raise NsysStatsError(
            f"`nsys stats` failed with exit code {result.returncode}: {details}"
        )

    report_paths = {
        report: output_dir / f"{output_base.name}_{report}.csv" for report in reports
    }
    missing_reports = [
        str(path) for path in report_paths.values() if not path.is_file()
    ]
    if missing_reports:
        raise NsysStatsError(
            "`nsys stats` did not create expected report files: "
            f"{', '.join(missing_reports)}"
        )

    return NsysStatsArtifacts(
        nsys_version=_read_nsys_version(resolved_executable),
        report_paths=report_paths,
    )


def parse_nvtx_pushpop_trace(path: Path) -> list[HostRange]:
    """Parse the ``nvtx_pushpop_trace`` CSV report."""
    rows = _read_csv_rows(
        path,
        required_columns=(
            "Start (ns)",
            "End (ns)",
            "Duration (ns)",
            "DurChild (ns)",
            "DurNonChild (ns)",
            "Name",
            "PID",
            "TID",
            "Lvl",
            "NumChild",
            "RangeId",
            "ParentId",
        ),
    )

    return [
        HostRange(
            name=_normalize_range_name(row["Name"]),
            raw_name=row["Name"],
            start_ns=_parse_int(row, "Start (ns)", path),
            end_ns=_parse_int(row, "End (ns)", path),
            duration_ns=_parse_int(row, "Duration (ns)", path),
            child_duration_ns=_parse_int(row, "DurChild (ns)", path),
            exclusive_duration_ns=_parse_int(row, "DurNonChild (ns)", path),
            process_id=_parse_int(row, "PID", path),
            thread_id=_parse_int(row, "TID", path),
            level=_parse_int(row, "Lvl", path),
            child_count=_parse_int(row, "NumChild", path),
            range_id=_parse_int(row, "RangeId", path),
            parent_id=_parse_optional_int(row, "ParentId", path),
        )
        for row in rows
    ]


def parse_nvtx_gpu_projection(path: Path) -> list[GpuProjectedRange]:
    """Parse the ``nvtx_gpu_proj_sum`` CSV report."""
    rows = _read_csv_rows(
        path,
        required_columns=(
            "Range",
            "Style",
            "Total Proj Time (ns)",
            "Total Range Time (ns)",
            "Range Instances",
            "Proj Avg (ns)",
            "Proj Med (ns)",
            "Proj Min (ns)",
            "Proj Max (ns)",
            "Proj StdDev (ns)",
            "Total GPU Ops",
            "Avg GPU Ops",
            "Avg Range Lvl",
            "Avg Num Child",
        ),
    )

    return [
        GpuProjectedRange(
            name=_normalize_range_name(row["Range"]),
            raw_name=row["Range"],
            style=row["Style"],
            projected_total_ns=_parse_int(row, "Total Proj Time (ns)", path),
            host_total_ns=_parse_int(row, "Total Range Time (ns)", path),
            instances=_parse_int(row, "Range Instances", path),
            projected_average_ns=_parse_float(row, "Proj Avg (ns)", path),
            projected_median_ns=_parse_float(row, "Proj Med (ns)", path),
            projected_minimum_ns=_parse_int(row, "Proj Min (ns)", path),
            projected_maximum_ns=_parse_int(row, "Proj Max (ns)", path),
            projected_stddev_ns=_parse_float(row, "Proj StdDev (ns)", path),
            gpu_operation_count=_parse_int(row, "Total GPU Ops", path),
            average_gpu_operations=_parse_float(row, "Avg GPU Ops", path),
            average_level=_parse_float(row, "Avg Range Lvl", path),
            average_child_count=_parse_float(row, "Avg Num Child", path),
        )
        for row in rows
    ]


def _read_nsys_version(executable: str) -> str:
    result = subprocess.run(
        [executable, "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return "unknown"

    return result.stdout.strip() or "unknown"


def _read_csv_rows(
    path: Path,
    *,
    required_columns: Sequence[str],
) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            columns = set(reader.fieldnames or ())
            missing_columns = [
                column for column in required_columns if column not in columns
            ]
            if missing_columns:
                raise NsysStatsError(
                    f"Nsight report {path} is missing required columns: "
                    f"{', '.join(missing_columns)}"
                )
            rows = list(reader)
    except OSError as error:
        raise NsysStatsError(f"Could not read Nsight report {path}: {error}") from error

    if not rows:
        raise NsysStatsError(f"Nsight report contains no rows: {path}")

    return rows


def _normalize_range_name(value: str) -> str:
    if value.startswith(":"):
        return value[1:]

    return value


def _parse_int(row: Mapping[str, str], column: str, path: Path) -> int:
    try:
        return int(row[column])
    except (KeyError, TypeError, ValueError) as error:
        raise NsysStatsError(
            f"Invalid integer in {path} column {column!r}: {row.get(column)!r}"
        ) from error


def _parse_optional_int(
    row: Mapping[str, str],
    column: str,
    path: Path,
) -> int | None:
    value = row.get(column)
    if value in (None, ""):
        return None

    return _parse_int(row, column, path)


def _parse_float(row: Mapping[str, str], column: str, path: Path) -> float:
    try:
        return float(row[column])
    except (KeyError, TypeError, ValueError) as error:
        raise NsysStatsError(
            f"Invalid number in {path} column {column!r}: {row.get(column)!r}"
        ) from error
