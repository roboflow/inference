import subprocess
from pathlib import Path

import pytest

from development.profiling.nsys_stats import (
    NsysStatsError,
    build_nsys_stats_command,
    parse_nvtx_gpu_projection_trace,
    parse_nvtx_pushpop_trace,
    run_nsys_stats,
)

FIXTURES = Path(__file__).parent / "fixtures" / "nsys_2025_1"


def test_parse_nvtx_pushpop_trace_from_real_nsys_output():
    ranges = parse_nvtx_pushpop_trace(FIXTURES / "nvtx_pushpop_trace.csv")

    assert len(ranges) == 11
    assert ranges[0].name == "profile-target"
    assert ranges[0].raw_name == ":profile-target"
    assert ranges[0].duration_ns == 700839
    assert ranges[0].exclusive_duration_ns == 130796
    assert ranges[0].parent_id is None

    second_iteration = next(item for item in ranges if item.name == "iteration 1")
    assert second_iteration.level == 1
    assert second_iteration.parent_id == 1
    assert second_iteration.child_count == 4


def test_parse_nvtx_gpu_projection_trace_from_real_nsys_output():
    ranges = parse_nvtx_gpu_projection_trace(FIXTURES / "nvtx_gpu_proj_trace.csv")

    assert len(ranges) == 11
    capture = next(item for item in ranges if item.name == "profile-target")
    assert capture.projected_duration_ns == 336423
    assert capture.original_duration_ns == 700839
    assert capture.gpu_operation_count == 8
    assert capture.range_id == 1
    assert capture.parent_id is None

    second_iteration = next(item for item in ranges if item.name == "iteration 1")
    assert second_iteration.level == 1
    assert second_iteration.parent_id == 1
    assert second_iteration.range_id == 7


def test_gpu_projection_parser_accepts_header_only_report(tmp_path):
    report = tmp_path / "gpu.csv"
    header = (
        (FIXTURES / "nvtx_gpu_proj_trace.csv")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    report.write_text(f"{header}\n", encoding="utf-8")

    assert parse_nvtx_gpu_projection_trace(report) == []


def test_gpu_projection_parser_accepts_empty_report(tmp_path):
    report = tmp_path / "gpu.csv"
    report.touch()

    assert parse_nvtx_gpu_projection_trace(report) == []


def test_host_parser_rejects_header_only_report(tmp_path):
    report = tmp_path / "host.csv"
    header = (
        (FIXTURES / "nvtx_pushpop_trace.csv")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    report.write_text(f"{header}\n", encoding="utf-8")

    with pytest.raises(NsysStatsError, match="contains no rows"):
        parse_nvtx_pushpop_trace(report)


def test_host_parser_rejects_empty_report(tmp_path):
    report = tmp_path / "host.csv"
    report.touch()

    with pytest.raises(NsysStatsError, match="contains no rows"):
        parse_nvtx_pushpop_trace(report)


def test_parser_rejects_report_without_required_columns(tmp_path):
    report = tmp_path / "invalid.csv"
    report.write_text("Name,Duration (ns)\n:range,10\n", encoding="utf-8")

    with pytest.raises(NsysStatsError, match="missing required columns"):
        parse_nvtx_pushpop_trace(report)

    with pytest.raises(NsysStatsError, match="missing required columns"):
        parse_nvtx_gpu_projection_trace(report)


def test_build_nsys_stats_command_uses_argument_list(tmp_path):
    command = build_nsys_stats_command(
        trace_path=tmp_path / "trace with spaces.nsys-rep",
        output_base=tmp_path / "stats" / "nsys",
        executable="/opt/nvidia/nsys",
    )

    assert command == [
        "/opt/nvidia/nsys",
        "stats",
        "--report",
        "nvtx_pushpop_trace",
        "--report",
        "nvtx_gpu_proj_trace",
        "--format",
        "csv",
        "--output",
        str(tmp_path / "stats" / "nsys"),
        "--force-overwrite=true",
        str(tmp_path / "trace with spaces.nsys-rep"),
    ]


def test_run_nsys_stats_returns_expected_artifacts(tmp_path, monkeypatch):
    trace_path = tmp_path / "trace.nsys-rep"
    trace_path.touch()
    output_dir = tmp_path / "stats"

    monkeypatch.setattr(
        "development.profiling.nsys_stats.shutil.which",
        lambda executable: "/usr/bin/nsys",
    )

    def fake_run(command, **kwargs):
        if command[-1] == "--version":
            return subprocess.CompletedProcess(
                command,
                returncode=0,
                stdout="NVIDIA Nsight Systems version 2025.1.3\n",
                stderr="",
            )
        output_base = Path(command[command.index("--output") + 1])
        for report in ("nvtx_pushpop_trace", "nvtx_gpu_proj_trace"):
            output_base.with_name(f"{output_base.name}_{report}.csv").touch()
        return subprocess.CompletedProcess(
            command,
            returncode=0,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(
        "development.profiling.nsys_stats.subprocess.run",
        fake_run,
    )

    artifacts = run_nsys_stats(trace_path=trace_path, output_dir=output_dir)

    assert artifacts.nsys_version == "NVIDIA Nsight Systems version 2025.1.3"
    assert artifacts.report_paths["nvtx_pushpop_trace"] == (
        output_dir / "nsys_nvtx_pushpop_trace.csv"
    )


def test_run_nsys_stats_rejects_missing_executable(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "development.profiling.nsys_stats.shutil.which",
        lambda executable: None,
    )

    with pytest.raises(NsysStatsError, match="executable not found"):
        run_nsys_stats(
            trace_path=tmp_path / "trace.nsys-rep",
            output_dir=tmp_path / "stats",
        )
