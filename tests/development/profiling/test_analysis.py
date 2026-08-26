import json
import shutil
from pathlib import Path

import yaml

from development.profiling.analysis import build_profile_analysis
from development.profiling.analyze import analyze_run
from development.profiling.nsys_stats import (
    NVTX_GPU_PROJECTION_REPORT,
    NVTX_PUSHPOP_TRACE_REPORT,
    NsysStatsArtifacts,
    parse_nvtx_gpu_projection,
    parse_nvtx_pushpop_trace,
)

FIXTURES = Path(__file__).parent / "fixtures" / "nsys_2025_1"


def test_build_profile_analysis_separates_host_and_gpu_timings(tmp_path):
    host_ranges = parse_nvtx_pushpop_trace(FIXTURES / "nvtx_pushpop_trace.csv")
    gpu_ranges = parse_nvtx_gpu_projection(FIXTURES / "nvtx_gpu_proj_sum.csv")
    manifest = _manifest()

    analysis = build_profile_analysis(
        manifest=manifest,
        host_ranges=host_ranges,
        gpu_projected_ranges=gpu_ranges,
        nsys_version="NVIDIA Nsight Systems version 2025.1.3",
        run_dir=tmp_path,
        trace_path=tmp_path / "trace.nsys-rep",
        report_paths={
            NVTX_PUSHPOP_TRACE_REPORT: tmp_path
            / "stats"
            / "nsys_nvtx_pushpop_trace.csv",
            NVTX_GPU_PROJECTION_REPORT: tmp_path
            / "stats"
            / "nsys_nvtx_gpu_proj_sum.csv",
        },
    )

    assert analysis["schema_version"] == 1
    assert analysis["run"]["run_id"] == "jarvis-smoke"
    assert analysis["run"]["record_count"] == 2
    assert analysis["run"]["cuda"] == {"synchronize_each_iteration": False}
    assert analysis["warnings"] == []

    host_capture = _range_by_name(analysis["host_ranges"], "profile-target")
    assert host_capture["inclusive"]["total_ns"] == 700839
    assert host_capture["exclusive"]["total_ns"] == 130796

    host_multiply = _range_by_name(analysis["host_ranges"], "smoke multiply")
    assert host_multiply["instances"] == 4
    assert host_multiply["inclusive"]["total_ns"] == 277292
    assert host_multiply["inclusive"]["stddev_ns"] > 0

    gpu_capture = _range_by_name(analysis["gpu_projected_ranges"], "profile-target")
    assert gpu_capture["projected"]["total_ns"] == 336423
    assert gpu_capture["projected"]["stddev_ns"] == 0.0
    assert gpu_capture["gpu_operation_count"] == 8

    assert analysis["iterations"] == [
        {
            "index": 0,
            "host": {
                "inclusive_ns": 431993,
                "exclusive_ns": 110648,
                "child_total_ns": 321345,
            },
            "gpu_projection": {
                "projected_ns": 191620,
                "host_range_total_ns": 431993,
                "gpu_operation_count": 4,
            },
        },
        {
            "index": 1,
            "host": {
                "inclusive_ns": 138050,
                "exclusive_ns": 50395,
                "child_total_ns": 87655,
            },
            "gpu_projection": {
                "projected_ns": 99714,
                "host_range_total_ns": 138050,
                "gpu_operation_count": 4,
            },
        },
    ]
    assert analysis["iteration_summary"]["host_inclusive"]["total_ns"] == 570043
    assert analysis["iteration_summary"]["gpu_projected"]["total_ns"] == 291334


def test_build_profile_analysis_warns_when_manifest_iteration_count_differs(
    tmp_path,
):
    host_ranges = parse_nvtx_pushpop_trace(FIXTURES / "nvtx_pushpop_trace.csv")
    gpu_ranges = parse_nvtx_gpu_projection(FIXTURES / "nvtx_gpu_proj_sum.csv")
    manifest = _manifest()
    manifest["workload"]["iterations"] = 3

    analysis = build_profile_analysis(
        manifest=manifest,
        host_ranges=host_ranges,
        gpu_projected_ranges=gpu_ranges,
        nsys_version="version",
        run_dir=tmp_path,
        trace_path=tmp_path / "trace.nsys-rep",
        report_paths={},
    )

    assert analysis["warnings"] == [
        "Manifest expected 3 iterations, but analysis found 2."
    ]


def test_analyze_run_writes_stable_json(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    stats_dir = run_dir / "stats"
    stats_dir.mkdir(parents=True)
    trace_path = run_dir / "trace.nsys-rep"
    trace_path.touch()
    (run_dir / "manifest.yaml").write_text(
        yaml.safe_dump(_manifest()),
        encoding="utf-8",
    )

    report_paths = {
        NVTX_PUSHPOP_TRACE_REPORT: stats_dir / "nsys_nvtx_pushpop_trace.csv",
        NVTX_GPU_PROJECTION_REPORT: stats_dir / "nsys_nvtx_gpu_proj_sum.csv",
    }
    shutil.copy(
        FIXTURES / "nvtx_pushpop_trace.csv", report_paths[NVTX_PUSHPOP_TRACE_REPORT]
    )
    shutil.copy(
        FIXTURES / "nvtx_gpu_proj_sum.csv", report_paths[NVTX_GPU_PROJECTION_REPORT]
    )

    monkeypatch.setattr(
        "development.profiling.analyze.run_nsys_stats",
        lambda **kwargs: NsysStatsArtifacts(
            nsys_version="NVIDIA Nsight Systems version 2025.1.3",
            report_paths=report_paths,
        ),
    )

    output_path = analyze_run(run_dir=run_dir)

    assert output_path == run_dir / "analysis.json"
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["provenance"]["manifest_path"] == "manifest.yaml"
    assert saved["provenance"]["trace_path"] == "trace.nsys-rep"
    assert saved["provenance"]["reports"][NVTX_PUSHPOP_TRACE_REPORT] == (
        "stats/nsys_nvtx_pushpop_trace.csv"
    )


def _manifest():
    return {
        "profile_name": "smoke-tensor",
        "run_id": "jarvis-smoke",
        "device": "cuda:0",
        "capture_range": "profile-target",
        "git_commit": "abc123",
        "record_ids": ["dummy-0", "dummy-1"],
        "workload": {
            "warmup": 1,
            "iterations": 2,
            "record_loading": "eager",
        },
        "cuda": {"synchronize_each_iteration": False},
    }


def _range_by_name(items, name):
    return next(item for item in items if item["name"] == name)
