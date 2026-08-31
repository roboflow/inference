import json
import shutil
from dataclasses import replace
from pathlib import Path

import yaml

from development.profiling.analysis import build_profile_analysis
from development.profiling.analyze import analyze_run
from development.profiling.nsys_stats import (
    NVTX_GPU_PROJECTION_TRACE_REPORT,
    NVTX_PUSHPOP_TRACE_REPORT,
    NsysStatsArtifacts,
    parse_nvtx_gpu_projection_trace,
    parse_nvtx_pushpop_trace,
)

FIXTURES = Path(__file__).parent / "fixtures" / "nsys_2025_1"
VALIDATED_NSYS_VERSION = "NVIDIA Nsight Systems version 2025.1.3"


def test_build_profile_analysis_separates_host_and_gpu_timings(tmp_path):
    host_ranges = parse_nvtx_pushpop_trace(FIXTURES / "nvtx_pushpop_trace.csv")
    gpu_ranges = parse_nvtx_gpu_projection_trace(FIXTURES / "nvtx_gpu_proj_trace.csv")
    manifest = _manifest()

    analysis = build_profile_analysis(
        manifest=manifest,
        host_ranges=host_ranges,
        gpu_projected_ranges=gpu_ranges,
        nsys_version=VALIDATED_NSYS_VERSION,
        run_dir=tmp_path,
        trace_path=tmp_path / "trace.nsys-rep",
        report_paths={
            NVTX_PUSHPOP_TRACE_REPORT: tmp_path
            / "stats"
            / "nsys_nvtx_pushpop_trace.csv",
            NVTX_GPU_PROJECTION_TRACE_REPORT: tmp_path
            / "stats"
            / "nsys_nvtx_gpu_proj_trace.csv",
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


def test_build_profile_analysis_preserves_host_data_without_gpu_ranges(tmp_path):
    host_ranges = parse_nvtx_pushpop_trace(FIXTURES / "nvtx_pushpop_trace.csv")

    analysis = build_profile_analysis(
        manifest=_manifest(),
        host_ranges=host_ranges,
        gpu_projected_ranges=[],
        nsys_version=VALIDATED_NSYS_VERSION,
        run_dir=tmp_path,
        trace_path=tmp_path / "trace.nsys-rep",
        report_paths={},
    )

    assert analysis["gpu_projected_ranges"] == []
    assert all(item["gpu_projection"] is None for item in analysis["iterations"])
    assert analysis["iteration_summary"]["gpu_projected_iterations"] == 0
    assert analysis["iteration_summary"]["gpu_projected"] is None
    assert analysis["warnings"] == [
        "Missing GPU projections for iterations: [0, 1].",
        "GPU projection report does not contain capture range 'profile-target'.",
    ]


def test_build_profile_analysis_warns_for_unvalidated_nsys_version(tmp_path):
    analysis = build_profile_analysis(
        manifest=_manifest(),
        host_ranges=parse_nvtx_pushpop_trace(FIXTURES / "nvtx_pushpop_trace.csv"),
        gpu_projected_ranges=parse_nvtx_gpu_projection_trace(
            FIXTURES / "nvtx_gpu_proj_trace.csv"
        ),
        nsys_version="NVIDIA Nsight Systems version 2026.2.0",
        run_dir=tmp_path,
        trace_path=tmp_path / "trace.nsys-rep",
        report_paths={},
    )

    assert analysis["warnings"] == [
        "NVIDIA Nsight Systems version 2026.2.0 has not been validated; report "
        "parsing is validated against Nsight Systems 2025.1."
    ]


def test_build_profile_analysis_warns_when_manifest_iteration_count_differs(
    tmp_path,
):
    host_ranges = parse_nvtx_pushpop_trace(FIXTURES / "nvtx_pushpop_trace.csv")
    gpu_ranges = parse_nvtx_gpu_projection_trace(FIXTURES / "nvtx_gpu_proj_trace.csv")
    manifest = _manifest()
    manifest["workload"]["iterations"] = 3

    analysis = build_profile_analysis(
        manifest=manifest,
        host_ranges=host_ranges,
        gpu_projected_ranges=gpu_ranges,
        nsys_version=VALIDATED_NSYS_VERSION,
        run_dir=tmp_path,
        trace_path=tmp_path / "trace.nsys-rep",
        report_paths={},
    )

    assert analysis["warnings"] == ["Missing expected iteration indexes: [2]."]


def test_build_profile_analysis_uses_hierarchy_for_harness_iterations(tmp_path):
    host_ranges = [
        (
            replace(item, name="iteration 0", raw_name=":iteration 0")
            if item.range_id == 3
            else item
        )
        for item in parse_nvtx_pushpop_trace(FIXTURES / "nvtx_pushpop_trace.csv")
    ]
    gpu_ranges = [
        (
            replace(item, name="iteration 0", raw_name=":iteration 0")
            if item.range_id == 3
            else item
        )
        for item in parse_nvtx_gpu_projection_trace(
            FIXTURES / "nvtx_gpu_proj_trace.csv"
        )
    ]

    analysis = build_profile_analysis(
        manifest=_manifest(),
        host_ranges=host_ranges,
        gpu_projected_ranges=gpu_ranges,
        nsys_version=VALIDATED_NSYS_VERSION,
        run_dir=tmp_path,
        trace_path=tmp_path / "trace.nsys-rep",
        report_paths={},
    )

    assert [item["index"] for item in analysis["iterations"]] == [0, 1]
    assert analysis["iterations"][0]["host"]["inclusive_ns"] == 431993
    assert analysis["iterations"][0]["gpu_projection"]["projected_ns"] == 191620
    assert _range_by_name(analysis["host_ranges"], "iteration 0")["instances"] == 1
    assert (
        _range_by_name(analysis["gpu_projected_ranges"], "iteration 0")["projected"][
            "total_ns"
        ]
        == 2112
    )


def test_build_profile_analysis_scopes_range_names_below_iterations(tmp_path):
    scope_updates = {
        3: ("preprocessing", 2, 2),
        4: ("resize", 3, 3),
        5: ("postprocessing", 2, 2),
        6: ("resize", 5, 3),
        8: ("preprocessing", 7, 2),
        9: ("resize", 8, 3),
        10: ("postprocessing", 7, 2),
        11: ("resize", 10, 3),
    }

    def apply_scope(item):
        update = scope_updates.get(item.range_id)
        if update is None:
            return item
        name, parent_id, level = update
        return replace(
            item,
            name=name,
            raw_name=f":{name}",
            parent_id=parent_id,
            level=level,
        )

    host_ranges = [
        apply_scope(item)
        for item in parse_nvtx_pushpop_trace(FIXTURES / "nvtx_pushpop_trace.csv")
    ]
    gpu_ranges = [
        apply_scope(item)
        for item in parse_nvtx_gpu_projection_trace(
            FIXTURES / "nvtx_gpu_proj_trace.csv"
        )
        if item.range_id not in {3, 5, 8, 10}
    ]

    analysis = build_profile_analysis(
        manifest=_manifest(),
        host_ranges=host_ranges,
        gpu_projected_ranges=gpu_ranges,
        nsys_version=VALIDATED_NSYS_VERSION,
        run_dir=tmp_path,
        trace_path=tmp_path / "trace.nsys-rep",
        report_paths={},
    )

    host_preprocessing = _range_by_name(analysis["host_ranges"], "preprocessing.resize")
    host_postprocessing = _range_by_name(
        analysis["host_ranges"], "postprocessing.resize"
    )
    assert host_preprocessing["instances"] == 2
    assert host_preprocessing["inclusive"]["total_ns"] == 87805
    assert host_postprocessing["instances"] == 2
    assert host_postprocessing["inclusive"]["total_ns"] == 43903

    gpu_preprocessing = _range_by_name(
        analysis["gpu_projected_ranges"], "preprocessing.resize"
    )
    gpu_postprocessing = _range_by_name(
        analysis["gpu_projected_ranges"], "postprocessing.resize"
    )
    assert gpu_preprocessing["instances"] == 2
    assert gpu_preprocessing["projected"]["total_ns"] == 8192
    assert gpu_postprocessing["instances"] == 2
    assert gpu_postprocessing["projected"]["total_ns"] == 8000


def test_build_profile_analysis_warns_for_wrong_iteration_index_set(tmp_path):
    host_ranges = [
        (
            replace(item, name="iteration 2", raw_name=":iteration 2")
            if item.range_id == 7
            else item
        )
        for item in parse_nvtx_pushpop_trace(FIXTURES / "nvtx_pushpop_trace.csv")
    ]
    gpu_ranges = [
        (
            replace(item, name="iteration 2", raw_name=":iteration 2")
            if item.range_id == 7
            else item
        )
        for item in parse_nvtx_gpu_projection_trace(
            FIXTURES / "nvtx_gpu_proj_trace.csv"
        )
    ]

    analysis = build_profile_analysis(
        manifest=_manifest(),
        host_ranges=host_ranges,
        gpu_projected_ranges=gpu_ranges,
        nsys_version=VALIDATED_NSYS_VERSION,
        run_dir=tmp_path,
        trace_path=tmp_path / "trace.nsys-rep",
        report_paths={},
    )

    assert analysis["warnings"] == [
        "Missing expected iteration indexes: [1].",
        "Unexpected iteration indexes: [2].",
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
        NVTX_GPU_PROJECTION_TRACE_REPORT: stats_dir / "nsys_nvtx_gpu_proj_trace.csv",
    }
    shutil.copy(
        FIXTURES / "nvtx_pushpop_trace.csv", report_paths[NVTX_PUSHPOP_TRACE_REPORT]
    )
    shutil.copy(
        FIXTURES / "nvtx_gpu_proj_trace.csv",
        report_paths[NVTX_GPU_PROJECTION_TRACE_REPORT],
    )

    stats_call = {}

    def fake_run_nsys_stats(**kwargs):
        stats_call.update(kwargs)
        return NsysStatsArtifacts(
            nsys_version=VALIDATED_NSYS_VERSION,
            report_paths=report_paths,
        )

    monkeypatch.setattr(
        "development.profiling.analyze.run_nsys_stats",
        fake_run_nsys_stats,
    )

    output_path = analyze_run(run_dir=run_dir)

    assert output_path == run_dir / "analysis.json"
    assert stats_call["trace_path"] == run_dir / "trace.nsys-rep"
    assert stats_call["output_dir"] == run_dir / "stats"
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
