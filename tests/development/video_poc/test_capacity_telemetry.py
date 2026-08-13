import sys
from pathlib import Path

import pytest


BENCHMARK_DIR = (
    Path(__file__).resolve().parents[3] / "development" / "video_poc" / "benchmarks"
)
sys.path.insert(0, str(BENCHMARK_DIR))

from collect_staging_capacity_telemetry import (  # noqa: E402
    metric_queries,
    report_processor_pods,
    summarize,
)


def test_report_pods_are_derived_only_from_sanitized_runtime_identity():
    report = {
        "jobs": [
            {"stats": {"runtime": {"hostname": "processor-b"}}},
            {"stats": {"runtime": {"hostname": "processor-a"}}},
            {"stats": {"runtime": {"hostname": "processor-b"}}},
        ]
    }

    assert report_processor_pods(report) == ["processor-a", "processor-b"]
    assert "apiKey" not in str(metric_queries(report_processor_pods(report)))


def test_report_without_runtime_identity_is_rejected():
    with pytest.raises(ValueError, match="runtime hostname"):
        report_processor_pods({"jobs": [{"stats": {}}]})


def test_queries_join_cadvisor_and_dcgm_to_exact_processor_pods():
    queries = metric_queries(["processor-a", "processor-b"])

    assert 'pod=~"processor-a|processor-b"' in queries["processorCpuCores"]
    assert (
        'exported_pod=~"processor-a|processor-b"'
        in queries["gpuDecoderUtilPercent"]
    )
    assert "paths_readers" in queries["relayReaders"]


def test_summary_uses_all_finite_samples_and_interpolated_p95():
    summary = summarize([[1, 1.0], [2, 2.0], [3, 3.0], [4, 4.0]])

    assert summary == {
        "count": 4,
        "min": 1.0,
        "mean": 2.5,
        "p95": 3.85,
        "max": 4.0,
    }
