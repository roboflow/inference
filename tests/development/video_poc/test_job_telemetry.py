from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
import sys


PROCESSOR_DIR = (
    Path(__file__).resolve().parents[3] / "development" / "video_poc" / "processor"
)
sys.path.insert(0, str(PROCESSOR_DIR))

from job_telemetry import (  # noqa: E402
    LATENCY_BOUNDS_MS,
    JobTelemetry,
    build_runtime_identity,
)


class FakeClock:
    def __init__(self, *values):
        self.values = iter(values)

    def __call__(self):
        return next(self.values)


def test_snapshot_keeps_legacy_fields_and_adds_bounded_job_counters():
    telemetry = JobTelemetry(monotonic_clock=FakeClock(100, 102, 103, 103.05, 105))
    telemetry.on_job()
    assert telemetry.on_pipeline_start() == 2

    for event_type in (
        "FRAME_CAPTURED",
        "FRAME_CAPTURED",
        "FRAME_DROPPED",
        "FRAME_CONSUMED",
    ):
        telemetry.on_source_event(event_type)
    telemetry.on_result(
        SimpleNamespace(frame_timestamp=datetime.now() - timedelta(milliseconds=20))
    )
    telemetry.on_result(
        SimpleNamespace(frame_timestamp=datetime.now() - timedelta(milliseconds=30))
    )
    telemetry.on_rendered()
    telemetry.on_published()
    telemetry.on_image_output_materialized()

    snapshot = telemetry.snapshot(runtime={"revision": "abc123"})

    assert snapshot["schemaVersion"] == 2
    assert snapshot["frames"] == 2
    assert snapshot["fps"] == 20.0
    assert snapshot["pipelineStartS"] == 2
    assert snapshot["timeToFirstResultS"] == 3
    assert snapshot["timing"]["runningS"] == 5
    assert snapshot["counters"] == {
        "captured": 2,
        "decoded": 1,
        "dropped": 1,
        "inferred": 2,
        "rendered": 1,
        "published": 1,
        "imageOutputHostMaterializations": 1,
    }
    assert snapshot["decodeToResultLatency"]["count"] == 2
    assert snapshot["runtime"] == {"revision": "abc123"}


def test_latency_histogram_is_fixed_size_mergeable_and_reports_approx_quantiles():
    telemetry = JobTelemetry(monotonic_clock=lambda: 0)
    for latency_ms in (1, 8, 18, 70, 120, 6000):
        telemetry.record_latency_ms(latency_ms)

    latency = telemetry.snapshot()["decodeToResultLatency"]

    assert latency["count"] == 6
    assert latency["sum"] == 6217.0
    assert latency["mean"] == 1036.2
    assert latency["min"] == 1.0
    assert latency["max"] == 6000.0
    assert latency["p50Approx"] == 20
    assert latency["p95Approx"] == 6000.0
    assert latency["histogram"]["bounds"] == list(LATENCY_BOUNDS_MS) + [None]
    cumulative = latency["histogram"]["cumulativeCounts"]
    assert len(cumulative) == len(LATENCY_BOUNDS_MS) + 1
    assert cumulative[-1] == 6


def test_runtime_identity_uses_only_allowlisted_bounded_environment(monkeypatch):
    monkeypatch.setenv("VIDEO_PROC_IMAGE", "registry/video-processor:benchmark")
    monkeypatch.setenv("VIDEO_PROC_GIT_SHA", "deadbeef")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-123")
    monkeypatch.setenv("VIDEO_PROC_SERVICE_SECRET", "must-not-leak")
    monkeypatch.setenv("ROBOFLOW_API_KEY", "must-not-leak-either")

    identity = build_runtime_identity("processor-a")

    assert identity["processorId"] == "processor-a"
    assert identity["image"] == "registry/video-processor:benchmark"
    assert identity["revision"] == "deadbeef"
    assert identity["gpuVisibleDevices"] == "GPU-123"
    assert identity["processId"] > 0
    assert "must-not-leak" not in repr(identity)


def test_processor_wires_job_telemetry_without_job_labeled_prometheus():
    processor = (PROCESSOR_DIR / "processor.py").read_text()
    dockerfile = (PROCESSOR_DIR / "Dockerfile").read_text()

    assert "self.stats = JobTelemetry()" in processor
    assert "self.stats.on_source_event(event_type)" in processor
    assert "on_published=self.stats.on_published" in processor
    assert "runtime=self.runtime_identity" in processor
    assert "COPY job_telemetry.py /app/job_telemetry.py" in dockerfile
