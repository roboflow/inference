import json
import sys
from pathlib import Path

import pytest

BENCHMARK_DIR = (
    Path(__file__).resolve().parents[3] / "development" / "video_poc" / "benchmarks"
)
sys.path.insert(0, str(BENCHMARK_DIR))

from cleanup_api_benchmark_run import cleanup_run, load_run_report  # noqa: E402


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def monotonic(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


class FakeClient:
    def __init__(self):
        self.jobs = {
            "job-active": {"id": "job-active", "state": "running"},
            "job-done": {"id": "job-done", "state": "completed"},
        }
        self.cancelled = []

    def get_job(self, job_id):
        job = self.jobs[job_id]
        if job.get("cancelRequested"):
            job["state"] = "cancelled"
        return dict(job)

    def cancel_job(self, job_id):
        self.cancelled.append(job_id)
        self.jobs[job_id]["cancelRequested"] = True
        return dict(self.jobs[job_id])


class TransientInspectClient(FakeClient):
    def __init__(self):
        super().__init__()
        self.first = True

    def get_job(self, job_id):
        if job_id == "job-active" and self.first:
            self.first = False
            raise RuntimeError("transient inspect failure")
        return super().get_job(job_id)


def write_checkpoint(tmp_path, run_id="run-001", api_base="https://api.roboflow.one"):
    path = tmp_path / f"api-corpus-{run_id}.json"
    path.write_text(
        json.dumps(
            {
                "schemaVersion": 2,
                "runId": run_id,
                "apiBase": api_base,
                "workspace": "benchmark-workspace",
                "starts": [{"job": {"id": "job-active", "state": "queued"}}],
                "jobs": [
                    {"id": "job-active", "state": "running"},
                    {"id": "job-done", "state": "completed"},
                ],
            }
        )
    )
    return path


def test_load_run_report_is_staging_only_and_uses_exact_run_id(tmp_path):
    path = write_checkpoint(tmp_path)
    loaded_path, _, api_base, workspace, jobs = load_run_report(tmp_path, "run-001")

    assert loaded_path == path
    assert api_base == "https://api.roboflow.one"
    assert workspace == "benchmark-workspace"
    assert sorted(jobs) == ["job-active", "job-done"]

    write_checkpoint(tmp_path, "prod-run", "https://api.roboflow.com")
    with pytest.raises(ValueError, match="staging"):
        load_run_report(tmp_path, "prod-run")
    with pytest.raises(ValueError, match="run id"):
        load_run_report(tmp_path, "../run-001")


def test_cleanup_only_cancels_active_captured_jobs():
    client = FakeClient()
    clock = FakeClock()

    result = cleanup_run(
        client=client,
        run_id="run-001",
        jobs={"job-active": {}, "job-done": {}},
        timeout_seconds=10,
        poll_interval_seconds=1,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert client.cancelled == ["job-active"]
    assert result["success"] is True
    assert result["expectedRecoveryState"] == "all captured jobs terminal"
    assert result["actualRecoveryState"] == "all captured jobs terminal"
    assert {job["state"] for job in result["jobs"]} == {"cancelled", "completed"}


def test_cleanup_still_cancels_exact_captured_job_after_transient_inspect_failure():
    client = TransientInspectClient()
    clock = FakeClock()

    result = cleanup_run(
        client=client,
        run_id="run-001",
        jobs={
            "job-active": {"id": "job-active", "state": "running"},
            "job-done": {"id": "job-done", "state": "completed"},
        },
        timeout_seconds=10,
        poll_interval_seconds=1,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert client.cancelled == ["job-active"]
    assert result["success"] is False
    assert result["errors"] == [
        {
            "phase": "inspect",
            "jobId": "job-active",
            "error": "transient inspect failure",
        }
    ]
    assert {job["state"] for job in result["jobs"]} == {"cancelled", "completed"}
