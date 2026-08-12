import json
import sys
from pathlib import Path

import pytest

BENCHMARK_DIR = (
    Path(__file__).resolve().parents[3] / "development" / "video_poc" / "benchmarks"
)
sys.path.insert(0, str(BENCHMARK_DIR))

from build_processor_jobs import load_corpus  # noqa: E402
from run_api_workflow_corpus import (  # noqa: E402
    build_run_plan,
    idempotency_key,
    report_job,
    run_benchmark,
    select_source,
    validate_api_base,
)

MANIFEST = BENCHMARK_DIR / "workflows" / "manifest.json"


def test_runner_refuses_production_and_builds_from_shared_corpus():
    with pytest.raises(ValueError, match="staging"):
        validate_api_base("https://api.roboflow.com")

    profiles = load_corpus(MANIFEST)
    plan = build_run_plan(
        profiles,
        ["single-detection", "detection-tracking"],
        repeat=2,
        publish_output=False,
    )

    assert len(plan) == 4
    assert [item["tier"] for item in plan] == ["gpu"] * 4
    assert all(item["imageOutput"] is None for item in plan)
    assert len({idempotency_key("staging-001", item) for item in plan}) == 4
    assert (
        len(
            {json.dumps(item["workflowSpecification"], sort_keys=True) for item in plan}
        )
        == 4
    )
    assert plan[0]["workflowSpecification"]["metadata"]["benchmark"] == {
        "profile": "single-detection",
        "instance": 1,
    }
    assert "metadata" not in profiles["single-detection"]["specification"]


def test_source_selection_requires_an_unambiguous_source():
    sources = [
        {"id": "source-a", "name": "Camera"},
        {"id": "source-b", "name": "Camera"},
    ]

    assert select_source(sources, source_id="source-b")["id"] == "source-b"
    with pytest.raises(ValueError, match="multiple"):
        select_source(sources, source_name="Camera")


def test_report_job_is_an_allowlist_that_drops_future_credentials():
    assert report_job(
        {
            "id": "job-a",
            "state": "running",
            "stats": {"fps": 30},
            "streamKey": "secret",
            "processorAccessToken": "secret",
        }
    ) == {"id": "job-a", "state": "running", "stats": {"fps": 30}}


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def monotonic(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


class FakeClient:
    api_base = "https://api.roboflow.one"
    workspace = "benchmark-workspace"

    def __init__(self):
        self.jobs = {}

    def start_job(self, source_id, item, key):
        job_id = f"job-{item['ordinal']}"
        job = {
            "id": job_id,
            "sourceId": source_id,
            "state": "queued",
            "tier": item["tier"],
            "attempts": 0,
            "stats": {"frames": 0},
        }
        self.jobs[job_id] = job
        return 201, dict(job)

    def get_job(self, job_id):
        job = self.jobs[job_id]
        if job["state"] == "queued":
            job.update({"state": "running", "stats": {"frames": 1, "fps": 30}})
        elif job.get("cancelRequested"):
            job["state"] = "cancelled"
        else:
            job["stats"]["frames"] += 30
        return dict(job)

    def cancel_job(self, job_id):
        self.jobs[job_id]["cancelRequested"] = True
        if self.jobs[job_id]["state"] == "queued":
            self.jobs[job_id]["state"] = "cancelled"
        return dict(self.jobs[job_id])


class PartialStartFailureClient(FakeClient):
    def start_job(self, source_id, item, key):
        if item["ordinal"] == 2:
            raise RuntimeError("synthetic start failure")
        return super().start_job(source_id, item, key)


def test_run_benchmark_starts_measures_and_cleans_up_every_job():
    clock = FakeClock()
    profiles = load_corpus(MANIFEST)
    plan = build_run_plan(
        profiles,
        ["single-detection"],
        repeat=2,
        publish_output=True,
    )

    report = run_benchmark(
        client=FakeClient(),
        source={"id": "source-a", "name": "Fixture"},
        plan=plan,
        run_id="test-run",
        duration_seconds=4,
        poll_interval_seconds=2,
        startup_timeout_seconds=10,
        cleanup_timeout_seconds=10,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert report["success"] is True
    assert report["plannedConcurrency"] == 2
    assert {job["state"] for job in report["jobs"]} == {"cancelled"}
    assert {sample["phase"] for sample in report["samples"]} == {
        "startup",
        "measurement",
    }
    assert all(start["httpStatus"] == 201 for start in report["starts"])


def test_partial_start_failure_still_cancels_started_jobs():
    clock = FakeClock()
    profiles = load_corpus(MANIFEST)
    plan = build_run_plan(
        profiles,
        ["single-detection"],
        repeat=2,
        publish_output=False,
    )

    report = run_benchmark(
        client=PartialStartFailureClient(),
        source={"id": "source-a", "name": "Fixture"},
        plan=plan,
        run_id="partial-start",
        duration_seconds=4,
        poll_interval_seconds=2,
        startup_timeout_seconds=10,
        cleanup_timeout_seconds=10,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert report["success"] is False
    assert report["jobs"] == [
        {
            "id": "job-1",
            "sourceId": "source-a",
            "state": "cancelled",
            "tier": "gpu",
            "attempts": 0,
            "cancelRequested": True,
            "stats": {"frames": 0},
        }
    ]
    assert report["errors"] == [
        {
            "phase": "start",
            "profile": "single-detection",
            "ordinal": 2,
            "error": "synthetic start failure",
        }
    ]
