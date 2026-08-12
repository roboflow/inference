import json
import sys
from pathlib import Path

import pytest

BENCHMARK_DIR = (
    Path(__file__).resolve().parents[3] / "development" / "video_poc" / "benchmarks"
)
sys.path.insert(0, str(BENCHMARK_DIR))

import run_api_workflow_corpus as runner  # noqa: E402
from build_processor_jobs import load_corpus  # noqa: E402
from run_api_workflow_corpus import (  # noqa: E402
    build_run_plan,
    idempotency_key,
    parse_workload,
    report_job,
    run_benchmark,
    select_source,
    validate_api_base,
)

MANIFEST = BENCHMARK_DIR / "workflows" / "manifest.json"


def test_list_sources_does_not_require_a_workload(monkeypatch, capsys):
    class SourceClient:
        def __init__(self, api_base, workspace, api_key):
            assert api_key == "test-key"

        def list_sources(self):
            return [{"id": "source-a", "name": "Fixture", "status": "ready"}]

    monkeypatch.setenv("VIDEO_BENCHMARK_API_KEY", "test-key")
    monkeypatch.setattr(runner, "VideoServiceClient", SourceClient)

    assert runner.main(["--workspace", "workspace-a", "--list-sources"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "sources": [{"id": "source-a", "name": "Fixture", "status": "ready"}]
    }


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


def test_runner_builds_staged_mixed_workloads_with_explicit_fps():
    profiles = load_corpus(MANIFEST)
    workloads = [
        parse_workload("single-detection=3"),
        parse_workload("instance-segmentation=1@30"),
    ]

    plan = build_run_plan(
        profiles,
        [],
        repeat=1,
        publish_output=False,
        workloads=workloads,
        max_fps=15,
    )

    assert [item["profile"] for item in plan] == [
        "single-detection",
        "single-detection",
        "single-detection",
        "instance-segmentation",
    ]
    assert [item["startAfterSeconds"] for item in plan] == [0, 0, 0, 30]
    assert {item["maxFps"] for item in plan} == {15}


@pytest.mark.parametrize(
    "value",
    ["single-detection", "single-detection=0", "single-detection=x@3"],
)
def test_invalid_workload_syntax_is_rejected(value):
    with pytest.raises(ValueError, match="workload"):
        parse_workload(value)


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


class PlacedClient(FakeClient):
    def __init__(self, split=False):
        super().__init__()
        self.split = split

    def start_job(self, source_id, item, key):
        status, job = super().start_job(source_id, item, key)
        processor_id = (
            f"processor-{item['ordinal']}" if self.split else "processor-shared"
        )
        self.jobs[job["id"]]["processorId"] = processor_id
        job["processorId"] = processor_id
        return status, job


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


def test_staged_arrival_records_baseline_and_arrival_samples():
    clock = FakeClock()
    profiles = load_corpus(MANIFEST)
    plan = build_run_plan(
        profiles,
        [],
        repeat=1,
        publish_output=False,
        workloads=[
            parse_workload("single-detection=2"),
            parse_workload("instance-segmentation=1@4"),
        ],
    )

    report = run_benchmark(
        client=PlacedClient(),
        source={"id": "source-a", "name": "Fixture"},
        plan=plan,
        run_id="staged-arrival",
        duration_seconds=2,
        poll_interval_seconds=2,
        startup_timeout_seconds=10,
        cleanup_timeout_seconds=10,
        require_single_processor=True,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert report["success"] is True
    assert report["processorIds"] == ["processor-shared"]
    assert [wave["startAfterSeconds"] for wave in report["waves"]] == [0, 4]
    assert {sample["phase"] for sample in report["samples"]} == {
        "startup",
        "baseline",
        "arrival",
        "measurement",
    }


def test_single_processor_requirement_rejects_spread_placement():
    clock = FakeClock()
    profiles = load_corpus(MANIFEST)
    plan = build_run_plan(
        profiles,
        ["single-detection"],
        repeat=2,
        publish_output=False,
    )

    report = run_benchmark(
        client=PlacedClient(split=True),
        source={"id": "source-a", "name": "Fixture"},
        plan=plan,
        run_id="split-placement",
        duration_seconds=2,
        poll_interval_seconds=1,
        startup_timeout_seconds=10,
        cleanup_timeout_seconds=10,
        require_single_processor=True,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert report["success"] is False
    assert report["processorIds"] == ["processor-1", "processor-2"]
    assert report["errors"][-1] == {
        "phase": "placement",
        "error": "expected exactly one processor, observed 2",
    }
