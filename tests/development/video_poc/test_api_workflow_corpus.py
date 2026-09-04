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
    DEFAULT_API_BASE,
    BenchmarkInterrupted,
    RunLock,
    _start_jobs,
    build_run_plan,
    corpus_bundle_sha256,
    idempotency_key,
    parse_workload,
    recovery_checkpoint,
    report_job,
    run_benchmark,
    select_source,
    validate_api_base,
    write_report_atomic,
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


def test_missing_api_key_error_does_not_echo_environment_name(monkeypatch, capsys):
    environment_name = "PRIVATE_BENCHMARK_CREDENTIAL"
    monkeypatch.delenv(environment_name, raising=False)

    assert (
        runner.main(
            [
                "--workspace",
                "workspace-a",
                "--list-sources",
                "--api-key-env",
                environment_name,
            ]
        )
        == 2
    )
    error = capsys.readouterr().err
    assert error == "error: benchmark API key is not configured\n"
    assert environment_name not in error


def test_runner_refuses_production_and_builds_from_shared_corpus():
    assert DEFAULT_API_BASE == "https://api.roboflow.one"
    assert validate_api_base(DEFAULT_API_BASE) == DEFAULT_API_BASE
    with pytest.raises(ValueError, match="staging"):
        validate_api_base("https://api.roboflow.com")
    with pytest.raises(ValueError, match="staging"):
        validate_api_base(
            "https://attacker-roboflow-staging.cloudfunctions.net/light-v2-device"
        )
    with pytest.raises(ValueError, match="staging"):
        validate_api_base("https://roboflow-api-staging.web.app.attacker.example")
    assert validate_api_base(
        "https://us-central1-roboflow-staging.cloudfunctions.net/light-v2-device"
    ).startswith("https://us-central1-")
    assert validate_api_base("https://roboflow-api-staging.web.app") == (
        "https://roboflow-api-staging.web.app"
    )
    assert validate_api_base("https://roboflow-api-staging.firebaseapp.com/") == (
        "https://roboflow-api-staging.firebaseapp.com"
    )

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
    assert len(plan[0]["workflowSpecificationSha256"]) == 64


def test_corpus_bundle_digest_binds_manifest_and_referenced_specifications(tmp_path):
    expected = json.loads(
        (BENCHMARK_DIR / "matrices" / "long-soak.staging.example.json").read_text()
    )["soakPolicy"]["corpusBundleSha256"]
    assert corpus_bundle_sha256(MANIFEST) == expected

    manifest = json.loads(MANIFEST.read_text())
    source_spec = BENCHMARK_DIR / "workflows" / manifest["profiles"][0]["spec"]
    copied_spec = tmp_path / source_spec.name
    copied_spec.write_text(source_spec.read_text())
    manifest["profiles"] = [manifest["profiles"][0]]
    copied_manifest = tmp_path / "manifest.json"
    copied_manifest.write_text(json.dumps(manifest))
    before = corpus_bundle_sha256(copied_manifest)
    specification = json.loads(copied_spec.read_text())
    specification["metadata"] = {"changed": True}
    copied_spec.write_text(json.dumps(specification))

    assert corpus_bundle_sha256(copied_manifest) != before


def test_report_profile_recomputes_and_rejects_inconsistent_spec_digest():
    plan = build_run_plan(
        load_corpus(MANIFEST),
        ["single-detection"],
        repeat=1,
        publish_output=False,
    )
    item = plan[0]
    item["workflowSpecificationSha256"] = "0" * 64

    with pytest.raises(ValueError, match="digest is inconsistent"):
        runner.report_profile(item)


def test_watch_api_uses_workspace_bound_job_route_and_credential_free_result(
    monkeypatch,
):
    client = runner.VideoServiceClient(
        DEFAULT_API_BASE, "workspace-a", "never-reported"
    )
    client.wall_time = lambda: 1_786_000_000
    calls = []

    def request(method, suffix, body=None, headers=None):
        calls.append((method, suffix, body, headers))
        return 200, {
            "watch": {
                "requestedUntil": 1_786_000_060_000,
                "output": "visualization",
            }
        }

    monkeypatch.setattr(client, "_request", request)

    result = client.watch_job("job/unsafe", "visualization")

    assert calls == [
        (
            "POST",
            "video-jobs/v1/job%2Funsafe/watch",
            {"output": "visualization"},
            None,
        )
    ]
    assert result == {
        "requestedUntil": 1_786_000_060_000,
        "output": "visualization",
    }


@pytest.mark.parametrize(
    "watch",
    [
        {},
        {"requestedUntil": "2099-01-01T00:00:00Z", "output": "visualization"},
        {"requestedUntil": 1_786_000_060_000, "output": "another-output"},
        {"requestedUntil": 1_786_000_001_000, "output": "visualization"},
        {"requestedUntil": float("nan"), "output": "visualization"},
    ],
)
def test_watch_api_rejects_invalid_lease_contract(monkeypatch, watch):
    client = runner.VideoServiceClient(
        DEFAULT_API_BASE, "workspace-a", "never-reported"
    )
    client.wall_time = lambda: 1_786_000_000
    monkeypatch.setattr(
        client, "_request", lambda *_args, **_kwargs: (200, {"watch": watch})
    )

    with pytest.raises(ValueError, match="invalid credential-free lease"):
        client.watch_job("job-a", "visualization")


def test_watch_api_requires_exact_success_status(monkeypatch):
    client = runner.VideoServiceClient(
        DEFAULT_API_BASE, "workspace-a", "never-reported"
    )
    client.wall_time = lambda: 1_786_000_000
    monkeypatch.setattr(
        client,
        "_request",
        lambda *_args, **_kwargs: (
            201,
            {
                "watch": {
                    "requestedUntil": 1_786_000_060_000,
                    "output": "visualization",
                }
            },
        ),
    )

    with pytest.raises(ValueError, match="invalid credential-free lease"):
        client.watch_job("job-a", "visualization")


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


def test_report_job_deep_copies_nested_stats_evidence():
    job = {"id": "job-a", "state": "running", "stats": {"frames": 1}}
    reported = report_job(job)
    job["stats"]["frames"] = 99

    assert reported["stats"]["frames"] == 1


def test_atomic_report_writer_replaces_complete_document(tmp_path):
    path = tmp_path / "report.json"
    write_report_atomic(path, {"schemaVersion": 2, "samples": [1]})
    write_report_atomic(path, {"schemaVersion": 2, "samples": [1, 2]})

    assert json.loads(path.read_text()) == {"schemaVersion": 2, "samples": [1, 2]}
    assert not list(tmp_path.glob("report.json.*.tmp"))


def test_recovery_checkpoint_bounds_sample_history_but_keeps_cleanup_identity():
    report = {
        "runId": "run-a",
        "samples": [{"phase": "measurement", "jobs": [{"id": "job-a"}]}] * 100,
        "starts": [{"job": {"id": "job-a"}}],
        "jobs": [{"id": "job-a", "state": "running"}],
    }

    checkpoint = recovery_checkpoint(report)

    assert "samples" not in checkpoint
    assert checkpoint["sampleCount"] == 100
    assert checkpoint["lastSample"]["jobs"] == [{"id": "job-a"}]
    assert checkpoint["starts"] == [{"job": {"id": "job-a"}}]
    assert checkpoint["jobs"] == [{"id": "job-a", "state": "running"}]


def test_run_lock_rejects_a_concurrent_duplicate(tmp_path):
    path = tmp_path / ".run-a.lock"
    with RunLock(path):
        with pytest.raises(ValueError, match="already active"):
            with RunLock(path):
                pass


def test_successful_concurrent_starts_are_reported_incrementally():
    profiles = load_corpus(MANIFEST)
    plan = build_run_plan(
        profiles, ["single-detection"], repeat=3, publish_output=False
    )
    client = FakeClient()
    observed = []

    started, errors = _start_jobs(
        client,
        "source-a",
        plan,
        "incremental",
        on_started=lambda item, _status, job: observed.append(
            (item["ordinal"], job["id"])
        ),
    )

    assert not errors
    assert len(started) == 3
    assert sorted(observed) == [(1, "job-1"), (2, "job-2"), (3, "job-3")]


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
        self.watch_calls = []

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

    def watch_job(self, job_id, output):
        self.watch_calls.append((job_id, output))
        return {"requestedUntil": 4102444800000, "output": output}


class WatchFailureClient(FakeClient):
    def watch_job(self, job_id, output):
        super().watch_job(job_id, output)
        raise RuntimeError("synthetic watch failure")


class SlowWatchClient(FakeClient):
    def __init__(self, clock):
        super().__init__()
        self.clock = clock

    def watch_job(self, job_id, output):
        self.clock.sleep(7)
        return super().watch_job(job_id, output)


class ErrorDuringCleanupClient(FakeClient):
    def get_job(self, job_id):
        job = self.jobs[job_id]
        if job.get("cancelRequested"):
            job["state"] = "error"
            return dict(job)
        return super().get_job(job_id)


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


class RecoveringClient(FakeClient):
    def __init__(self, recover=True):
        super().__init__()
        self.recover = recover
        self.polls = {}

    def get_job(self, job_id):
        job = self.jobs[job_id]
        if job.get("cancelRequested"):
            job["state"] = "cancelled"
            return dict(job)
        poll = self.polls.get(job_id, 0) + 1
        self.polls[job_id] = poll
        if poll == 1:
            job.update(
                {
                    "state": "running",
                    "processorId": "processor-before",
                    "stats": {"frames": 1, "fps": 30},
                }
            )
        elif poll == 2:
            job.update({"state": "queued", "attempts": 1})
        elif self.recover and poll == 3:
            job.update({"state": "claimed", "processorId": "processor-after"})
        elif self.recover and poll >= 4:
            job.update({"state": "running", "processorId": "processor-after"})
            job["stats"]["frames"] += 30
        return dict(job)


class FastRecoveringClient(FakeClient):
    def __init__(self):
        super().__init__()
        self.polls = {}

    def get_job(self, job_id):
        job = self.jobs[job_id]
        if job.get("cancelRequested"):
            job["state"] = "cancelled"
            return dict(job)
        poll = self.polls.get(job_id, 0) + 1
        self.polls[job_id] = poll
        if poll == 1:
            job.update(
                {
                    "state": "running",
                    "processorId": "processor-before",
                    "stats": {"frames": 10, "fps": 30},
                }
            )
        elif poll == 2:
            job.update(
                {
                    "state": "running",
                    "processorId": "processor-after",
                    "attempts": 1,
                    "stats": {"frames": 1, "fps": 30},
                }
            )
        else:
            job["stats"]["frames"] += 30
        return dict(job)


class ClaimedThenRunningClient(FakeClient):
    def __init__(self):
        super().__init__()
        self.polls = {}

    def get_job(self, job_id):
        job = self.jobs[job_id]
        if job.get("cancelRequested"):
            job["state"] = "cancelled"
            return dict(job)
        poll = self.polls.get(job_id, 0) + 1
        self.polls[job_id] = poll
        if poll == 1:
            job.update({"state": "claimed", "processorId": "processor-before"})
        else:
            job.update(
                {
                    "state": "running",
                    "processorId": "processor-before",
                    "stats": {"frames": poll * 10, "fps": 30},
                }
            )
        return dict(job)


class InitialClaimAdvancesAttemptClient(ClaimedThenRunningClient):
    def get_job(self, job_id):
        job = super().get_job(job_id)
        if self.polls[job_id] == 1:
            job["attempts"] = 1
            self.jobs[job_id]["attempts"] = 1
        return job


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
    assert all(sample.get("sampledAt") for sample in report["samples"])
    assert all(start["httpStatus"] == 201 for start in report["starts"])
    assert set(report["watchLeases"]) == {"job-1", "job-2"}
    assert all(
        item["renewalCount"] == 1 and not item["errors"]
        for item in report["watchLeases"].values()
    )


def test_output_watch_is_renewed_and_report_never_retains_response_credentials():
    clock = FakeClock()
    client = FakeClient()
    profiles = load_corpus(MANIFEST)
    plan = build_run_plan(profiles, ["single-detection"], repeat=1, publish_output=True)

    report = run_benchmark(
        client=client,
        source={"id": "source-a", "name": "Fixture"},
        plan=plan,
        run_id="watch-renewal",
        duration_seconds=45,
        poll_interval_seconds=5,
        startup_timeout_seconds=10,
        cleanup_timeout_seconds=10,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    lease = report["watchLeases"]["job-1"]
    assert report["success"] is True
    assert len(client.watch_calls) == 3
    assert lease["renewalCount"] == 3
    assert lease["maximumRenewalGapSeconds"] == 20
    assert "requestedUntil" not in json.dumps(report)
    assert "processorAccessToken" not in json.dumps(report)


def test_watch_gap_is_measured_between_completed_renewals():
    clock = FakeClock()
    evidence = {}
    renewer = runner.WatchLeaseRenewer(
        SlowWatchClient(clock), evidence, monotonic=clock.monotonic
    )
    renewer.register("job-1", "visualization")
    running = {"job-1": {"state": "running"}}

    renewer.renew(running)
    clock.sleep(20)
    renewer.renew(running)

    assert evidence["job-1"]["renewalCount"] == 2
    assert evidence["job-1"]["maximumRenewalGapSeconds"] == 27


def test_output_watch_failure_fails_run_but_still_cleans_up():
    clock = FakeClock()
    client = WatchFailureClient()
    profiles = load_corpus(MANIFEST)
    plan = build_run_plan(profiles, ["single-detection"], repeat=1, publish_output=True)

    report = run_benchmark(
        client=client,
        source={"id": "source-a", "name": "Fixture"},
        plan=plan,
        run_id="watch-failure",
        duration_seconds=2,
        poll_interval_seconds=1,
        startup_timeout_seconds=10,
        cleanup_timeout_seconds=10,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert report["success"] is False
    assert {job["state"] for job in report["jobs"]} == {"cancelled"}
    assert report["watchLeases"]["job-1"]["renewalCount"] == 0
    assert len(report["watchLeases"]["job-1"]["errors"]) == 1
    assert "watch lease renewal failed" in report["errors"][0]["error"]


def test_output_watch_rejects_poll_interval_that_cannot_safely_renew_lease():
    profiles = load_corpus(MANIFEST)
    plan = build_run_plan(profiles, ["single-detection"], repeat=1, publish_output=True)

    with pytest.raises(ValueError, match="watch lease"):
        run_benchmark(
            client=FakeClient(),
            source={"id": "source-a", "name": "Fixture"},
            plan=plan,
            run_id="watch-too-slow",
            duration_seconds=60,
            poll_interval_seconds=21,
            startup_timeout_seconds=10,
            cleanup_timeout_seconds=10,
        )


def test_error_terminal_during_cleanup_fails_an_otherwise_successful_run():
    clock = FakeClock()
    profiles = load_corpus(MANIFEST)
    plan = build_run_plan(
        profiles, ["single-detection"], repeat=1, publish_output=False
    )

    report = run_benchmark(
        client=ErrorDuringCleanupClient(),
        source={"id": "source-a", "name": "Fixture"},
        plan=plan,
        run_id="cleanup-error",
        duration_seconds=2,
        poll_interval_seconds=1,
        startup_timeout_seconds=10,
        cleanup_timeout_seconds=10,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert report["success"] is False
    assert report["jobs"][0]["state"] == "error"
    assert "error terminal state" in report["errors"][-1]["error"]


def test_run_benchmark_checkpoints_every_poll_and_on_completion():
    clock = FakeClock()
    profiles = load_corpus(MANIFEST)
    plan = build_run_plan(
        profiles, ["single-detection"], repeat=1, publish_output=False
    )
    checkpoints = []

    report = run_benchmark(
        client=FakeClient(),
        source={"id": "source-a", "name": "Fixture"},
        plan=plan,
        run_id="checkpointed",
        duration_seconds=2,
        poll_interval_seconds=1,
        startup_timeout_seconds=10,
        cleanup_timeout_seconds=10,
        checkpoint=lambda value: checkpoints.append(json.loads(json.dumps(value))),
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    sampled_phases = [sample["phase"] for sample in report["samples"]]
    checkpoint_phases = [item["checkpoint"]["phase"] for item in checkpoints]
    for phase in sampled_phases:
        assert checkpoint_phases.count(phase) >= sampled_phases.count(phase)
    assert checkpoint_phases[0] == "initialized"
    assert checkpoint_phases[-1] == "complete"
    assert checkpoints[-1]["success"] is True


def test_recovery_tolerance_records_requeue_and_new_processor():
    clock = FakeClock()
    profiles = load_corpus(MANIFEST)
    plan = build_run_plan(
        profiles, ["single-detection"], repeat=1, publish_output=False
    )

    report = run_benchmark(
        client=RecoveringClient(),
        source={"id": "source-a", "name": "Fixture"},
        plan=plan,
        run_id="recovery-success",
        duration_seconds=5,
        poll_interval_seconds=1,
        startup_timeout_seconds=10,
        cleanup_timeout_seconds=10,
        recovery_timeout_seconds=3,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert report["success"] is True
    assert {sample["phase"] for sample in report["samples"]} >= {
        "startup",
        "measurement",
        "recovery",
    }
    event = report["recoveries"][0]
    assert event["sourcePhase"] == "measurement"
    assert event["startedElapsedSeconds"] == 1.0
    assert event["observedControlPlaneRecoverySeconds"] == 3.0
    assert event["outcome"] == "recovered"
    assert event["before"][0]["processorId"] == "processor-before"
    assert event["before"][0]["state"] == "running"
    assert event["firstObserved"][0]["state"] == "queued"
    assert event["after"][0]["processorId"] == "processor-after"
    assert event["after"][0]["state"] == "running"
    assert event["after"][0]["attempts"] == 1
    assert event["after"][0]["stats"]["frames"] > (
        event["runningObserved"][0]["stats"]["frames"]
    )
    assert event["assertions"]["job-1"] == {
        "processorChanged": True,
        "attemptAdvanced": True,
        "framesAdvancedAfterRunning": True,
        "requeueIdentityChanged": True,
    }


def test_recovery_timeout_fails_and_still_cleans_up():
    clock = FakeClock()
    profiles = load_corpus(MANIFEST)
    plan = build_run_plan(
        profiles, ["single-detection"], repeat=1, publish_output=False
    )

    report = run_benchmark(
        client=RecoveringClient(recover=False),
        source={"id": "source-a", "name": "Fixture"},
        plan=plan,
        run_id="recovery-timeout",
        duration_seconds=5,
        poll_interval_seconds=1,
        startup_timeout_seconds=10,
        cleanup_timeout_seconds=10,
        recovery_timeout_seconds=2,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert report["success"] is False
    assert report["recoveries"][0]["outcome"] == "timeout"
    assert report["errors"][0] == {
        "phase": "measurement",
        "error": "job did not recover during measurement",
    }
    assert {job["state"] for job in report["jobs"]} == {"cancelled"}


def test_fast_requeue_between_polls_is_detected_and_progress_verified():
    clock = FakeClock()
    profiles = load_corpus(MANIFEST)
    plan = build_run_plan(
        profiles, ["single-detection"], repeat=1, publish_output=False
    )

    report = run_benchmark(
        client=FastRecoveringClient(),
        source={"id": "source-a", "name": "Fixture"},
        plan=plan,
        run_id="fast-recovery",
        duration_seconds=3,
        poll_interval_seconds=1,
        startup_timeout_seconds=10,
        cleanup_timeout_seconds=10,
        recovery_timeout_seconds=2,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert report["success"] is True
    event = report["recoveries"][0]
    assert event["firstObserved"][0]["state"] == "running"
    assert event["firstObserved"][0]["processorId"] == "processor-after"
    assert event["outcome"] == "recovered"
    assert event["observedControlPlaneRecoverySeconds"] == 1.0
    assert event["assertions"]["job-1"]["processorChanged"] is True
    assert event["assertions"]["job-1"]["framesAdvancedAfterRunning"] is True


def test_recovery_timeout_cli_is_opt_in_and_nonnegative():
    args = runner.parse_args(
        [
            "--workspace",
            "workspace-a",
            "--source-id",
            "source-a",
            "--profile",
            "single-detection",
        ]
    )
    assert args.recovery_timeout_seconds == 0
    assert args.startup_fault_ready_seconds == 0

    with pytest.raises(SystemExit):
        runner.parse_args(
            [
                "--workspace",
                "workspace-a",
                "--source-id",
                "source-a",
                "--profile",
                "single-detection",
                "--recovery-timeout-seconds",
                "-1",
            ]
        )
    with pytest.raises(SystemExit):
        runner.parse_args(
            [
                "--workspace",
                "workspace-a",
                "--source-id",
                "source-a",
                "--profile",
                "single-detection",
                "--startup-fault-ready-seconds",
                "30",
            ]
        )


def test_startup_fault_ready_checkpoint_holds_claimed_assignment():
    clock = FakeClock()
    profiles = load_corpus(MANIFEST)
    plan = build_run_plan(
        profiles, ["single-detection"], repeat=1, publish_output=False
    )
    checkpoints = []

    report = run_benchmark(
        client=ClaimedThenRunningClient(),
        source={"id": "source-a", "name": "Fixture"},
        plan=plan,
        run_id="startup-fault-ready",
        duration_seconds=1,
        poll_interval_seconds=1,
        startup_timeout_seconds=10,
        cleanup_timeout_seconds=10,
        startup_fault_ready_seconds=3,
        checkpoint=lambda value: checkpoints.append(json.loads(json.dumps(value))),
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    fault_ready = [
        item for item in checkpoints if item["checkpoint"]["phase"] == "fault-ready"
    ]
    assert report["success"] is True
    assert clock.value >= 3
    assert len(fault_ready) == 1
    assert fault_ready[0]["jobs"][0]["state"] == "claimed"
    assert fault_ready[0]["jobs"][0]["processorId"] == "processor-before"


def test_normal_first_claim_attempt_increment_is_not_a_recovery():
    clock = FakeClock()
    profiles = load_corpus(MANIFEST)
    plan = build_run_plan(
        profiles, ["single-detection"], repeat=1, publish_output=False
    )

    report = run_benchmark(
        client=InitialClaimAdvancesAttemptClient(),
        source={"id": "source-a", "name": "Fixture"},
        plan=plan,
        run_id="ordinary-first-claim",
        duration_seconds=1,
        poll_interval_seconds=1,
        startup_timeout_seconds=10,
        cleanup_timeout_seconds=10,
        recovery_timeout_seconds=3,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert report["success"] is True
    assert report.get("recoveries", []) == []


def test_initial_checkpoint_failure_prevents_any_job_start():
    clock = FakeClock()
    client = FakeClient()
    profiles = load_corpus(MANIFEST)
    plan = build_run_plan(
        profiles, ["single-detection"], repeat=1, publish_output=False
    )

    with pytest.raises(BenchmarkInterrupted, match="checkpoint write failure"):
        run_benchmark(
            client=client,
            source={"id": "source-a", "name": "Fixture"},
            plan=plan,
            run_id="no-checkpoint-no-start",
            duration_seconds=2,
            poll_interval_seconds=1,
            startup_timeout_seconds=10,
            cleanup_timeout_seconds=10,
            checkpoint=lambda _value: (_ for _ in ()).throw(OSError("disk full")),
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )

    assert client.jobs == {}


def test_stop_request_reaches_cleanup_and_records_interruption():
    clock = FakeClock()
    profiles = load_corpus(MANIFEST)
    plan = build_run_plan(
        profiles, ["single-detection"], repeat=1, publish_output=False
    )

    def stop_after_start():
        raise BenchmarkInterrupted("SIGTERM")

    report = run_benchmark(
        client=FakeClient(),
        source={"id": "source-a", "name": "Fixture"},
        plan=plan,
        run_id="interrupted",
        duration_seconds=2,
        poll_interval_seconds=1,
        startup_timeout_seconds=10,
        cleanup_timeout_seconds=10,
        should_stop=stop_after_start,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert report["interrupted"] is True
    assert report["success"] is False
    assert report["errors"][0] == {
        "phase": "run",
        "error": "interrupted by SIGTERM",
    }
    assert {job["state"] for job in report["jobs"]} == {"cancelled"}


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
