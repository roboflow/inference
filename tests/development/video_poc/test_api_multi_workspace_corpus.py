import json
import sys
from pathlib import Path

import pytest

BENCHMARK_DIR = (
    Path(__file__).resolve().parents[3] / "development" / "video_poc" / "benchmarks"
)
sys.path.insert(0, str(BENCHMARK_DIR))

import run_api_multi_workspace_corpus as multi  # noqa: E402
from build_processor_jobs import load_corpus  # noqa: E402
from cleanup_api_multi_workspace_run import (  # noqa: E402
    cleanup_run as cleanup_multi_run,
    load_run_report as load_multi_run_report,
)
from run_api_experiment_matrix import build_command, load_matrix  # noqa: E402
from run_api_workflow_corpus import BenchmarkInterrupted  # noqa: E402

MANIFEST = BENCHMARK_DIR / "workflows" / "manifest.json"


def matrix_document():
    return {
        "schemaVersion": 1,
        "environment": "staging",
        "defaults": {
            "apiBase": "https://api.roboflow.one",
            "durationSeconds": 4,
            "pollIntervalSeconds": 2,
            "startupTimeoutSeconds": 10,
            "cleanupTimeoutSeconds": 10,
            "maxPlannedJobs": 8,
        },
        "scenarios": [
            {
                "name": "two-tenant-fairness",
                "workloads": [
                    {
                        "profile": "single-detection",
                        "count": 2,
                        "workspaceLabel": "tenant-a",
                        "workspace": "workspace-private-a",
                        "apiKeyEnv": "VIDEO_KEY_A",
                        "sourceId": "source-a",
                        "tier": "gpu",
                        "maxFps": 15,
                        "mode": "stream",
                        "publishOutput": False,
                    },
                    {
                        "profile": "cpu-blur",
                        "count": 1,
                        "startAfterSeconds": 2,
                        "workspaceLabel": "tenant-b",
                        "workspace": "workspace-private-b",
                        "apiKeyEnv": "VIDEO_KEY_B",
                        "sourceName": "Fixture B",
                        "tier": "cpu",
                        "maxFps": 5,
                        "mode": "batch",
                        "publishOutput": True,
                    },
                ],
            }
        ],
    }


def write_matrix(tmp_path, update=None):
    document = matrix_document()
    if update:
        update(document)
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(document))
    return path


def test_matrix_dispatches_object_workloads_without_credentials_in_command(tmp_path):
    path = write_matrix(tmp_path)
    matrix = load_matrix(path)
    scenario = matrix["scenarios"][0]

    assert scenario["multiWorkspace"] is True
    assert scenario["plannedJobs"] == 3
    assert scenario["requiredApiKeyEnvs"] == ["VIDEO_KEY_A", "VIDEO_KEY_B"]
    command = build_command(
        BENCHMARK_DIR / "run_api_workflow_corpus.py",
        matrix,
        scenario,
        "suite-two-tenant-r1",
        tmp_path,
        execute=True,
    )

    assert Path(command[1]).name == "run_api_multi_workspace_corpus.py"
    assert command[command.index("--scenario") + 1] == "two-tenant-fairness"
    assert command[command.index("--expected-matrix-sha256") + 1] == scenario[
        "matrixSha256"
    ]
    assert "--execute" in command
    assert "workspace-private-a" not in command
    assert "VIDEO_KEY_A" not in command
    assert "secret-a" not in command


def test_single_workspace_matrix_keeps_existing_runner_command(tmp_path):
    document = {
        "schemaVersion": 1,
        "environment": "staging",
        "defaults": {"workspace": "workspace-a", "sourceId": "source-a"},
        "scenarios": [{"name": "legacy", "workloads": ["cpu-blur=1"]}],
    }
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(document))
    matrix = load_matrix(path)
    command = build_command(
        BENCHMARK_DIR / "run_api_workflow_corpus.py",
        matrix,
        matrix["scenarios"][0],
        "suite-legacy-r1",
        tmp_path,
        execute=False,
    )

    assert matrix["scenarios"][0]["multiWorkspace"] is False
    assert Path(command[1]).name == "run_api_workflow_corpus.py"
    assert command[command.index("--workspace") + 1] == "workspace-a"


def test_multi_workspace_matrix_rejects_production_and_inline_secrets(tmp_path):
    def production(document):
        document["scenarios"][0]["workloads"][0]["apiBase"] = (
            "https://api.roboflow.com"
        )

    with pytest.raises(ValueError, match="staging"):
        load_matrix(write_matrix(tmp_path, production))

    def inline_secret(document):
        document["scenarios"][0]["workloads"][0]["apiKey"] = "secret-a"

    with pytest.raises(ValueError, match="forbidden"):
        load_matrix(write_matrix(tmp_path, inline_secret))

    def inline_password(document):
        document["scenarios"][0]["workloads"][0]["x-api-password"] = "secret-a"

    with pytest.raises(ValueError, match="forbidden"):
        load_matrix(write_matrix(tmp_path, inline_password))


def test_workspace_labels_are_bijective_tenant_identities(tmp_path):
    def duplicate_workspace(document):
        second = document["scenarios"][0]["workloads"][1]
        second["workspace"] = "workspace-private-a"

    with pytest.raises(ValueError, match="multiple workspaceLabel"):
        load_matrix(write_matrix(tmp_path, duplicate_workspace))


def test_plan_preserves_per_workload_routing_and_runtime_options(tmp_path):
    scenario = multi.load_scenario(write_matrix(tmp_path), "two-tenant-fairness")
    plan = multi.build_plan(load_corpus(MANIFEST), scenario)

    assert [item["workspaceLabel"] for item in plan] == [
        "tenant-a",
        "tenant-a",
        "tenant-b",
    ]
    assert [item["tier"] for item in plan] == ["gpu", "gpu", "cpu"]
    assert [item["maxFps"] for item in plan] == [15, 15, 5]
    assert [item["mode"] for item in plan] == ["stream", "stream", "batch"]
    assert plan[2]["imageOutput"] == "visualization"
    assert plan[2]["_routing"]["sourceName"] == "Fixture B"
    assert len(
        {
            json.dumps(item["workflowSpecification"], sort_keys=True)
            for item in plan
        }
    ) == 3


def test_error_sanitizer_removes_workspace_routing_and_credential(
    monkeypatch, tmp_path
):
    scenario = multi.load_scenario(write_matrix(tmp_path), "two-tenant-fairness")
    item = multi.build_plan(load_corpus(MANIFEST), scenario)[0]
    monkeypatch.setenv("VIDEO_KEY_A", "secret-a")

    assert multi._safe_error(
        RuntimeError("secret-a rejected for workspace-private-a"), item
    ) == "[redacted credential] rejected for tenant-a"


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def monotonic(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


class FakeClient:
    def __init__(self, workspace):
        self.workspace = workspace
        self.jobs = {}

    def start_job(self, source_id, item, key):
        # Deliberately reuse a job ID across workspaces to exercise composite handles.
        job_id = f"job-{item['copy']}"
        job = {
            "id": job_id,
            "sourceId": source_id,
            "state": "queued",
            "tier": item["tier"],
            "processorId": "processor-shared",
            "stats": {"frames": 0},
        }
        self.jobs[job_id] = job
        return 201, dict(job)

    def get_job(self, job_id):
        job = self.jobs[job_id]
        if job["state"] == "queued":
            job["state"] = "running"
        elif job.get("cancelRequested"):
            job["state"] = "cancelled"
        else:
            job["stats"]["frames"] += 1
        return dict(job)

    def cancel_job(self, job_id):
        self.jobs[job_id]["cancelRequested"] = True
        return dict(self.jobs[job_id])


class TransientInspectClient(FakeClient):
    def __init__(self, workspace):
        super().__init__(workspace)
        self.first = True

    def get_job(self, job_id):
        if self.first:
            self.first = False
            raise RuntimeError("transient inspect failure")
        return super().get_job(job_id)


def test_report_uses_labels_and_never_serializes_credentials_or_workspace_ids(tmp_path):
    scenario = multi.load_scenario(write_matrix(tmp_path), "two-tenant-fairness")
    # Avoid same-client collisions while retaining a cross-workspace collision.
    scenario["workloads"][0]["count"] = 1
    scenario["plannedJobs"] = 2
    plan = multi.build_plan(load_corpus(MANIFEST), scenario)
    clients, sources = {}, {}
    for item in plan:
        routing = item["_routing"]
        client_key = (routing["apiBase"], routing["workspace"], routing["apiKeyEnv"])
        clients.setdefault(client_key, FakeClient(routing["workspace"]))
        source_key = (client_key, routing.get("sourceId"), routing.get("sourceName"))
        sources[source_key] = {
            "id": routing.get("sourceId") or "source-b",
            "name": routing.get("sourceName") or "Fixture A",
            "status": "ready",
        }
        item["_clientKey"] = client_key
        item["_sourceKey"] = source_key

    clock = FakeClock()
    report = multi.run_benchmark(
        clients,
        sources,
        plan,
        scenario,
        "multi-test",
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )
    serialized = json.dumps(report, sort_keys=True)

    assert report["success"] is True
    assert {job["workspaceLabel"] for job in report["jobs"]} == {
        "tenant-a",
        "tenant-b",
    }
    assert "workspace-private-a" not in serialized
    assert "workspace-private-b" not in serialized
    assert "VIDEO_KEY_A" not in serialized
    assert "secret-a" not in serialized


def test_multi_workspace_run_checkpoints_each_poll_and_cleans_up_on_stop(tmp_path):
    scenario = multi.load_scenario(write_matrix(tmp_path), "two-tenant-fairness")
    scenario["workloads"][0]["count"] = 1
    scenario["plannedJobs"] = 2
    plan = multi.build_plan(load_corpus(MANIFEST), scenario)
    clients, sources = {}, {}
    for item in plan:
        routing = item["_routing"]
        client_key = (routing["apiBase"], routing["workspace"], routing["apiKeyEnv"])
        clients.setdefault(client_key, FakeClient(routing["workspace"]))
        source_key = (client_key, routing.get("sourceId"), routing.get("sourceName"))
        sources[source_key] = {
            "id": routing.get("sourceId") or "source-b",
            "name": routing.get("sourceName") or "Fixture A",
            "status": "ready",
        }
        item["_clientKey"] = client_key
        item["_sourceKey"] = source_key

    clock = FakeClock()
    checkpoints = []
    checks = 0

    def stop_after_first_poll():
        nonlocal checks
        checks += 1
        if checks >= 2:
            raise BenchmarkInterrupted("SIGTERM")

    report = multi.run_benchmark(
        clients,
        sources,
        plan,
        scenario,
        "multi-interrupted",
        checkpoint=lambda value: checkpoints.append(json.loads(json.dumps(value))),
        should_stop=stop_after_first_poll,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    phases = [item["checkpoint"]["phase"] for item in checkpoints]
    assert phases[0] == "initialized"
    assert "started" in phases
    assert "startup" in phases
    assert "interrupted" in phases
    assert phases[-1] == "complete"
    assert report["success"] is False
    assert report["errors"][-1] == {
        "phase": "run",
        "error": "interrupted by SIGTERM",
    }
    assert {job["state"] for job in report["jobs"]} == {"cancelled"}


def test_multi_workspace_janitor_rehydrates_routing_by_safe_plan_ordinal(tmp_path):
    matrix = write_matrix(tmp_path)
    scenario = multi.load_scenario(matrix, "two-tenant-fairness")
    checkpoint = tmp_path / "api-multi-workspace-multi-lost.json"
    checkpoint.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "kind": "multi-workspace-api-corpus",
                "environment": "staging",
                "runId": "multi-lost",
                "scenarioName": "two-tenant-fairness",
                "matrixSha256": scenario["matrixSha256"],
                "jobs": [
                    {
                        "id": "job-a",
                        "ordinal": 1,
                        "workspaceLabel": "tenant-a",
                        "profile": "single-detection",
                        "state": "running",
                        "stats": {"frames": 1},
                    },
                    {
                        "id": "job-b",
                        "ordinal": 3,
                        "workspaceLabel": "tenant-b",
                        "profile": "cpu-blur",
                        "state": "running",
                        "stats": {"frames": 1},
                    },
                ],
            }
        )
    )
    path, captured = load_multi_run_report(
        tmp_path,
        "multi-lost",
        matrix,
        "two-tenant-fairness",
        MANIFEST,
    )

    assert path == checkpoint
    assert sorted(captured) == ["1:job-a", "3:job-b"]
    assert {
        record["item"]["_routing"]["workspace"] for record in captured.values()
    } == {"workspace-private-a", "workspace-private-b"}

    clients = {}
    for record in captured.values():
        item = record["item"]
        key = item["workspaceLabel"]
        item["_clientKey"] = key
        client = FakeClient(key)
        client.jobs[record["job"]["id"]] = dict(record["job"])
        clients[key] = client
    clock = FakeClock()
    result = cleanup_multi_run(
        clients,
        "multi-lost",
        captured,
        timeout_seconds=10,
        poll_interval_seconds=1,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    serialized = json.dumps(result, sort_keys=True)
    assert result["success"] is True
    assert {job["state"] for job in result["jobs"]} == {"cancelled"}
    assert "workspace-private-a" not in serialized
    assert "VIDEO_KEY_A" not in serialized


def test_multi_workspace_janitor_cancels_after_transient_inspect_failure(tmp_path):
    matrix = write_matrix(tmp_path)
    scenario = multi.load_scenario(matrix, "two-tenant-fairness")
    item = multi.build_plan(load_corpus(MANIFEST), scenario)[0]
    item["_clientKey"] = "tenant-a"
    captured = {
        "1:job-a": {
            "item": item,
            "job": {
                "id": "job-a",
                "ordinal": 1,
                "workspaceLabel": "tenant-a",
                "profile": "single-detection",
                "state": "running",
                "stats": {"frames": 1},
            },
        }
    }
    client = TransientInspectClient("tenant-a")
    client.jobs["job-a"] = dict(captured["1:job-a"]["job"])
    clock = FakeClock()

    result = cleanup_multi_run(
        {"tenant-a": client},
        "multi-lost",
        captured,
        timeout_seconds=10,
        poll_interval_seconds=1,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert result["success"] is False
    assert result["errors"][0]["phase"] == "inspect"
    assert result["jobs"][0]["state"] == "cancelled"


def test_multi_workspace_janitor_rejects_matrix_checkpoint_identity_drift(tmp_path):
    matrix = write_matrix(tmp_path)
    scenario = multi.load_scenario(matrix, "two-tenant-fairness")
    (tmp_path / "api-multi-workspace-multi-drift.json").write_text(
        json.dumps(
            {
                "kind": "multi-workspace-api-corpus",
                "environment": "staging",
                "runId": "multi-drift",
                "scenarioName": "two-tenant-fairness",
                "matrixSha256": scenario["matrixSha256"],
                "jobs": [
                    {
                        "id": "job-a",
                        "ordinal": 1,
                        "workspaceLabel": "another-tenant",
                        "profile": "single-detection",
                    }
                ],
            }
        )
    )

    with pytest.raises(ValueError, match="identity"):
        load_multi_run_report(
            tmp_path,
            "multi-drift",
            matrix,
            "two-tenant-fairness",
            MANIFEST,
        )


def test_multi_workspace_janitor_rejects_changed_routing_with_same_safe_identity(tmp_path):
    matrix = write_matrix(tmp_path)
    scenario = multi.load_scenario(matrix, "two-tenant-fairness")
    (tmp_path / "api-multi-workspace-multi-route-drift.json").write_text(
        json.dumps(
            {
                "kind": "multi-workspace-api-corpus",
                "environment": "staging",
                "runId": "multi-route-drift",
                "scenarioName": "two-tenant-fairness",
                "matrixSha256": scenario["matrixSha256"],
                "jobs": [
                    {
                        "id": "job-collision",
                        "ordinal": 1,
                        "workspaceLabel": "tenant-a",
                        "profile": "single-detection",
                    }
                ],
            }
        )
    )
    document = json.loads(matrix.read_text())
    document["scenarios"][0]["workloads"][0]["workspace"] = "substituted-workspace"
    matrix.write_text(json.dumps(document))

    with pytest.raises(ValueError, match="matrix digest"):
        load_multi_run_report(
            tmp_path,
            "multi-route-drift",
            matrix,
            "two-tenant-fairness",
            MANIFEST,
        )
