import copy
import json
import sys
from pathlib import Path

import pytest


FAULT_DIR = (
    Path(__file__).resolve().parents[3]
    / "development"
    / "video_poc"
    / "benchmarks"
    / "fault_injection"
)
BENCHMARK_DIR = FAULT_DIR.parent
sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(FAULT_DIR))

import run_staging_fault as controller  # noqa: E402
import run_api_workflow_corpus as corpus  # noqa: E402


def scenario(tmp_path, fault_type="processor-pod-loss", phase="steady-state"):
    run_id = "recovery-run-001"
    checkpoint = tmp_path / f"api-corpus-{run_id}.json"
    fault = {
        "name": "delete-worker",
        "type": fault_type,
        "phase": phase,
        "gracePeriodSeconds": 0,
    }
    if fault_type == "processor-pod-loss":
        fault["jobOrdinal"] = 1
    else:
        fault["selector"] = {"app": "mediamtx", "instance": "staging-relay"}
    return {
        "schemaVersion": 1,
        "environment": "staging",
        "name": "recovery-test",
        "clusterContext": "gke_roboflow-staging_us-central1_k8s-staging-v3",
        "namespace": "video-proc",
        "benchmark": {
            "runId": run_id,
            "checkpoint": str(checkpoint),
            "apiHost": "us-central1-roboflow-staging.cloudfunctions.net",
        },
        "fault": fault,
        "deadlines": {
            "triggerSeconds": 1,
            "recoverySeconds": 1,
            "pollSeconds": 0.01,
        },
    }


def pod(name, uid, labels=None, ready=True, managed=True):
    owners = (
        [
            {
                "apiVersion": "apps/v1",
                "kind": "ReplicaSet",
                "name": "worker-abc",
                "uid": "owner-uid",
                "controller": True,
            }
        ]
        if managed
        else []
    )
    return {
        "metadata": {
            "name": name,
            "uid": uid,
            "resourceVersion": "42",
            "labels": labels or {"app": "worker", "pod-template-hash": "abc"},
            "ownerReferences": owners,
        },
        "status": {
            "conditions": [
                {"type": "Ready", "status": "True" if ready else "False"}
            ],
            "containerStatuses": [
                {"imageID": "registry/image@sha256:" + "a" * 64}
            ],
        },
    }


def write_checkpoint(path, phase="measurement", state="running"):
    path.write_text(
        json.dumps(
            {
                "schemaVersion": 2,
                "runId": "recovery-run-001",
                "apiBase": (
                    "https://us-central1-roboflow-staging.cloudfunctions.net/"
                    "light-v2-device"
                ),
                "checkpoint": {"phase": phase},
                "starts": [
                    {"ordinal": 1, "job": {"id": "job-a", "state": "queued"}}
                ],
                "jobs": [
                    {
                        "id": "job-a",
                        "state": state,
                        "processorId": "worker-abc-123",
                    }
                ],
            }
        )
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("environment", "production", "restricted to staging"),
        ("clusterContext", "gke_roboflow-prod_us-central1_prod", "staging"),
        ("namespace", "video-proc-production", "namespace"),
        ("namespace", "default", "namespace"),
    ],
)
def test_production_and_unscoped_namespaces_are_hard_refusals(
    tmp_path, field, value, message
):
    raw = scenario(tmp_path)
    raw[field] = value
    with pytest.raises(ValueError, match=message):
        controller.validate_scenario(raw)


def test_relay_selector_only_accepts_exact_equality_labels(tmp_path):
    raw = scenario(tmp_path, fault_type="relay-pod-loss")
    raw["fault"]["selector"] = {"app": "mediamtx,other"}
    with pytest.raises(ValueError, match="exact equality"):
        controller.validate_scenario(raw)


def test_deadlines_are_positive_and_capped(tmp_path):
    raw = scenario(tmp_path)
    raw["deadlines"]["recoverySeconds"] = 3601
    with pytest.raises(ValueError, match="cannot exceed"):
        controller.validate_scenario(raw)


def test_checkpoint_requires_exact_staging_hostname(tmp_path):
    raw = scenario(tmp_path)
    normalized = controller.validate_scenario(raw)
    checkpoint_path = Path(normalized["benchmark"]["checkpoint"])
    write_checkpoint(checkpoint_path)
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["apiBase"] = "https://api.roboflow.one.attacker.example"
    checkpoint_path.write_text(json.dumps(checkpoint))

    with pytest.raises(RuntimeError, match="not staging"):
        controller.read_checkpoint(checkpoint_path, "recovery-run-001")


def test_checkpoint_allows_existing_staging_alias(tmp_path):
    raw = scenario(tmp_path)
    raw["benchmark"]["apiHost"] = "api.roboflow.one"
    normalized = controller.validate_scenario(raw)
    checkpoint_path = Path(normalized["benchmark"]["checkpoint"])
    write_checkpoint(checkpoint_path)
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["apiBase"] = "https://api.roboflow.one"
    checkpoint_path.write_text(json.dumps(checkpoint))

    assert (
        controller.read_checkpoint(
            checkpoint_path, "recovery-run-001", "api.roboflow.one"
        )["runId"]
        == "recovery-run-001"
    )


def test_relative_checkpoint_resolves_from_scenario_directory(tmp_path):
    raw = scenario(tmp_path)
    raw["benchmark"]["checkpoint"] = "../results/api-corpus-recovery-run-001.json"
    scenario_dir = tmp_path / "scenarios"
    scenario_dir.mkdir()
    scenario_path = scenario_dir / "fault.json"
    scenario_path.write_text(json.dumps(raw))

    normalized = controller.load_scenario(scenario_path)

    assert normalized["benchmark"]["checkpoint"] == str(
        (tmp_path / "results" / "api-corpus-recovery-run-001.json").resolve()
    )


def test_default_main_only_renders_and_does_not_construct_kubectl(
    tmp_path, monkeypatch, capsys
):
    raw = scenario(tmp_path)
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text(json.dumps(raw))

    class ForbiddenKubectl:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("dry-run constructed a Kubernetes client")

    monkeypatch.setattr(controller, "Kubectl", ForbiddenKubectl)
    assert controller.main([str(scenario_path)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["mode"] == "dry-run"
    assert plan["benchmarkRunId"] == "recovery-run-001"
    assert "planDigest" in plan


def test_startup_trigger_requires_explicit_claimed_fault_ready_window(tmp_path):
    normalized = controller.validate_scenario(
        scenario(tmp_path, phase="startup")
    )
    checkpoint_path = Path(normalized["benchmark"]["checkpoint"])
    write_checkpoint(checkpoint_path, phase="startup", state="running")
    checkpoint = controller.read_checkpoint(checkpoint_path, "recovery-run-001")
    assert controller.trigger_target(normalized, checkpoint) is None

    checkpoint["checkpoint"]["phase"] = "fault-ready"
    checkpoint["jobs"][0]["state"] = "claimed"
    target = controller.trigger_target(normalized, checkpoint)
    assert target["podName"] == "worker-abc-123"

    checkpoint["jobs"][0]["state"] = "running"
    with pytest.raises(RuntimeError, match="must be claimed"):
        controller.trigger_target(normalized, checkpoint)


def test_capture_requires_one_managed_pod():
    with pytest.raises(RuntimeError, match="exactly one"):
        controller.capture_exact_pod([])
    with pytest.raises(RuntimeError, match="controller owner"):
        controller.capture_exact_pod([pod("worker", "uid", managed=False)])


class FakeKubectl:
    def __init__(self, context, original, replacement):
        self.context = context
        self.original = original
        self.replacement = replacement
        self.deleted = []

    def current_context(self):
        return self.context

    def cluster_identity(self):
        return controller.ALLOWED_STAGING_CLUSTERS.get(self.context)

    def pods(self, selector):
        if selector == {"metadata.name": "worker-abc-123"}:
            return [self.original] if not self.deleted else []
        if selector == {"metadata.name": "worker-abc-456"}:
            return [self.replacement] if self.deleted else []
        raise AssertionError(f"unexpected selector: {selector}")

    def delete_pod(self, captured, grace_period):
        assert captured["name"] == "worker-abc-123"
        assert captured["uid"] == "old-uid"
        assert grace_period == 0
        self.deleted.append(copy.deepcopy(captured))
        checkpoint = json.loads(self.checkpoint_path.read_text())
        checkpoint["jobs"][0]["processorId"] = "worker-abc-456"
        self.checkpoint_path.write_text(json.dumps(checkpoint))


def test_execute_deletes_only_captured_pod_and_writes_hash_chained_evidence(
    tmp_path,
):
    normalized = controller.validate_scenario(scenario(tmp_path))
    checkpoint_path = Path(normalized["benchmark"]["checkpoint"])
    write_checkpoint(checkpoint_path)
    original = pod("worker-abc-123", "old-uid")
    original["metadata"]["labels"]["pool"] = "working"
    replacement = pod("worker-abc-456", "new-uid")
    replacement["metadata"]["labels"]["pool"] = "ready"
    kube = FakeKubectl(normalized["clusterContext"], original, replacement)
    kube.checkpoint_path = checkpoint_path
    evidence_dir = tmp_path / "evidence"

    result = controller.execute(
        normalized,
        evidence_dir,
        confirm_run_id="recovery-run-001",
        kube=kube,
    )

    assert result["outcome"] == "passed"
    assert result["replacement"]["uid"] == "new-uid"
    assert len(kube.deleted) == 1
    events = [
        json.loads(line)
        for line in (evidence_dir / "events.jsonl").read_text().splitlines()
    ]
    assert [event["sequence"] for event in events] == list(range(len(events)))
    assert events[0]["previousDigest"] == "0" * 64
    assert all(
        event["previousDigest"] == events[index - 1]["digest"]
        for index, event in enumerate(events[1:], start=1)
    )
    assert events[-1]["payload"]["outcome"] == "passed"
    assert not ((evidence_dir / "events.jsonl").stat().st_mode & 0o222)


def test_execute_requires_exact_confirmation_and_context(tmp_path):
    normalized = controller.validate_scenario(scenario(tmp_path))
    original = pod("worker-abc-123", "old-uid")
    kube = FakeKubectl("some-other-staging", original, original)

    with pytest.raises(ValueError, match="confirm-run-id"):
        controller.execute(normalized, tmp_path / "evidence-a", "wrong", kube=kube)
    with pytest.raises(RuntimeError, match="current kubectl context"):
        controller.execute(
            normalized,
            tmp_path / "evidence-b",
            "recovery-run-001",
            kube=kube,
        )


def test_execute_rejects_context_alias_with_wrong_cluster_identity(tmp_path):
    normalized = controller.validate_scenario(scenario(tmp_path))
    original = pod("worker-abc-123", "old-uid")
    kube = FakeKubectl(normalized["clusterContext"], original, original)
    kube.cluster_identity = lambda: {
        "cluster": normalized["clusterContext"],
        "server": "https://production.example.invalid",
    }

    with pytest.raises(RuntimeError, match="allowlisted staging identity"):
        controller.execute(
            normalized,
            tmp_path / "evidence-wrong-cluster",
            "recovery-run-001",
            kube=kube,
        )


def test_execute_rejects_job_migration_between_capture_and_delete(tmp_path):
    normalized = controller.validate_scenario(scenario(tmp_path))
    checkpoint_path = Path(normalized["benchmark"]["checkpoint"])
    write_checkpoint(checkpoint_path)
    original = pod("worker-abc-123", "old-uid")
    kube = FakeKubectl(normalized["clusterContext"], original, original)
    kube.checkpoint_path = checkpoint_path
    original_pods = kube.pods
    reads = {"old": 0}

    def migrating_pods(selector):
        result = original_pods(selector)
        if selector == {"metadata.name": "worker-abc-123"}:
            reads["old"] += 1
            if reads["old"] == 2:
                checkpoint = json.loads(checkpoint_path.read_text())
                checkpoint["jobs"][0]["processorId"] = "worker-abc-456"
                checkpoint_path.write_text(json.dumps(checkpoint))
        return result

    kube.pods = migrating_pods

    with pytest.raises(RuntimeError, match="assignment changed"):
        controller.execute(
            normalized,
            tmp_path / "evidence-migrated",
            "recovery-run-001",
            kube=kube,
        )
    assert kube.deleted == []


def test_startup_fault_ready_checkpoint_drives_runner_recovery_contract(tmp_path):
    class ContractClient:
        api_base = (
            "https://us-central1-roboflow-staging.cloudfunctions.net/"
            "light-v2-device"
        )
        workspace = "benchmark-workspace"

        def __init__(self):
            self.job = None
            self.poll = 0
            self.faulted = False

        def start_job(self, source_id, item, key):
            del key
            self.job = {
                "id": "job-a",
                "sourceId": source_id,
                "state": "queued",
                "tier": item["tier"],
                "attempts": 0,
                "stats": {"frames": 0},
            }
            return 201, copy.deepcopy(self.job)

        def get_job(self, _job_id):
            if self.job.get("cancelRequested"):
                self.job["state"] = "cancelled"
                return copy.deepcopy(self.job)
            self.poll += 1
            if self.poll == 1:
                self.job.update(
                    {"state": "claimed", "processorId": "worker-abc-123"}
                )
            elif self.faulted and self.poll == 2:
                self.job.update({"state": "queued", "attempts": 1})
            elif self.faulted and self.poll == 3:
                self.job.update(
                    {"state": "claimed", "processorId": "worker-abc-456"}
                )
            elif self.faulted and self.poll >= 4:
                self.job.update(
                    {"state": "running", "processorId": "worker-abc-456"}
                )
                self.job["stats"]["frames"] += 1
            return copy.deepcopy(self.job)

        def cancel_job(self, _job_id):
            self.job["cancelRequested"] = True
            return copy.deepcopy(self.job)

    clock = type(
        "Clock",
        (),
        {
            "value": 0.0,
            "monotonic": lambda self: self.value,
            "sleep": lambda self, seconds: setattr(
                self, "value", self.value + seconds
            ),
        },
    )()
    client = ContractClient()
    normalized = controller.validate_scenario(
        scenario(tmp_path, phase="startup")
    )
    targets = []

    def checkpoint(report):
        if report.get("checkpoint", {}).get("phase") == "fault-ready":
            target = controller.trigger_target(normalized, report)
            targets.append(target)
            client.faulted = True

    report = corpus.run_benchmark(
        client=client,
        source={"id": "source-a", "name": "Fixture"},
        plan=[
            {
                "ordinal": 1,
                "copy": 1,
                "profile": "single-detection",
                "provisionalClass": "light",
                "tier": "gpu",
                "mode": "stream",
                "imageOutput": None,
                "maxFps": 10,
                "startAfterSeconds": 0,
                "workflowSpecification": {"version": "1.0", "steps": []},
            }
        ],
        run_id="recovery-run-001",
        duration_seconds=1,
        poll_interval_seconds=1,
        startup_timeout_seconds=20,
        cleanup_timeout_seconds=10,
        recovery_timeout_seconds=10,
        startup_fault_ready_seconds=2,
        checkpoint=checkpoint,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert report["success"] is True
    assert targets[0]["podName"] == "worker-abc-123"
    assert report["recoveries"][0]["sourcePhase"] == "startup"
    assert report["recoveries"][0]["outcome"] == "recovered"
    assert report["recoveries"][0]["assertions"]["job-a"] == {
        "processorChanged": True,
        "attemptAdvanced": True,
        "framesAdvancedAfterRunning": True,
        "requeueIdentityChanged": True,
    }
