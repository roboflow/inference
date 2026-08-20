import hashlib
import json
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

BENCHMARK_DIR = (
    Path(__file__).resolve().parents[3] / "development" / "video_poc" / "benchmarks"
)
sys.path.insert(0, str(BENCHMARK_DIR / "analysis"))

from soak import (  # noqa: E402
    _canonical_sha256,
    certify_ladder,
    certify_run,
    certify_stage,
    load_policy,
)

SEQUENCE = ["15m", "01h", "04h", "12h"]


def write_matrix(tmp_path):
    workflows = tmp_path / "workflows"
    shutil.copytree(BENCHMARK_DIR / "workflows", workflows)
    checked_policy = json.loads(
        (BENCHMARK_DIR / "matrices" / "long-soak.staging.example.json").read_text()
    )["soakPolicy"]
    stages = {}
    scenarios = []
    for index, stage in enumerate(SEQUENCE, 1):
        duration = index * 10
        stages[stage] = {
            "durationSeconds": duration,
            "maxProcessorMemoryGrowthMiB": 10,
            "maxRelayMemoryGrowthMiB": 10,
            "maxProcessorMemorySlopeMiBPerHour": 10,
            "maxRelayMemorySlopeMiBPerHour": 10,
            "maxGpuFramebufferGrowthMiB": 10,
            "maxGpuFramebufferSlopeMiBPerHour": 10,
        }
        for family in ("gpu", "cpu"):
            scenarios.append(
                {
                    "name": f"{family}-soak-{stage}",
                    "soakFamily": family,
                    "soakStage": stage,
                    "workloads": [
                        (
                            "single-detection=1"
                            if family == "gpu"
                            else "single-detection-cpu=1"
                        )
                    ],
                    "durationSeconds": duration,
                    "publishOutput": True,
                }
            )
    matrix = {
        "schemaVersion": 1,
        "environment": "staging",
        "defaults": {
            "apiBase": "https://api.roboflow.one",
            "workspace": "benchmark",
            "sourceId": "source-a",
            "pollIntervalSeconds": 5,
        },
        "soakPolicy": {
            "schemaVersion": 1,
            "requiredSequence": SEQUENCE,
            "minimumMeasurementCoverageRatio": 0.9,
            "minimumTelemetryCoverageRatio": 0.5,
            "corpusManifestPath": "../workflows/manifest.json",
            "corpusBundleSha256": checked_policy["corpusBundleSha256"],
            "maximumWatchRenewalGapSeconds": 30,
            "maxConsecutiveNonadvancingOutputIntervals": 1,
            "maxFrameLatencyP95Ms": 50,
            "maxFrameLatencyP99Ms": 150,
            "requiredCounters": [
                "captured",
                "decoded",
                "inferred",
                "rendered",
                "published",
            ],
            "stages": stages,
        },
        "scenarios": scenarios,
    }
    path = tmp_path / "matrices" / "soak.json"
    path.parent.mkdir()
    path.write_text(json.dumps(matrix))
    return path


def write_artifacts(
    tmp_path, run_id="suite-gpu-soak-15m-r1", duration=10, family="gpu"
):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(seconds=duration)
    counters = [index * 10 for index in range(duration // 5 + 1)]
    profile_id = "single-detection" if family == "gpu" else "single-detection-cpu"
    manifest = json.loads((BENCHMARK_DIR / "workflows" / "manifest.json").read_text())
    raw_profile = next(
        item for item in manifest["profiles"] if item["id"] == profile_id
    )
    specification = json.loads(
        (BENCHMARK_DIR / "workflows" / raw_profile["spec"]).read_text()
    )
    metadata = dict(specification.get("metadata") or {})
    metadata["benchmark"] = {"profile": profile_id, "instance": 1}
    specification["metadata"] = metadata
    checked_policy = json.loads(
        (BENCHMARK_DIR / "matrices" / "long-soak.staging.example.json").read_text()
    )["soakPolicy"]

    def job(value, state="running"):
        return {
            "id": "job-a",
            "state": state,
            "processorId": "processor-a",
            "attempts": 0,
            "stats": {
                "frames": value,
                "runtime": {"hostname": "processor-a"},
                "counters": {
                    "captured": value,
                    "decoded": value,
                    "inferred": value,
                    "rendered": value,
                    "published": value,
                },
                "decodeToResultLatency": {
                    "count": value,
                    "sum": value * 15,
                    "max": 15,
                    "histogram": {
                        "bounds": [10, 20, None],
                        "cumulativeCounts": [0, value, value],
                    },
                },
            },
        }

    report = {
        "schemaVersion": 2,
        "runId": run_id,
        "success": True,
        "startedAt": start.isoformat(),
        "endedAt": (end + timedelta(seconds=1)).isoformat(),
        "measurementStartedAt": start.isoformat(),
        "measurementEndedAt": end.isoformat(),
        "checkpoint": {"phase": "complete"},
        "apiBase": "https://api.roboflow.one",
        "workspace": "benchmark",
        "source": {"id": "source-a"},
        "plannedConcurrency": 1,
        "corpusBundleSha256": checked_policy["corpusBundleSha256"],
        "profiles": [
            {
                "ordinal": 1,
                "copy": 1,
                "profile": profile_id,
                "provisionalClass": raw_profile["provisionalClass"],
                "tier": family,
                "mode": "stream",
                "imageOutput": raw_profile["imageOutput"],
                "maxFps": None,
                "startAfterSeconds": 0.0,
                "workflowSpecificationSha256": _canonical_sha256(specification),
            }
        ],
        "watchLeases": {
            "job-a": {
                "jobId": "job-a",
                "output": "visualization",
                "renewalIntervalSeconds": 20,
                "renewalCount": 2,
                "firstRequestedAt": (start - timedelta(seconds=1)).isoformat(),
                "lastRequestedAt": end.isoformat(),
                "maximumRenewalGapSeconds": 20,
                "errors": [],
            }
        },
        "starts": [{"ordinal": 1, "job": {"id": "job-a"}}],
        "samples": [
            {
                "phase": "measurement",
                "elapsedSeconds": index * 5,
                "jobs": [job(value)],
            }
            for index, value in enumerate(counters)
        ],
        "jobs": [job(counters[-1], "cancelled")],
        "errors": [],
        "cancelErrors": [],
    }
    report_path = tmp_path / f"api-corpus-{run_id}.json"
    report_path.write_text(json.dumps(report))

    metric_names = {
        "processorCpuCores",
        "processorMemoryWorkingSetBytes",
        "processorContainerRestarts",
        "relayCpuCores",
        "relayMemoryWorkingSetBytes",
        "relayContainerRestarts",
        "relayPodIdentity",
        "relayReaders",
        "relayIngressBytesPerSecond",
        "relayEgressBytesPerSecond",
        "relayOutputPathCount",
        "relayOutputIngressBytesPerSecond",
        "relayRtspPacketsLostPerSecond",
        "relayRtspPacketsInErrorPerSecond",
    }
    if family == "gpu":
        metric_names.update(
            {
                "gpuUtilPercent",
                "gpuFramebufferUsedMiB",
                "gpuDecoderUtilPercent",
                "gpuMemoryCopyUtilPercent",
            }
        )
    timestamps = [start.timestamp() + index * 5 for index in range(duration // 5 + 1)]
    metrics = {}
    for name in metric_names:
        if "Packets" in name or "Restarts" in name:
            values = [0] * len(timestamps)
        elif name == "relayReaders":
            values = [1] * len(timestamps)
        elif "MemoryWorkingSetBytes" in name:
            values = [100 * 1024 * 1024] * len(timestamps)
        else:
            values = [1] * len(timestamps)
        labels = {}
        if name == "relayPodIdentity":
            labels = {"pod": "mediamtx-a", "uid": "relay-uid-a"}
        elif name in {
            "relayContainerRestarts",
            "relayOutputPathCount",
            "relayOutputIngressBytesPerSecond",
        }:
            labels = {"pod": "mediamtx-a"}
        metrics[name] = {
            "series": [
                {
                    "metric": labels,
                    "values": [
                        [timestamp, str(value)]
                        for timestamp, value in zip(timestamps, values)
                    ],
                }
            ]
        }
    resources = {
        "schemaVersion": 1,
        "environment": "staging",
        "clusterContext": "ck8s-stg",
        "sourceReportSha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "runId": run_id,
        "measurementStartedAt": report["measurementStartedAt"],
        "measurementEndedAt": report["measurementEndedAt"],
        "sampleStepSeconds": 5,
        "processorPods": ["processor-a"],
        "metrics": metrics,
    }
    approved_identity = {
        "schemaVersion": 1,
        "environment": "staging",
        "context": "ck8s-stg",
        "apiServer": "https://staging-kubernetes.example.test",
        "kubeSystemNamespaceUid": "staging-cluster-uid",
        "approvedAt": "2025-12-31T00:00:00Z",
        "validUntil": "2026-01-02T00:00:00Z",
        "approvedBy": "staging-platform-owner",
    }
    identity_path = tmp_path / "staging-cluster-identity.approved.json"
    identity_path.write_text(json.dumps(approved_identity))
    resources["clusterIdentity"] = {
        "approved": approved_identity,
        "approvedPath": str(identity_path.resolve()),
        "approvedFileSha256": hashlib.sha256(identity_path.read_bytes()).hexdigest(),
        "approvedSha256": _canonical_sha256(approved_identity),
        "observed": {
            "apiServer": approved_identity["apiServer"],
            "kubeSystemNamespaceUid": approved_identity["kubeSystemNamespaceUid"],
        },
    }
    resources_path = tmp_path / f"api-corpus-{run_id}-resources.json"
    resources_path.write_text(json.dumps(resources))
    return report_path, resources_path


def write_suite(tmp_path, matrix_path, runs):
    suite = {
        "schemaVersion": 2,
        "suiteId": "suite",
        "environment": "staging",
        "execute": True,
        "matrixSha256": hashlib.sha256(matrix_path.read_bytes()).hexdigest(),
        "runs": runs,
    }
    path = tmp_path / "suite-suite.json"
    path.write_text(json.dumps(suite))
    return path


def test_certifies_bound_output_and_resource_evidence(tmp_path):
    matrix_path = write_matrix(tmp_path)
    report, resources = write_artifacts(tmp_path)

    result = certify_run(report, resources, load_policy(matrix_path), "gpu-soak-15m")

    assert result["passed"] is True
    assert all(result["checks"].values())
    assert result["jobEvidence"]["streams"][0]["counterDeltas"]["published"] == 20
    assert result["resourceEvidence"]["memory"]["processor"] == {
        "endpointGrowthMiB": 0.0,
        "linearSlopeMiBPerHour": 0.0,
    }


def test_rejects_counter_reset_and_tampered_resource_binding(tmp_path):
    matrix_path = write_matrix(tmp_path)
    report_path, resources_path = write_artifacts(tmp_path)
    report = json.loads(report_path.read_text())
    report["samples"][1]["jobs"][0]["stats"]["counters"]["published"] = 30
    report_path.write_text(json.dumps(report))

    result = certify_run(
        report_path,
        resources_path,
        load_policy(matrix_path),
        "gpu-soak-15m",
    )

    assert result["passed"] is False
    assert result["checks"]["binding.resourceSourceHash"] is False
    assert result["checks"]["report.noCounterResets"] is False


def test_rejects_workload_profile_or_cluster_identity_substitution(tmp_path):
    matrix_path = write_matrix(tmp_path)
    report_path, resources_path = write_artifacts(tmp_path)
    report = json.loads(report_path.read_text())
    report["profiles"][0]["tier"] = "cpu"
    report["workspace"] = "another-workspace"
    report["apiBase"] = "https://roboflow-api-staging.web.app"
    report_path.write_text(json.dumps(report))
    resources = json.loads(resources_path.read_text())
    resources["sourceReportSha256"] = hashlib.sha256(
        report_path.read_bytes()
    ).hexdigest()
    resources["clusterIdentity"]["observed"][
        "kubeSystemNamespaceUid"
    ] = "another-cluster-uid"
    Path(resources["clusterIdentity"]["approvedPath"]).write_text(
        json.dumps({**resources["clusterIdentity"]["approved"], "approvedBy": "other"})
    )
    resources_path.write_text(json.dumps(resources))

    result = certify_run(
        report_path,
        resources_path,
        load_policy(matrix_path),
        "gpu-soak-15m",
    )

    assert result["passed"] is False
    assert result["checks"]["report.exactWorkloadProfiles"] is False
    assert result["checks"]["binding.workspace"] is False
    assert result["checks"]["binding.apiBase"] is False
    assert result["checks"]["binding.clusterIdentityObserved"] is False
    assert result["checks"]["binding.clusterIdentityFileBinding"] is False


def test_rejects_stalled_output_expired_watch_and_restart_delta(tmp_path):
    matrix_path = write_matrix(tmp_path)
    report_path, resources_path = write_artifacts(tmp_path)
    report = json.loads(report_path.read_text())
    for sample in report["samples"]:
        sample["jobs"][0]["stats"]["counters"]["published"] = 1
    report["watchLeases"]["job-a"]["lastRequestedAt"] = datetime(
        2025, 12, 31, tzinfo=timezone.utc
    ).isoformat()
    report_path.write_text(json.dumps(report))
    resources = json.loads(resources_path.read_text())
    resources["sourceReportSha256"] = hashlib.sha256(
        report_path.read_bytes()
    ).hexdigest()
    restart_values = resources["metrics"]["processorContainerRestarts"]["series"][0][
        "values"
    ]
    restart_values[-1][1] = "1"
    resources_path.write_text(json.dumps(resources))

    result = certify_run(
        report_path,
        resources_path,
        load_policy(matrix_path),
        "gpu-soak-15m",
    )

    assert result["passed"] is False
    assert result["checks"]["report.outputNeverStalled"] is False
    assert result["checks"]["watch.allLeasesRenewed"] is False
    assert result["checks"]["resources.processorRestartCounterStable"] is False


def test_rejects_relay_replacement_hidden_by_reset_restart_counter(tmp_path):
    matrix_path = write_matrix(tmp_path)
    report_path, resources_path = write_artifacts(tmp_path)
    resources = json.loads(resources_path.read_text())
    identity_metric = resources["metrics"]["relayPodIdentity"]
    old_series = identity_metric["series"][0]
    split = len(old_series["values"]) // 2 + 1
    old_series["values"] = old_series["values"][:split]
    identity_metric["series"].append(
        {
            "metric": {"pod": "mediamtx-b", "uid": "relay-uid-b"},
            "values": [
                list(item)
                for item in resources["metrics"]["relayReaders"]["series"][0]["values"][
                    split - 1 :
                ]
            ],
        }
    )
    resources["metrics"]["relayContainerRestarts"]["series"] = [
        {
            "metric": {"pod": "mediamtx-b"},
            "values": [
                [timestamp, "0"]
                for timestamp, _value in resources["metrics"]["relayReaders"]["series"][
                    0
                ]["values"]
            ],
        }
    ]
    resources_path.write_text(json.dumps(resources))

    result = certify_run(
        report_path,
        resources_path,
        load_policy(matrix_path),
        "gpu-soak-15m",
    )

    assert result["passed"] is False
    assert result["checks"]["resources.relayRestartCounterStable"] is True
    assert result["checks"]["resources.relayPodIdentityStable"] is False


def test_ladder_blocks_higher_stage_without_predecessor(tmp_path):
    matrix_path = write_matrix(tmp_path)
    run_id = "suite-gpu-soak-01h-r1"
    write_artifacts(tmp_path, run_id=run_id, duration=20)
    suite_path = write_suite(
        tmp_path,
        matrix_path,
        [
            {
                "scenario": "gpu-soak-01h",
                "runId": run_id,
                "status": "completed",
                "returnCode": 0,
            }
        ],
    )

    result = certify_ladder(matrix_path, suite_path, tmp_path)
    gpu = result["families"]["gpu"]

    assert gpu["nextRequiredStage"] == "15m"
    assert gpu["stages"][1]["evidencePassed"] is True
    assert gpu["stages"][1]["predecessorsPassed"] is False
    assert gpu["stages"][1]["passed"] is False


def test_stage_promotion_recomputes_every_required_predecessor(tmp_path):
    matrix_path = write_matrix(tmp_path)
    config = load_policy(matrix_path)
    first_report, first_resources = write_artifacts(tmp_path)
    first = certify_stage(first_report, first_resources, config, "gpu-soak-15m")
    first_path = tmp_path / "gpu-soak-15m-certification.json"
    first_path.write_text(json.dumps(first))
    second_report, second_resources = write_artifacts(
        tmp_path, run_id="suite-gpu-soak-01h-r1", duration=20
    )

    with pytest.raises(ValueError, match="missing required prior"):
        certify_stage(second_report, second_resources, config, "gpu-soak-01h")

    second = certify_stage(
        second_report,
        second_resources,
        config,
        "gpu-soak-01h",
        [first_path],
    )
    assert second["passed"] is True
    assert second["nextStage"] == "04h"
    assert (
        second["priorCertifications"][0]["sha256"]
        == hashlib.sha256(first_path.read_bytes()).hexdigest()
    )

    first_report.write_text(first_report.read_text() + "\n")
    with pytest.raises(ValueError, match="not reproducible"):
        certify_stage(
            second_report,
            second_resources,
            config,
            "gpu-soak-01h",
            [first_path],
        )


def test_checked_in_ladder_has_exact_order_and_output_enabled():
    config = load_policy(BENCHMARK_DIR / "matrices" / "long-soak.staging.example.json")

    assert config["policy"]["requiredSequence"] == SEQUENCE
    assert len(config["scenarios"]) == 8
    assert all(item["publishOutput"] for item in config["scenarios"].values())


def test_rejects_ladder_policy_drift(tmp_path):
    matrix_path = write_matrix(tmp_path)
    matrix = json.loads(matrix_path.read_text())
    matrix["soakPolicy"]["requiredSequence"] = ["15m", "12h"]
    matrix_path.write_text(json.dumps(matrix))

    with pytest.raises(ValueError, match="exact 15m/01h/04h/12h"):
        load_policy(matrix_path)
