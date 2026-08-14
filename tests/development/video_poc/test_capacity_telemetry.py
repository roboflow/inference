import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

BENCHMARK_DIR = (
    Path(__file__).resolve().parents[3] / "development" / "video_poc" / "benchmarks"
)
sys.path.insert(0, str(BENCHMARK_DIR))

from collect_staging_capacity_telemetry import (  # noqa: E402
    canonical_sha256,
    load_approved_cluster_identity,
    metric_queries,
    report_processor_pods,
    summarize,
    validate_cluster_identity,
)


def write_cluster_identity(tmp_path, **updates):
    identity = {
        "schemaVersion": 1,
        "environment": "staging",
        "context": "ck8s-stg",
        "apiServer": "https://staging-kubernetes.example.test",
        "kubeSystemNamespaceUid": "staging-cluster-uid",
        "approvedAt": "2026-08-13T00:00:00Z",
        "validUntil": "2026-08-15T00:00:00Z",
        "approvedBy": "staging-platform-owner",
    }
    identity.update(updates)
    path = tmp_path / "staging-cluster-identity.json"
    path.write_text(json.dumps(identity))
    return path, identity


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
    assert 'exported_pod=~"processor-a|processor-b"' in queries["gpuDecoderUtilPercent"]
    assert "paths_readers" in queries["relayReaders"]
    assert (
        "kube_pod_container_status_restarts_total"
        in queries["processorContainerRestarts"]
    )
    assert (
        "kube_pod_container_status_restarts_total" in queries["relayContainerRestarts"]
    )
    assert "max by (pod, uid) (kube_pod_info" in queries["relayPodIdentity"]
    assert 'name=~"out-.*"' in queries["relayOutputIngressBytesPerSecond"]
    assert "sum by (pod)" in queries["relayOutputIngressBytesPerSecond"]


def test_summary_uses_all_finite_samples_and_interpolated_p95():
    summary = summarize([[1, 1.0], [2, 2.0], [3, 3.0], [4, 4.0]])

    assert summary == {
        "count": 4,
        "min": 1.0,
        "mean": 2.5,
        "p95": 3.85,
        "max": 4.0,
    }


def test_cluster_identity_requires_current_independent_approval(tmp_path):
    path, identity = write_cluster_identity(tmp_path)
    now = datetime(2026, 8, 14, tzinfo=timezone.utc)

    assert load_approved_cluster_identity(path, now=now) == identity
    assert len(canonical_sha256(identity)) == 64

    path, _identity = write_cluster_identity(
        tmp_path, validUntil="2026-08-13T12:00:00Z"
    )
    with pytest.raises(ValueError, match="not currently valid"):
        load_approved_cluster_identity(path, now=now)


def test_cluster_identity_checks_local_server_before_any_live_request(
    tmp_path, monkeypatch
):
    path, _identity = write_cluster_identity(tmp_path)
    calls = []

    def run(command, **_kwargs):
        calls.append(command)
        return SimpleNamespace(
            stdout=json.dumps(
                {
                    "contexts": [
                        {"name": "ck8s-stg", "context": {"cluster": "staging"}}
                    ],
                    "clusters": [
                        {
                            "name": "staging",
                            "cluster": {"server": "https://production.example.test"},
                        }
                    ],
                }
            ).encode()
        )

    monkeypatch.setattr("collect_staging_capacity_telemetry.subprocess.run", run)

    with pytest.raises(ValueError, match="not the approved staging server"):
        validate_cluster_identity(
            "ck8s-stg",
            path,
            now=datetime(2026, 8, 14, tzinfo=timezone.utc),
        )
    assert calls == [["kubectl", "config", "view", "--raw", "-o", "json"]]


def test_cluster_identity_binds_live_immutable_namespace_uid(tmp_path, monkeypatch):
    path, identity = write_cluster_identity(tmp_path)
    responses = [
        {
            "contexts": [{"name": "ck8s-stg", "context": {"cluster": "staging"}}],
            "clusters": [
                {"name": "staging", "cluster": {"server": identity["apiServer"]}}
            ],
        },
        {"metadata": {"uid": identity["kubeSystemNamespaceUid"]}},
    ]
    calls = []

    def run(command, **_kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=json.dumps(responses.pop(0)).encode())

    monkeypatch.setattr("collect_staging_capacity_telemetry.subprocess.run", run)

    result = validate_cluster_identity(
        "ck8s-stg",
        path,
        now=datetime(2026, 8, 14, tzinfo=timezone.utc),
    )

    assert result["approvedSha256"] == canonical_sha256(identity)
    assert result["approvedPath"] == str(path.resolve())
    assert result["approvedFileSha256"]
    assert (
        result["observed"]["kubeSystemNamespaceUid"]
        == identity["kubeSystemNamespaceUid"]
    )
    assert calls[1] == [
        "kubectl",
        "--context",
        "ck8s-stg",
        "get",
        "namespace",
        "kube-system",
        "-o",
        "json",
    ]
