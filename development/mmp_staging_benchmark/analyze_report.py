#!/usr/bin/env python3
"""Apply the strict MMP staging capacity gates to one completed JSON report."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

CUDA_FAILURE = re.compile(
    r"cuda (?:error|out of memory)|illegal memory access|nvrm:\s*xid",
    re.IGNORECASE,
)


def _worker_pids(report: Mapping[str, Any]) -> dict[str, set[int]]:
    pids: dict[str, set[int]] = {}
    for sample in report.get("metrics_samples", []):
        if sample.get("status") != 200:
            continue
        for model_id, model in sample.get("metrics", {}).get("mmp_models", {}).items():
            pid = model.get("worker_pid")
            if pid is not None:
                pids.setdefault(model_id, set()).add(int(pid))
    return pids


def _environment(container: Mapping[str, Any]) -> dict[str, str]:
    return {
        item["name"]: str(item["value"])
        for item in container.get("env", [])
        if "name" in item and "value" in item
    }


def _gpu_identity(capability: Mapping[str, Any]) -> tuple[str | None, str | None]:
    query = capability.get("nvidia_runtime", {}).get("gpu_query", {})
    if query.get("exit_code") != 0:
        return None, None
    rows = [row.strip() for row in str(query.get("stdout", "")).splitlines() if row]
    if len(rows) != 1:
        return None, None
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) < 6:
        return None, None
    return fields[1], fields[2]


def _pod_evidence(
    pod: Mapping[str, Any], capability: Mapping[str, Any]
) -> dict[str, Any]:
    containers = pod.get("spec", {}).get("containers", [])
    statuses = pod.get("status", {}).get("containerStatuses", [])
    server = next((item for item in containers if item.get("name") == "server"), {})
    server_status = next(
        (item for item in statuses if item.get("name") == "server"), {}
    )
    name, uuid = _gpu_identity(capability)
    return {
        "pod": pod.get("metadata", {}).get("name"),
        "node": pod.get("spec", {}).get("nodeName"),
        "annotations": pod.get("metadata", {}).get("annotations", {}),
        "desired_image": server.get("image"),
        "runtime_image_id": server_status.get("imageID"),
        "restart_count": server_status.get("restartCount"),
        "ready": server_status.get("ready"),
        "env": _environment(server),
        "gpu_name": name,
        "gpu_uuid": uuid,
    }


def analyze(
    report: Mapping[str, Any],
    *,
    phase: str,
    latency_p95_ms_max: float = 50,
    jain_fairness_min: float = 0.95,
    minimum_metrics_coverage: float = 0.9,
    worker_restarts_max: int = 0,
    server_log: str = "",
    pod: Mapping[str, Any] | None = None,
    capability: Mapping[str, Any] | None = None,
    expected_node: str | None = None,
    expected_gpu_uuid: str | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    clients = report.get("clients", [])
    aggregate = report.get("aggregate", {})
    run = report.get("run", {})
    experiment = report.get("experiment", {})
    expected_mps = phase.endswith("-mps") and not phase.endswith("-no-mps")
    if pod is None or capability is None:
        failures.append("Kubernetes pod and capability evidence are required")
        identity: dict[str, Any] = {}
    else:
        identity = _pod_evidence(pod, capability)
    if not clients:
        failures.append("no client reports")
    if int(aggregate.get("errors", 0)) != 0:
        failures.append("request errors recorded")
    if int(aggregate.get("successes", 0)) != int(aggregate.get("requests", -1)):
        failures.append("success rate below 100%")
    for client in clients:
        p95 = client.get("latency_ms", {}).get("p95")
        if p95 is None or float(p95) > latency_p95_ms_max:
            failures.append(f"{client.get('client_id')}: latency p95 failed")
    if "mixed" not in phase and len(clients) > 1:
        fairness = aggregate.get("jain_fairness_delivered_fps")
        if fairness is None or float(fairness) < jain_fairness_min:
            failures.append("same-model Jain fairness failed")
    elapsed = float(run.get("warmup_s", 0)) + float(run.get("duration_s", 0))
    sample_count = int(report.get("metrics_evidence", {}).get("sample_count", 0))
    sample_interval = float(run.get("sample_interval_s", 1))
    expected_samples = max(1.0, elapsed / sample_interval)
    coverage = sample_count / expected_samples
    if coverage < minimum_metrics_coverage:
        failures.append("metrics coverage failed")
    rejects = int(
        report.get("metrics_evidence", {}).get("mmp_pool_full_rejects_delta", 0)
    )
    if rejects:
        failures.append("MMP pool-full rejects recorded")
    pids = _worker_pids(report)
    expected_routes = {client.get("routing_key") for client in clients}
    worker_restarts = sum(
        max(0, len(pids.get(route, set())) - 1) for route in expected_routes
    )
    if worker_restarts > worker_restarts_max:
        failures.append("model worker PID missing or changed")
    if not expected_routes.issubset(pids):
        failures.append("expected model worker routing key is missing")
    model_deltas = report.get("metrics_evidence", {}).get("model_deltas", {})
    run_batch_sizes: dict[str, float | None] = {}
    for route in expected_routes:
        delta = model_deltas.get(route, {})
        inference_count = int(delta.get("inference_count_delta", 0))
        batch_count = int(delta.get("batch_count_delta", 0))
        if inference_count <= 0 or batch_count <= 0:
            failures.append(f"{route}: no positive MMP inference/batch delta")
            run_batch_sizes[route] = None
        else:
            run_batch_sizes[route] = inference_count / batch_count
        if int(delta.get("error_count_delta", 0)) != 0:
            failures.append(f"{route}: model worker errors recorded")
    total_concurrency = sum(int(client.get("concurrency", 0)) for client in clients)
    if "shared" in phase and total_concurrency > 1:
        shared_route = next(iter(expected_routes), None)
        if shared_route is None or (run_batch_sizes.get(shared_route) or 0) <= 1.0:
            failures.append("shared backend did not produce a multi-request batch")
    if str(experiment.get("mps_enabled")) != ("1" if expected_mps else "0"):
        failures.append("report MPS identity does not match phase")
    if experiment.get("mode") != phase:
        failures.append("report phase identity does not match analyzer phase")
    if not experiment.get("run_id") or experiment.get("run_id") == "unknown":
        failures.append("run identity missing")
    if not re.fullmatch(r"[0-9a-f]{40}", str(experiment.get("harness_revision", ""))):
        failures.append("harness revision identity missing")
    if experiment.get("decoder") != "imagecodecs":
        failures.append("initial matrix must use imagecodecs decoder")
    expected_runtime = {
        "slots": 128,
        "input_mb_per_slot": 20,
        "batch_max_size": 0,
        "batch_max_wait_ms": 5,
    }
    for field, expected in expected_runtime.items():
        try:
            actual = int(experiment.get(field))
        except (TypeError, ValueError):
            actual = None
        if actual != expected:
            failures.append(f"runtime identity mismatch: {field}")
    if run.get("image_ref") != experiment.get("server_image_ref"):
        failures.append("server image identity mismatch")
    if run.get("source_revision") != experiment.get("server_source_revision"):
        failures.append("server source identity mismatch")
    if run.get("pod_name") in {None, "", "unknown"}:
        failures.append("server pod identity missing")
    if run.get("node_name") in {None, "", "unknown"}:
        failures.append("server node identity missing")
    elif run.get("node_name") != experiment.get("server_node"):
        failures.append("server node identity mismatch")
    if run.get("pod_name") != experiment.get("server_pod"):
        failures.append("server pod identity mismatch")
    fixture_sha256 = experiment.get("fixture_sha256")
    if not fixture_sha256 or fixture_sha256 == "unknown":
        failures.append("fixture digest identity missing")
    elif run.get("image_sha256") != fixture_sha256:
        failures.append("fixture digest does not match report image")
    if identity:
        annotations = identity["annotations"]
        env = identity["env"]
        if identity["ready"] is not True:
            failures.append("server container was not ready in pod evidence")
        if int(identity["restart_count"] or 0) != 0:
            failures.append("server container restart recorded")
        if identity["pod"] != run.get("pod_name"):
            failures.append("pod evidence identity mismatch")
        if identity["node"] != run.get("node_name"):
            failures.append("node evidence identity mismatch")
        if expected_node is None or identity["node"] != expected_node:
            failures.append("pod is not pinned to the expected capability node")
        if expected_gpu_uuid is None or identity["gpu_uuid"] != expected_gpu_uuid:
            failures.append("GPU UUID does not match capability baseline")
        if identity["gpu_name"] != "NVIDIA L40S":
            failures.append("capability evidence is not one L40S")
        if capability.get("checks", {}).get("passed") is not True:
            failures.append("capability checks did not pass")
        if capability.get("source_revision") != experiment.get(
            "server_source_revision"
        ):
            failures.append("capability source revision mismatch")
        if capability.get("image_ref") != experiment.get("server_image_ref"):
            failures.append("capability image reference mismatch")
        if identity["desired_image"] != experiment.get("server_image_ref"):
            failures.append("pod desired image mismatch")
        digest = str(experiment.get("server_image_ref", "")).split("@", 1)[-1]
        if digest not in str(identity["runtime_image_id"]):
            failures.append("pod runtime image digest mismatch")
        expected_annotations = {
            "roboflow.com/source-revision": experiment.get("server_source_revision"),
            "roboflow.com/run-id": experiment.get("run_id"),
            "roboflow.com/mps": "enabled" if expected_mps else "disabled",
            "roboflow.com/decoder": experiment.get("decoder"),
        }
        if any(
            annotations.get(key) != value for key, value in expected_annotations.items()
        ):
            failures.append("pod annotations do not match report identity")
        expected_env = {
            "NVIDIA_MPS": "1" if expected_mps else "0",
            "INFERENCE_DECODER": "imagecodecs",
            "INFERENCE_N_SLOTS": "128",
            "INFERENCE_INPUT_MB": "20",
            "INFERENCE_BATCH_MAX_SIZE": "0",
            "INFERENCE_BATCH_MAX_WAIT_MS": "5",
        }
        if any(env.get(key) != value for key, value in expected_env.items()):
            failures.append("pod runtime environment does not match matrix")
    if CUDA_FAILURE.search(server_log):
        failures.append("CUDA failure found in server log")
    if not server_log.strip():
        failures.append("server log evidence is empty")
    return {
        "schema_version": 1,
        "phase": phase,
        "success": not failures,
        "failures": failures,
        "evidence": {
            "requests": aggregate.get("requests"),
            "successes": aggregate.get("successes"),
            "latency_p95_ms_by_client": {
                client.get("client_id"): client.get("latency_ms", {}).get("p95")
                for client in clients
            },
            "jain_fairness_delivered_fps": aggregate.get("jain_fairness_delivered_fps"),
            "metrics_coverage": coverage,
            "mmp_pool_full_rejects_delta": rejects,
            "worker_pids": {
                model_id: sorted(values) for model_id, values in pids.items()
            },
            "worker_restarts": worker_restarts,
            "run_average_batch_size": run_batch_sizes,
            "total_concurrency": total_concurrency,
            "run_id": experiment.get("run_id"),
            "log_sha256": hashlib.sha256(server_log.encode()).hexdigest(),
            "extra_loaded_routes": sorted(set(pids) - expected_routes),
            "mps_enabled": experiment.get("mps_enabled"),
            "fixture_sha256": fixture_sha256,
            "pod_evidence": identity,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--pod-evidence", type=Path, required=True)
    parser.add_argument("--capability-report", type=Path, required=True)
    parser.add_argument("--expected-node", required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = json.loads(args.report.read_text())
    result = analyze(
        report,
        phase=args.phase,
        server_log=args.server_log.read_text(),
        pod=json.loads(args.pod_evidence.read_text()),
        capability=json.loads(args.capability_report.read_text()),
        expected_node=args.expected_node,
        expected_gpu_uuid=args.expected_gpu_uuid,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")
    return 0 if result["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
