#!/usr/bin/env python3
"""Apply the strict MMP staging capacity gates to one completed JSON report."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from development.mmp_staging_benchmark.render_run_spec import (  # noqa: E402
    current_clean_revision,
    render_point,
)
from development.mmp_staging_benchmark.run_concurrent_clients import (  # noqa: E402
    ClientSpec,
)
from development.mmp_staging_benchmark.validate_staging_plan import (  # noqa: E402
    canonical_sha256,
    load_and_validate,
)

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
        "pod_uid": pod.get("metadata", {}).get("uid"),
        "namespace": pod.get("metadata", {}).get("namespace"),
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


def _raw_metrics(report: Mapping[str, Any], routes: set[str]) -> dict[str, Any]:
    samples = [
        sample
        for sample in report.get("metrics_samples", [])
        if sample.get("status") == 200 and isinstance(sample.get("metrics"), dict)
    ]
    rejects = []
    route_samples: dict[str, list[tuple[float, Mapping[str, Any]]]] = {
        route: [] for route in routes
    }
    for sample in samples:
        offset = float(sample.get("offset_s", 0))
        metrics = sample["metrics"]
        if metrics.get("mmp_rejects_pool_full") is not None:
            rejects.append(int(metrics["mmp_rejects_pool_full"]))
        models = metrics.get("mmp_models") or {}
        for route in routes:
            if isinstance(models.get(route), dict):
                route_samples[route].append((offset, models[route]))
    deltas = {}
    route_evidence = {}
    counter_fields = ("inference_count", "batch_count", "error_count")
    first_global_models = (
        samples[0]["metrics"].get("mmp_models") or {} if samples else {}
    )
    for route, timed_models in route_samples.items():
        models = [model for _offset, model in timed_models]
        offsets = [offset for offset, _model in timed_models]
        complete = all(
            all(field in model for field in (*counter_fields, "worker_pid"))
            for model in models
        )
        monotonic = complete and all(
            int(later[field]) >= int(earlier[field])
            for earlier, later in zip(models, models[1:])
            for field in counter_fields
        )
        route_evidence[route] = {
            "sample_count": len(models),
            "span_s": max(offsets) - min(offsets) if len(offsets) >= 2 else 0.0,
            "complete": complete,
            "monotonic": monotonic,
        }
        if len(models) < 2:
            continue
        first = first_global_models.get(route) or {}
        last = models[-1]
        if not complete:
            continue
        deltas[route] = {
            "inference_count_delta": int(last["inference_count"])
            - int(first.get("inference_count", 0)),
            "batch_count_delta": int(last["batch_count"])
            - int(first.get("batch_count", 0)),
            "error_count_delta": int(last["error_count"])
            - int(first.get("error_count", 0)),
        }
    offsets = [float(sample.get("offset_s", 0)) for sample in samples]
    return {
        "sample_count": len(samples),
        "span_s": max(offsets) - min(offsets) if len(offsets) >= 2 else 0.0,
        "rejects_complete": len(rejects) == len(samples) and len(rejects) >= 2,
        "rejects_monotonic": all(
            later >= earlier for earlier, later in zip(rejects, rejects[1:])
        ),
        "rejects_delta": rejects[-1] - rejects[0] if len(rejects) >= 2 else None,
        "model_deltas": deltas,
        "route_evidence": route_evidence,
        "offsets_monotonic": all(
            later > earlier for earlier, later in zip(offsets, offsets[1:])
        ),
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
    matrix: Mapping[str, Any] | None = None,
    matrix_dir: Path | None = None,
    mps_evidence: Mapping[str, Any] | None = None,
    cache_evidence: Mapping[str, Any] | None = None,
    expected_harness_revision: str | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    clients = report.get("clients", [])
    aggregate = report.get("aggregate", {})
    run = report.get("run", {})
    experiment = report.get("experiment", {})
    expected_mps = phase.endswith("-mps") and not phase.endswith("-no-mps")
    phase_config = next(
        (item for item in (matrix or {}).get("phases", []) if item.get("id") == phase),
        None,
    )
    if phase_config is None or matrix_dir is None:
        failures.append("checked-in matrix evidence is required")
    else:
        gates = matrix["strict_gates"]
        latency_p95_ms_max = float(gates["latency_p95_ms_max"])
        jain_fairness_min = float(gates["jain_fairness_min"])
        minimum_metrics_coverage = float(gates["minimum_metrics_coverage"])
        worker_restarts_max = int(gates["worker_restarts_max"])
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
    raw = _raw_metrics(report, expected_routes)
    raw_coverage = raw["sample_count"] / expected_samples
    if raw_coverage < minimum_metrics_coverage:
        failures.append("raw metrics coverage failed")
    if raw["span_s"] < max(0.0, elapsed - sample_interval) * minimum_metrics_coverage:
        failures.append("raw metrics interval span failed")
    if sample_count != raw["sample_count"]:
        failures.append("derived metrics sample count differs from raw samples")
    if (
        not raw["rejects_complete"]
        or not raw["rejects_monotonic"]
        or raw["rejects_delta"] != 0
    ):
        failures.append("raw MMP pool-reject evidence missing or nonzero")
    if not raw["offsets_monotonic"]:
        failures.append("raw metrics sample offsets are not strictly increasing")
    worker_restarts = sum(
        max(0, len(pids.get(route, set())) - 1) for route in expected_routes
    )
    if worker_restarts > worker_restarts_max:
        failures.append("model worker PID missing or changed")
    if not expected_routes.issubset(pids):
        failures.append("expected model worker routing key is missing")
    cache_state = experiment.get("cache_state")
    if not cache_evidence:
        failures.append("pre-run MMP/cache evidence is required")
    else:
        captured = float(cache_evidence.get("captured_unix_s", 0))
        run_started = float(run.get("started_unix_s", 0))
        if not 0 <= run_started - captured <= 60:
            failures.append("cache evidence was not captured immediately before run")
        if (
            cache_evidence.get("context") != "ck8s-stg"
            or cache_evidence.get("namespace") != "video-proc-bench-mmp"
            or cache_evidence.get("pod") != run.get("pod_name")
            or cache_evidence.get("pod_uid") != identity.get("pod_uid")
        ):
            failures.append("cache evidence staging pod UID mismatch")
        cache_routes = cache_evidence.get("routes") or {}
        cache_count = cache_evidence.get("cache_file_count")
        cache_digest = str(cache_evidence.get("cache_paths_sha256", ""))
        if not isinstance(cache_routes, dict) or not re.fullmatch(
            r"[0-9a-f]{64}", cache_digest
        ):
            failures.append("cache evidence schema is invalid")
        elif cache_state == "cold":
            if (
                cache_routes
                or cache_count != 0
                or cache_digest != hashlib.sha256(b"").hexdigest()
            ):
                failures.append("cold run did not start with empty routes/cache")
        elif cache_state == "warm":
            if (
                set(cache_routes) != expected_routes
                or not isinstance(cache_count, int)
                or cache_count <= 0
            ):
                failures.append("warm run did not start with exact loaded routes/cache")
            for route in expected_routes:
                model = cache_routes.get(route) or {}
                if (
                    not isinstance(model.get("worker_pid"), int)
                    or int(model.get("inference_count") or 0) <= 0
                    or int(model.get("batch_count") or 0) <= 0
                    or int(model.get("error_count") or 0) != 0
                ):
                    failures.append(f"{route}: warm pre-run route evidence is invalid")
                if model.get("worker_pid") not in pids.get(route, set()):
                    failures.append(
                        f"{route}: warm pre-run worker PID differs from measured worker"
                    )
    model_deltas = report.get("metrics_evidence", {}).get("model_deltas", {})
    run_batch_sizes: dict[str, float | None] = {}
    for route in expected_routes:
        route_raw = raw["route_evidence"].get(route, {})
        if (
            not route_raw.get("complete")
            or not route_raw.get("monotonic")
            or route_raw.get("sample_count", 0) / expected_samples
            < minimum_metrics_coverage
            or route_raw.get("span_s", 0)
            < max(0.0, elapsed - sample_interval) * minimum_metrics_coverage
        ):
            failures.append(f"{route}: raw model metric coverage/continuity failed")
        delta = model_deltas.get(route, {})
        raw_delta = raw["model_deltas"].get(route)
        if raw_delta is None or any(
            int(delta.get(field, 0)) != value for field, value in raw_delta.items()
        ):
            failures.append(f"{route}: derived model deltas differ from raw samples")
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
    if ("isolated" in phase or "mixed" in phase) and len(expected_routes) > 1:
        if len(expected_routes) != len(clients):
            failures.append("isolated/mixed clients do not have distinct routing keys")
        stable = [next(iter(pids.get(route, set())), None) for route in expected_routes]
        if None in stable or len(set(stable)) != len(stable):
            failures.append("isolated/mixed routes do not have distinct worker PIDs")
    if str(experiment.get("mps_enabled")) != ("1" if expected_mps else "0"):
        failures.append("report MPS identity does not match phase")
    if experiment.get("mode") != phase:
        failures.append("report phase identity does not match analyzer phase")
    if phase_config is not None and matrix_dir is not None:
        fixed = matrix["fixed_runtime"]
        artifact = matrix["artifact"]
        template = matrix_dir / phase_config["spec"]
        if total_concurrency not in phase_config["total_concurrency"]:
            failures.append("concurrency is absent from matrix phase")
        expected_workload = render_point(
            template,
            total_concurrency,
            fixed["duration_s"],
            fixed["warmup_s"],
        )
        expected_clients = [
            ClientSpec(**client).__dict__ for client in expected_workload["clients"]
        ]
        if run.get("clients") != expected_clients:
            failures.append("report clients differ from matrix workload")
        expected_runtime_fields = {
            "server_url": expected_workload["server_url"],
            "duration_s": float(expected_workload["duration_s"]),
            "warmup_s": float(expected_workload["warmup_s"]),
            "sample_interval_s": float(expected_workload.get("sample_interval_s", 1)),
            "request_timeout_s": float(expected_workload.get("request_timeout_s", 60)),
            "max_latency_samples_per_client": int(
                expected_workload.get("max_latency_samples_per_client", 250_000)
            ),
        }
        if any(
            run.get(field) != expected
            for field, expected in expected_runtime_fields.items()
        ):
            failures.append("report timing differs from matrix workload")
        if experiment.get("matrix_sha256") != canonical_sha256(matrix):
            failures.append("matrix digest identity mismatch")
        if (
            experiment.get("template_sha256")
            != hashlib.sha256(template.read_bytes()).hexdigest()
        ):
            failures.append("template digest identity mismatch")
        if experiment.get("workload_sha256") != canonical_sha256(expected_workload):
            failures.append("workload digest identity mismatch")
        if (
            experiment.get("server_image_ref") != artifact["image"]
            or experiment.get("server_source_revision") != artifact["source_revision"]
        ):
            failures.append("server artifact differs from checked-in matrix")
        if experiment.get("fixture_sha256") != fixed["fixture_sha256"]:
            failures.append("fixture identity differs from checked-in matrix")
    if cache_state not in {"cold", "warm"}:
        failures.append("cache state identity missing")
    if not experiment.get("run_id") or experiment.get("run_id") == "unknown":
        failures.append("run identity missing")
    if not re.fullmatch(r"[0-9a-f]{40}", str(experiment.get("harness_revision", ""))):
        failures.append("harness revision identity missing")
    if (
        expected_harness_revision is None
        or experiment.get("harness_revision") != expected_harness_revision
    ):
        failures.append("harness revision differs from clean analyzer checkout")
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
        if identity["namespace"] != "video-proc-bench-mmp" or not identity["pod_uid"]:
            failures.append("pod evidence namespace/UID mismatch")
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
    if expected_mps:
        if not mps_evidence:
            failures.append("live measured-interval MPS evidence is required")
        else:
            captured = float(mps_evidence.get("captured_unix_s", 0))
            if not (
                float(run.get("started_unix_s", 0)) + float(run.get("warmup_s", 0))
                <= captured
                <= float(run.get("finished_unix_s", 0))
            ):
                failures.append("MPS evidence was not captured during the run")
            if mps_evidence.get("pod") != run.get("pod_name"):
                failures.append("MPS evidence pod identity mismatch")
            if (
                mps_evidence.get("context") != "ck8s-stg"
                or mps_evidence.get("namespace") != "video-proc-bench-mmp"
                or mps_evidence.get("pod_uid") != identity.get("pod_uid")
            ):
                failures.append("MPS evidence staging pod UID mismatch")
            if mps_evidence.get("gpu_uuid") != identity.get("gpu_uuid"):
                failures.append("MPS evidence GPU identity mismatch")
            server_list = str(mps_evidence.get("server_list", "")).strip()
            if (
                mps_evidence.get("command") != "get_server_list"
                or mps_evidence.get("exit_code") != 0
            ):
                failures.append("MPS control query did not succeed")
            if not server_list or not all(
                line.strip().isdigit() for line in server_list.splitlines()
            ):
                failures.append("MPS server list is empty")
            server_pids = {line.strip() for line in server_list.splitlines()}
            clients_by_server = mps_evidence.get("clients_by_server") or {}
            if set(clients_by_server) != server_pids or not all(
                isinstance(clients, list)
                and all(isinstance(pid, int) and pid > 0 for pid in clients)
                for clients in clients_by_server.values()
            ):
                failures.append("MPS client-list evidence is incomplete")
            else:
                mps_clients = {
                    pid for clients in clients_by_server.values() for pid in clients
                }
                measured_workers = {
                    pid for route in expected_routes for pid in pids.get(route, set())
                }
                if not measured_workers.issubset(mps_clients):
                    failures.append("measured model workers are not all MPS clients")
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
            "raw_metrics_coverage": raw_coverage,
            "raw_metrics_span_s": raw["span_s"],
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
            "matrix_sha256": experiment.get("matrix_sha256"),
            "template_sha256": experiment.get("template_sha256"),
            "workload_sha256": experiment.get("workload_sha256"),
            "cache_state": experiment.get("cache_state"),
            "cache_evidence_sha256": (
                canonical_sha256(cache_evidence) if cache_evidence else None
            ),
            "server_image_ref": experiment.get("server_image_ref"),
            "server_source_revision": experiment.get("server_source_revision"),
            "harness_revision": experiment.get("harness_revision"),
            "server_node": run.get("node_name"),
            "gpu_uuid": identity.get("gpu_uuid"),
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
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--mps-evidence", type=Path)
    parser.add_argument("--cache-evidence", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = json.loads(args.report.read_text())
    matrix = load_and_validate(args.matrix)
    harness_revision = current_clean_revision(Path(__file__).resolve().parents[2])
    result = analyze(
        report,
        phase=args.phase,
        server_log=args.server_log.read_text(),
        pod=json.loads(args.pod_evidence.read_text()),
        capability=json.loads(args.capability_report.read_text()),
        expected_node=args.expected_node,
        expected_gpu_uuid=args.expected_gpu_uuid,
        matrix=matrix,
        matrix_dir=args.matrix.parent,
        mps_evidence=(
            json.loads(args.mps_evidence.read_text()) if args.mps_evidence else None
        ),
        cache_evidence=json.loads(args.cache_evidence.read_text()),
        expected_harness_revision=harness_revision,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")
    return 0 if result["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
