#!/usr/bin/env python3
"""Dry-run-first fault controller for staging video benchmark recovery.

The default mode only validates and renders a plan. Execution deletes one exact,
controller-owned pod and observes bounded recovery. It never lists or mutates a
production context and it never sends requests to the video service API.
"""

import argparse
import hashlib
import json
import os
import re
import signal
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


DNS_LABEL = re.compile(r"^[a-z0-9](?:[-a-z0-9]*[a-z0-9])?$")
STAGING_CONTEXT = re.compile(
    r"(?:^|[-_.])(stg|staging)(?:$|[-_.])", re.IGNORECASE
)
ALLOWED_NAMESPACE = re.compile(r"^video-proc(?:-bench(?:-[a-z0-9-]+)?)?$")
PRODUCTION_MARKER = re.compile(r"(?:^|[-_.])(prod|production)(?:$|[-_.])", re.I)
STAGING_API_HOSTS = {
    "api.roboflow.one",
    "us-central1-roboflow-staging.cloudfunctions.net",
}
ALLOWED_STAGING_CLUSTERS = {
    "ck8s-stg": {
        "cluster": "ck8s-stg",
        "server": (
            "https://ck8s-stg-83c07ac7.us-east1-a."
            "cmk.crusoecloudcompute.com"
        ),
    }
}
FAULT_TYPES = {"processor-pod-loss", "relay-pod-loss"}
PHASES = {"startup", "steady-state"}
READY_STATES = {"running"}
TERMINAL_STATES = {
    "cancelled",
    "canceled",
    "complete",
    "completed",
    "error",
    "failed",
    "stopped",
}
MANAGED_POD_OWNERS = {"ReplicaSet", "StatefulSet"}


def canonical_digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _dns_label(value, field):
    value = str(value or "")
    if len(value) > 63 or not DNS_LABEL.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase Kubernetes DNS label")
    return value


def _positive(value, field):
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric") from error
    if value <= 0:
        raise ValueError(f"{field} must be positive")
    return value


def _bounded_positive(value, field, maximum):
    value = _positive(value, field)
    if value > maximum:
        raise ValueError(f"{field} cannot exceed {maximum} seconds")
    return value


def _selector(raw):
    if not isinstance(raw, dict) or not raw:
        raise ValueError("fault.selector must be a non-empty equality-label object")
    result = {}
    for key, value in raw.items():
        key, value = str(key), str(value)
        if not key or not value or any(token in key + value for token in ",=!()"):
            raise ValueError("fault.selector only supports exact equality labels")
        result[key] = value
    return dict(sorted(result.items()))


def selector_text(selector):
    return ",".join(f"{key}={value}" for key, value in sorted(selector.items()))


def validate_scenario(raw, source_path=None):
    if not isinstance(raw, dict) or raw.get("schemaVersion") != 1:
        raise ValueError("scenario schemaVersion must be 1")
    if raw.get("environment") != "staging":
        raise ValueError("fault injection is restricted to staging")

    context = str(raw.get("clusterContext") or "")
    if (
        PRODUCTION_MARKER.search(context)
        or not STAGING_CONTEXT.search(context)
        or context not in ALLOWED_STAGING_CLUSTERS
    ):
        raise ValueError(
            "clusterContext must be an explicitly allowlisted staging context"
        )
    namespace = str(raw.get("namespace") or "")
    if PRODUCTION_MARKER.search(namespace) or not ALLOWED_NAMESPACE.fullmatch(
        namespace
    ):
        raise ValueError("namespace must be video-proc or a video-proc-bench namespace")

    benchmark = raw.get("benchmark") or {}
    run_id = _dns_label(benchmark.get("runId"), "benchmark.runId")
    checkpoint = Path(str(benchmark.get("checkpoint") or "")).expanduser()
    if not checkpoint.is_absolute() and source_path:
        checkpoint = Path(source_path).resolve().parent / checkpoint
    if not checkpoint.name or run_id not in checkpoint.name:
        raise ValueError("benchmark.checkpoint filename must contain the exact runId")
    api_host = str(benchmark.get("apiHost") or "")
    if api_host not in STAGING_API_HOSTS:
        raise ValueError("benchmark.apiHost must be an allowlisted staging API host")

    fault = raw.get("fault") or {}
    fault_type = str(fault.get("type") or "")
    if fault_type not in FAULT_TYPES:
        raise ValueError(f"fault.type must be one of {sorted(FAULT_TYPES)}")
    phase = str(fault.get("phase") or "")
    if phase not in PHASES:
        raise ValueError(f"fault.phase must be one of {sorted(PHASES)}")
    normalized_fault = {
        "name": _dns_label(fault.get("name"), "fault.name"),
        "type": fault_type,
        "phase": phase,
        "gracePeriodSeconds": int(fault.get("gracePeriodSeconds", 0)),
    }
    if normalized_fault["gracePeriodSeconds"] < 0:
        raise ValueError("fault.gracePeriodSeconds cannot be negative")
    if fault_type == "processor-pod-loss":
        ordinal = int(fault.get("jobOrdinal", 1))
        if ordinal < 1:
            raise ValueError("fault.jobOrdinal must be positive")
        normalized_fault["jobOrdinal"] = ordinal
        if fault.get("selector") is not None:
            raise ValueError("processor target is derived from the exact benchmark job")
    else:
        normalized_fault["selector"] = _selector(fault.get("selector"))

    deadlines = raw.get("deadlines") or {}
    normalized = {
        "schemaVersion": 1,
        "environment": "staging",
        "name": _dns_label(raw.get("name"), "name"),
        "clusterContext": context,
        "namespace": namespace,
        "benchmark": {
            "runId": run_id,
            "checkpoint": str(checkpoint.resolve()),
            "apiHost": api_host,
        },
        "fault": normalized_fault,
        "deadlines": {
            "triggerSeconds": _bounded_positive(
                deadlines.get("triggerSeconds", 180),
                "deadlines.triggerSeconds",
                3600,
            ),
            "recoverySeconds": _bounded_positive(
                deadlines.get("recoverySeconds", 180),
                "deadlines.recoverySeconds",
                3600,
            ),
            "pollSeconds": _bounded_positive(
                deadlines.get("pollSeconds", 2), "deadlines.pollSeconds", 60
            ),
        },
    }
    if source_path:
        normalized["scenarioPath"] = str(Path(source_path).resolve())
    normalized["scenarioDigest"] = canonical_digest(normalized)
    return normalized


def load_scenario(path):
    with Path(path).open() as source:
        return validate_scenario(json.load(source), path)


def render_plan(scenario):
    fault = scenario["fault"]
    target = (
        {
            "source": "benchmark-checkpoint",
            "runId": scenario["benchmark"]["runId"],
            "jobOrdinal": fault["jobOrdinal"],
        }
        if fault["type"] == "processor-pod-loss"
        else {"source": "exact-label-selector", "selector": fault["selector"]}
    )
    plan = {
        "schemaVersion": 1,
        "mode": "dry-run",
        "scenarioDigest": scenario["scenarioDigest"],
        "environment": scenario["environment"],
        "clusterContext": scenario["clusterContext"],
        "namespace": scenario["namespace"],
        "benchmarkRunId": scenario["benchmark"]["runId"],
        "fault": {
            "name": fault["name"],
            "type": fault["type"],
            "phase": fault["phase"],
            "target": target,
        },
        "executionGuards": [
            "--execute is present",
            "--confirm-run-id exactly matches benchmarkRunId",
            "current kubectl context exactly matches clusterContext",
            "target resolves to exactly one controller-owned pod",
            "captured pod UID is rechecked immediately before deletion",
        ],
        "writeBoundary": (
            "delete exactly one captured pod by name after an immediate UID recheck"
        ),
        "deadlines": scenario["deadlines"],
    }
    plan["planDigest"] = canonical_digest(plan)
    return plan


def read_checkpoint(path, run_id, expected_host=None):
    try:
        with Path(path).open() as source:
            checkpoint = json.load(source)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if checkpoint.get("runId") != run_id:
        raise RuntimeError("checkpoint runId does not match the authorized run")
    api_base = urlparse(str(checkpoint.get("apiBase") or ""))
    if api_base.scheme != "https" or api_base.hostname not in STAGING_API_HOSTS:
        raise RuntimeError("checkpoint API base is not staging")
    if expected_host and api_base.hostname != expected_host:
        raise RuntimeError("checkpoint API host does not match the scenario")
    return checkpoint


def checkpoint_job(checkpoint, ordinal):
    starts = checkpoint.get("starts") or []
    matching = [item for item in starts if item.get("ordinal") == ordinal]
    if len(matching) != 1:
        return None
    job_id = (matching[0].get("job") or {}).get("id")
    jobs = [job for job in checkpoint.get("jobs") or [] if job.get("id") == job_id]
    return jobs[0] if len(jobs) == 1 else None


def trigger_target(scenario, checkpoint):
    fault = scenario["fault"]
    phase = (checkpoint.get("checkpoint") or {}).get("phase")
    jobs = checkpoint.get("jobs") or []
    if fault["phase"] == "startup":
        if phase != "fault-ready":
            return None
    else:
        if phase != "measurement" or not jobs:
            return None
        if any(job.get("state") not in READY_STATES for job in jobs):
            return None

    if fault["type"] == "relay-pod-loss":
        return {"selector": fault["selector"], "benchmarkJob": None}
    job = checkpoint_job(checkpoint, fault["jobOrdinal"])
    if not job or not job.get("processorId"):
        return None
    if fault["phase"] == "startup" and job.get("state") != "claimed":
        raise RuntimeError(
            "startup target must be claimed in the explicit fault-ready window"
        )
    if job.get("state") in TERMINAL_STATES:
        raise RuntimeError("selected benchmark job is already terminal")
    pod_name = _dns_label(job["processorId"], "checkpoint processorId")
    return {
        "selector": {"metadata.name": pod_name},
        "podName": pod_name,
        "benchmarkJob": {"id": job["id"], "ordinal": fault["jobOrdinal"]},
    }


def pod_ready(pod):
    conditions = (pod.get("status") or {}).get("conditions") or []
    return any(
        item.get("type") == "Ready" and item.get("status") == "True"
        for item in conditions
    )


def capture_exact_pod(pods, expected_name=None):
    if len(pods) != 1:
        raise RuntimeError(f"target must resolve to exactly one pod, got {len(pods)}")
    pod = pods[0]
    metadata = pod.get("metadata") or {}
    if expected_name and metadata.get("name") != expected_name:
        raise RuntimeError("resolved pod does not match checkpoint processorId")
    owners = metadata.get("ownerReferences") or []
    controllers = [owner for owner in owners if owner.get("controller") is True]
    if len(controllers) != 1 or controllers[0].get("kind") not in MANAGED_POD_OWNERS:
        raise RuntimeError("target must have one supported Kubernetes controller owner")
    uid = str(metadata.get("uid") or "")
    if not uid:
        raise RuntimeError("target pod has no UID")
    labels = metadata.get("labels") or {}
    if not labels:
        raise RuntimeError("target pod has no labels for replacement selection")
    image_ids = sorted(
        status.get("imageID")
        for status in (pod.get("status") or {}).get("containerStatuses") or []
        if status.get("imageID")
    )
    return {
        "name": metadata["name"],
        "uid": uid,
        "resourceVersion": metadata.get("resourceVersion"),
        "labels": dict(sorted(labels.items())),
        "owner": {
            "kind": controllers[0]["kind"],
            "name": controllers[0].get("name"),
            "uid": controllers[0].get("uid"),
        },
        "imageIds": image_ids,
        "ready": pod_ready(pod),
    }


class Kubectl:
    def __init__(self, context, namespace):
        self.context = context
        self.namespace = namespace

    def _run(self, args):
        command = [
            "kubectl",
            "--context",
            self.context,
            "--namespace",
            self.namespace,
            "--request-timeout=15s",
        ]
        result = subprocess.run(
            command + list(args), check=True, capture_output=True, text=True
        )
        return result.stdout

    def current_context(self):
        result = subprocess.run(
            ["kubectl", "config", "current-context"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def cluster_identity(self):
        output = self._run_config(
            [
                "view",
                "--minify",
                "-o",
                "jsonpath={.contexts[0].context.cluster}{'\\t'}"
                "{.clusters[0].cluster.server}",
            ]
        )
        cluster, separator, server = output.partition("\t")
        if not separator:
            raise RuntimeError("could not resolve active Kubernetes cluster identity")
        return {"cluster": cluster, "server": server}

    def _run_config(self, args):
        result = subprocess.run(
            ["kubectl", "--context", self.context, "config"] + list(args),
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def pods(self, selector):
        if set(selector) == {"metadata.name"}:
            data = json.loads(
                self._run(
                    [
                        "get",
                        "pods",
                        "--field-selector",
                        f"metadata.name={selector['metadata.name']}",
                        "-o",
                        "json",
                    ]
                )
            )
            return data.get("items") or []
        data = json.loads(
            self._run(["get", "pods", "-l", selector_text(selector), "-o", "json"])
        )
        return data.get("items") or []

    def delete_pod(self, captured, grace_period):
        self._run(
            [
                "delete",
                "pod",
                captured["name"],
                "--wait=false",
                f"--grace-period={grace_period}",
            ]
        )


class Evidence:
    def __init__(self, directory, plan):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.path = self.directory / "events.jsonl"
        self.stream = self.path.open("x", buffering=1)
        self.previous = "0" * 64
        self.append("plan", plan)

    def append(self, event_type, payload):
        event = {
            "schemaVersion": 1,
            "sequence": getattr(self, "sequence", 0),
            "at": datetime.now(timezone.utc).isoformat(),
            "type": event_type,
            "payload": payload,
            "previousDigest": self.previous,
        }
        event["digest"] = canonical_digest(event)
        self.stream.write(
            json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
        )
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.previous = event["digest"]
        self.sequence = event["sequence"] + 1

    def close(self, outcome):
        self.append("complete", {"outcome": outcome, "chainHead": self.previous})
        self.stream.close()
        self.path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def wait_until(predicate, timeout, poll, sleep=time.sleep, monotonic=time.monotonic):
    deadline = monotonic() + timeout
    while True:
        value = predicate()
        if value:
            return value
        if monotonic() >= deadline:
            raise TimeoutError("bounded wait deadline expired")
        sleep(min(poll, max(0, deadline - monotonic())))


def execute(scenario, evidence_dir, confirm_run_id, kube=None):
    run_id = scenario["benchmark"]["runId"]
    if confirm_run_id != run_id:
        raise ValueError("--confirm-run-id must exactly match benchmark.runId")
    kube = kube or Kubectl(scenario["clusterContext"], scenario["namespace"])
    if kube.current_context() != scenario["clusterContext"]:
        raise RuntimeError(
            "current kubectl context does not exactly match clusterContext"
        )
    expected_identity = ALLOWED_STAGING_CLUSTERS[scenario["clusterContext"]]
    if kube.cluster_identity() != expected_identity:
        raise RuntimeError(
            "kubectl cluster name/server do not match the allowlisted staging identity"
        )
    plan = render_plan(scenario)
    evidence = Evidence(evidence_dir, plan)
    old_handlers = {}

    def interrupt(signum, _frame):
        raise RuntimeError(f"interrupted by signal {signum}")

    if hasattr(signal, "SIGTERM"):
        for signum in (signal.SIGINT, signal.SIGTERM):
            old_handlers[signum] = signal.signal(signum, interrupt)
    try:
        def resolve_trigger():
            checkpoint = read_checkpoint(
                scenario["benchmark"]["checkpoint"],
                run_id,
                scenario["benchmark"]["apiHost"],
            )
            return checkpoint and trigger_target(scenario, checkpoint)

        target = wait_until(
            resolve_trigger,
            scenario["deadlines"]["triggerSeconds"],
            scenario["deadlines"]["pollSeconds"],
        )
        evidence.append("trigger", target)
        captured = capture_exact_pod(
            kube.pods(target["selector"]), target.get("podName")
        )
        evidence.append("target-captured", captured)

        # Re-read by exact name and compare UID immediately before the write.
        verified = capture_exact_pod(
            kube.pods({"metadata.name": captured["name"]}), captured["name"]
        )
        if verified["uid"] != captured["uid"]:
            raise RuntimeError("target pod changed after capture; refusing deletion")
        recovery_event_offset = 0
        if target.get("benchmarkJob"):
            checkpoint = read_checkpoint(
                scenario["benchmark"]["checkpoint"],
                run_id,
                scenario["benchmark"]["apiHost"],
            )
            revalidated = checkpoint and trigger_target(scenario, checkpoint)
            if (
                not revalidated
                or revalidated.get("podName") != captured["name"]
                or revalidated.get("benchmarkJob") != target.get("benchmarkJob")
            ):
                raise RuntimeError(
                    "benchmark job assignment changed before deletion"
                )
            recovery_event_offset = len(checkpoint.get("recoveries") or [])
        evidence.append("target-verified", verified)
        evidence.append(
            "fault-requested",
            {"podName": captured["name"], "podUid": captured["uid"]},
        )
        kube.delete_pod(captured, scenario["fault"]["gracePeriodSeconds"])
        evidence.append(
            "fault-applied", {"podName": captured["name"], "podUid": captured["uid"]}
        )

        def recovered():
            if target.get("benchmarkJob"):
                checkpoint = read_checkpoint(
                    scenario["benchmark"]["checkpoint"],
                    run_id,
                    scenario["benchmark"]["apiHost"],
                )
                job = checkpoint and checkpoint_job(
                    checkpoint, target["benchmarkJob"]["ordinal"]
                )
                new_processor = (job or {}).get("processorId")
                if (
                    not job
                    or job.get("state") != "running"
                    or not new_processor
                    or new_processor == captured["name"]
                ):
                    matching_events = []
                    for recovery_event in (
                        (checkpoint or {}).get("recoveries") or []
                    )[recovery_event_offset:]:
                        if (
                            recovery_event.get("outcome") != "recovered"
                            or target["benchmarkJob"]["id"]
                            not in (recovery_event.get("jobIds") or [])
                        ):
                            continue
                        after = [
                            item
                            for item in recovery_event.get("after") or []
                            if item.get("id") == target["benchmarkJob"]["id"]
                            and item.get("processorId") != captured["name"]
                        ]
                        if len(after) == 1:
                            matching_events.append(after[0])
                    if len(matching_events) != 1:
                        return None
                    new_processor = matching_events[0].get("processorId")
                candidates = kube.pods({"metadata.name": new_processor})
                if len(candidates) != 1 or not pod_ready(candidates[0]):
                    return None
                replacement = capture_exact_pod(candidates, new_processor)
                if replacement["uid"] == captured["uid"]:
                    return None
                return replacement

            # The relay is expected to be a singleton. Bind recovery to the
            # same controller revision so an unrelated rollout cannot pass.
            candidates = kube.pods(target["selector"])
            ready = [
                capture_exact_pod([pod])
                for pod in candidates
                if (pod.get("metadata") or {}).get("uid") != captured["uid"]
                and pod_ready(pod)
            ]
            matching = [
                pod
                for pod in ready
                if pod["owner"].get("uid") == captured["owner"].get("uid")
            ]
            return matching[0] if len(matching) == 1 else None

        replacement = wait_until(
            recovered,
            scenario["deadlines"]["recoverySeconds"],
            scenario["deadlines"]["pollSeconds"],
        )
        evidence.append("recovered", replacement)
        evidence.close("passed")
        return {"outcome": "passed", "replacement": replacement, "plan": plan}
    except Exception as error:
        evidence.append("error", {"type": type(error).__name__, "message": str(error)})
        evidence.close("failed")
        raise
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario")
    parser.add_argument("--output", help="write the dry-run plan to this path")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-run-id")
    parser.add_argument("--evidence-dir")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    scenario = load_scenario(args.scenario)
    plan = render_plan(scenario)
    if not args.execute:
        rendered = json.dumps(plan, indent=2, sort_keys=True) + "\n"
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(rendered)
        else:
            sys.stdout.write(rendered)
        return 0
    if not args.evidence_dir:
        raise ValueError("--evidence-dir is required with --execute")
    result = execute(scenario, args.evidence_dir, args.confirm_run_id)
    sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
