#!/usr/bin/env python3
"""Run the video workflow corpus through the staging service API.

The runner is deliberately staging-only and dry-run by default. API credentials
are read from an environment variable and are never accepted on the command line
or written to the result report.
"""

import argparse
import concurrent.futures
import copy
import fcntl
import json
import math
import os
import re
import signal
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from build_processor_jobs import load_corpus

DEFAULT_API_BASE = "https://roboflow-api-staging.web.app"
DEFAULT_MANIFEST = Path(__file__).with_name("workflows") / "manifest.json"
SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
WORKLOAD = re.compile(
    r"^(?P<profile>[A-Za-z0-9][A-Za-z0-9_.-]*)="
    r"(?P<count>[1-9][0-9]*)(?:@(?P<delay>[0-9]+(?:\.[0-9]+)?))?$"
)
STAGING_HOSTS = {
    "api.roboflow.one",
    "roboflow-api-staging.firebaseapp.com",
    "roboflow-api-staging.web.app",
    "us-central1-roboflow-staging.cloudfunctions.net",
}
TERMINAL_STATES = {"cancelled", "completed", "error"}
REPORT_JOB_FIELDS = {
    "id",
    "sourceId",
    "sourceName",
    "workflowUrl",
    "imageOutput",
    "mode",
    "tier",
    "maxFps",
    "state",
    "attempts",
    "cancelRequested",
    "processorId",
    "stats",
    "created_at",
    "updated_at",
    "heartbeatAt",
    "error",
    "lastError",
}
REPORT_SOURCE_FIELDS = {
    "id",
    "name",
    "kind",
    "status",
    "connectorId",
    "localId",
    "videoUploadId",
    "lastSeen",
}


class VideoServiceError(RuntimeError):
    def __init__(self, method, path, status, payload):
        message = payload.get("error") if isinstance(payload, dict) else payload
        if isinstance(message, dict):
            message = message.get("message") or json.dumps(message, sort_keys=True)
        super().__init__(f"{method} {path} returned HTTP {status}: {message}")
        self.status = status
        self.payload = payload


class BenchmarkInterrupted(RuntimeError):
    """Raised at a safe point after SIGINT or SIGTERM requests cleanup."""


class SignalStop:
    """Turn process stop signals into a flag so the runner reaches ``finally``."""

    def __init__(self):
        self.requested = threading.Event()
        self.signal_name = None
        self.signum = None
        self._handlers = {}

    def _request_stop(self, signum, _frame):
        self.signal_name = signal.Signals(signum).name
        self.signum = signum
        self.requested.set()

    def __enter__(self):
        for signum in (signal.SIGINT, signal.SIGTERM):
            self._handlers[signum] = signal.signal(signum, self._request_stop)
        return self

    def __exit__(self, _type, _value, _traceback):
        for signum, handler in self._handlers.items():
            signal.signal(signum, handler)

    def raise_if_requested(self):
        if self.requested.is_set():
            raise BenchmarkInterrupted(self.signal_name or "signal")


class RunLock:
    """Hold a non-blocking advisory lock for one run/suite identifier."""

    def __init__(self, path):
        self.path = Path(path)
        self._file = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a+")
        try:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self._file.close()
            self._file = None
            raise ValueError(f"run is already active: {self.path.stem}") from error
        return self

    def __exit__(self, _type, _value, _traceback):
        if self._file is not None:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
            self._file.close()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def validate_api_base(api_base):
    parsed = urllib.parse.urlparse(api_base)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or host not in STAGING_HOSTS:
        raise ValueError(
            "--api-base must be an allowlisted staging API or "
            "roboflow-staging Cloud Function"
        )
    return api_base.rstrip("/")


def validate_run_id(run_id):
    if not SAFE_RUN_ID.fullmatch(run_id or ""):
        raise ValueError(
            "run id must contain only letters, numbers, dot, dash, or underscore"
        )
    if len(run_id) > 64:
        raise ValueError("run id must be at most 64 characters")
    return run_id


def default_run_id():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def report_job(job):
    # Keep every sample immutable even when a test/client mutates a cached job
    # object in place. This also prevents a later stats update from rewriting
    # the recovery event's before/running-observed evidence.
    return {
        key: copy.deepcopy(job[key]) for key in REPORT_JOB_FIELDS if key in job
    }


def report_source(source):
    return {key: source[key] for key in REPORT_SOURCE_FIELDS if key in source}


def write_report_atomic(path, report):
    """Persist an already-redacted report without exposing partial JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(
        path.suffix + f".{os.getpid()}.{threading.get_ident()}.tmp"
    )
    with temporary.open("w") as output:
        json.dump(report, output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def recovery_checkpoint(report):
    """Bounded checkpoint for cleanup without rewriting the sample history."""
    bounded = {
        key: value
        for key, value in report.items()
        if key not in {"samples", "jobs"}
    }
    bounded["sampleCount"] = len(report.get("samples") or [])
    if report.get("samples"):
        bounded["lastSample"] = report["samples"][-1]
    bounded["jobs"] = list(report.get("jobs") or [])
    return bounded


def parse_workload(value):
    """Parse PROFILE=COUNT[@START_AFTER_SECONDS]."""
    match = WORKLOAD.fullmatch(value or "")
    if not match:
        raise ValueError(
            "workload must use PROFILE=COUNT or PROFILE=COUNT@START_AFTER_SECONDS"
        )
    return {
        "profile": match.group("profile"),
        "count": int(match.group("count")),
        "startAfterSeconds": float(match.group("delay") or 0),
    }


def build_run_plan(
    profiles,
    selections,
    repeat,
    publish_output,
    mode="stream",
    workloads=None,
    max_fps=None,
):
    if repeat < 1:
        raise ValueError("repeat must be at least 1")
    if workloads and selections:
        raise ValueError("use either profile selections or workloads, not both")
    requested = workloads or [
        {"profile": profile_id, "count": repeat, "startAfterSeconds": 0.0}
        for profile_id in selections
    ]
    if not requested:
        raise ValueError("at least one workload is required")
    if min(item["startAfterSeconds"] for item in requested) != 0:
        raise ValueError("at least one workload must start at zero seconds")

    plan = []
    ordinal = 0
    for workload in sorted(
        enumerate(requested), key=lambda item: (item[1]["startAfterSeconds"], item[0])
    ):
        _, workload = workload
        profile_id = workload["profile"]
        if profile_id not in profiles:
            raise ValueError(f"unknown workflow profile: {profile_id}")
        profile = profiles[profile_id]
        for copy_index in range(workload["count"]):
            ordinal += 1
            specification = copy.deepcopy(profile["specification"])
            metadata = dict(specification.get("metadata") or {})
            metadata["benchmark"] = {"profile": profile_id, "instance": ordinal}
            specification["metadata"] = metadata
            plan.append(
                {
                    "ordinal": ordinal,
                    "copy": copy_index + 1,
                    "profile": profile_id,
                    "provisionalClass": profile["provisionalClass"],
                    "tier": profile["tier"],
                    "mode": mode,
                    "maxFps": max_fps,
                    "startAfterSeconds": workload["startAfterSeconds"],
                    "imageOutput": (
                        profile.get("imageOutput") if publish_output else None
                    ),
                    # The platform prevents the exact same workflow from running
                    # twice on one source. Metadata makes logical copies distinct
                    # without changing their executable steps or model sharing.
                    "workflowSpecification": specification,
                }
            )
    return plan


def select_source(sources, source_id=None, source_name=None):
    if source_id:
        matches = [source for source in sources if source.get("id") == source_id]
    else:
        matches = [source for source in sources if source.get("name") == source_name]
    if not matches:
        selector = source_id or source_name
        raise ValueError(f"video source not found: {selector}")
    if len(matches) > 1:
        raise ValueError(
            f"multiple video sources are named {source_name!r}; select one with --source-id"
        )
    return matches[0]


def idempotency_key(run_id, item):
    key = f"corpus-{run_id}-{item['profile']}-{item['ordinal']}"
    if len(key) > 128:
        raise ValueError("generated idempotency key exceeds the API limit")
    return key


class VideoServiceClient:
    def __init__(self, api_base, workspace, api_key, timeout_seconds=30):
        self.api_base = validate_api_base(api_base)
        self.workspace = workspace
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def _path(self, suffix):
        workspace = urllib.parse.quote(self.workspace, safe="")
        return f"/{workspace}/{suffix.lstrip('/')}"

    def _request(self, method, suffix, body=None, headers=None):
        path = self._path(suffix)
        request_headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "roboflow-video-workflow-corpus/1",
        }
        request_headers.update(headers or {})
        data = None
        if body is not None:
            data = json.dumps(body, separators=(",", ":")).encode()
            request_headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.api_base + path,
            data=data,
            headers=request_headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout_seconds
            ) as response:
                payload = json.loads(response.read() or b"{}")
                return response.status, payload
        except urllib.error.HTTPError as error:
            try:
                payload = json.loads(error.read() or b"{}")
            except (json.JSONDecodeError, UnicodeDecodeError):
                payload = {"error": "non-JSON API response"}
            raise VideoServiceError(method, path, error.code, payload) from error

    def list_sources(self):
        _, payload = self._request("GET", "video-sources/v1")
        return payload.get("sources") or []

    def start_job(self, source_id, item, key):
        source_id = urllib.parse.quote(source_id, safe="")
        body = {
            "workflowSpecification": item["workflowSpecification"],
            "imageOutput": item["imageOutput"],
            "mode": item["mode"],
            "tier": item["tier"],
        }
        if item.get("maxFps") is not None:
            body["maxFps"] = item["maxFps"]
        status, payload = self._request(
            "POST",
            f"video-sources/v1/{source_id}/jobs",
            body=body,
            headers={"Idempotency-Key": key},
        )
        return status, payload["job"]

    def get_job(self, job_id):
        job_id = urllib.parse.quote(job_id, safe="")
        _, payload = self._request("GET", f"video-jobs/v1/{job_id}")
        return payload["job"]

    def cancel_job(self, job_id):
        job_id = urllib.parse.quote(job_id, safe="")
        _, payload = self._request("POST", f"video-jobs/v1/{job_id}/cancel")
        return payload["job"]


def _start_jobs(client, source_id, plan, run_id, on_started=None):
    def start(item):
        status, job = client.start_job(source_id, item, idempotency_key(run_id, item))
        return item, status, job

    started = []
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(plan)) as executor:
        futures = {executor.submit(start, item): item for item in plan}
        for future in concurrent.futures.as_completed(futures):
            item = futures[future]
            try:
                result = future.result()
                started.append(result)
                if on_started is not None:
                    on_started(*result)
            except Exception as error:
                errors.append(
                    {
                        "phase": "start",
                        "profile": item["profile"],
                        "ordinal": item["ordinal"],
                        "error": str(error),
                    }
                )
    started.sort(key=lambda result: result[0]["ordinal"])
    errors.sort(key=lambda error: error["ordinal"])
    return started, errors


def _poll_jobs(client, job_ids):
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(job_ids)) as executor:
        jobs = executor.map(client.get_job, job_ids)
        return {job["id"]: job for job in jobs}


def _cancel_jobs(client, jobs):
    active_ids = [
        job_id
        for job_id, job in jobs.items()
        if job.get("state") not in TERMINAL_STATES
    ]
    if not active_ids:
        return []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(active_ids)) as executor:
        futures = {
            executor.submit(client.cancel_job, job_id): job_id for job_id in active_ids
        }
        errors = []
        for future, job_id in ((future, futures[future]) for future in futures):
            try:
                future.result()
            except Exception as error:  # cleanup must continue for the other jobs
                errors.append({"jobId": job_id, "error": str(error)})
        return errors


def run_benchmark(
    client,
    source,
    plan,
    run_id,
    duration_seconds,
    poll_interval_seconds,
    startup_timeout_seconds,
    cleanup_timeout_seconds,
    recovery_timeout_seconds=0.0,
    startup_fault_ready_seconds=0.0,
    require_single_processor=False,
    checkpoint=None,
    should_stop=None,
    sleep=time.sleep,
    monotonic=time.monotonic,
):
    checkpoint = checkpoint or (lambda _report: None)
    should_stop = should_stop or (lambda: None)
    report = {
        "schemaVersion": 2,
        "runId": run_id,
        "startedAt": utc_now(),
        "apiBase": client.api_base,
        "workspace": client.workspace,
        "source": report_source(source),
        "plannedConcurrency": len(plan),
        "recoveryTimeoutSeconds": recovery_timeout_seconds,
        "startupFaultReadySeconds": startup_fault_ready_seconds,
        "profiles": [
            {
                key: item[key]
                for key in (
                    "ordinal",
                    "copy",
                    "profile",
                    "provisionalClass",
                    "tier",
                    "mode",
                    "imageOutput",
                    "maxFps",
                    "startAfterSeconds",
                )
            }
            for item in plan
        ],
        "samples": [],
        "errors": [],
    }
    jobs = {}
    benchmark_started = monotonic()
    success = True
    interrupted = False

    def save_checkpoint(phase, fail_run=True):
        report["checkpoint"] = {"phase": phase, "updatedAt": utc_now()}
        report["jobs"] = [report_job(job) for job in jobs.values()]
        try:
            checkpoint(report)
        except Exception as error:
            report["checkpointError"] = "checkpoint write failed"
            if fail_run:
                raise BenchmarkInterrupted("checkpoint write failure") from error

    def recovery_candidates(previous_jobs, established):
        candidates = {}
        for job_id, current in jobs.items():
            previous = previous_jobs.get(job_id) or {}
            previous_attempts = int(previous.get("attempts") or 0)
            current_attempts = int(current.get("attempts") or 0)
            processor_changed = bool(previous.get("processorId")) and (
                current.get("processorId") != previous.get("processorId")
            )
            ownership_observed = (
                established
                or previous.get("state") in {"claimed", "running"}
                or bool(previous.get("processorId"))
            )
            if ownership_observed and (
                (
                    current.get("state") not in {"claimed", "running"}
                    and previous.get("state") in {"claimed", "running"}
                )
                or processor_changed
                or current_attempts > previous_attempts
            ):
                candidates[job_id] = report_job(previous or current)
        return candidates

    def frame_count(job):
        value = (job.get("stats") or {}).get("frames")
        return value if isinstance(value, (int, float)) else None

    def recover_running_jobs(source_phase, previous_jobs, established):
        """Wait for a non-terminal requeue without hiding its downtime.

        Recovery tolerance is opt-in. Every poll remains in the sample stream,
        and the event records the exact before/after processor and attempt
        metadata so a fault run can measure re-placement rather than merely
        surviving it.
        """

        nonlocal jobs
        affected = recovery_candidates(previous_jobs, established)
        if not affected:
            return None
        if any(job.get("state") in TERMINAL_STATES for job in jobs.values()):
            return False
        if recovery_timeout_seconds <= 0:
            return False

        started = monotonic()
        event = {
            "index": len(report.setdefault("recoveries", [])) + 1,
            "sourcePhase": source_phase,
            "startedAt": utc_now(),
            "startedElapsedSeconds": round(started - benchmark_started, 3),
            "jobIds": sorted(affected),
            "before": [affected[job_id] for job_id in sorted(affected)],
            "firstObserved": [
                report_job(jobs[job_id]) for job_id in sorted(affected)
            ],
        }
        report["recoveries"].append(event)
        save_checkpoint("recovery")
        deadline = started + recovery_timeout_seconds
        running_baseline = None
        if {job.get("state") for job in jobs.values()} == {"running"}:
            running_baseline = {
                job_id: frame_count(jobs[job_id]) for job_id in affected
            }
            event["runningObservedAt"] = utc_now()
            event["runningObservedElapsedSeconds"] = round(
                monotonic() - benchmark_started, 3
            )
            event["runningObserved"] = [
                report_job(jobs[job_id]) for job_id in sorted(affected)
            ]
            save_checkpoint("recovery")
        while monotonic() < deadline:
            should_stop()
            sleep(min(poll_interval_seconds, deadline - monotonic()))
            should_stop()
            previous_recovery_jobs = {
                job_id: report_job(job) for job_id, job in jobs.items()
            }
            jobs = _poll_jobs(client, list(jobs))
            report["samples"].append(
                {
                    "phase": "recovery",
                    "elapsedSeconds": round(monotonic() - benchmark_started, 3),
                    "jobs": [report_job(job) for job in jobs.values()],
                }
            )
            save_checkpoint("recovery")
            secondary = set(
                recovery_candidates(previous_recovery_jobs, established=True)
            ) - set(affected)
            if secondary:
                event.update(
                    {
                        "outcome": "secondary-requeue",
                        "endedAt": utc_now(),
                        "observedControlPlaneRecoverySeconds": round(
                            monotonic() - started, 3
                        ),
                        "secondaryAffectedJobIds": sorted(secondary),
                        "after": [report_job(job) for job in jobs.values()],
                    }
                )
                save_checkpoint("recovery")
                return False
            states = {job.get("state") for job in jobs.values()}
            if states & TERMINAL_STATES:
                event.update(
                    {
                        "outcome": "terminal",
                        "endedAt": utc_now(),
                        "observedControlPlaneRecoverySeconds": round(
                            monotonic() - started, 3
                        ),
                        "after": [report_job(job) for job in jobs.values()],
                    }
                )
                save_checkpoint("recovery")
                return False
            if states == {"running"}:
                if running_baseline is None:
                    running_baseline = {
                        job_id: frame_count(jobs[job_id]) for job_id in affected
                    }
                    event["runningObservedAt"] = utc_now()
                    event["runningObservedElapsedSeconds"] = round(
                        monotonic() - benchmark_started, 3
                    )
                    event["runningObserved"] = [
                        report_job(jobs[job_id]) for job_id in sorted(affected)
                    ]
                    save_checkpoint("recovery")
                    continue
                progress = {
                    job_id: (
                        frame_count(jobs[job_id]) is not None
                        and running_baseline[job_id] is not None
                        and frame_count(jobs[job_id]) > running_baseline[job_id]
                    )
                    for job_id in affected
                }
                if all(progress.values()):
                    assertions = {}
                    for job_id, before in affected.items():
                        after = jobs[job_id]
                        previous_attempts = int(before.get("attempts") or 0)
                        current_attempts = int(after.get("attempts") or 0)
                        assertions[job_id] = {
                            "processorChanged": bool(before.get("processorId"))
                            and after.get("processorId")
                            != before.get("processorId"),
                            "attemptAdvanced": current_attempts
                            > previous_attempts,
                            "framesAdvancedAfterRunning": progress[job_id],
                        }
                        assertions[job_id]["requeueIdentityChanged"] = (
                            assertions[job_id]["processorChanged"]
                            or assertions[job_id]["attemptAdvanced"]
                        )
                    if all(
                        item["requeueIdentityChanged"]
                        and item["framesAdvancedAfterRunning"]
                        for item in assertions.values()
                    ):
                        event.update(
                            {
                                "outcome": "recovered",
                                "progressVerifiedAt": utc_now(),
                                "endedAt": utc_now(),
                                "observedControlPlaneRecoverySeconds": round(
                                    monotonic() - started, 3
                                ),
                                "after": [
                                    report_job(jobs[job_id])
                                    for job_id in sorted(affected)
                                ],
                                "assertions": assertions,
                            }
                        )
                        save_checkpoint(source_phase)
                        return True
            else:
                running_baseline = None

        event.update(
            {
                "outcome": "timeout",
                "endedAt": utc_now(),
                "observedControlPlaneRecoverySeconds": round(
                    monotonic() - started, 3
                ),
                "after": [report_job(job) for job in jobs.values()],
            }
        )
        save_checkpoint("recovery")
        return False

    save_checkpoint("initialized")
    try:
        waves = {}
        for item in plan:
            waves.setdefault(item.get("startAfterSeconds", 0.0), []).append(item)
        baseline_started = None

        for wave_index, (start_after, wave_plan) in enumerate(sorted(waves.items())):
            if not success:
                break
            if baseline_started is not None:
                target = baseline_started + start_after
                while monotonic() < target:
                    should_stop()
                    sleep(min(poll_interval_seconds, target - monotonic()))
                    should_stop()
                    previous_jobs = {
                        job_id: report_job(job) for job_id, job in jobs.items()
                    }
                    jobs = _poll_jobs(client, list(jobs))
                    report["samples"].append(
                        {
                            "phase": "baseline",
                            "elapsedSeconds": round(
                                monotonic() - benchmark_started, 3
                            ),
                            "jobs": [report_job(job) for job in jobs.values()],
                        }
                    )
                    save_checkpoint("baseline")
                    recovery = recover_running_jobs(
                        "baseline", previous_jobs, established=True
                    )
                    if recovery is False:
                        success = False
                        report["errors"].append(
                            {
                                "phase": "baseline",
                                "error": "job stopped before a later workload arrived",
                            }
                        )
                        break
            if not success:
                break

            report.setdefault("waves", []).append(
                {
                    "index": wave_index,
                    "startAfterSeconds": start_after,
                    "startedAt": utc_now(),
                    "ordinals": [item["ordinal"] for item in wave_plan],
                }
            )

            def record_started(item, status, job):
                jobs[job["id"]] = job
                report.setdefault("starts", []).append(
                    {
                        "profile": item["profile"],
                        "ordinal": item["ordinal"],
                        "httpStatus": status,
                        "job": report_job(job),
                    }
                )
                save_checkpoint("started")

            _started, start_errors = _start_jobs(
                client,
                source["id"],
                wave_plan,
                run_id,
                on_started=record_started,
            )
            if start_errors:
                success = False
                report["errors"].extend(start_errors)
                break

            startup_deadline = monotonic() + startup_timeout_seconds
            startup_phase = "startup" if wave_index == 0 else "arrival"
            fault_ready_emitted = False
            while True:
                should_stop()
                previous_jobs = {
                    job_id: report_job(job) for job_id, job in jobs.items()
                }
                jobs = _poll_jobs(client, list(jobs))
                report["samples"].append(
                    {
                        "phase": startup_phase,
                        "elapsedSeconds": round(monotonic() - benchmark_started, 3),
                        "jobs": [report_job(job) for job in jobs.values()],
                    }
                )
                save_checkpoint(startup_phase)
                if (
                    startup_fault_ready_seconds > 0
                    and not fault_ready_emitted
                    and any(
                        job.get("state") == "claimed" and job.get("processorId")
                        for job in jobs.values()
                    )
                    and not any(
                        job.get("state") == "running" for job in jobs.values()
                    )
                ):
                    fault_ready_emitted = True
                    report["startupFaultReadyAt"] = utc_now()
                    save_checkpoint("fault-ready")
                    pause_deadline = monotonic() + startup_fault_ready_seconds
                    while monotonic() < pause_deadline:
                        should_stop()
                        sleep(min(0.5, pause_deadline - monotonic()))
                    should_stop()
                recovery = recover_running_jobs(
                    startup_phase, previous_jobs, established=False
                )
                if recovery is False:
                    success = False
                    report["errors"].append(
                        {
                            "phase": startup_phase,
                            "error": "job did not recover during startup",
                        }
                    )
                    break
                states = {job.get("state") for job in jobs.values()}
                if states == {"running"}:
                    break
                if states & TERMINAL_STATES:
                    success = False
                    report["errors"].append(
                        {
                            "phase": startup_phase,
                            "error": "job became terminal before all jobs ran",
                        }
                    )
                    break
                if monotonic() >= startup_deadline:
                    success = False
                    report["errors"].append(
                        {"phase": startup_phase, "error": "startup timeout"}
                    )
                    break
                sleep(poll_interval_seconds)
                should_stop()
            if baseline_started is None and success:
                baseline_started = monotonic()

        if success:
            report["measurementStartedAt"] = utc_now()
            measurement_deadline = monotonic() + duration_seconds
            while monotonic() < measurement_deadline:
                should_stop()
                sleep(min(poll_interval_seconds, measurement_deadline - monotonic()))
                should_stop()
                previous_jobs = {
                    job_id: report_job(job) for job_id, job in jobs.items()
                }
                jobs = _poll_jobs(client, list(jobs))
                report["samples"].append(
                    {
                        "phase": "measurement",
                        "elapsedSeconds": round(monotonic() - benchmark_started, 3),
                        "jobs": [report_job(job) for job in jobs.values()],
                    }
                )
                save_checkpoint("measurement")
                recovery = recover_running_jobs(
                    "measurement", previous_jobs, established=True
                )
                if recovery is False:
                    success = False
                    report["errors"].append(
                        {
                            "phase": "measurement",
                            "error": (
                                "job did not recover during measurement"
                                if recovery_timeout_seconds > 0
                                else "job stopped during measurement"
                            ),
                        }
                    )
                    break
            report["measurementEndedAt"] = utc_now()
    except (KeyboardInterrupt, BenchmarkInterrupted) as error:
        interrupted = True
        success = False
        detail = str(error)
        message = "interrupted" if not detail else f"interrupted by {detail}"
        report["errors"].append({"phase": "run", "error": message})
        save_checkpoint("interrupted")
    except Exception as error:
        success = False
        report["errors"].append({"phase": "run", "error": str(error)})
    finally:
        report["cancelErrors"] = _cancel_jobs(client, jobs)
        save_checkpoint("cleanup-requested", fail_run=False)
        cleanup_deadline = monotonic() + cleanup_timeout_seconds
        while jobs and monotonic() < cleanup_deadline:
            try:
                jobs = _poll_jobs(client, list(jobs))
            except Exception as error:
                success = False
                report["errors"].append({"phase": "cleanup", "error": str(error)})
                break
            save_checkpoint("cleanup", fail_run=False)
            if all(job.get("state") in TERMINAL_STATES for job in jobs.values()):
                break
            sleep(poll_interval_seconds)
        if jobs and not all(
            job.get("state") in TERMINAL_STATES for job in jobs.values()
        ):
            success = False
            report["errors"].append({"phase": "cleanup", "error": "cleanup timeout"})
        if report["cancelErrors"]:
            success = False
        report["jobs"] = [report_job(job) for job in jobs.values()]
        report["processorIds"] = sorted(
            {
                job.get("processorId")
                for job in jobs.values()
                if job.get("processorId")
            }
        )
        if require_single_processor and len(report["processorIds"]) != 1:
            success = False
            report["errors"].append(
                {
                    "phase": "placement",
                    "error": (
                        "expected exactly one processor, observed "
                        f"{len(report['processorIds'])}"
                    ),
                }
            )
        report["interrupted"] = interrupted
        if report.get("checkpointError"):
            success = False
            if not any(item.get("phase") == "checkpoint" for item in report["errors"]):
                report["errors"].append(
                    {"phase": "checkpoint", "error": report["checkpointError"]}
                )
        report["success"] = success
        report["endedAt"] = utc_now()
        save_checkpoint("complete", fail_run=False)
    return report


def positive_float(parser, name, value):
    try:
        parsed = float(value)
    except ValueError:
        parser.error(f"{name} must be a number")
    if not math.isfinite(parsed) or parsed <= 0:
        parser.error(f"{name} must be greater than zero")
    return parsed


def nonnegative_float(parser, name, value):
    try:
        parsed = float(value)
    except ValueError:
        parser.error(f"{name} must be a number")
    if not math.isfinite(parsed) or parsed < 0:
        parser.error(f"{name} must be finite and nonnegative")
    return parsed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--api-key-env", default="VIDEO_BENCHMARK_API_KEY")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--profile", action="append")
    parser.add_argument(
        "--workload",
        action="append",
        help=(
            "PROFILE=COUNT[@START_AFTER_SECONDS]; repeat to build mixed and "
            "staged-arrival experiments"
        ),
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--mode", choices=["stream", "batch"], default="stream")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--source-id")
    source.add_argument("--source-name")
    parser.add_argument("--list-sources", action="store_true")
    parser.add_argument("--publish-output", action="store_true")
    parser.add_argument("--max-fps")
    parser.add_argument("--require-single-processor", action="store_true")
    parser.add_argument("--duration-seconds", default="60")
    parser.add_argument("--poll-interval-seconds", default="2")
    parser.add_argument("--startup-timeout-seconds", default="120")
    parser.add_argument("--cleanup-timeout-seconds", default="30")
    parser.add_argument(
        "--recovery-timeout-seconds",
        default="0",
        help=(
            "allow queued/claimed jobs this many seconds to return to running "
            "during baseline/measurement; 0 preserves fail-fast capacity runs"
        ),
    )
    parser.add_argument(
        "--startup-fault-ready-seconds",
        default="0",
        help=(
            "after a claimed processor is checkpointed, hold a fault-ready "
            "window this many seconds before polling again; staging faults only"
        ),
    )
    parser.add_argument("--run-id", default=default_run_id())
    parser.add_argument("--output-dir", default=Path(__file__).with_name("results"))
    parser.add_argument(
        "--execute",
        action="store_true",
        help="start staging jobs; without this flag, print the plan without network access",
    )
    args = parser.parse_args(argv)
    try:
        args.api_base = validate_api_base(args.api_base)
        args.run_id = validate_run_id(args.run_id)
    except ValueError as error:
        parser.error(str(error))
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    if args.profile and args.workload:
        parser.error("use either --profile or --workload, not both")
    if not args.list_sources and not (args.profile or args.workload):
        parser.error("at least one --profile or --workload is required")
    if not args.list_sources and not (args.source_id or args.source_name):
        parser.error("--source-id or --source-name is required")
    for name in (
        "duration_seconds",
        "poll_interval_seconds",
        "startup_timeout_seconds",
        "cleanup_timeout_seconds",
    ):
        option = f"--{name.replace('_', '-')}"
        setattr(args, name, positive_float(parser, option, getattr(args, name)))
    args.recovery_timeout_seconds = nonnegative_float(
        parser,
        "--recovery-timeout-seconds",
        args.recovery_timeout_seconds,
    )
    args.startup_fault_ready_seconds = nonnegative_float(
        parser,
        "--startup-fault-ready-seconds",
        args.startup_fault_ready_seconds,
    )
    if args.recovery_timeout_seconds > 3600:
        parser.error("--recovery-timeout-seconds cannot exceed 3600")
    if args.startup_fault_ready_seconds > 300:
        parser.error("--startup-fault-ready-seconds cannot exceed 300")
    if 0 < args.startup_fault_ready_seconds < 60:
        parser.error("--startup-fault-ready-seconds must be at least 60 when enabled")
    if args.max_fps is not None:
        args.max_fps = positive_float(parser, "--max-fps", args.max_fps)
    try:
        args.workloads = [parse_workload(item) for item in args.workload or []]
    except ValueError as error:
        parser.error(str(error))
    return args


def main(argv=None):
    args = parse_args(argv)
    profiles = load_corpus(args.manifest)
    try:
        plan = (
            []
            if args.list_sources
            else build_run_plan(
                profiles,
                args.profile or [],
                args.repeat,
                args.publish_output,
                args.mode,
                workloads=args.workloads,
                max_fps=args.max_fps,
            )
        )
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if args.startup_fault_ready_seconds > 0 and len(plan) != 1:
        print(
            "error: --startup-fault-ready-seconds requires exactly one job",
            file=sys.stderr,
        )
        return 2

    if not args.execute and not args.list_sources:
        print(
            json.dumps(
                {
                    "dryRun": True,
                    "apiBase": args.api_base,
                    "workspace": args.workspace,
                    "sourceId": args.source_id,
                    "sourceName": args.source_name,
                    "runId": args.run_id,
                    "plannedConcurrency": len(plan),
                    "profiles": [
                        {
                            key: item[key]
                            for key in item
                            if key != "workflowSpecification"
                        }
                        for item in plan
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        print("error: benchmark API key is not configured", file=sys.stderr)
        return 2
    client = VideoServiceClient(args.api_base, args.workspace, api_key)
    output_dir = Path(args.output_dir).resolve()
    output_path = output_dir / f"api-corpus-{args.run_id}.json"
    if not args.list_sources and output_path.exists():
        print(
            f"error: result already exists for run id {args.run_id}; "
            "use a new run id or the exact-run cleanup tool",
            file=sys.stderr,
        )
        return 2
    try:
        sources = client.list_sources()
        if args.list_sources:
            print(
                json.dumps(
                    {"sources": [report_source(item) for item in sources]}, indent=2
                )
            )
            return 0
        source = select_source(sources, args.source_id, args.source_name)
        lock_path = output_dir / f".api-corpus-{args.run_id}.lock"
        with RunLock(lock_path), SignalStop() as stop:
            report = run_benchmark(
                client=client,
                source=source,
                plan=plan,
                run_id=args.run_id,
                duration_seconds=args.duration_seconds,
                poll_interval_seconds=args.poll_interval_seconds,
                startup_timeout_seconds=args.startup_timeout_seconds,
                cleanup_timeout_seconds=args.cleanup_timeout_seconds,
                recovery_timeout_seconds=args.recovery_timeout_seconds,
                startup_fault_ready_seconds=args.startup_fault_ready_seconds,
                require_single_processor=args.require_single_processor,
                checkpoint=lambda value: write_report_atomic(
                    output_path, recovery_checkpoint(value)
                ),
                should_stop=stop.raise_if_requested,
            )
    except (ValueError, VideoServiceError, urllib.error.URLError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    write_report_atomic(output_path, report)
    print(output_path)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
