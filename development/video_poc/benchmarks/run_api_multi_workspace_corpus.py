#!/usr/bin/env python3
"""Run a concurrent, staging-only workflow corpus across workspace boundaries.

This is the multi-workspace companion to ``run_api_workflow_corpus.py``. Its
matrix stores only environment-variable names; credential values are resolved
in memory immediately before execution and never enter commands or reports.
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

from build_processor_jobs import load_corpus
from run_api_workflow_corpus import (
    BenchmarkInterrupted,
    DEFAULT_API_BASE,
    RunLock,
    SAFE_RUN_ID,
    SignalStop,
    TERMINAL_STATES,
    VideoServiceClient,
    VideoServiceError,
    build_run_plan,
    recovery_checkpoint,
    report_job,
    report_source,
    select_source,
    validate_api_base,
    validate_run_id,
    write_report_atomic,
)

DEFAULT_MANIFEST = Path(__file__).with_name("workflows") / "manifest.json"
SAFE_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SAFE_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SECRET_FIELD_MARKERS = {
    "apikey",
    "authorization",
    "credential",
    "password",
    "secret",
    "token",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def _positive(value, field):
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a number") from error
    if parsed <= 0:
        raise ValueError(f"{field} must be greater than zero")
    return parsed


def _reject_inline_secrets(value, path="matrix"):
    if isinstance(value, dict):
        for key, child in value.items():
            normalized_key = re.sub(r"[^a-z0-9]", "", key.lower())
            if key != "apiKeyEnv" and any(
                marker in normalized_key for marker in SECRET_FIELD_MARKERS
            ):
                raise ValueError(
                    f"{path}.{key} is forbidden; use an apiKeyEnv name instead"
                )
            _reject_inline_secrets(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_inline_secrets(child, f"{path}[{index}]")


def is_multi_workspace_scenario(raw):
    """Return whether a scenario opts into object-based workload placement."""
    workloads = raw.get("workloads") or []
    return any(isinstance(item, dict) for item in workloads)


def matrix_digest(document):
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def normalize_scenario(document, raw):
    """Validate and normalize one object-workload scenario without secrets."""
    _reject_inline_secrets(document)
    if document.get("schemaVersion") != 1:
        raise ValueError("matrix schemaVersion must be 1")
    if document.get("environment") != "staging":
        raise ValueError("matrix environment must be staging")
    if not is_multi_workspace_scenario(raw):
        raise ValueError("multi-workspace scenarios require object workloads")

    defaults = dict(document.get("defaults") or {})
    scenario_defaults = {**defaults, **dict(raw.get("defaults") or {})}
    name = str(raw.get("name") or "")
    if not SAFE_RUN_ID.fullmatch(name):
        raise ValueError(f"invalid scenario name: {name!r}")

    normalized = []
    workspace_labels = {}
    labels_by_workspace = {}
    for index, item in enumerate(raw.get("workloads") or []):
        if not isinstance(item, dict):
            raise ValueError(
                f"scenario {name} cannot mix string and object workloads"
            )
        merged = {**scenario_defaults, **item}
        field = f"scenario {name} workload {index + 1}"
        profile = str(merged.get("profile") or "")
        if not SAFE_LABEL.fullmatch(profile):
            raise ValueError(f"{field} has an invalid profile")
        label = str(merged.get("workspaceLabel") or "")
        if not SAFE_LABEL.fullmatch(label) or len(label) > 64:
            raise ValueError(
                f"{field} workspaceLabel must be a safe label of at most 64 characters"
            )
        workspace = str(merged.get("workspace") or "")
        if not workspace:
            raise ValueError(f"{field} workspace is required")
        prior_workspace = workspace_labels.setdefault(label, workspace)
        if prior_workspace != workspace:
            raise ValueError(
                f"{field} reuses workspaceLabel {label!r} for another workspace"
            )
        prior_label = labels_by_workspace.setdefault(workspace, label)
        if prior_label != label:
            raise ValueError(
                f"{field} maps one workspace to multiple workspaceLabel values"
            )
        api_key_env = str(merged.get("apiKeyEnv") or "VIDEO_BENCHMARK_API_KEY")
        if not SAFE_ENV_NAME.fullmatch(api_key_env):
            raise ValueError(f"{field} apiKeyEnv is invalid")
        source_selectors = [
            key for key in ("sourceId", "sourceName") if merged.get(key)
        ]
        if len(source_selectors) != 1:
            raise ValueError(f"{field} must set exactly one sourceId or sourceName")
        count = int(merged.get("count", 1))
        if count < 1:
            raise ValueError(f"{field} count must be positive")
        start_after = float(merged.get("startAfterSeconds", 0))
        if start_after < 0:
            raise ValueError(f"{field} startAfterSeconds cannot be negative")
        mode = str(merged.get("mode", "stream"))
        if mode not in {"stream", "batch"}:
            raise ValueError(f"{field} mode must be stream or batch")
        tier = merged.get("tier")
        if tier is not None and tier not in {"cpu", "gpu"}:
            raise ValueError(f"{field} tier must be cpu or gpu")
        max_fps = merged.get("maxFps")
        if max_fps is not None:
            max_fps = _positive(max_fps, f"{field} maxFps")
        api_base = validate_api_base(
            merged.get("apiBase") or DEFAULT_API_BASE
        )
        normalized.append(
            {
                "profile": profile,
                "count": count,
                "startAfterSeconds": start_after,
                "workspaceLabel": label,
                # Runtime-only routing fields are omitted from reports.
                "workspace": workspace,
                "apiKeyEnv": api_key_env,
                "apiBase": api_base,
                "sourceId": merged.get("sourceId"),
                "sourceName": merged.get("sourceName"),
                "tier": tier,
                "maxFps": max_fps,
                "publishOutput": bool(merged.get("publishOutput", False)),
                "mode": mode,
            }
        )
    if not normalized:
        raise ValueError(f"scenario {name} must define workloads")
    if min(item["startAfterSeconds"] for item in normalized) != 0:
        raise ValueError(f"scenario {name} needs at least one workload at zero seconds")

    max_jobs = int(defaults.get("maxPlannedJobs", 32))
    planned_jobs = sum(item["count"] for item in normalized)
    if max_jobs < 1 or planned_jobs > max_jobs:
        raise ValueError(
            f"scenario {name} plans {planned_jobs} jobs, above safety cap {max_jobs}"
        )
    return {
        "name": name,
        "workloads": normalized,
        "plannedJobs": planned_jobs,
        "durationSeconds": _positive(
            raw.get("durationSeconds", defaults.get("durationSeconds", 300)),
            f"scenario {name} durationSeconds",
        ),
        "startupTimeoutSeconds": _positive(
            raw.get(
                "startupTimeoutSeconds", defaults.get("startupTimeoutSeconds", 300)
            ),
            f"scenario {name} startupTimeoutSeconds",
        ),
        "cleanupTimeoutSeconds": _positive(
            raw.get(
                "cleanupTimeoutSeconds", defaults.get("cleanupTimeoutSeconds", 60)
            ),
            f"scenario {name} cleanupTimeoutSeconds",
        ),
        "pollIntervalSeconds": _positive(
            raw.get("pollIntervalSeconds", defaults.get("pollIntervalSeconds", 2)),
            f"scenario {name} pollIntervalSeconds",
        ),
        "requireSingleProcessor": bool(raw.get("requireSingleProcessor", True)),
        "requiredApiKeyEnvs": sorted({item["apiKeyEnv"] for item in normalized}),
        "matrixSha256": matrix_digest(document),
    }


def load_scenario(path, scenario_name):
    path = Path(path).resolve()
    with path.open() as source:
        document = json.load(source)
    matches = [
        raw
        for raw in document.get("scenarios") or []
        if raw.get("name") == scenario_name
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one scenario named {scenario_name!r}")
    return normalize_scenario(document, matches[0])


def build_plan(profiles, scenario):
    plan = []
    ordinal = 0
    for workload_index, workload in sorted(
        enumerate(scenario["workloads"]),
        key=lambda pair: (pair[1]["startAfterSeconds"], pair[0]),
    ):
        local = build_run_plan(
            profiles,
            [workload["profile"]],
            workload["count"],
            workload["publishOutput"],
            workload["mode"],
            max_fps=workload["maxFps"],
        )
        for item in local:
            ordinal += 1
            item["ordinal"] = ordinal
            item["startAfterSeconds"] = workload["startAfterSeconds"]
            if workload["tier"] is not None:
                item["tier"] = workload["tier"]
            item["workflowSpecification"]["metadata"]["benchmark"] = {
                "profile": item["profile"],
                "instance": ordinal,
            }
            item["workspaceLabel"] = workload["workspaceLabel"]
            item["_routing"] = {
                key: workload[key]
                for key in (
                    "workspace",
                    "apiKeyEnv",
                    "apiBase",
                    "sourceId",
                    "sourceName",
                )
            }
            item["_workloadIndex"] = workload_index
            plan.append(item)
    return plan


def _safe_plan_item(item, source):
    return {
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
            "workspaceLabel",
        )
    } | {"source": report_source(source)}


def prepare_runtime(plan, execute):
    """Resolve credentials and sources; return clients and safe source metadata."""
    clients = {}
    sources = {}
    if not execute:
        return clients, sources
    for item in plan:
        routing = item["_routing"]
        client_key = (
            routing["apiBase"],
            routing["workspace"],
            routing["apiKeyEnv"],
        )
        if client_key not in clients:
            api_key = os.environ.get(routing["apiKeyEnv"])
            if not api_key:
                raise ValueError("benchmark API key is not configured")
            clients[client_key] = VideoServiceClient(
                routing["apiBase"], routing["workspace"], api_key
            )
        source_key = (client_key, routing.get("sourceId"), routing.get("sourceName"))
        if source_key not in sources:
            sources[source_key] = select_source(
                clients[client_key].list_sources(),
                routing.get("sourceId"),
                routing.get("sourceName"),
            )
        item["_clientKey"] = client_key
        item["_sourceKey"] = source_key
    return clients, sources


def _job_view(job, item):
    return {
        **report_job(job),
        "workspaceLabel": item["workspaceLabel"],
        "profile": item["profile"],
        "ordinal": item["ordinal"],
    }


def _safe_error(error, item):
    """Redact routing identity and any credential echoed by an upstream error."""
    message = str(error)
    routing = item["_routing"]
    credential = os.environ.get(routing["apiKeyEnv"])
    if credential:
        message = message.replace(credential, "[redacted credential]")
    workspace = routing["workspace"]
    message = message.replace(workspace, item["workspaceLabel"])
    message = message.replace(
        urllib.parse.quote(workspace, safe=""), item["workspaceLabel"]
    )
    return message


def _start_wave(clients, sources, wave, run_id, on_started=None):
    def start(item):
        client = clients[item["_clientKey"]]
        source = sources[item["_sourceKey"]]
        key = f"multi-{run_id}-{item['workspaceLabel']}-{item['ordinal']}"
        if len(key) > 128:
            raise ValueError("generated idempotency key exceeds the API limit")
        status, job = client.start_job(source["id"], item, key)
        return item, status, job

    started, errors = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(wave)) as executor:
        futures = {executor.submit(start, item): item for item in wave}
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
                        "workspaceLabel": item["workspaceLabel"],
                        "profile": item["profile"],
                        "ordinal": item["ordinal"],
                        "error": _safe_error(error, item),
                    }
                )
    return started, errors


def _poll(clients, active):
    def get(handle_and_record):
        handle, record = handle_and_record
        try:
            job = clients[record["item"]["_clientKey"]].get_job(
                record["job"]["id"]
            )
        except Exception as error:
            raise RuntimeError(
                f"{record['item']['workspaceLabel']}: "
                f"{_safe_error(error, record['item'])}"
            ) from error
        return handle, {"item": record["item"], "job": job}

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(active)) as executor:
        return dict(executor.map(get, active.items()))


def _cancel(clients, active):
    errors = []
    for handle, record in active.items():
        if record["job"].get("state") in TERMINAL_STATES:
            continue
        try:
            clients[record["item"]["_clientKey"]].cancel_job(record["job"]["id"])
        except Exception as error:
            errors.append(
                {
                    "workspaceLabel": record["item"]["workspaceLabel"],
                    "jobId": record["job"].get("id"),
                    "error": _safe_error(error, record["item"]),
                }
            )
    return errors


def run_benchmark(
    clients,
    sources,
    plan,
    scenario,
    run_id,
    checkpoint=None,
    should_stop=None,
    sleep=time.sleep,
    monotonic=time.monotonic,
):
    checkpoint = checkpoint or (lambda _report: None)
    should_stop = should_stop or (lambda: None)
    report = {
        "schemaVersion": 1,
        "kind": "multi-workspace-api-corpus",
        "runId": run_id,
        "scenarioName": scenario["name"],
        "matrixSha256": scenario["matrixSha256"],
        "environment": "staging",
        "startedAt": utc_now(),
        "plannedConcurrency": len(plan),
        "workloads": [
            _safe_plan_item(item, sources[item["_sourceKey"]]) for item in plan
        ],
        "samples": [],
        "errors": [],
    }
    active = {}
    success = True
    started_at = monotonic()
    baseline_started = None
    waves = {}
    for item in plan:
        waves.setdefault(item["startAfterSeconds"], []).append(item)

    def save_checkpoint(phase, fail_run=True):
        report["checkpoint"] = {"phase": phase, "updatedAt": utc_now()}
        report["jobs"] = [
            _job_view(record["job"], record["item"])
            for record in active.values()
        ]
        try:
            checkpoint(report)
        except Exception as error:
            report["checkpointError"] = "checkpoint write failed"
            if fail_run:
                raise BenchmarkInterrupted("checkpoint write failure") from error

    save_checkpoint("initialized")
    try:
        for wave_index, (start_after, wave) in enumerate(sorted(waves.items())):
            if baseline_started is not None:
                target = baseline_started + start_after
                while monotonic() < target:
                    should_stop()
                    sleep(min(scenario["pollIntervalSeconds"], target - monotonic()))
                    should_stop()
                    active = _poll(clients, active)
                    report["samples"].append(
                        {
                            "phase": "baseline",
                            "elapsedSeconds": round(monotonic() - started_at, 3),
                            "jobs": [
                                _job_view(record["job"], record["item"])
                                for record in active.values()
                            ],
                        }
                    )
                    save_checkpoint("baseline")
                    if any(
                        record["job"].get("state") != "running"
                        for record in active.values()
                    ):
                        raise RuntimeError(
                            "job stopped before a later workload arrived"
                        )

            report.setdefault("waves", []).append(
                {
                    "index": wave_index,
                    "startAfterSeconds": start_after,
                    "ordinals": [item["ordinal"] for item in wave],
                }
            )

            def record_started(item, status, job):
                handle = f"{item['ordinal']}:{job['id']}"
                active[handle] = {"item": item, "job": job}
                report.setdefault("starts", []).append(
                    {
                        "httpStatus": status,
                        "job": _job_view(job, item),
                    }
                )
                save_checkpoint("started")

            _started, errors = _start_wave(
                clients, sources, wave, run_id, on_started=record_started
            )
            if errors:
                report["errors"].extend(errors)
                success = False
                break

            deadline = monotonic() + scenario["startupTimeoutSeconds"]
            phase = "startup" if wave_index == 0 else "arrival"
            while True:
                should_stop()
                active = _poll(clients, active)
                report["samples"].append(
                    {
                        "phase": phase,
                        "elapsedSeconds": round(monotonic() - started_at, 3),
                        "jobs": [
                            _job_view(record["job"], record["item"])
                            for record in active.values()
                        ],
                    }
                )
                save_checkpoint(phase)
                states = {record["job"].get("state") for record in active.values()}
                if states == {"running"}:
                    break
                if states & TERMINAL_STATES or monotonic() >= deadline:
                    report["errors"].append(
                        {
                            "phase": phase,
                            "error": (
                                "job became terminal before all jobs ran"
                                if states & TERMINAL_STATES
                                else "startup timeout"
                            ),
                        }
                    )
                    success = False
                    break
                sleep(scenario["pollIntervalSeconds"])
                should_stop()
            if not success:
                break
            if baseline_started is None:
                baseline_started = monotonic()

        if success:
            deadline = monotonic() + scenario["durationSeconds"]
            while monotonic() < deadline:
                should_stop()
                sleep(min(scenario["pollIntervalSeconds"], deadline - monotonic()))
                should_stop()
                active = _poll(clients, active)
                report["samples"].append(
                    {
                        "phase": "measurement",
                        "elapsedSeconds": round(monotonic() - started_at, 3),
                        "jobs": [
                            _job_view(record["job"], record["item"])
                            for record in active.values()
                        ],
                    }
                )
                save_checkpoint("measurement")
                if any(
                    record["job"].get("state") != "running"
                    for record in active.values()
                ):
                    report["errors"].append(
                        {
                            "phase": "measurement",
                            "error": "job stopped during measurement",
                        }
                    )
                    success = False
                    break
    except (KeyboardInterrupt, BenchmarkInterrupted) as error:
        detail = str(error)
        message = "interrupted" if not detail else f"interrupted by {detail}"
        report["errors"].append({"phase": "run", "error": message})
        success = False
        save_checkpoint("interrupted")
    except Exception as error:
        report["errors"].append({"phase": "run", "error": str(error)})
        success = False
    finally:
        report["cancelErrors"] = _cancel(clients, active)
        save_checkpoint("cleanup-requested", fail_run=False)
        deadline = monotonic() + scenario["cleanupTimeoutSeconds"]
        while active and monotonic() < deadline:
            try:
                active = _poll(clients, active)
            except Exception as error:
                report["errors"].append(
                    {"phase": "cleanup", "error": str(error)}
                )
                success = False
                break
            save_checkpoint("cleanup", fail_run=False)
            if all(
                record["job"].get("state") in TERMINAL_STATES
                for record in active.values()
            ):
                break
            sleep(scenario["pollIntervalSeconds"])
        if active and not all(
            record["job"].get("state") in TERMINAL_STATES
            for record in active.values()
        ):
            report["errors"].append({"phase": "cleanup", "error": "cleanup timeout"})
            success = False
        if report["cancelErrors"]:
            success = False
        report["jobs"] = [
            _job_view(record["job"], record["item"]) for record in active.values()
        ]
        report["processorIds"] = sorted(
            {
                record["job"].get("processorId")
                for record in active.values()
                if record["job"].get("processorId")
            }
        )
        if scenario["requireSingleProcessor"] and len(report["processorIds"]) != 1:
            report["errors"].append(
                {
                    "phase": "placement",
                    "error": (
                        "expected exactly one processor, observed "
                        f"{len(report['processorIds'])}"
                    ),
                }
            )
            success = False
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-matrix-sha256")
    parser.add_argument("--output-dir", default=Path(__file__).with_name("results"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        run_id = validate_run_id(args.run_id)
        scenario = load_scenario(args.matrix, args.scenario)
        if (
            args.expected_matrix_sha256
            and args.expected_matrix_sha256 != scenario["matrixSha256"]
        ):
            raise ValueError("matrix digest changed after suite validation")
        plan = build_plan(load_corpus(args.manifest), scenario)
        clients, sources = prepare_runtime(plan, args.execute)
        if not args.execute:
            print(
                json.dumps(
                    {
                        "dryRun": True,
                        "environment": "staging",
                        "runId": run_id,
                        "plannedConcurrency": len(plan),
                        "workloads": [
                            {
                                key: item[key]
                                for key in (
                                    "ordinal",
                                    "profile",
                                    "tier",
                                    "mode",
                                    "imageOutput",
                                    "maxFps",
                                    "startAfterSeconds",
                                    "workspaceLabel",
                                )
                            }
                            for item in plan
                        ],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        output_dir = Path(args.output_dir).resolve()
        output_path = output_dir / f"api-multi-workspace-{run_id}.json"
        if output_path.exists():
            raise ValueError(
                f"result already exists for run id {run_id}; use a new run id "
                "or the exact-run cleanup tool"
            )
        lock_path = output_dir / f".api-multi-workspace-{run_id}.lock"
        with RunLock(lock_path), SignalStop() as stop:
            report = run_benchmark(
                clients,
                sources,
                plan,
                scenario,
                run_id,
                checkpoint=lambda value: write_report_atomic(
                    output_path, recovery_checkpoint(value)
                ),
                should_stop=stop.raise_if_requested,
            )
    except (OSError, ValueError, VideoServiceError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    write_report_atomic(output_path, report)
    print(output_path)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
