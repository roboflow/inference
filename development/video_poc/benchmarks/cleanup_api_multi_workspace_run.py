#!/usr/bin/env python3
"""Cancel exactly the jobs captured by one multi-workspace staging run."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from build_processor_jobs import load_corpus
from run_api_multi_workspace_corpus import (
    _job_view,
    _safe_error,
    build_plan,
    load_scenario,
    utc_now,
)
from run_api_workflow_corpus import (
    TERMINAL_STATES,
    VideoServiceClient,
    validate_run_id,
    write_report_atomic,
)

DEFAULT_MANIFEST = Path(__file__).with_name("workflows") / "manifest.json"


def load_run_report(output_dir, run_id, matrix, scenario_name, manifest):
    """Join credential-free checkpoint jobs to matrix routing by ordinal."""
    run_id = validate_run_id(run_id)
    path = Path(output_dir).resolve() / f"api-multi-workspace-{run_id}.json"
    with path.open() as source:
        report = json.load(source)
    if report.get("runId") != run_id:
        raise ValueError("checkpoint runId does not match the requested run")
    if report.get("kind") != "multi-workspace-api-corpus":
        raise ValueError("checkpoint is not a multi-workspace API corpus report")
    if report.get("environment") != "staging":
        raise ValueError("checkpoint environment must be staging")
    if report.get("scenarioName") != scenario_name:
        raise ValueError("checkpoint scenarioName does not match the requested scenario")

    scenario = load_scenario(matrix, scenario_name)
    if report.get("matrixSha256") != scenario["matrixSha256"]:
        raise ValueError("checkpoint matrix digest does not match the current matrix")
    plan = build_plan(load_corpus(manifest), scenario)
    plan_by_ordinal = {item["ordinal"]: item for item in plan}
    captured = {}
    records = list(report.get("jobs") or [])
    records.extend(
        start.get("job")
        for start in report.get("starts") or []
        if isinstance(start, dict)
    )
    for job in records:
        if not isinstance(job, dict) or not isinstance(job.get("id"), str):
            continue
        ordinal = job.get("ordinal")
        item = plan_by_ordinal.get(ordinal)
        if item is None:
            raise ValueError(f"captured job has unknown plan ordinal: {ordinal!r}")
        if (
            job.get("workspaceLabel") != item["workspaceLabel"]
            or job.get("profile") != item["profile"]
        ):
            raise ValueError("checkpoint job identity does not match the matrix plan")
        captured[f"{ordinal}:{job['id']}"] = {"item": item, "job": job}
    return path, captured


def build_clients(captured):
    clients = {}
    for record in captured.values():
        routing = record["item"]["_routing"]
        key = (routing["apiBase"], routing["workspace"], routing["apiKeyEnv"])
        record["item"]["_clientKey"] = key
        if key in clients:
            continue
        credential = os.environ.get(routing["apiKeyEnv"])
        if not credential:
            raise ValueError("benchmark API key is not configured")
        clients[key] = VideoServiceClient(
            routing["apiBase"], routing["workspace"], credential
        )
    return clients


def _safe_identity(record):
    return {
        "jobId": record["job"]["id"],
        "ordinal": record["item"]["ordinal"],
        "workspaceLabel": record["item"]["workspaceLabel"],
    }


def _safe_failure(phase, error, record):
    return {
        "phase": phase,
        **_safe_identity(record),
        "error": _safe_error(error, record["item"]),
    }


def cleanup_run(
    clients,
    run_id,
    captured,
    timeout_seconds,
    poll_interval_seconds,
    sleep=time.sleep,
    monotonic=time.monotonic,
):
    result = {
        "schemaVersion": 1,
        "kind": "multi-workspace-api-corpus-cleanup",
        "runId": run_id,
        "environment": "staging",
        "startedAt": utc_now(),
        "expectedRecoveryState": "all captured jobs terminal",
        "requestedJobs": [
            _safe_identity(record) for _, record in sorted(captured.items())
        ],
        "errors": [],
    }
    current = {
        handle: {"item": record["item"], "job": dict(record["job"])}
        for handle, record in captured.items()
    }
    for handle, record in sorted(captured.items()):
        item = record["item"]
        try:
            job = clients[item["_clientKey"]].get_job(record["job"]["id"])
            current[handle]["job"] = job
        except Exception as error:
            result["errors"].append(_safe_failure("inspect", error, record))

    for record in current.values():
        if record["job"].get("state") in TERMINAL_STATES:
            continue
        item = record["item"]
        try:
            clients[item["_clientKey"]].cancel_job(record["job"]["id"])
        except Exception as error:
            result["errors"].append(_safe_failure("cancel", error, record))

    deadline = monotonic() + timeout_seconds
    active = {
        handle
        for handle, record in current.items()
        if record["job"].get("state") not in TERMINAL_STATES
    }
    while active and monotonic() < deadline:
        for handle in sorted(active):
            record = current[handle]
            item = record["item"]
            try:
                record["job"] = clients[item["_clientKey"]].get_job(
                    record["job"]["id"]
                )
            except Exception as error:
                result["errors"].append(_safe_failure("poll", error, record))
        active = {
            handle
            for handle in active
            if current[handle]["job"].get("state") not in TERMINAL_STATES
        }
        if active:
            sleep(poll_interval_seconds)

    if active:
        result["errors"].append(
            {
                "phase": "cleanup",
                "error": "cleanup timeout",
                "activeJobs": [
                    _safe_identity(current[handle]) for handle in sorted(active)
                ],
            }
        )
    result["jobs"] = [
        _job_view(record["job"], record["item"])
        for _, record in sorted(current.items())
    ]
    result["actualRecoveryState"] = (
        "all captured jobs terminal" if not active else "captured jobs still active"
    )
    result["success"] = not result["errors"] and not active
    result["endedAt"] = utc_now()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default=Path(__file__).with_name("results"))
    parser.add_argument("--timeout-seconds", type=float, default=60)
    parser.add_argument("--poll-interval-seconds", type=float, default=2)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.timeout_seconds <= 0 or args.poll_interval_seconds <= 0:
            raise ValueError("timeouts must be greater than zero")
        path, captured = load_run_report(
            args.output_dir,
            args.run_id,
            args.matrix,
            args.scenario,
            args.manifest,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    plan = {
        "dryRun": not args.execute,
        "environment": "staging",
        "runId": args.run_id,
        "checkpoint": str(path),
        "scenario": args.scenario,
        "jobs": [_safe_identity(record) for _, record in sorted(captured.items())],
    }
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    try:
        clients = build_clients(captured)
        result = cleanup_run(
            clients,
            args.run_id,
            captured,
            args.timeout_seconds,
            args.poll_interval_seconds,
        )
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    result_path = (
        Path(args.output_dir).resolve()
        / f"cleanup-api-multi-workspace-{args.run_id}.json"
    )
    write_report_atomic(result_path, result)
    print(result_path)
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
