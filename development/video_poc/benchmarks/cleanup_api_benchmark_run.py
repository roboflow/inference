#!/usr/bin/env python3
"""Safely cancel jobs captured by one staging benchmark run checkpoint.

This is a recovery tool, not a general job sweeper. It resolves exactly one
``api-corpus-RUN_ID.json`` report, verifies that the report points at staging,
and only touches the allowlisted job IDs already persisted in that report.
Execution is opt-in so an operator can inspect the cleanup plan first.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from run_api_workflow_corpus import (
    TERMINAL_STATES,
    VideoServiceClient,
    report_job,
    utc_now,
    validate_api_base,
    validate_run_id,
    write_report_atomic,
)


def load_run_report(output_dir, run_id):
    run_id = validate_run_id(run_id)
    path = Path(output_dir).resolve() / f"api-corpus-{run_id}.json"
    with path.open() as source:
        report = json.load(source)
    if report.get("runId") != run_id:
        raise ValueError("checkpoint runId does not match the requested run")
    api_base = validate_api_base(report.get("apiBase") or "")
    workspace = report.get("workspace")
    if not isinstance(workspace, str) or not workspace:
        raise ValueError("checkpoint workspace is missing")

    jobs = {}
    for job in report.get("jobs") or []:
        if isinstance(job, dict) and isinstance(job.get("id"), str):
            jobs[job["id"]] = report_job(job)
    for start in report.get("starts") or []:
        job = start.get("job") if isinstance(start, dict) else None
        if isinstance(job, dict) and isinstance(job.get("id"), str):
            jobs[job["id"]] = report_job(job)
    return path, report, api_base, workspace, jobs


def cleanup_run(
    client,
    run_id,
    jobs,
    timeout_seconds,
    poll_interval_seconds,
    sleep=time.sleep,
    monotonic=time.monotonic,
):
    result = {
        "schemaVersion": 1,
        "runId": run_id,
        "environment": "staging",
        "startedAt": utc_now(),
        "expectedRecoveryState": "all captured jobs terminal",
        "requestedJobIds": sorted(jobs),
        "errors": [],
    }
    current = {}
    for job_id in sorted(jobs):
        try:
            current[job_id] = client.get_job(job_id)
        except Exception as error:
            result["errors"].append(
                {"phase": "inspect", "jobId": job_id, "error": str(error)}
            )

    for job_id, job in current.items():
        if job.get("state") in TERMINAL_STATES:
            continue
        try:
            client.cancel_job(job_id)
        except Exception as error:
            result["errors"].append(
                {"phase": "cancel", "jobId": job_id, "error": str(error)}
            )

    deadline = monotonic() + timeout_seconds
    active = {
        job_id
        for job_id, job in current.items()
        if job.get("state") not in TERMINAL_STATES
    }
    while active and monotonic() < deadline:
        for job_id in sorted(active):
            try:
                current[job_id] = client.get_job(job_id)
            except Exception as error:
                result["errors"].append(
                    {"phase": "poll", "jobId": job_id, "error": str(error)}
                )
        active = {
            job_id
            for job_id in active
            if current.get(job_id, {}).get("state") not in TERMINAL_STATES
        }
        if active:
            sleep(poll_interval_seconds)

    if active:
        result["errors"].append(
            {
                "phase": "cleanup",
                "error": "cleanup timeout",
                "activeJobIds": sorted(active),
            }
        )
    result["jobs"] = [report_job(current[job_id]) for job_id in sorted(current)]
    result["actualRecoveryState"] = (
        "all captured jobs terminal" if not active else "captured jobs still active"
    )
    result["success"] = not result["errors"] and not active
    result["endedAt"] = utc_now()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default=Path(__file__).with_name("results"))
    parser.add_argument("--api-key-env", default="VIDEO_BENCHMARK_API_KEY")
    parser.add_argument("--timeout-seconds", type=float, default=60)
    parser.add_argument("--poll-interval-seconds", type=float, default=2)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)

    try:
        if args.timeout_seconds <= 0 or args.poll_interval_seconds <= 0:
            raise ValueError("timeouts must be greater than zero")
        path, _report, api_base, workspace, jobs = load_run_report(
            args.output_dir, args.run_id
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    plan = {
        "dryRun": not args.execute,
        "environment": "staging",
        "runId": args.run_id,
        "checkpoint": str(path),
        "workspace": workspace,
        "apiBase": api_base,
        "jobIds": sorted(jobs),
    }
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        print(f"error: {args.api_key_env} is not set", file=sys.stderr)
        return 2
    client = VideoServiceClient(api_base, workspace, api_key)
    result = cleanup_run(
        client,
        args.run_id,
        jobs,
        args.timeout_seconds,
        args.poll_interval_seconds,
    )
    result_path = (
        Path(args.output_dir).resolve() / f"cleanup-api-corpus-{args.run_id}.json"
    )
    write_report_atomic(result_path, result)
    print(result_path)
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
