#!/usr/bin/env python3
"""Join fault-controller evidence with a corpus recovery report.

The joined duration is an upper bound from the recorded pod deletion request to
the runner's first poll that proves frames advanced after the replacement job
returned to running. It is not an exact per-frame outage measurement.
"""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path


def canonical_digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def parse_time(value):
    if not value:
        raise ValueError("evidence timestamp is missing")
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def read_evidence(path):
    events = []
    previous = "0" * 64
    for index, line in enumerate(Path(path).read_text().splitlines()):
        event = json.loads(line)
        if event.get("sequence") != index:
            raise ValueError("fault evidence sequence is invalid")
        if event.get("previousDigest") != previous:
            raise ValueError("fault evidence digest chain is broken")
        claimed = event.get("digest")
        unsigned = dict(event)
        unsigned.pop("digest", None)
        if claimed != canonical_digest(unsigned):
            raise ValueError("fault evidence event digest is invalid")
        events.append(event)
        previous = claimed
    if not events or events[-1].get("type") != "complete":
        raise ValueError("fault evidence is incomplete")
    if events[-1].get("payload", {}).get("chainHead") != events[-1].get(
        "previousDigest"
    ):
        raise ValueError("fault evidence final chain head is invalid")
    return events


def _one_event(evidence, event_type):
    matching = [event for event in evidence if event.get("type") == event_type]
    if len(matching) != 1:
        raise ValueError(f"evidence must contain one {event_type} event")
    return matching[0]


def _job_snapshot(event, field, job_id):
    matching = [
        item for item in event.get(field) or [] if item.get("id") == job_id
    ]
    if len(matching) != 1:
        return None
    return matching[0]


def join_recovery(report, evidence):
    plan_event = _one_event(evidence, "plan")
    trigger_event = _one_event(evidence, "trigger")
    captured_event = _one_event(evidence, "target-captured")
    verified_event = _one_event(evidence, "target-verified")
    requested_event = _one_event(evidence, "fault-requested")
    applied_event = _one_event(evidence, "fault-applied")
    recovered_event = _one_event(evidence, "recovered")
    complete_event = _one_event(evidence, "complete")

    if complete_event.get("payload", {}).get("outcome") != "passed":
        raise ValueError("fault controller did not complete successfully")
    plan = plan_event.get("payload") or {}
    if (plan.get("fault") or {}).get("type") != "processor-pod-loss":
        raise ValueError("frame recovery join only supports processor-pod-loss")
    fault_phase = (plan.get("fault") or {}).get("phase")
    allowed_source_phases = {
        "startup": {"startup", "arrival"},
        "steady-state": {"measurement"},
    }
    if fault_phase not in allowed_source_phases:
        raise ValueError("fault evidence has an invalid processor fault phase")
    run_id = report.get("runId")
    planned_run_id = plan.get("benchmarkRunId")
    if not run_id or run_id != planned_run_id:
        raise ValueError("benchmark run ID does not match fault evidence")
    checkpoint = report.get("checkpoint") or {}
    if (
        checkpoint.get("phase") != "complete"
        or not report.get("endedAt")
        or not isinstance(report.get("success"), bool)
    ):
        raise ValueError("benchmark report is not complete")

    trigger = trigger_event.get("payload") or {}
    benchmark_job = trigger.get("benchmarkJob") or {}
    job_id = benchmark_job.get("id")
    if not job_id or not isinstance(benchmark_job.get("ordinal"), int):
        raise ValueError("fault evidence has no exact benchmark job target")

    captured = captured_event.get("payload") or {}
    verified = verified_event.get("payload") or {}
    requested = requested_event.get("payload") or {}
    applied = applied_event.get("payload") or {}
    replacement = recovered_event.get("payload") or {}
    old_processor = captured.get("name")
    new_processor = replacement.get("name")
    if not old_processor or not new_processor or old_processor == new_processor:
        raise ValueError("fault evidence has invalid processor replacement identity")
    if trigger.get("podName") != old_processor:
        raise ValueError("fault trigger does not match captured processor pod")
    if verified.get("name") != old_processor or verified.get("uid") != captured.get(
        "uid"
    ):
        raise ValueError("verified target does not match captured processor pod")
    for event_name, payload in (
        ("fault-requested", requested),
        ("fault-applied", applied),
    ):
        if payload.get("podName") != old_processor or payload.get(
            "podUid"
        ) != captured.get("uid"):
            raise ValueError(f"{event_name} does not match captured processor pod")
    if replacement.get("uid") == captured.get("uid"):
        raise ValueError("recovered processor pod did not change UID")

    fault_requested_at = parse_time(requested_event.get("at"))
    ordered_evidence_times = [
        parse_time(event.get("at"))
        for event in (
            verified_event,
            requested_event,
            applied_event,
            recovered_event,
        )
    ]
    if ordered_evidence_times != sorted(ordered_evidence_times):
        raise ValueError("fault evidence timestamps are out of order")
    matching_recoveries = []
    for event in report.get("recoveries") or []:
        progress_at_raw = event.get("progressVerifiedAt")
        if event.get("outcome") != "recovered" or not progress_at_raw:
            continue
        if event.get("sourcePhase") not in allowed_source_phases[fault_phase]:
            continue
        if job_id not in (event.get("jobIds") or []):
            continue
        before = _job_snapshot(event, "before", job_id)
        after = _job_snapshot(event, "after", job_id)
        assertions = (event.get("assertions") or {}).get(job_id) or {}
        if not before or not after:
            continue
        if before.get("processorId") != old_processor:
            continue
        if after.get("processorId") != new_processor:
            continue
        if not all(
            assertions.get(key) is True
            for key in (
                "processorChanged",
                "framesAdvancedAfterRunning",
                "requeueIdentityChanged",
            )
        ):
            continue
        progress_at = parse_time(progress_at_raw)
        duration = (progress_at - fault_requested_at).total_seconds()
        if duration < 0:
            raise ValueError("verified recovery precedes fault request")
        matching_recoveries.append(
            {
                "index": event.get("index"),
                "sourcePhase": event.get("sourcePhase"),
                "jobIds": event.get("jobIds") or [],
                "targetJobId": job_id,
                "oldProcessorId": old_processor,
                "newProcessorId": new_processor,
                "faultRequestedAt": requested_event["at"],
                "faultAppliedAt": applied_event["at"],
                "runningObservedAt": event.get("runningObservedAt"),
                "progressVerifiedAt": progress_at_raw,
                "verifiedFrameRecoveryUpperBoundSeconds": round(duration, 3),
                "observedControlPlaneRecoverySeconds": event.get(
                    "observedControlPlaneRecoverySeconds"
                ),
                "assertions": {job_id: assertions},
            }
        )
    if len(matching_recoveries) != 1:
        raise ValueError(
            "report must contain exactly one recovery matching the injected target"
        )
    return {
        "schemaVersion": 1,
        "runId": run_id,
        "faultType": "processor-pod-loss",
        "faultPhase": fault_phase,
        "faultOutcome": "passed",
        "benchmarkSuccess": report["success"],
        "benchmarkEndedAt": report["endedAt"],
        "benchmarkErrors": report.get("errors") or [],
        "recoveryCount": 1,
        "recoveries": matching_recoveries,
        "measurementSemantics": (
            "upper bound from immediately before the pod deletion request to a "
            "runner poll that verified post-replacement frame progress; evidence "
            "timestamps preserve subsecond precision"
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    with Path(args.report).open() as source:
        report = json.load(source)
    result = join_recovery(report, read_evidence(args.evidence))
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(rendered)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
