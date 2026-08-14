#!/usr/bin/env python3
"""Join fault-controller evidence with a corpus recovery report.

The joined duration is an upper bound from the recorded pod deletion request to
the runner's first poll that proves frames advanced after the replacement job
returned to running. It is not an exact per-frame outage measurement.
"""

import argparse
import hashlib
import json
import math
from datetime import datetime, timedelta
from pathlib import Path

RELAY_RATE_WINDOW_SECONDS = 60


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
    matching = [item for item in event.get(field) or [] if item.get("id") == job_id]
    if len(matching) != 1:
        return None
    return matching[0]


def _validate_common(report, evidence, expected_fault):
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
    if (plan.get("fault") or {}).get("type") != expected_fault:
        raise ValueError(f"fault evidence is not {expected_fault}")
    fault_phase = (plan.get("fault") or {}).get("phase")
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
    captured = captured_event.get("payload") or {}
    verified = verified_event.get("payload") or {}
    requested = requested_event.get("payload") or {}
    applied = applied_event.get("payload") or {}
    replacement = recovered_event.get("payload") or {}
    trigger = trigger_event.get("payload") or {}
    old_processor = captured.get("name")
    new_processor = replacement.get("name")
    if not old_processor or not new_processor or old_processor == new_processor:
        raise ValueError("fault evidence has invalid processor replacement identity")
    if (
        expected_fault == "processor-pod-loss"
        and trigger.get("podName") != old_processor
    ):
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
    return {
        "plan": plan,
        "faultPhase": fault_phase,
        "runId": run_id,
        "trigger": trigger_event.get("payload") or {},
        "captured": captured,
        "replacement": replacement,
        "requestedEvent": requested_event,
        "appliedEvent": applied_event,
        "recoveredEvent": recovered_event,
        "faultRequestedAt": fault_requested_at,
    }


def _validate_successful_benchmark_cleanup(report):
    if (
        report.get("success") is not True
        or report.get("errors")
        or report.get("cancelErrors")
    ):
        raise ValueError("benchmark did not complete successfully after fault")
    planned = report.get("plannedConcurrency")
    starts = report.get("starts") or []
    started_ids = [((start.get("job") or {}).get("id")) for start in starts]
    if (
        not isinstance(planned, int)
        or isinstance(planned, bool)
        or planned <= 0
        or len(started_ids) != planned
        or any(not job_id for job_id in started_ids)
        or len(set(started_ids)) != planned
    ):
        raise ValueError("benchmark report has no exact planned job identity set")
    jobs = report.get("jobs") or []
    final_by_id = {job.get("id"): job for job in jobs if job.get("id")}
    if (
        len(final_by_id) != len(jobs)
        or set(final_by_id) != set(started_ids)
        or any(job.get("state") != "cancelled" for job in final_by_id.values())
    ):
        raise ValueError("benchmark did not clean up the exact job set")
    return set(started_ids)


def _join_processor_recovery(report, evidence):
    common = _validate_common(report, evidence, "processor-pod-loss")
    benchmark_job_ids = _validate_successful_benchmark_cleanup(report)
    fault_phase = common["faultPhase"]
    allowed_source_phases = {
        "startup": {"startup", "arrival"},
        "steady-state": {"measurement"},
    }
    if fault_phase not in allowed_source_phases:
        raise ValueError("fault evidence has an invalid processor fault phase")
    trigger = common["trigger"]
    benchmark_job = trigger.get("benchmarkJob") or {}
    job_id = benchmark_job.get("id")
    if not job_id or not isinstance(benchmark_job.get("ordinal"), int):
        raise ValueError("fault evidence has no exact benchmark job target")
    if job_id not in benchmark_job_ids:
        raise ValueError("fault target is not one of the benchmark jobs")
    old_processor = common["captured"]["name"]
    new_processor = common["replacement"]["name"]
    fault_requested_at = common["faultRequestedAt"]
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
                "faultRequestedAt": common["requestedEvent"]["at"],
                "faultAppliedAt": common["appliedEvent"]["at"],
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
        "runId": common["runId"],
        "faultType": "processor-pod-loss",
        "faultPhase": fault_phase,
        "faultOutcome": "passed",
        "benchmarkSuccess": True,
        "benchmarkEndedAt": report["endedAt"],
        "benchmarkErrors": [],
        "recoveryCount": 1,
        "recoveries": matching_recoveries,
        "measurementSemantics": (
            "upper bound from immediately before the pod deletion request to a "
            "runner poll that verified post-replacement frame progress; evidence "
            "timestamps preserve subsecond precision"
        ),
    }


def _sample_time(report, sample):
    if sample.get("sampledAt"):
        return parse_time(sample["sampledAt"])
    return parse_time(report.get("startedAt")) + timedelta(
        seconds=float(sample.get("elapsedSeconds"))
    )


def _sample_jobs(sample, expected_ids):
    jobs = sample.get("jobs") or []
    by_id = {job.get("id"): job for job in jobs if job.get("id")}
    if len(by_id) != len(jobs) or set(by_id) != expected_ids:
        raise ValueError("relay recovery sample does not cover the exact job set")
    return by_id


def _numeric_stat(job, field):
    value = (job.get("stats") or {}).get(field)
    if not isinstance(value, (int, float)):
        raise ValueError(f"relay recovery sample has no numeric {field}")
    return value


def _published(job):
    value = ((job.get("stats") or {}).get("counters") or {}).get("published")
    if not isinstance(value, (int, float)):
        raise ValueError("relay recovery sample has no published counter")
    return value


def _prometheus_series_samples(series):
    samples = []
    for raw_time, raw_value in series.get("values") or []:
        try:
            timestamp = float(raw_time)
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(timestamp) and math.isfinite(value):
            samples.append((timestamp, value))
    return sorted(samples)


def _validate_relay_media_evidence(
    report, resources, source_report_sha256, common, job_ids
):
    if not isinstance(resources, dict):
        raise ValueError("relay recovery requires MediaMTX resource evidence")
    if (
        resources.get("environment") != "staging"
        or resources.get("clusterContext") != "ck8s-stg"
        or resources.get("runId") != report.get("runId")
        or resources.get("sourceReportSha256") != source_report_sha256
        or resources.get("measurementStartedAt") != report.get("measurementStartedAt")
        or resources.get("measurementEndedAt") != report.get("measurementEndedAt")
    ):
        raise ValueError("relay resource evidence is not bound to the benchmark")
    requested_at = common["faultRequestedAt"].timestamp()
    replacement_ready_at = parse_time(common["recoveredEvent"].get("at")).timestamp()
    measurement_start = parse_time(report.get("measurementStartedAt")).timestamp()
    measurement_end = parse_time(report.get("measurementEndedAt")).timestamp()
    if not (
        measurement_start <= requested_at < replacement_ready_at <= measurement_end
    ):
        raise ValueError("relay fault is outside the resource measurement window")

    identity_series = (
        (resources.get("metrics") or {}).get("relayPodIdentity") or {}
    ).get("series") or []
    observed_identities = {}
    for series in identity_series:
        labels = series.get("metric") or {}
        identity = (labels.get("pod"), labels.get("uid"))
        if not all(identity) or identity in observed_identities:
            raise ValueError("relay identity resource evidence is ambiguous")
        observed_identities[identity] = _prometheus_series_samples(series)
    expected_identities = {
        (common["captured"].get("name"), common["captured"].get("uid")),
        (common["replacement"].get("name"), common["replacement"].get("uid")),
    }
    if set(observed_identities) != expected_identities:
        raise ValueError("relay identity metrics do not match the injected replacement")
    old_identity = (
        common["captured"].get("name"),
        common["captured"].get("uid"),
    )
    new_identity = (
        common["replacement"].get("name"),
        common["replacement"].get("uid"),
    )
    if not any(
        timestamp <= requested_at
        for timestamp, _value in observed_identities[old_identity]
    ) or not any(
        timestamp >= replacement_ready_at
        for timestamp, _value in observed_identities[new_identity]
    ):
        raise ValueError("relay identity metrics do not span deletion and replacement")

    uncontaminated_after = replacement_ready_at + RELAY_RATE_WINDOW_SECONDS
    post_replacement = {}
    for name in (
        "relayReaders",
        "relayIngressBytesPerSecond",
        "relayEgressBytesPerSecond",
        "relayOutputPathCount",
        "relayOutputIngressBytesPerSecond",
    ):
        metric_series = ((resources.get("metrics") or {}).get(name) or {}).get(
            "series"
        ) or []
        samples = []
        for series in metric_series:
            pod = (series.get("metric") or {}).get("pod")
            post = [
                (timestamp, value)
                for timestamp, value in _prometheus_series_samples(series)
                if timestamp >= uncontaminated_after
            ]
            if post and pod != new_identity[0]:
                raise ValueError(f"{name} post-fault samples are not from replacement")
            if pod == new_identity[0]:
                samples.extend(post)
        samples.sort()
        if len(samples) < 2:
            raise ValueError(
                f"relay resource evidence has insufficient post-fault {name}"
            )
        post_replacement[name] = samples
    if min(value for _time, value in post_replacement["relayReaders"]) < len(job_ids):
        raise ValueError("relay readers did not recover for every benchmark output")
    for name in ("relayIngressBytesPerSecond", "relayEgressBytesPerSecond"):
        if any(value <= 0 for _time, value in post_replacement[name]):
            raise ValueError(f"{name} did not remain positive after relay replacement")
    if min(value for _time, value in post_replacement["relayOutputPathCount"]) < len(
        job_ids
    ):
        raise ValueError(
            "MediaMTX output paths did not recover for every benchmark job"
        )
    if any(
        value <= 0
        for _time, value in post_replacement["relayOutputIngressBytesPerSecond"]
    ):
        raise ValueError("MediaMTX output ingress did not recover after replacement")
    return {
        "sourceReportSha256": source_report_sha256,
        "oldRelayPod": {"name": old_identity[0], "uid": old_identity[1]},
        "newRelayPod": {"name": new_identity[0], "uid": new_identity[1]},
        "replacementReadyAt": common["recoveredEvent"]["at"],
        "rateWindowSeconds": RELAY_RATE_WINDOW_SECONDS,
        "uncontaminatedEvidenceAfter": datetime.fromtimestamp(
            uncontaminated_after, tz=common["faultRequestedAt"].tzinfo
        ).isoformat(),
        "postReplacementSampleCounts": {
            name: len(samples) for name, samples in post_replacement.items()
        },
        "minimumPostReplacementReaders": min(
            value for _time, value in post_replacement["relayReaders"]
        ),
        "minimumPostReplacementIngressBytesPerSecond": min(
            value for _time, value in post_replacement["relayIngressBytesPerSecond"]
        ),
        "minimumPostReplacementEgressBytesPerSecond": min(
            value for _time, value in post_replacement["relayEgressBytesPerSecond"]
        ),
        "minimumPostReplacementOutputPathCount": min(
            value for _time, value in post_replacement["relayOutputPathCount"]
        ),
        "minimumPostReplacementOutputIngressBytesPerSecond": min(
            value
            for _time, value in post_replacement["relayOutputIngressBytesPerSecond"]
        ),
    }


def _join_relay_recovery(report, evidence, resources, source_report_sha256):
    common = _validate_common(report, evidence, "relay-pod-loss")
    if common["faultPhase"] != "steady-state":
        raise ValueError("relay recovery join requires a steady-state fault")
    if common["trigger"].get("benchmarkJob") is not None:
        raise ValueError("relay fault must not target a benchmark processor job")
    job_ids = _validate_successful_benchmark_cleanup(report)
    relay_media = _validate_relay_media_evidence(
        report, resources, source_report_sha256, common, job_ids
    )
    if report.get("recoveries"):
        raise ValueError("relay loss unexpectedly caused a processor requeue")
    captured = common["captured"]
    replacement = common["replacement"]
    if (captured.get("owner") or {}).get("uid") != (replacement.get("owner") or {}).get(
        "uid"
    ):
        raise ValueError("relay replacement is not owned by the captured controller")

    samples = sorted(
        [
            sample
            for sample in report.get("samples") or []
            if sample.get("phase") == "measurement"
        ],
        key=lambda sample: float(sample.get("elapsedSeconds")),
    )
    requested_at = common["faultRequestedAt"]
    replacement_ready_at = parse_time(common["recoveredEvent"].get("at"))
    before_candidates = [
        sample for sample in samples if _sample_time(report, sample) <= requested_at
    ]
    if not before_candidates:
        raise ValueError("report has no measurement sample before relay loss")
    before_sample = before_candidates[-1]
    before_jobs = _sample_jobs(before_sample, job_ids)
    output_required = any(
        profile.get("imageOutput") for profile in report.get("profiles") or []
    )
    baseline = {}
    for job_id, job in before_jobs.items():
        if job.get("state") != "running" or not job.get("processorId"):
            raise ValueError("job was not running on a processor before relay loss")
        baseline[job_id] = {
            "processorId": job["processorId"],
            "attempts": job.get("attempts"),
            "frames": _numeric_stat(job, "frames"),
            "published": _published(job) if output_required else None,
        }

    progress_sample = None
    post_replacement_baseline = None
    previous = baseline
    observations = 0
    for sample in samples[samples.index(before_sample) + 1 :]:
        jobs = _sample_jobs(sample, job_ids)
        current = {}
        for job_id, job in jobs.items():
            if (
                job.get("state") != "running"
                or job.get("processorId") != baseline[job_id]["processorId"]
                or job.get("attempts") != baseline[job_id]["attempts"]
            ):
                raise ValueError("relay loss changed processor ownership or job state")
            current[job_id] = {
                "processorId": job["processorId"],
                "attempts": job.get("attempts"),
                "frames": _numeric_stat(job, "frames"),
                "published": _published(job) if output_required else None,
            }
            if current[job_id]["frames"] < previous[job_id]["frames"]:
                raise ValueError("frame counter reset during relay recovery")
            if output_required and (
                current[job_id]["published"] < previous[job_id]["published"]
            ):
                raise ValueError("published counter reset during relay recovery")
        previous = current
        observations += 1
        if _sample_time(report, sample) < replacement_ready_at:
            continue
        if post_replacement_baseline is None:
            post_replacement_baseline = current
            continue
        if all(
            current[job_id]["frames"] > post_replacement_baseline[job_id]["frames"]
            and (
                not output_required
                or current[job_id]["published"]
                > post_replacement_baseline[job_id]["published"]
            )
            for job_id in job_ids
        ):
            progress_sample = sample
            break
    if progress_sample is None:
        raise ValueError(
            "frames and requested outputs did not advance after relay recovery"
        )
    progress_at = _sample_time(report, progress_sample)
    before_at = _sample_time(report, before_sample)
    return {
        "schemaVersion": 1,
        "runId": common["runId"],
        "faultType": "relay-pod-loss",
        "faultPhase": "steady-state",
        "faultOutcome": "passed",
        "benchmarkSuccess": True,
        "benchmarkEndedAt": report["endedAt"],
        "benchmarkErrors": [],
        "recoveryCount": 1,
        "recoveries": [
            {
                "oldRelayPod": captured["name"],
                "newRelayPod": replacement["name"],
                "controllerUid": (captured.get("owner") or {}).get("uid"),
                "faultRequestedAt": common["requestedEvent"]["at"],
                "faultAppliedAt": common["appliedEvent"]["at"],
                "replacementReadyAt": common["recoveredEvent"]["at"],
                "firstPostReplacementSampleAt": next(
                    _sample_time(report, sample).isoformat()
                    for sample in samples
                    if _sample_time(report, sample) >= replacement_ready_at
                ),
                "lastPreFaultSampleAt": before_at.isoformat(),
                "progressVerifiedAt": progress_at.isoformat(),
                "faultToFrameProgressUpperBoundSeconds": round(
                    (progress_at - requested_at).total_seconds(), 3
                ),
                "sampleGapUpperBoundSeconds": round(
                    (progress_at - before_at).total_seconds(), 3
                ),
                "jobsVerified": sorted(job_ids),
                "outputProgressRequired": output_required,
                "postFaultSamplesInspected": observations,
                "processorOwnershipStable": True,
                "attemptIdentityStable": True,
            }
        ],
        "relayMediaEvidence": relay_media,
        "measurementSemantics": (
            "upper bounds derived from runner polls around the relay deletion; "
            "the result proves stable processor ownership plus frame and requested "
            "output progress between two polls after replacement readiness, not "
            "gapless media"
        ),
    }


def join_recovery(report, evidence, resources=None, source_report_sha256=None):
    source_report_sha256 = source_report_sha256 or canonical_digest(report)
    plan = _one_event(evidence, "plan").get("payload") or {}
    fault_type = (plan.get("fault") or {}).get("type")
    if fault_type == "processor-pod-loss":
        return _join_processor_recovery(report, evidence)
    if fault_type == "relay-pod-loss":
        return _join_relay_recovery(report, evidence, resources, source_report_sha256)
    raise ValueError("unsupported fault type for recovery join")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--resources")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    report_bytes = Path(args.report).read_bytes()
    report = json.loads(report_bytes)
    resources = json.loads(Path(args.resources).read_text()) if args.resources else None
    result = join_recovery(
        report,
        read_evidence(args.evidence),
        resources=resources,
        source_report_sha256=hashlib.sha256(report_bytes).hexdigest(),
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(rendered)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
