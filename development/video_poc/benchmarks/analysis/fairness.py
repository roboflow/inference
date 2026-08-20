#!/usr/bin/env python3
"""Analyze tenant fairness and noisy-neighbor effects in multi-workspace runs."""

import argparse
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

from .report import _frame_rate, _latency_summary, _rounded


@dataclass(frozen=True)
class FairnessConfig:
    warmup_seconds: float = 10.0
    min_steady_intervals: int = 2
    min_incumbent_baseline_intervals: int = 1
    min_target_attainment_ratio: float = 0.90
    max_target_overshoot_ratio: float = 1.10
    max_tenant_attainment_spread_ratio: float = 0.10
    min_incumbent_retention_ratio: float = 0.90
    max_latency_p95_ms: float = 50.0
    require_shared_processor: bool = True


def _jain(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return None
    squared_sum = sum(values) ** 2
    sum_of_squares = sum(value * value for value in values)
    if sum_of_squares == 0:
        return 1.0
    return squared_sum / (len(values) * sum_of_squares)


def _spread_ratio(values):
    values = [float(value) for value in values if value is not None]
    if len(values) < 2:
        return 0.0 if values else None
    median = statistics.median(values)
    if median == 0:
        return 0.0 if max(values) == min(values) else math.inf
    return (max(values) - min(values)) / median


def _metadata(report):
    metadata = {}
    for workload in report.get("workloads", []):
        ordinal = workload.get("ordinal")
        if ordinal is not None:
            metadata[ordinal] = {
                "ordinal": ordinal,
                "workspaceLabel": workload.get("workspaceLabel"),
                "profile": workload.get("profile"),
                "maxFps": workload.get("maxFps"),
                "startAfterSeconds": workload.get("startAfterSeconds", 0),
            }
    for start in report.get("starts", []):
        job = start.get("job") or {}
        ordinal = job.get("ordinal") or start.get("ordinal")
        if ordinal is None:
            continue
        item = metadata.setdefault(ordinal, {"ordinal": ordinal})
        for key in ("workspaceLabel", "profile"):
            if job.get(key) is not None:
                item[key] = job[key]
        item["jobId"] = job.get("id")
    return metadata


def _phase_records(records, phase, warmup_seconds=0.0):
    selected = [
        record
        for record in records
        if record.get("phase") == phase
        and record.get("job", {}).get("state") == "running"
    ]
    if not selected:
        return []
    cutoff = selected[0]["elapsedSeconds"] + warmup_seconds
    return [record for record in selected if record["elapsedSeconds"] >= cutoff]


def _processor_sequence(records):
    processors = []
    for record in records:
        processor = record.get("job", {}).get("processorId")
        if processor and (not processors or processors[-1] != processor):
            processors.append(processor)
    return processors


def _job_records_by_ordinal(report):
    """Keep workspace-local job IDs distinct by using the plan ordinal."""
    records = {}
    for sample in sorted(
        report.get("samples", []), key=lambda item: item["elapsedSeconds"]
    ):
        for job in sample.get("jobs", []):
            ordinal = job.get("ordinal")
            if ordinal is None:
                continue
            records.setdefault(ordinal, []).append(
                {
                    "elapsedSeconds": sample["elapsedSeconds"],
                    "phase": sample.get("phase"),
                    "job": job,
                }
            )
    return records


def _job_summary(item, records, config):
    measurement = _phase_records(records, "measurement", config.warmup_seconds)
    baseline = _phase_records(records, "baseline")
    measurement_rate = _frame_rate(measurement)
    baseline_rate = _frame_rate(baseline)
    latency = _latency_summary(measurement)
    delivered = measurement_rate["deliveredFps"]
    baseline_fps = baseline_rate["deliveredFps"]
    target = item.get("maxFps")
    processors = _processor_sequence(records)
    job_ids = sorted(
        {record["job"].get("id") for record in records if record["job"].get("id")}
    )
    return {
        "jobId": job_ids[0] if len(job_ids) == 1 else None,
        "jobIdsSeen": job_ids,
        "ordinal": item.get("ordinal"),
        "workspaceLabel": item.get("workspaceLabel"),
        "profile": item.get("profile"),
        "maxFps": target,
        "startAfterSeconds": item.get("startAfterSeconds", 0),
        "deliveredFps": delivered,
        "targetAttainmentRatio": (
            _rounded(delivered / target) if delivered is not None and target else None
        ),
        "baselineDeliveredFps": baseline_fps,
        "incumbentRetentionRatio": (
            _rounded(delivered / baseline_fps)
            if delivered is not None and baseline_fps
            else None
        ),
        "steadyIntervals": measurement_rate["steadyIntervals"],
        "baselineIntervals": baseline_rate["steadyIntervals"],
        "measurementFrameCounterResets": measurement_rate["frameCounterResets"],
        "baselineFrameCounterResets": baseline_rate["frameCounterResets"],
        "frameCounterResets": (
            measurement_rate["frameCounterResets"] + baseline_rate["frameCounterResets"]
        ),
        "latencyP95Ms": latency["latencyP95ForSloMs"],
        "latencySource": latency["latencySource"],
        "processorsSeen": processors,
        "processorMigrations": max(0, len(processors) - 1),
    }


def _tenant_summary(label, jobs):
    delivered = [job["deliveredFps"] for job in jobs]
    targets = [job["maxFps"] for job in jobs]
    all_targets_known = all(value is not None for value in targets)
    aggregate = sum(value for value in delivered if value is not None)
    target = sum(targets) if all_targets_known else None
    incumbent_jobs = [
        job for job in jobs if job.get("baselineDeliveredFps") is not None
    ]
    baseline = sum(job["baselineDeliveredFps"] for job in incumbent_jobs)
    incumbent_delivered = sum(
        job["deliveredFps"] for job in incumbent_jobs if job["deliveredFps"] is not None
    )
    normalized_job_rates = [
        (
            job["targetAttainmentRatio"]
            if job["targetAttainmentRatio"] is not None
            else job["deliveredFps"]
        )
        for job in jobs
    ]
    latencies = [job["latencyP95Ms"] for job in jobs if job["latencyP95Ms"]]
    processors = sorted(
        {processor for job in jobs for processor in job["processorsSeen"]}
    )
    return {
        "workspaceLabel": label,
        "jobCount": len(jobs),
        "aggregateDeliveredFps": _rounded(aggregate),
        "aggregateTargetFps": _rounded(target),
        "targetAttainmentRatio": _rounded(aggregate / target) if target else None,
        "withinTenantJainIndex": _rounded(_jain(normalized_job_rates), 6),
        "maxLatencyP95Ms": max(latencies) if latencies else None,
        "incumbentJobCount": len(incumbent_jobs),
        "incumbentBaselineFps": _rounded(baseline) if incumbent_jobs else None,
        "incumbentMeasurementFps": (
            _rounded(incumbent_delivered) if incumbent_jobs else None
        ),
        "incumbentRetentionRatio": (
            _rounded(incumbent_delivered / baseline) if baseline else None
        ),
        "processorsSeen": processors,
    }


def analyze_fairness(report, config=None):
    config = config or FairnessConfig()
    if report.get("kind") != "multi-workspace-api-corpus":
        raise ValueError("fairness analysis requires a multi-workspace report")
    if report.get("schemaVersion") != 1:
        raise ValueError("unsupported multi-workspace report schemaVersion")

    records_by_ordinal = _job_records_by_ordinal(report)
    metadata_by_ordinal = _metadata(report)
    jobs = []
    for ordinal, records in records_by_ordinal.items():
        item = metadata_by_ordinal.get(ordinal, {"ordinal": ordinal})
        jobs.append(_job_summary(item, records, config))
    jobs.sort(key=lambda item: (item.get("ordinal") is None, item.get("ordinal")))

    planned_concurrency = report.get("plannedConcurrency")
    planned_ordinals = set(metadata_by_ordinal)
    start_records = [
        start
        for start in report.get("starts", [])
        if isinstance(start, dict)
        and (start.get("job") or {}).get("ordinal") is not None
    ]
    started_ordinals = {
        (start.get("job") or {}).get("ordinal") for start in start_records
    }
    start_ids_by_ordinal = {}
    for start in start_records:
        job = start.get("job") or {}
        if job.get("id"):
            start_ids_by_ordinal.setdefault(job["ordinal"], set()).add(job["id"])
    sampled_ordinals = {
        job.get("ordinal") for job in jobs if job.get("ordinal") is not None
    }
    expected_ordinals = (
        set(range(1, planned_concurrency + 1))
        if isinstance(planned_concurrency, int) and planned_concurrency > 0
        else set()
    )
    measurement_samples = [
        sample
        for sample in report.get("samples", [])
        if sample.get("phase") == "measurement"
    ]
    complete_measurement_samples = bool(measurement_samples) and all(
        len(sample.get("jobs", [])) == planned_concurrency
        and {job.get("ordinal") for job in sample.get("jobs", [])} == expected_ordinals
        for sample in measurement_samples
    )

    jobs_by_tenant = {}
    for job in jobs:
        label = job.get("workspaceLabel")
        if label:
            jobs_by_tenant.setdefault(label, []).append(job)
    tenants = [
        _tenant_summary(label, tenant_jobs)
        for label, tenant_jobs in sorted(jobs_by_tenant.items())
    ]
    attainments = [
        tenant["targetAttainmentRatio"]
        for tenant in tenants
        if tenant["targetAttainmentRatio"] is not None
    ]
    job_attainments = [
        job["targetAttainmentRatio"]
        for job in jobs
        if job["targetAttainmentRatio"] is not None
    ]
    retention = [
        tenant["incumbentRetentionRatio"]
        for tenant in tenants
        if tenant["incumbentRetentionRatio"] is not None
    ]
    latency = [
        tenant["maxLatencyP95Ms"]
        for tenant in tenants
        if tenant["maxLatencyP95Ms"] is not None
    ]
    all_processors = sorted(
        {processor for tenant in tenants for processor in tenant["processorsSeen"]}
    )
    shared_processor = len(all_processors) == 1 and len(tenants) >= 2
    delayed_arrival = any(job.get("startAfterSeconds", 0) > 0 for job in jobs)
    incumbent_jobs = [job for job in jobs if job.get("startAfterSeconds", 0) == 0]
    checks = {
        "reportSucceeded": report.get("success") is True,
        "plannedWorkloadsObserved": bool(expected_ordinals)
        and planned_ordinals == expected_ordinals,
        "allJobsStarted": bool(expected_ordinals)
        and started_ordinals == expected_ordinals
        and len(start_records) == planned_concurrency,
        "allJobsSampled": bool(expected_ordinals)
        and sampled_ordinals == expected_ordinals
        and len(jobs) == planned_concurrency,
        "completeMeasurementSamples": complete_measurement_samples,
        "stableJobIdentity": bool(jobs)
        and all(
            len(job["jobIdsSeen"]) == 1
            and start_ids_by_ordinal.get(job["ordinal"]) == set(job["jobIdsSeen"])
            for job in jobs
        ),
        "multipleTenantsObserved": len(tenants) >= 2,
        "enoughSteadyIntervals": bool(jobs)
        and all(job["steadyIntervals"] >= config.min_steady_intervals for job in jobs),
        "allTargetsDefined": bool(jobs)
        and all(job.get("maxFps") is not None for job in jobs),
        "targetFpsPropagated": len(job_attainments) == len(jobs)
        and all(
            value <= config.max_target_overshoot_ratio for value in job_attainments
        ),
        "targetAttainment": len(job_attainments) == len(jobs)
        and all(
            value >= config.min_target_attainment_ratio for value in job_attainments
        ),
        "tenantAttainmentSpread": len(attainments) >= 2
        and _spread_ratio(attainments) <= config.max_tenant_attainment_spread_ratio,
        "incumbentBaselineCoverage": not delayed_arrival
        or (
            bool(incumbent_jobs)
            and all(
                job["baselineIntervals"] >= config.min_incumbent_baseline_intervals
                and job["incumbentRetentionRatio"] is not None
                for job in incumbent_jobs
            )
        ),
        "incumbentRetention": not delayed_arrival
        or (
            len(retention) == len({job["workspaceLabel"] for job in incumbent_jobs})
            and all(
                value >= config.min_incumbent_retention_ratio for value in retention
            )
        ),
        "frameLatencyHistogramAvailable": bool(jobs)
        and all(job["latencySource"] == "frame_histogram" for job in jobs),
        "latencyP95": bool(latency) and max(latency) <= config.max_latency_p95_ms,
        "sharedProcessor": (
            shared_processor if config.require_shared_processor else True
        ),
        "noProcessorMigrations": all(job["processorMigrations"] == 0 for job in jobs),
        "noFrameCounterResets": all(job["frameCounterResets"] == 0 for job in jobs),
    }
    return {
        "analysisSchemaVersion": 2,
        "runId": report.get("runId"),
        "scenarioName": report.get("scenarioName"),
        "success": all(checks.values()),
        "checks": checks,
        "config": asdict(config),
        "crossTenant": {
            "tenantCount": len(tenants),
            "targetAttainmentJainIndex": _rounded(_jain(attainments), 6),
            "targetAttainmentSpreadRatio": _rounded(_spread_ratio(attainments)),
            "processorsSeen": all_processors,
            "allTenantsShareOneProcessor": shared_processor,
        },
        "tenants": tenants,
        "jobs": jobs,
        "sourceReportSuccess": report.get("success"),
        "sourceErrorSummary": {
            "count": len(report.get("errors", [])),
            "phases": sorted(
                {
                    item.get("phase")
                    for item in report.get("errors", [])
                    if isinstance(item, dict) and item.get("phase")
                }
            ),
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report")
    parser.add_argument("--warmup-seconds", type=float, default=10.0)
    parser.add_argument("--allow-distributed", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    with Path(args.report).open() as source:
        report = json.load(source)
    result = analyze_fairness(
        report,
        FairnessConfig(
            warmup_seconds=args.warmup_seconds,
            require_shared_processor=not args.allow_distributed,
        ),
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(rendered)
    else:
        print(rendered, end="")
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
