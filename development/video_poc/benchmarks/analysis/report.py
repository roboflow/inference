#!/usr/bin/env python3
"""Analyze reports emitted by ``run_api_workflow_corpus.py``.

The source report contains snapshots of cumulative frame counters and a rolling
decode-to-result latency EMA. This module deliberately calls the derived values
``delivered FPS`` and ``sampled EMA latency``: the snapshots are not per-frame
latency observations and therefore cannot produce a true latency percentile.
"""

import argparse
import glob
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class AnalysisConfig:
    """Steady-state window and default capacity certification gates."""

    warmup_seconds: float = 10.0
    min_steady_intervals: int = 2
    min_fps_retention_ratio: float = 0.90
    max_sampled_ema_latency_p95_ms: float = 50.0
    max_fps_spread_ratio: float = 0.10
    max_time_to_first_result_s: float = 30.0
    require_single_processor: bool = True


def _percentile(values, percentile):
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _rounded(value, digits=3):
    return None if value is None else round(value, digits)


def _job_metadata(report):
    profiles_by_ordinal = {
        item.get("ordinal"): item for item in report.get("profiles", [])
    }
    metadata = {}
    for start in report.get("starts", []):
        job = start.get("job") or {}
        job_id = job.get("id")
        if not job_id:
            continue
        profile = profiles_by_ordinal.get(start.get("ordinal"), {})
        metadata[job_id] = {
            "jobId": job_id,
            "ordinal": start.get("ordinal"),
            "profile": start.get("profile") or profile.get("profile"),
            "tier": job.get("tier") or profile.get("tier"),
            "mode": job.get("mode") or profile.get("mode"),
            "imageOutput": profile.get("imageOutput"),
            "maxFps": profile.get("maxFps"),
        }
    return metadata


def _job_records(report):
    records = {}
    for sample in sorted(
        report.get("samples", []), key=lambda item: item["elapsedSeconds"]
    ):
        for job in sample.get("jobs", []):
            job_id = job.get("id")
            if job_id:
                records.setdefault(job_id, []).append(
                    {
                        "elapsedSeconds": sample["elapsedSeconds"],
                        "phase": sample.get("phase"),
                        "job": job,
                    }
                )
    return records


def _first_elapsed(records, predicate):
    for record in records:
        if predicate(record["job"]):
            return record["elapsedSeconds"]
    return None


def _last_stat(records, name):
    for record in reversed(records):
        value = (record["job"].get("stats") or {}).get(name)
        if value is not None:
            return value
    return None


def _processor_sequence(records):
    sequence = []
    for record in records:
        processor = record["job"].get("processorId")
        if processor and (not sequence or sequence[-1] != processor):
            sequence.append(processor)
    return sequence


def _steady_records(records, warmup_seconds):
    measurement = [record for record in records if record["phase"] == "measurement"]
    if not measurement:
        return []
    cutoff = measurement[0]["elapsedSeconds"] + warmup_seconds
    return [
        record
        for record in measurement
        if record["elapsedSeconds"] >= cutoff
        and record["job"].get("state") == "running"
    ]


def _frame_rate(records):
    intervals = []
    frame_resets = 0
    for previous, current in zip(records, records[1:]):
        previous_stats = previous["job"].get("stats") or {}
        current_stats = current["job"].get("stats") or {}
        previous_frames = previous_stats.get("frames")
        current_frames = current_stats.get("frames")
        elapsed = current["elapsedSeconds"] - previous["elapsedSeconds"]
        if previous_frames is None or current_frames is None or elapsed <= 0:
            continue
        if current_frames < previous_frames:
            frame_resets += 1
            continue
        frame_delta = current_frames - previous_frames
        intervals.append((elapsed, frame_delta, frame_delta / elapsed))
    total_seconds = sum(item[0] for item in intervals)
    total_frames = sum(item[1] for item in intervals)
    rates = [item[2] for item in intervals]
    return {
        "deliveredFps": _rounded(
            total_frames / total_seconds if total_seconds else None
        ),
        "intervalFpsP05": _rounded(_percentile(rates, 0.05)),
        "intervalFpsP50": _rounded(_percentile(rates, 0.50)),
        "intervalFpsP95": _rounded(_percentile(rates, 0.95)),
        "steadyIntervals": len(intervals),
        "steadyObservedSeconds": _rounded(total_seconds),
        "steadyDeliveredFrames": total_frames,
        "frameCounterResets": frame_resets,
    }


def _latency_summary(records):
    values = [
        (record["job"].get("stats") or {}).get("decodeToResultLatencyMs")
        for record in records
    ]
    values = [value for value in values if value is not None]
    summary = {
        "sampledEmaLatencyMeanMs": _rounded(
            statistics.mean(values) if values else None
        ),
        "sampledEmaLatencyP50Ms": _rounded(_percentile(values, 0.50)),
        "sampledEmaLatencyP95Ms": _rounded(_percentile(values, 0.95)),
        "sampledEmaLatencyMaxMs": _rounded(max(values) if values else None),
        "latencySamples": len(values),
    }
    summary.update(_histogram_latency_delta(records))
    histogram_p95 = summary["frameLatencyP95ApproxMs"]
    summary["latencyP95ForSloMs"] = (
        histogram_p95
        if histogram_p95 is not None
        else summary["sampledEmaLatencyP95Ms"]
    )
    summary["latencySource"] = (
        "frame_histogram" if histogram_p95 is not None else "sampled_ema"
    )
    return summary


def _histogram_latency_delta(records):
    snapshots = []
    for record in records:
        latency = (record["job"].get("stats") or {}).get(
            "decodeToResultLatency"
        )
        histogram = (latency or {}).get("histogram") or {}
        bounds = histogram.get("bounds")
        counts = histogram.get("cumulativeCounts")
        if (
            isinstance(bounds, list)
            and isinstance(counts, list)
            and len(bounds) == len(counts)
            and latency.get("count") is not None
        ):
            snapshots.append((latency, bounds, counts))
    empty = {
        "frameLatencyHistogramCount": 0,
        "frameLatencyMeanMs": None,
        "frameLatencyP50ApproxMs": None,
        "frameLatencyP95ApproxMs": None,
        "frameLatencyP99ApproxMs": None,
    }
    if len(snapshots) < 2:
        return empty
    first, first_bounds, first_counts = snapshots[0]
    last, last_bounds, last_counts = snapshots[-1]
    if first_bounds != last_bounds:
        return empty
    count = int(last["count"]) - int(first["count"])
    delta_counts = [
        int(current) - int(previous)
        for previous, current in zip(first_counts, last_counts)
    ]
    if count <= 0 or any(value < 0 for value in delta_counts):
        return empty

    def quantile(value):
        target = max(1, math.ceil(count * value))
        for bound, cumulative in zip(last_bounds, delta_counts):
            if cumulative >= target:
                # The overflow bucket is represented by null. The cumulative
                # snapshot's max may predate the steady window, so label this
                # as an approximation rather than pretending it is exact.
                return bound if bound is not None else last.get("max")
        return None

    delta_sum = float(last.get("sum") or 0) - float(first.get("sum") or 0)
    return {
        "frameLatencyHistogramCount": count,
        "frameLatencyMeanMs": _rounded(delta_sum / count),
        "frameLatencyP50ApproxMs": _rounded(quantile(0.50)),
        "frameLatencyP95ApproxMs": _rounded(quantile(0.95)),
        "frameLatencyP99ApproxMs": _rounded(quantile(0.99)),
    }


def _counter_deltas(records):
    snapshots = [
        (record["job"].get("stats") or {}).get("counters")
        for record in records
    ]
    snapshots = [item for item in snapshots if isinstance(item, dict)]
    if len(snapshots) < 2:
        return {}
    first, last = snapshots[0], snapshots[-1]
    counters = {}
    for name in sorted(set(first) & set(last)):
        try:
            delta = int(last[name]) - int(first[name])
        except (TypeError, ValueError):
            continue
        if delta >= 0:
            counters[name] = delta
    return counters


def _placement_summary(report, job_ids):
    jobs_by_processor = {}
    peak_by_processor = {}
    measurement_processors = set()
    for sample in report.get("samples", []):
        counts = {}
        for job in sample.get("jobs", []):
            processor = job.get("processorId")
            job_id = job.get("id")
            if not processor or job_id not in job_ids:
                continue
            jobs_by_processor.setdefault(processor, set()).add(job_id)
            if sample.get("phase") == "measurement" and job.get("state") == "running":
                measurement_processors.add(processor)
                counts[processor] = counts.get(processor, 0) + 1
        for processor, count in counts.items():
            peak_by_processor[processor] = max(
                peak_by_processor.get(processor, 0), count
            )
    processor_count = len(measurement_processors)
    return {
        "distinctProcessorCount": processor_count,
        "allStreamsCoLocated": processor_count == 1 and bool(job_ids),
        "jobsByProcessor": {
            processor: sorted(jobs)
            for processor, jobs in sorted(jobs_by_processor.items())
        },
        "peakConcurrentJobsByProcessor": dict(sorted(peak_by_processor.items())),
    }


def _fairness(streams):
    fps_values = [stream["steadyState"]["deliveredFps"] for stream in streams]
    fps_values = [value for value in fps_values if value is not None]
    latency_values = [
        stream["steadyState"]["latencyP95ForSloMs"] for stream in streams
    ]
    latency_values = [value for value in latency_values if value is not None]
    fps_median = statistics.median(fps_values) if fps_values else None
    latency_median = statistics.median(latency_values) if latency_values else None
    for stream in streams:
        fps = stream["steadyState"]["deliveredFps"]
        latency = stream["steadyState"]["latencyP95ForSloMs"]
        stream["fairness"] = {
            "fpsVsCohortMedianPct": _rounded(
                100 * (fps / fps_median - 1) if fps is not None and fps_median else None
            ),
            "latencyVsCohortMedianPct": _rounded(
                100 * (latency / latency_median - 1)
                if latency is not None and latency_median
                else None
            ),
        }
    if fps_values:
        denominator = len(fps_values) * sum(value * value for value in fps_values)
        jain = sum(fps_values) ** 2 / denominator if denominator else None
        spread = (
            (max(fps_values) - min(fps_values)) / fps_median if fps_median else None
        )
    else:
        jain = None
        spread = None
    return {
        "deliveredFpsMedian": _rounded(fps_median),
        "deliveredFpsSpreadRatio": _rounded(spread),
        "deliveredFpsJainIndex": _rounded(jain, 6),
        "sampledEmaLatencyP95MedianMs": _rounded(latency_median),
        "latencyP95MedianMs": _rounded(latency_median),
    }


def _recovery_summary(report):
    events = report.get("recoveries") or []
    durations = [
        float(event["observedControlPlaneRecoverySeconds"])
        for event in events
        if event.get("observedControlPlaneRecoverySeconds") is not None
    ]
    return {
        "toleranceSeconds": report.get("recoveryTimeoutSeconds", 0) or 0,
        "eventCount": len(events),
        "recoveredCount": sum(
            event.get("outcome") == "recovered" for event in events
        ),
        "failedCount": sum(
            bool(event.get("outcome")) and event.get("outcome") != "recovered"
            for event in events
        ),
        "incompleteCount": sum(not event.get("outcome") for event in events),
        "totalObservedControlPlaneRecoverySeconds": _rounded(sum(durations)),
        "maxObservedControlPlaneRecoverySeconds": _rounded(
            max(durations) if durations else None
        ),
    }


def analyze_report(report, config=None):
    """Return a deterministic analysis of one corpus report."""

    config = config or AnalysisConfig()
    metadata = _job_metadata(report)
    records_by_job = _job_records(report)
    for job_id in records_by_job:
        metadata.setdefault(
            job_id,
            {
                "jobId": job_id,
                "ordinal": None,
                "profile": None,
                "tier": None,
                "mode": None,
                "imageOutput": None,
                "maxFps": None,
            },
        )
    streams = []
    for job_id, item in sorted(
        metadata.items(), key=lambda pair: (pair[1].get("ordinal") or math.inf, pair[0])
    ):
        records = records_by_job.get(job_id, [])
        steady = _steady_records(records, config.warmup_seconds)
        processor_sequence = _processor_sequence(records)
        startup = {
            "observedClaimedAtS": _rounded(
                _first_elapsed(
                    records,
                    lambda job: job.get("state") in {"claimed", "running"},
                )
            ),
            "observedRunningAtS": _rounded(
                _first_elapsed(records, lambda job: job.get("state") == "running")
            ),
            "observedFirstResultAtS": _rounded(
                _first_elapsed(
                    records,
                    lambda job: (job.get("stats") or {}).get("frames", 0) > 0,
                )
            ),
            "pipelineStartS": _rounded(_last_stat(records, "pipelineStartS")),
            "timeToFirstResultS": _rounded(_last_stat(records, "timeToFirstResultS")),
        }
        steady_summary = _frame_rate(steady)
        steady_summary.update(_latency_summary(steady))
        steady_summary["counterDeltas"] = _counter_deltas(steady)
        streams.append(
            {
                **item,
                "startup": startup,
                "placement": {
                    "processorsSeen": processor_sequence,
                    "processorMigrations": max(0, len(processor_sequence) - 1),
                },
                "steadyState": steady_summary,
                "maxAttempts": max(
                    (record["job"].get("attempts", 0) or 0 for record in records),
                    default=0,
                ),
            }
        )
    fairness = _fairness(streams)
    placement = _placement_summary(report, set(metadata))
    delivered_fps = [
        stream["steadyState"]["deliveredFps"]
        for stream in streams
        if stream["steadyState"]["deliveredFps"] is not None
    ]
    return {
        "runId": report.get("runId"),
        "reportSchemaVersion": report.get("schemaVersion"),
        "success": bool(report.get("success")),
        "plannedConcurrency": report.get("plannedConcurrency"),
        "observedStreamCount": len(streams),
        "profiles": sorted(
            {stream["profile"] for stream in streams if stream.get("profile")}
        ),
        "source": report.get("source"),
        "errors": report.get("errors", []),
        "steadyStateWarmupSeconds": config.warmup_seconds,
        "streams": streams,
        "placement": placement,
        "fairness": fairness,
        "aggregate": {
            "totalDeliveredFps": _rounded(sum(delivered_fps)),
            "streamsWithSteadyFps": len(delivered_fps),
        },
        "recovery": _recovery_summary(report),
    }


def _workload_signature(run):
    if len(run["profiles"]) != 1:
        return None
    stream = run["streams"][0] if run["streams"] else {}
    return (
        run["profiles"][0],
        stream.get("tier"),
        stream.get("mode"),
        bool(stream.get("imageOutput")),
        stream.get("maxFps"),
    )


def _capacity_summary(signature, runs, config):
    successful = [run for run in runs if run["success"] and run["streams"]]
    baseline_concurrency = min(
        (run["plannedConcurrency"] for run in successful), default=None
    )
    baseline_runs = [
        run for run in successful if run["plannedConcurrency"] == baseline_concurrency
    ]
    baseline_fps_values = [
        stream["steadyState"]["deliveredFps"]
        for run in baseline_runs
        for stream in run["streams"]
        if stream["steadyState"]["deliveredFps"] is not None
    ]
    baseline_latency_values = [
        stream["steadyState"]["latencyP95ForSloMs"]
        for run in baseline_runs
        for stream in run["streams"]
        if stream["steadyState"]["latencyP95ForSloMs"] is not None
    ]
    baseline_fps = (
        statistics.median(baseline_fps_values) if baseline_fps_values else None
    )
    baseline_latency = (
        statistics.median(baseline_latency_values)
        if baseline_latency_values
        else None
    )
    results = []
    for run in sorted(
        runs,
        key=lambda item: (item["plannedConcurrency"], item["runId"] or ""),
    ):
        fps_values = []
        latency_values = []
        startup_values = []
        intervals = []
        for stream in run["streams"]:
            fps = stream["steadyState"]["deliveredFps"]
            latency = stream["steadyState"]["latencyP95ForSloMs"]
            startup = stream["startup"]["timeToFirstResultS"]
            intervals.append(stream["steadyState"]["steadyIntervals"])
            if fps is not None:
                fps_values.append(fps)
            if latency is not None:
                latency_values.append(latency)
            if startup is not None:
                startup_values.append(startup)
            stream["baselineComparison"] = {
                "fpsRetentionRatio": _rounded(
                    fps / baseline_fps if fps is not None and baseline_fps else None
                ),
                "latencyP95VsBaselinePct": _rounded(
                    100 * (latency / baseline_latency - 1)
                    if latency is not None and baseline_latency
                    else None
                ),
            }
        min_retention = (
            min(value / baseline_fps for value in fps_values)
            if fps_values and baseline_fps
            else None
        )
        max_latency = max(latency_values) if latency_values else None
        max_startup = max(startup_values) if startup_values else None
        checks = {
            "reportSucceeded": run["success"],
            "enoughSteadyIntervals": bool(intervals)
            and min(intervals) >= config.min_steady_intervals,
            "baselineAvailable": baseline_fps is not None,
            "fpsRetention": min_retention is not None
            and min_retention >= config.min_fps_retention_ratio,
            "latencyP95": max_latency is not None
            and max_latency <= config.max_sampled_ema_latency_p95_ms,
            "fpsFairness": run["fairness"]["deliveredFpsSpreadRatio"] is not None
            and run["fairness"]["deliveredFpsSpreadRatio"]
            <= config.max_fps_spread_ratio,
            "startup": max_startup is not None
            and max_startup <= config.max_time_to_first_result_s,
            "singleProcessor": (
                run["placement"]["allStreamsCoLocated"]
                if config.require_single_processor
                else True
            ),
        }
        passed = all(checks.values())
        run["capacitySlo"] = {
            "passed": passed,
            "checks": checks,
            "minFpsRetentionRatio": _rounded(min_retention),
            "maxSampledEmaLatencyP95Ms": _rounded(max_latency),
            "maxLatencyP95Ms": _rounded(max_latency),
            "maxTimeToFirstResultS": _rounded(max_startup),
        }
        results.append(
            {
                "runId": run["runId"],
                "concurrency": run["plannedConcurrency"],
                **run["capacitySlo"],
            }
        )
    passing = [item["concurrency"] for item in results if item["passed"]]
    return {
        "profile": signature[0],
        "tier": signature[1],
        "mode": signature[2],
        "outputPublished": signature[3],
        "maxFps": signature[4],
        "baselineConcurrency": baseline_concurrency,
        "baselineDeliveredFps": _rounded(baseline_fps),
        "baselineSampledEmaLatencyP95Ms": _rounded(baseline_latency),
        "baselineLatencyP95Ms": _rounded(baseline_latency),
        "maxTestedConcurrency": max(
            (run["plannedConcurrency"] for run in runs), default=None
        ),
        "maxPassingConcurrency": max(passing, default=None),
        "runs": results,
    }


def analyze_reports(reports, config=None):
    """Analyze reports together and derive homogeneous-workload capacity curves."""

    config = config or AnalysisConfig()
    runs = [analyze_report(report, config) for report in reports]
    groups = {}
    for run in runs:
        if (
            run["recovery"]["toleranceSeconds"] > 0
            or run["recovery"]["eventCount"] > 0
        ):
            run["capacityExcludedReason"] = "recovery-tolerant fault run"
            continue
        signature = _workload_signature(run)
        if signature is not None:
            groups.setdefault(signature, []).append(run)
    capacity = [
        _capacity_summary(signature, grouped_runs, config)
        for signature, grouped_runs in sorted(
            groups.items(), key=lambda item: str(item[0])
        )
    ]
    return {
        "analysisSchemaVersion": 1,
        "config": asdict(config),
        "runs": runs,
        "capacitySummaries": capacity,
    }


def render_markdown(analysis):
    """Render a compact human-readable capacity table."""

    lines = [
        "# Video workflow benchmark analysis",
        "",
        "Schema-v2 reports use approximate percentiles from fixed per-frame "
        "histograms. Legacy reports use percentiles of sampled rolling EMAs, "
        "not per-frame percentiles.",
        "",
    ]
    for capacity in analysis["capacitySummaries"]:
        output = "published" if capacity["outputPublished"] else "disabled"
        fps_limit = (
            f", max {capacity['maxFps']} FPS"
            if capacity["maxFps"] is not None
            else ", unbounded input"
        )
        lines.extend(
            [
                f"## {capacity['profile']} ({capacity['tier']}, output "
                f"{output}{fps_limit})",
                "",
                "| Run | Streams | Total FPS | FPS spread | Max latency p95 | "
                "Processors | SLO |",
                "| --- | ---: | ---: | ---: | ---: | ---: | :---: |",
            ]
        )
        runs_by_id = {run["runId"]: run for run in analysis["runs"]}
        for result in capacity["runs"]:
            run = runs_by_id[result["runId"]]
            lines.append(
                "| {run_id} | {concurrency} | {fps} | {spread} | {latency} ms | "
                "{processors} | {slo} |".format(
                    run_id=result["runId"],
                    concurrency=result["concurrency"],
                    fps=run["aggregate"]["totalDeliveredFps"],
                    spread=run["fairness"]["deliveredFpsSpreadRatio"],
                    latency=result["maxLatencyP95Ms"],
                    processors=run["placement"]["distinctProcessorCount"],
                    slo="pass" if result["passed"] else "fail",
                )
            )
        lines.append("")
    return "\n".join(lines)


def _input_paths(values):
    paths = []
    for value in values:
        path = Path(value)
        if path.is_dir():
            paths.extend(sorted(path.glob("api-corpus-*.json")))
        else:
            matches = [Path(match) for match in glob.glob(value)]
            paths.extend(matches or [path])
    unique = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "reports", nargs="+", help="report files, globs, or directories"
    )
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    parser.add_argument("--output")
    parser.add_argument("--warmup-seconds", type=float, default=10.0)
    parser.add_argument("--min-fps-retention", type=float, default=0.90)
    parser.add_argument("--max-latency-ms", type=float, default=50.0)
    parser.add_argument("--max-fps-spread", type=float, default=0.10)
    parser.add_argument("--max-ttfr-seconds", type=float, default=30.0)
    parser.add_argument("--allow-multiple-processors", action="store_true")
    args = parser.parse_args(argv)
    paths = _input_paths(args.reports)
    if not paths:
        parser.error("no report files matched")
    reports = []
    for path in paths:
        with path.open() as report_file:
            reports.append(json.load(report_file))
    config = AnalysisConfig(
        warmup_seconds=args.warmup_seconds,
        min_fps_retention_ratio=args.min_fps_retention,
        max_sampled_ema_latency_p95_ms=args.max_latency_ms,
        max_fps_spread_ratio=args.max_fps_spread,
        max_time_to_first_result_s=args.max_ttfr_seconds,
        require_single_processor=not args.allow_multiple_processors,
    )
    analysis = analyze_reports(reports, config)
    rendered = (
        json.dumps(analysis, indent=2, sort_keys=True) + "\n"
        if args.format == "json"
        else render_markdown(analysis) + "\n"
    )
    if args.output:
        Path(args.output).write_text(rendered)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
