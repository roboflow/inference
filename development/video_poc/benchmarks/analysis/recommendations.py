#!/usr/bin/env python3
"""Derive provisional worker-class recommendations from corpus reports.

Recommendations remain evidence-bounded: missing controlled input, repetitions,
resource telemetry, cost, soak, or a failing boundary are emitted as gaps rather
than filled with assumptions.
"""

import argparse
import glob
import json
from dataclasses import replace
from pathlib import Path

from report import AnalysisConfig, analyze_reports


def signature(capacity):
    return (
        capacity["profile"],
        capacity["tier"],
        capacity["mode"],
        capacity["outputPublished"],
        capacity.get("maxFps"),
    )


def run_at_concurrency(analysis, capacity, concurrency):
    run_ids = {
        item["runId"]
        for item in capacity["runs"]
        if item["concurrency"] == concurrency and item["passed"]
    }
    candidates = [
        run
        for run in analysis["runs"]
        if run["runId"] in run_ids and run["success"]
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda run: run["aggregate"]["totalDeliveredFps"] or 0,
    )


def _measurement_seconds(report):
    samples = [
        float(item["elapsedSeconds"])
        for item in report.get("samples") or []
        if item.get("phase") == "measurement"
    ]
    return max(samples) - min(samples) if len(samples) >= 2 else 0


def _has_resource_telemetry(report):
    return bool(
        report.get("resourceTelemetry")
        or report.get("prometheusSamples")
        or report.get("environment", {}).get("resources")
    )


def build_recommendations(
    reports,
    strict_config=None,
    relaxed_max_latency_ms=75.0,
):
    strict_config = strict_config or AnalysisConfig()
    relaxed_config = replace(
        strict_config,
        max_sampled_ema_latency_p95_ms=relaxed_max_latency_ms,
    )
    strict = analyze_reports(reports, strict_config)
    relaxed = analyze_reports(reports, relaxed_config)
    relaxed_by_signature = {
        signature(item): item for item in relaxed["capacitySummaries"]
    }
    reports_by_run = {report.get("runId"): report for report in reports}
    recommendations = []
    for capacity in strict["capacitySummaries"]:
        key = signature(capacity)
        relaxed_capacity = relaxed_by_signature[key]
        strict_max = capacity["maxPassingConcurrency"]
        relaxed_max = relaxed_capacity["maxPassingConcurrency"]
        strict_run = (
            run_at_concurrency(strict, capacity, strict_max)
            if strict_max is not None
            else None
        )
        tested = sorted({item["concurrency"] for item in capacity["runs"]})
        failures_above = [
            item["concurrency"]
            for item in capacity["runs"]
            if not item["passed"]
            and strict_max is not None
            and item["concurrency"] > strict_max
        ]
        related_reports = [
            reports_by_run.get(item["runId"])
            for item in capacity["runs"]
            if reports_by_run.get(item["runId"])
        ]
        repetitions = {
            concurrency: sum(
                item["concurrency"] == concurrency for item in capacity["runs"]
            )
            for concurrency in tested
        }
        gaps = []
        if capacity.get("maxFps") is None:
            gaps.append("controlled input FPS curve")
        if any(value < 2 for value in repetitions.values()):
            gaps.append("at least two repetitions at every concurrency")
        if not failures_above:
            gaps.append("observed failing boundary above the certified point")
        if not related_reports or not all(
            report.get("schemaVersion", 1) >= 2 for report in related_reports
        ):
            gaps.append("schema-v2 frame counters and latency histogram")
        if not any(_has_resource_telemetry(report) for report in related_reports):
            gaps.append("CPU, GPU, memory, and network utilization evidence")
        if not any(_measurement_seconds(report) >= 900 for report in related_reports):
            gaps.append("15-minute-or-longer stability run")
        if not any(report.get("costInputs") for report in related_reports):
            gaps.append("environment-specific worker cost inputs")
        if not capacity["outputPublished"]:
            companion = (
                capacity["profile"],
                capacity["tier"],
                capacity["mode"],
                True,
                capacity.get("maxFps"),
            )
            if companion not in {
                signature(item) for item in strict["capacitySummaries"]
            }:
                gaps.append("matched annotated-output publishing curve")

        if strict_max is None:
            boundary = "no-certified-point"
        elif failures_above:
            boundary = "bounded-by-observed-failure"
        else:
            boundary = "lower-bound-only"
        recommendations.append(
            {
                "workload": {
                    "profile": capacity["profile"],
                    "tier": capacity["tier"],
                    "mode": capacity["mode"],
                    "outputPublished": capacity["outputPublished"],
                    "maxFps": capacity.get("maxFps"),
                },
                "evidence": {
                    "testedConcurrencies": tested,
                    "repetitionsByConcurrency": repetitions,
                    "strictMaxPassingConcurrency": strict_max,
                    "relaxedMaxPassingConcurrency": relaxed_max,
                    "relaxedLatencyGateMs": relaxed_max_latency_ms,
                    "firstObservedFailureAboveStrict": min(
                        failures_above, default=None
                    ),
                    "strictAggregateDeliveredFps": (
                        strict_run["aggregate"]["totalDeliveredFps"]
                        if strict_run
                        else None
                    ),
                    "strictMedianPerStreamDeliveredFps": (
                        strict_run["fairness"]["deliveredFpsMedian"]
                        if strict_run
                        else None
                    ),
                    "boundaryClassification": boundary,
                },
                "recommendation": {
                    "status": "provisional" if strict_max else "insufficient",
                    "maxStreamsPerWorker": strict_max,
                    "relaxedLatencyMaxStreamsPerWorker": relaxed_max,
                    "missingEvidence": gaps,
                    "pricingReady": not gaps,
                },
            }
        )
    return {
        "schemaVersion": 1,
        "strictSlo": strict["config"],
        "relaxedLatencyGateMs": relaxed_max_latency_ms,
        "recommendations": recommendations,
        "economics": {
            "status": "not-computed",
            "reason": (
                "worker cost and utilization attribution must be supplied by "
                "the measured environment; throughput alone is not pricing"
            ),
        },
    }


def render_markdown(result):
    lines = [
        "# Provisional video worker-class recommendations",
        "",
        "These are evidence-bounded capacity recommendations, not pricing.",
        "",
    ]
    for item in result["recommendations"]:
        workload = item["workload"]
        evidence = item["evidence"]
        recommendation = item["recommendation"]
        fps = workload["maxFps"] if workload["maxFps"] is not None else "unbounded"
        lines.extend(
            [
                f"## {workload['profile']} ({workload['tier']}, {fps} FPS)",
                "",
                f"- Strict max streams/worker: {recommendation['maxStreamsPerWorker']}",
                "- Relaxed-latency max streams/worker: "
                f"{recommendation['relaxedLatencyMaxStreamsPerWorker']}",
                f"- Boundary: {evidence['boundaryClassification']}",
                "- Missing evidence: "
                + (
                    "; ".join(recommendation["missingEvidence"])
                    if recommendation["missingEvidence"]
                    else "none"
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Economics",
            "",
            result["economics"]["reason"],
            "",
        ]
    )
    return "\n".join(lines)


def _paths(values):
    paths = []
    for value in values:
        path = Path(value)
        paths.extend(
            sorted(path.glob("api-corpus-*.json"))
            if path.is_dir()
            else [Path(item) for item in glob.glob(value)] or [path]
        )
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    parser.add_argument("--output")
    parser.add_argument("--relaxed-max-latency-ms", type=float, default=75.0)
    args = parser.parse_args(argv)
    reports = [json.loads(path.read_text()) for path in _paths(args.reports)]
    if not reports:
        parser.error("no report files matched")
    result = build_recommendations(
        reports, relaxed_max_latency_ms=args.relaxed_max_latency_ms
    )
    rendered = (
        json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.format == "json"
        else render_markdown(result) + "\n"
    )
    if args.output:
        Path(args.output).write_text(rendered)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
