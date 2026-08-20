#!/usr/bin/env python3
"""Fail-closed incumbent-retention analysis for delayed CPU workload arrivals."""

import argparse
import json
from pathlib import Path


def _job_ordinals(report):
    result = {}
    for start in report.get("starts") or []:
        job_id = (start.get("job") or {}).get("id")
        if job_id:
            result[start.get("ordinal")] = job_id
    return result


def _phase_rate(report, job_id, phase):
    points = []
    for sample in report.get("samples") or []:
        if sample.get("phase") != phase:
            continue
        job = next(
            (item for item in sample.get("jobs") or [] if item.get("id") == job_id),
            None,
        )
        frames = ((job or {}).get("stats") or {}).get("frames")
        if frames is not None:
            points.append((float(sample["elapsedSeconds"]), int(frames)))
    resets = sum(current[1] < prior[1] for prior, current in zip(points, points[1:]))
    intervals = max(0, len(points) - 1)
    if intervals == 0 or resets:
        return {"fps": None, "intervals": intervals, "counterResets": resets}
    elapsed = points[-1][0] - points[0][0]
    fps = (points[-1][1] - points[0][1]) / elapsed if elapsed > 0 else None
    return {
        "fps": round(fps, 6) if fps is not None else None,
        "intervals": intervals,
        "counterResets": resets,
    }


def analyze(report, min_intervals=10, min_retention=0.90):
    ordinals = _job_ordinals(report)
    incumbents = [
        item
        for item in report.get("profiles") or []
        if float(item.get("startAfterSeconds") or 0) == 0
    ]
    arrivals = [
        item
        for item in report.get("profiles") or []
        if float(item.get("startAfterSeconds") or 0) > 0
    ]
    jobs = []
    for incumbent in incumbents:
        job_id = ordinals.get(incumbent.get("ordinal"))
        baseline = _phase_rate(report, job_id, "baseline") if job_id else {}
        post = _phase_rate(report, job_id, "measurement") if job_id else {}
        retention = (
            post.get("fps") / baseline.get("fps")
            if baseline.get("fps") and post.get("fps") is not None
            else None
        )
        checks = {
            "jobMapped": job_id is not None,
            "baselineCoverage": baseline.get("intervals", 0) >= min_intervals,
            "postArrivalCoverage": post.get("intervals", 0) >= min_intervals,
            "noCounterReset": baseline.get("counterResets", 0) == 0
            and post.get("counterResets", 0) == 0,
            "retention": retention is not None and retention >= min_retention,
        }
        jobs.append(
            {
                "ordinal": incumbent.get("ordinal"),
                "jobId": job_id,
                "profile": incumbent.get("profile"),
                "baseline": baseline,
                "postArrival": post,
                "retentionRatio": (
                    round(retention, 6) if retention is not None else None
                ),
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    checks = {
        "reportSucceeded": report.get("success") is True,
        "hasIncumbent": bool(incumbents),
        "hasDelayedArrival": bool(arrivals),
        "allIncumbentsPassed": bool(jobs) and all(item["passed"] for item in jobs),
    }
    return {
        "schemaVersion": 1,
        "runId": report.get("runId"),
        "minIntervals": min_intervals,
        "minRetentionRatio": min_retention,
        "checks": checks,
        "incumbents": jobs,
        "passed": all(checks.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(json.loads(args.report.read_text()))
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
