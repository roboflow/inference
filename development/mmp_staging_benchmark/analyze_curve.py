#!/usr/bin/env python3
"""Certify a capacity boundary from paired single-report analyzer results."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from development.mmp_staging_benchmark.validate_staging_plan import (  # noqa: E402
    load_and_validate,
)

COHORT_FIELDS = (
    "server_image_ref",
    "server_source_revision",
    "harness_revision",
    "fixture_sha256",
    "matrix_sha256",
    "template_sha256",
    "cache_state",
    "server_node",
    "gpu_uuid",
)

CAPACITY_FAILURE_PATTERNS = {
    "latency": ("latency p95 failed",),
    "throughput": (
        "success rate below 100%",
        "request errors recorded",
        "no positive MMP inference/batch delta",
    ),
    "fairness": ("Jain fairness failed",),
    "pool_admission": (
        "pool-full rejects",
        "pool-reject evidence missing or nonzero",
    ),
    "model_errors": ("model worker errors recorded",),
}


def _capacity_failures(result: Mapping[str, Any]) -> tuple[set[str], list[str]]:
    failures = result.get("failures") or []
    categories = set()
    invalid = []
    for failure in failures:
        matched = {
            category
            for category, patterns in CAPACITY_FAILURE_PATTERNS.items()
            if any(pattern in str(failure) for pattern in patterns)
        }
        if matched:
            categories.update(matched)
        else:
            invalid.append(str(failure))
    return categories, invalid


def analyze_curve(
    results: Sequence[Mapping[str, Any]],
    *,
    phase: str,
    allowed_points: Sequence[int],
    repetitions: int = 2,
) -> dict[str, Any]:
    failures: list[str] = []
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    run_ids: set[str] = set()
    cohorts: set[tuple[Any, ...]] = set()
    for result in results:
        if result.get("phase") != phase:
            failures.append("single-report analyzer phase mismatch")
            continue
        evidence = result.get("evidence", {})
        point = int(evidence.get("total_concurrency", 0))
        run_id = str(evidence.get("run_id", ""))
        if point not in allowed_points:
            failures.append(f"unexpected concurrency point {point}")
            continue
        if not run_id or run_id in run_ids:
            failures.append("run IDs must be nonempty and unique")
        run_ids.add(run_id)
        cohort = tuple(evidence.get(field) for field in COHORT_FIELDS)
        if any(value in {None, "", "unknown"} for value in cohort):
            failures.append("single-report cohort identity is incomplete")
        cohorts.add(cohort)
        grouped[point].append(result)
    if len(cohorts) != 1:
        failures.append("single-report results do not form one exact cohort")

    outcomes: dict[int, str] = {}
    for point in allowed_points:
        point_results = grouped.get(point, [])
        if not point_results:
            continue
        if len(point_results) != repetitions:
            failures.append(f"concurrency {point}: expected {repetitions} repetitions")
            outcomes[point] = "incomplete"
        elif (
            len(
                {
                    item.get("evidence", {}).get("workload_sha256")
                    for item in point_results
                }
            )
            != 1
        ):
            outcomes[point] = "split"
            failures.append(f"concurrency {point}: workload identities differ")
        elif all(bool(item.get("success")) for item in point_results):
            outcomes[point] = "pass"
        elif all(not bool(item.get("success")) for item in point_results):
            classified = [_capacity_failures(item) for item in point_results]
            shared_failures = set.intersection(
                *(categories for categories, _invalid in classified)
            )
            invalid_failures = [
                failure for _categories, invalid in classified for failure in invalid
            ]
            if invalid_failures:
                outcomes[point] = "invalid"
                failures.append(
                    f"concurrency {point}: failed repetition contains non-capacity failure"
                )
            elif shared_failures:
                outcomes[point] = "fail"
            else:
                outcomes[point] = "split"
                failures.append(
                    f"concurrency {point}: repetitions lack one shared capacity failure"
                )
        else:
            outcomes[point] = "split"
            failures.append(f"concurrency {point}: repetitions disagree")

    passing = [point for point, outcome in outcomes.items() if outcome == "pass"]
    capacity = max(passing) if passing else None
    next_point = None
    if capacity is None:
        failures.append("no concurrency point passed twice")
    else:
        capacity_index = list(allowed_points).index(capacity)
        if capacity_index + 1 >= len(allowed_points):
            failures.append("capacity is right-censored; test the next higher point")
        else:
            next_point = allowed_points[capacity_index + 1]
            if outcomes.get(next_point) != "fail":
                failures.append("next higher concurrency did not fail twice")
        if any(
            outcomes.get(point) != "pass"
            for point in allowed_points[: capacity_index + 1]
        ):
            failures.append("all tested points through capacity must pass twice")

    return {
        "schema_version": 1,
        "phase": phase,
        "success": not failures,
        "failures": failures,
        "capacity_total_concurrency": capacity if not failures else None,
        "next_failed_total_concurrency": next_point if not failures else None,
        "repetitions": repetitions,
        "allowed_points": list(allowed_points),
        "outcomes": {
            str(point): outcomes[point] for point in allowed_points if point in outcomes
        },
        "cohort": (
            dict(zip(COHORT_FIELDS, next(iter(cohorts)))) if len(cohorts) == 1 else None
        ),
    }


def compare_mps_pair(
    no_mps: Mapping[str, Any], mps: Mapping[str, Any]
) -> dict[str, Any]:
    failures = []
    if not no_mps.get("success") or not mps.get("success"):
        failures.append("both curves must be certified before A/B comparison")
    no_phase = str(no_mps.get("phase", ""))
    mps_phase = str(mps.get("phase", ""))
    if no_phase.removesuffix("-no-mps") != mps_phase.removesuffix("-mps"):
        failures.append("curves are not the matched no-MPS/MPS phase pair")
    if no_mps.get("cohort") != mps.get("cohort"):
        failures.append("no-MPS/MPS curves do not share one exact cohort")
    if no_mps.get("allowed_points") != mps.get("allowed_points"):
        failures.append("no-MPS/MPS curves do not use the same concurrency points")
    if no_mps.get("repetitions") != mps.get("repetitions"):
        failures.append("no-MPS/MPS curves do not use the same repetitions")
    return {"success": not failures, "failures": failures}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare-with", type=Path)
    args = parser.parse_args(argv)
    matrix = load_and_validate(args.matrix)
    phase = next((item for item in matrix["phases"] if item["id"] == args.phase), None)
    if phase is None:
        raise SystemExit(f"unknown phase {args.phase!r}")
    analyzed = [json.loads(path.read_text()) for path in args.results]
    result = analyze_curve(
        analyzed,
        phase=args.phase,
        allowed_points=phase["total_concurrency"],
        repetitions=int(matrix["fixed_runtime"]["repetitions"]),
    )
    if args.compare_with:
        paired = json.loads(args.compare_with.read_text())
        result["mps_pair"] = (
            compare_mps_pair(paired, result)
            if args.phase.endswith("-mps")
            else compare_mps_pair(result, paired)
        )
        if not result["mps_pair"]["success"]:
            result["success"] = False
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")
    return 0 if result["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
