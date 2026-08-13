#!/usr/bin/env python3
"""Certify a capacity boundary from paired single-report analyzer results."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


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
        grouped[point].append(result)

    outcomes: dict[int, str] = {}
    for point in allowed_points:
        point_results = grouped.get(point, [])
        if not point_results:
            continue
        if len(point_results) != repetitions:
            failures.append(f"concurrency {point}: expected {repetitions} repetitions")
            outcomes[point] = "incomplete"
        elif all(bool(item.get("success")) for item in point_results):
            outcomes[point] = "pass"
        elif all(not bool(item.get("success")) for item in point_results):
            outcomes[point] = "fail"
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
        "outcomes": {
            str(point): outcomes[point] for point in allowed_points if point in outcomes
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    matrix = json.loads(args.matrix.read_text())
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
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")
    return 0 if result["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
