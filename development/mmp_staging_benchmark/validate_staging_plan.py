#!/usr/bin/env python3
"""Validate and render the immutable, staging-only MMP/MPS experiment plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from development.mmp_staging_benchmark.render_staging_deployment import (  # noqa: E402
    render,
)

EXPECTED_MODES = {
    "mmp-shared-no-mps": False,
    "mmp-isolated-no-mps": False,
    "mmp-mixed-no-mps": False,
    "mmp-shared-mps": True,
    "mmp-isolated-mps": True,
    "mmp-mixed-mps": True,
}
SHA256 = re.compile(r"^[0-9a-f]{64}$")
REPO_ROOT = Path(__file__).resolve().parents[2]


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_and_validate(path: Path) -> dict[str, Any]:
    matrix = json.loads(path.read_text())
    if matrix.get("schema_version") != 1:
        raise ValueError("matrix schema_version must be 1")
    artifact = matrix.get("artifact", {})
    fixed = matrix.get("fixed_runtime", {})
    phases = matrix.get("phases", [])
    gates = matrix.get("strict_gates", {})
    phase_map = {phase.get("id"): phase for phase in phases}
    if set(phase_map) != set(EXPECTED_MODES) or len(phase_map) != len(phases):
        raise ValueError("matrix must define each required phase exactly once")
    if fixed.get("exclusive_gpu") is not True or fixed.get("gpu") != "L40S":
        raise ValueError("matrix must reserve one exclusive L40S")
    if fixed.get("shm_size_limit") != "4Gi":
        raise ValueError("matrix must retain the certified 4Gi /dev/shm")
    if fixed.get("decoder") not in {"imagecodecs", "nvjpeg"}:
        raise ValueError("matrix decoder is invalid")
    fixture_sha256 = fixed.get("fixture_sha256", "")
    if not SHA256.fullmatch(fixture_sha256):
        raise ValueError("matrix must pin a lowercase fixture SHA256")
    fixture_path = fixed.get("fixture_path", "")
    fixture = REPO_ROOT / fixture_path
    if not fixture.is_file():
        raise ValueError("matrix fixture_path must resolve to a checked-in file")
    if hashlib.sha256(fixture.read_bytes()).hexdigest() != fixture_sha256:
        raise ValueError("matrix fixture digest does not match fixture_path")
    if int(fixed.get("repetitions", 0)) < 2:
        raise ValueError("matrix requires at least two repetitions")
    for phase_id, expected_mps in EXPECTED_MODES.items():
        phase = phase_map[phase_id]
        if phase.get("mps") is not expected_mps:
            raise ValueError(f"{phase_id}: incorrect MPS mode")
        points = phase.get("total_concurrency", [])
        if not points or points != sorted(set(points)) or min(points) <= 0:
            raise ValueError(f"{phase_id}: concurrency must be positive and ordered")
    if gates.get("success_rate") != 1.0:
        raise ValueError("strict capacity must require a 100% success rate")
    if float(gates.get("latency_p95_ms_max", 0)) <= 0:
        raise ValueError("strict latency gate must be positive")
    return matrix


def render_plan(matrix: dict[str, Any], run_prefix: str) -> dict[str, Any]:
    artifact = matrix["artifact"]
    fixed = matrix["fixed_runtime"]
    deployments = {}
    for mps in (False, True):
        mode = "mps" if mps else "no-mps"
        run_id = f"{run_prefix}-{mode}"
        deployments[mode] = render(
            artifact["image"],
            artifact["source_revision"],
            run_id=run_id,
            mps=mps,
            decoder=fixed["decoder"],
        )
    return {
        "schema_version": 1,
        "matrix_sha256": canonical_sha256(matrix),
        "run_prefix": run_prefix,
        "artifact": artifact,
        "deployments": deployments,
        "phases": matrix["phases"],
        "strict_gates": matrix["strict_gates"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    plan = render_plan(load_and_validate(args.matrix), args.run_prefix)
    rendered = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
