#!/usr/bin/env python3
"""Render one exact, loopback-only MMP benchmark point from a checked-in spec."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from development.mmp_staging_benchmark.run_concurrent_clients import (  # noqa: E402
    load_spec,
)
from development.mmp_staging_benchmark.validate_staging_plan import (  # noqa: E402
    load_and_validate,
)


def render_point(
    template: Path,
    total_concurrency: int,
    duration_s: float,
    warmup_s: float,
    experiment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if total_concurrency <= 0:
        raise ValueError("total concurrency must be positive")
    raw = json.loads(template.read_text())
    clients = raw.get("clients", [])
    if len(clients) != 2:
        raise ValueError("point templates must contain exactly two clients")
    if total_concurrency == 1:
        clients = [clients[0]]
        clients[0]["concurrency"] = 1
    else:
        if total_concurrency % 2:
            raise ValueError("two-client points require even total concurrency")
        per_client = total_concurrency // 2
        for client in clients:
            client["concurrency"] = per_client
    raw["server_url"] = "http://127.0.0.1:18000"
    raw["duration_s"] = duration_s
    raw["warmup_s"] = warmup_s
    raw["clients"] = clients
    if experiment is not None:
        raw["experiment"] = experiment
    return raw


def current_clean_revision(repo_root: Path) -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise ValueError("benchmark harness checkout must be clean")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(revision) != 40:
        raise ValueError("could not resolve exact harness revision")
    return revision


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--total-concurrency", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--server-pod", required=True)
    parser.add_argument("--server-node", required=True)
    args = parser.parse_args(argv)
    harness_revision = current_clean_revision(Path(__file__).resolve().parents[2])
    matrix = load_and_validate(args.matrix)
    phase = next((item for item in matrix["phases"] if item["id"] == args.phase), None)
    if phase is None:
        raise ValueError(f"unknown phase {args.phase!r}")
    if args.total_concurrency not in phase["total_concurrency"]:
        raise ValueError("concurrency point is not in the selected matrix phase")
    fixed = matrix["fixed_runtime"]
    artifact = matrix["artifact"]
    template = args.matrix.parent / phase["spec"]
    rendered = render_point(
        template,
        args.total_concurrency,
        fixed["duration_s"],
        fixed["warmup_s"],
        experiment={
            "run_id": args.run_id,
            "phase": args.phase,
            "server_image_ref": artifact["image"],
            "server_source_revision": artifact["source_revision"],
            "harness_revision": harness_revision,
            "server_pod": args.server_pod,
            "server_node": args.server_node,
            "mps_enabled": "1" if phase["mps"] else "0",
            "decoder": fixed["decoder"],
            "slots": fixed["shm_slots"],
            "input_mb_per_slot": fixed["shm_input_mb_per_slot"],
            "batch_max_size": fixed["batch_max_size"],
            "batch_max_wait_ms": fixed["batch_max_wait_ms"],
            "fixture_sha256": fixed["fixture_sha256"],
        },
    )
    args.output.write_text(json.dumps(rendered, indent=2, sort_keys=True) + "\n")
    load_spec(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
