#!/usr/bin/env python3
"""Capture a live, read-only MPS server-list observation from the staging pod."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Sequence

CONTEXT = "ck8s-stg"
NAMESPACE = "video-proc-bench-mmp"


def _run(command: list[str], *, stdin: str | None = None) -> str:
    completed = subprocess.run(
        command,
        input=stdin,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _mps_control(prefix: list[str], pod: str, command: str) -> str:
    return _run(
        [
            *prefix,
            "exec",
            "-i",
            pod,
            "-c",
            "server",
            "--",
            "nvidia-cuda-mps-control",
        ],
        stdin=f"{command}\n",
    )


def capture(pod: str) -> dict[str, object]:
    current = _run(["kubectl", "config", "current-context"])
    if current != CONTEXT:
        raise ValueError(f"refusing context {current!r}; expected {CONTEXT!r}")
    prefix = ["kubectl", "--context", CONTEXT, "-n", NAMESPACE]
    pod_document = json.loads(_run([*prefix, "get", "pod", pod, "-o", "json"]))
    if pod_document.get("metadata", {}).get("name") != pod:
        raise ValueError("pod response identity mismatch")
    gpu_rows = _run(
        [
            *prefix,
            "exec",
            pod,
            "-c",
            "server",
            "--",
            "nvidia-smi",
            "--query-gpu=uuid",
            "--format=csv,noheader",
        ]
    ).splitlines()
    if len(gpu_rows) != 1 or not gpu_rows[0].strip().startswith("GPU-"):
        raise ValueError("expected exactly one GPU UUID")
    server_list = _mps_control(prefix, pod, "get_server_list")
    if not server_list or not all(
        line.strip().isdigit() for line in server_list.splitlines()
    ):
        raise ValueError("MPS get_server_list returned no numeric server PIDs")
    clients_by_server = {}
    for server_pid in server_list.splitlines():
        clients = _mps_control(prefix, pod, f"get_client_list {server_pid.strip()}")
        if clients and not all(line.strip().isdigit() for line in clients.splitlines()):
            raise ValueError("MPS get_client_list returned a nonnumeric client PID")
        clients_by_server[server_pid.strip()] = [
            int(line.strip()) for line in clients.splitlines() if line.strip()
        ]
    return {
        "schema_version": 1,
        "context": CONTEXT,
        "namespace": NAMESPACE,
        "pod": pod,
        "pod_uid": pod_document.get("metadata", {}).get("uid"),
        "gpu_uuid": gpu_rows[0].strip(),
        "captured_unix_s": time.time(),
        "command": "get_server_list",
        "exit_code": 0,
        "server_list": server_list,
        "clients_by_server": clients_by_server,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    evidence = capture(args.pod)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
