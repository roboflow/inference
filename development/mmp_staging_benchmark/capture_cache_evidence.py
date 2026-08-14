#!/usr/bin/env python3
"""Capture read-only pre-run MMP route and model-cache evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from development.mmp_staging_benchmark.run_concurrent_clients import (  # noqa: E402
    validate_staging_url,
)

CONTEXT = "ck8s-stg"
NAMESPACE = "video-proc-bench-mmp"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


def _run(command: list[str]) -> str:
    return subprocess.run(
        command, check=True, capture_output=True, text=True
    ).stdout.strip()


def _metrics(server_url: str, api_key: str) -> Mapping[str, Any]:
    request = urllib.request.Request(
        f"{validate_staging_url(server_url)}/v2/server/metrics",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    opener = urllib.request.build_opener(_NoRedirect)
    with opener.open(request, timeout=10) as response:
        if response.status != 200:
            raise ValueError(f"metrics endpoint returned HTTP {response.status}")
        return json.loads(response.read())


def capture(pod: str, server_url: str, api_key: str) -> dict[str, object]:
    if _run(["kubectl", "config", "current-context"]) != CONTEXT:
        raise ValueError(f"current context must be {CONTEXT!r}")
    prefix = ["kubectl", "--context", CONTEXT, "-n", NAMESPACE]
    pod_document = json.loads(_run([*prefix, "get", "pod", pod, "-o", "json"]))
    if pod_document.get("metadata", {}).get("name") != pod:
        raise ValueError("pod response identity mismatch")
    cache_output = _run(
        [
            *prefix,
            "exec",
            pod,
            "-c",
            "server",
            "--",
            "find",
            "/models/cache",
            "-type",
            "f",
        ]
    )
    cache_files = sorted(
        line.strip() for line in cache_output.splitlines() if line.strip()
    )
    metrics = _metrics(server_url, api_key)
    raw_routes = metrics.get("mmp_models") or {}
    routes = {
        route: {
            field: model.get(field)
            for field in ("worker_pid", "inference_count", "batch_count", "error_count")
        }
        for route, model in sorted(raw_routes.items())
        if isinstance(model, Mapping)
    }
    for route, model in routes.items():
        pid = model.get("worker_pid")
        if not isinstance(pid, int) or pid <= 0:
            raise ValueError(f"route {route!r} does not report a positive worker PID")
        _run([*prefix, "exec", pod, "-c", "server", "--", "kill", "-0", str(pid)])
    return {
        "schema_version": 1,
        "context": CONTEXT,
        "namespace": NAMESPACE,
        "pod": pod,
        "pod_uid": pod_document.get("metadata", {}).get("uid"),
        "captured_unix_s": time.time(),
        "cache_file_count": len(cache_files),
        "cache_paths_sha256": hashlib.sha256(
            "\n".join(cache_files).encode()
        ).hexdigest(),
        "routes": routes,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod", required=True)
    parser.add_argument("--server-url", default="http://127.0.0.1:18000")
    parser.add_argument("--api-key-env", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise ValueError(f"missing API key environment variable {args.api_key_env}")
    evidence = capture(args.pod, args.server_url, api_key)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
