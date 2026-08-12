#!/usr/bin/env python3
"""Emit a JSON CUDA/MPS and shared-memory capability report.

The default probe is read-only. ``--start-stop-mps`` is intentionally explicit
and must only be used in a dedicated pod that exclusively owns the GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

SLOT_HEADER_BYTES = 64
MIB = 1024 * 1024


def _run(command: Sequence[str], *, timeout_s: float = 10.0) -> dict[str, Any]:
    started = time.monotonic()
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return {
            "command": list(command),
            "exit_code": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "duration_ms": round((time.monotonic() - started) * 1000, 3),
        }
    except FileNotFoundError:
        return {
            "command": list(command),
            "exit_code": None,
            "error": "not found",
            "duration_ms": round((time.monotonic() - started) * 1000, 3),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": list(command),
            "exit_code": None,
            "error": "timeout",
            "stdout": (exc.stdout or "").strip(),
            "stderr": (exc.stderr or "").strip(),
            "duration_ms": round((time.monotonic() - started) * 1000, 3),
        }


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise SystemExit(f"{name} must be an integer") from exc


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise SystemExit(f"{name} must be numeric") from exc


def shared_memory_report(path: str = "/dev/shm") -> dict[str, Any]:
    slots = _int_env("INFERENCE_N_SLOTS", 32)
    input_mb = _float_env("INFERENCE_INPUT_MB", 25.0)
    required = int(slots * (input_mb * MIB + SLOT_HEADER_BYTES))
    reserve_factor = _float_env("MMP_SHM_RESERVE_FACTOR", 1.25)
    recommended = int(required * reserve_factor)

    target = Path(path)
    exists = target.exists()
    report: dict[str, Any] = {
        "path": path,
        "exists": exists,
        "writable": os.access(path, os.W_OK) if exists else False,
        "slots": slots,
        "input_mb_per_slot": input_mb,
        "slot_header_bytes": SLOT_HEADER_BYTES,
        "required_bytes": required,
        "recommended_bytes": recommended,
        "reserve_factor": reserve_factor,
    }
    if exists:
        usage = shutil.disk_usage(path)
        report.update(
            {
                "total_bytes": usage.total,
                "free_bytes": usage.free,
                "satisfies_required": usage.free >= required,
                "satisfies_recommended": usage.free >= recommended,
            }
        )
    else:
        report.update({"satisfies_required": False, "satisfies_recommended": False})
    return report


def _torch_report() -> dict[str, Any]:
    try:
        import torch

        available = torch.cuda.is_available()
        result: dict[str, Any] = {
            "installed": True,
            "version": torch.__version__,
            "compiled_cuda": torch.version.cuda,
            "cuda_available": available,
            "device_count": torch.cuda.device_count() if available else 0,
        }
        if available:
            result["devices"] = [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
                for index in range(torch.cuda.device_count())
            ]
        return result
    except Exception as exc:  # probe must report import/driver failures
        return {
            "installed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _mps_start_stop() -> dict[str, Any]:
    control = shutil.which("nvidia-cuda-mps-control")
    server = shutil.which("nvidia-cuda-mps-server")
    if not control or not server:
        return {
            "attempted": False,
            "success": False,
            "error": "MPS control/server binary missing",
        }

    with tempfile.TemporaryDirectory(prefix="mmp-mps-") as root:
        pipe_dir = Path(root) / "pipe"
        log_dir = Path(root) / "log"
        pipe_dir.mkdir()
        log_dir.mkdir()
        env = dict(os.environ)
        env["CUDA_MPS_PIPE_DIRECTORY"] = str(pipe_dir)
        env["CUDA_MPS_LOG_DIRECTORY"] = str(log_dir)

        started = time.monotonic()
        start = subprocess.run(
            [control, "-d"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
            env=env,
        )
        query: dict[str, Any] | None = None
        stop: dict[str, Any] | None = None
        try:
            if start.returncode == 0:
                query_proc = subprocess.run(
                    [control],
                    input="get_server_list\n",
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    env=env,
                )
                query = {
                    "exit_code": query_proc.returncode,
                    "stdout": query_proc.stdout.strip(),
                    "stderr": query_proc.stderr.strip(),
                }
        finally:
            if start.returncode == 0:
                stop_proc = subprocess.run(
                    [control],
                    input="quit\n",
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    env=env,
                )
                stop = {
                    "exit_code": stop_proc.returncode,
                    "stdout": stop_proc.stdout.strip(),
                    "stderr": stop_proc.stderr.strip(),
                }

        logs = {}
        for log_file in sorted(log_dir.glob("*")):
            try:
                logs[log_file.name] = log_file.read_text(errors="replace")[-4000:]
            except OSError as exc:
                logs[log_file.name] = f"unreadable: {exc}"
        return {
            "attempted": True,
            "success": start.returncode == 0
            and stop is not None
            and stop["exit_code"] == 0,
            "duration_ms": round((time.monotonic() - started) * 1000, 3),
            "pipe_directory": str(pipe_dir),
            "log_directory": str(log_dir),
            "start": {
                "exit_code": start.returncode,
                "stdout": start.stdout.strip(),
                "stderr": start.stderr.strip(),
            },
            "query": query,
            "stop": stop,
            "logs": logs,
        }


def build_report(start_stop_mps: bool = False) -> dict[str, Any]:
    mps_control = shutil.which("nvidia-cuda-mps-control")
    mps_server = shutil.which("nvidia-cuda-mps-server")
    nvidia_smi = shutil.which("nvidia-smi")
    report: dict[str, Any] = {
        "schema_version": 1,
        "timestamp_unix_s": time.time(),
        "source_revision": os.environ.get("MMP_BENCHMARK_SOURCE_REVISION", "unknown"),
        "image_ref": os.environ.get("MMP_BENCHMARK_IMAGE_REF", "unknown"),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "nvidia_runtime": {
            "visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
            "driver_capabilities": os.environ.get("NVIDIA_DRIVER_CAPABILITIES"),
            "device_nodes": sorted(str(p) for p in Path("/dev").glob("nvidia*")),
            "nvidia_smi_path": nvidia_smi,
            "mps_control_path": mps_control,
            "mps_server_path": mps_server,
            "mps_binaries_available": bool(mps_control and mps_server),
            "cuda_mps_pipe_directory": os.environ.get("CUDA_MPS_PIPE_DIRECTORY"),
            "cuda_mps_log_directory": os.environ.get("CUDA_MPS_LOG_DIRECTORY"),
            "cuda_mps_active_thread_percentage": os.environ.get(
                "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"
            ),
        },
        "shared_memory": shared_memory_report(),
        "torch": _torch_report(),
    }

    package_manifest = Path("/opt/mmp-benchmark/python-packages.txt")
    if package_manifest.is_file():
        manifest_bytes = package_manifest.read_bytes()
        report["python_packages"] = {
            "path": str(package_manifest),
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "line_count": len(manifest_bytes.splitlines()),
        }

    if nvidia_smi:
        report["nvidia_runtime"]["gpu_query"] = _run(
            [
                nvidia_smi,
                "--query-gpu=index,name,uuid,driver_version,memory.total,compute_mode",
                "--format=csv,noheader,nounits",
            ]
        )
        report["nvidia_runtime"]["mig_query"] = _run(
            [nvidia_smi, "--query-gpu=index,mig.mode.current", "--format=csv,noheader"]
        )
    if start_stop_mps:
        report["mps_start_stop"] = _mps_start_stop()
    return report


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="write JSON report to this path")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--require-mps", action="store_true")
    parser.add_argument("--require-shm", action="store_true")
    parser.add_argument(
        "--start-stop-mps",
        action="store_true",
        help="start and stop MPS; dedicated single-GPU experiment pod only",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_report(start_stop_mps=args.start_stop_mps)
    failures: list[str] = []
    runtime = report["nvidia_runtime"]
    if args.require_gpu:
        if not runtime["nvidia_smi_path"]:
            failures.append("nvidia-smi missing")
        if not report["torch"].get("cuda_available", False):
            failures.append("torch CUDA unavailable")
    if args.require_mps and not runtime["mps_binaries_available"]:
        failures.append("MPS control/server binaries missing")
    if args.start_stop_mps and not report.get("mps_start_stop", {}).get(
        "success", False
    ):
        failures.append("MPS start/stop smoke failed")
    if args.require_shm and not report["shared_memory"]["satisfies_recommended"]:
        failures.append("/dev/shm below recommended capacity")

    report["checks"] = {"passed": not failures, "failures": failures}
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n")
    print(rendered)
    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())
