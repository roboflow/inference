"""Persistent model-weight cache for the self-hosted GPU test runner.

Every server test stage mounts a per-leg directory as the container's
MODEL_CACHE_DIR, so later stages and later runs skip the weight downloads.
The cache is discarded when MODEL_CACHE_KEY changes, when it is older than
MODEL_CACHE_MAX_AGE_DAYS, or when MODEL_CACHE_RESET is requested. The server
writes the files as root, so wiping runs inside the test image rather than
with sudo. Never print subprocess output.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path

KEY_FILE = ".cache-key"
CREATED_FILE = ".created"
CONTAINER_CACHE_DIR = "/tmp/cache"
DEFAULT_MAX_AGE_DAYS = "14"
DEFAULT_IMAGE = "roboflow/roboflow-inference-server-gpu:test"


def _run(args: list[str]) -> str:
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode:
        raise ValueError("Model cache operation failed; private output withheld")

    return result.stdout


def _emit(path_variable: str, name: str, value: str) -> None:
    if "\n" in value or "\r" in value:
        raise ValueError("Invalid output")

    with Path(os.environ[path_variable]).open("a") as handle:
        handle.write(f"{name}={value}\n")


def _cache_dir() -> Path:
    mode = os.environ.get("MODEL_MODE")
    if mode not in {"true", "false"}:
        raise ValueError("Invalid model mode")

    root = Path(os.environ["HOME"]) / ".cache" / "roboflow-inference-ci"
    cache_dir = root / "model-weights" / f"use-inference-models-{mode}"
    # The path is word-split into docker arguments by the Makefile.
    if re.search(r"\s", str(cache_dir)):
        raise ValueError("Model cache path must not contain whitespace")

    return cache_dir


def _key() -> str:
    key = os.environ.get("MODEL_CACHE_KEY", "")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", key):
        raise ValueError("Invalid model cache key")

    return key


def _max_age_seconds() -> int:
    days = os.environ.get("MODEL_CACHE_MAX_AGE_DAYS", DEFAULT_MAX_AGE_DAYS)
    if not re.fullmatch(r"[0-9]{1,4}", days):
        raise ValueError("Invalid model cache age")

    return int(days) * 86400


def _invalidation_reason(cache_dir: Path, *, key: str, max_age: int) -> str | None:
    if os.environ.get("MODEL_CACHE_RESET") == "true":
        return "reset requested"

    key_file = cache_dir / KEY_FILE
    created_file = cache_dir / CREATED_FILE
    if not key_file.is_file() or not created_file.is_file():
        return "cache markers missing"

    if key_file.read_text().strip() != key:
        return "cache key changed"

    try:
        created = int(created_file.read_text().strip())
    except ValueError:
        return "cache age unreadable"

    if time.time() - created > max_age:
        return "cache older than the maximum age"

    return None


def _wipe(cache_dir: Path) -> None:
    repository = os.environ.get("INFERENCE_SERVER_REPO", "")
    image = os.environ.get("INFERENCE_TEST_IMAGE", "")
    if not image:
        image = f"roboflow/{repository}:test" if repository else DEFAULT_IMAGE

    # Files inside belong to root (the server's user), so delete them through
    # the already-present test image instead of sudo.
    _run(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "find",
            "-v",
            f"{cache_dir}:/cache",
            image,
            "/cache",
            "-mindepth",
            "1",
            "-delete",
        ]
    )


def _write_markers(cache_dir: Path, *, key: str) -> None:
    (cache_dir / KEY_FILE).write_text(f"{key}\n")
    (cache_dir / CREATED_FILE).write_text(f"{int(time.time())}\n")


def _size_bytes(cache_dir: Path) -> int:
    total = 0
    for directory, _, files in os.walk(cache_dir, onerror=lambda error: None):
        for name in files:
            try:
                total += (Path(directory) / name).stat().st_size
            except OSError:
                continue

    return total


def _prepare() -> None:
    key = _key()
    max_age = _max_age_seconds()
    cache_dir = _cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    reason = _invalidation_reason(cache_dir, key=key, max_age=max_age)
    if reason is None:
        status = "reused"
    else:
        if any(cache_dir.iterdir()):
            _wipe(cache_dir)
        _write_markers(cache_dir, key=key)
        status = f"started fresh ({reason})"

    _emit(
        "GITHUB_ENV",
        "MODEL_CACHE_DOCKER_ARGS",
        f"-v {cache_dir}:{CONTAINER_CACHE_DIR} -e DISK_CACHE_CLEANUP=False",
    )
    used_gib = _size_bytes(cache_dir) / 2**30
    free_gib = shutil.disk_usage(cache_dir).free / 2**30
    print(
        f"Model cache {cache_dir}: {status}; "
        f"{used_gib:.1f} GiB cached, {free_gib:.1f} GiB free on disk"
    )


if __name__ == "__main__":
    try:
        _prepare()
    except (ValueError, KeyError, OSError):
        # Do not render the exception, command or user input.
        print("::error::Model cache preparation failed; no fallback performed")
        raise SystemExit(1) from None
