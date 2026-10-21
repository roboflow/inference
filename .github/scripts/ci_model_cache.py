"""Persistent model-weight cache for the self-hosted GPU test runner.

Every server test stage mounts a per-leg directory as the container's
MODEL_CACHE_DIR, so later stages and later runs skip the weight downloads.
The cache is discarded when MODEL_CACHE_KEY changes, when it is older than
MODEL_CACHE_MAX_AGE_DAYS, when free disk drops below MODEL_CACHE_MIN_FREE_GIB,
or when MODEL_CACHE_RESET is requested. A run marks the directory in use and
releases it at the end; a concurrent run of the same leg falls back to the
container-local cache instead of sharing files with a live server. A run also
marks the directory dirty and cleans it only when it was not cancelled, so a
container killed mid-download (LoRA bases extract a tar in place and cannot be
made atomic from outside) never leaves a partial cache behind. The server
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
IN_USE_FILE = ".in-use"
DIRTY_FILE = ".dirty"
CONTAINER_CACHE_DIR = "/tmp/cache"
DEFAULT_MAX_AGE_DAYS = "14"
DEFAULT_MIN_FREE_GIB = "20"
# A job may hold the cache for at most the workflow's timeout-minutes.
DEFAULT_BUSY_TIMEOUT_MINUTES = "120"
DEFAULT_IMAGE = "roboflow/roboflow-inference-server-gpu:test"
CONTAINER_ENV = "-e DISK_CACHE_CLEANUP=False -e ATOMIC_CACHE_WRITES_ENABLED=True"


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


def _number(name: str, default: str) -> int:
    value = os.environ.get(name, default)
    if not re.fullmatch(r"[0-9]{1,5}", value):
        raise ValueError(f"Invalid {name}")

    return int(value)


def _identity() -> str:
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "")
    if not re.fullmatch(r"[0-9]{1,20}", run_id) or not re.fullmatch(
        r"[0-9]{1,5}", attempt
    ):
        raise ValueError("Invalid run identity")

    return f"{run_id}-{attempt}"


def _read_marker(path: Path) -> tuple[str, int] | None:
    if not path.is_file():
        return None

    fields = path.read_text().split()
    if len(fields) != 2 or not re.fullmatch(r"[0-9]+", fields[1]):
        return None

    return fields[0], int(fields[1])


def _held_by_other_run(cache_dir: Path, *, identity: str, timeout: int) -> bool:
    marker = _read_marker(cache_dir / IN_USE_FILE)
    if marker is None:
        return False

    owner, since = marker
    # A marker older than the job timeout belongs to a run the runner lost.
    return owner != identity and time.time() - since <= timeout


def _invalidation_reason(cache_dir: Path, *, key: str, max_age: int) -> str | None:
    if os.environ.get("MODEL_CACHE_RESET") == "true":
        return "reset requested"

    if (cache_dir / DIRTY_FILE).is_file():
        return "previous run was interrupted"

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


def _free_gib(cache_dir: Path) -> float:
    return shutil.disk_usage(cache_dir).free / 2**30


def _prepare() -> None:
    key = _key()
    max_age = _number("MODEL_CACHE_MAX_AGE_DAYS", DEFAULT_MAX_AGE_DAYS) * 86400
    min_free = _number("MODEL_CACHE_MIN_FREE_GIB", DEFAULT_MIN_FREE_GIB)
    timeout = _number("MODEL_CACHE_BUSY_TIMEOUT_MINUTES", DEFAULT_BUSY_TIMEOUT_MINUTES)
    identity = _identity()
    cache_dir = _cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    if _held_by_other_run(cache_dir, identity=identity, timeout=timeout * 60):
        _emit("GITHUB_ENV", "MODEL_CACHE_DOCKER_ARGS", "")
        print(f"Model cache {cache_dir}: in use by another run; not mounted")
        return

    reason = _invalidation_reason(cache_dir, key=key, max_age=max_age)
    if reason is None and _free_gib(cache_dir) < min_free:
        reason = "free disk below the minimum"
    if reason is None:
        status = "reused"
    else:
        if any(cache_dir.iterdir()):
            _wipe(cache_dir)
        _write_markers(cache_dir, key=key)
        status = f"started fresh ({reason})"

    free_gib = _free_gib(cache_dir)
    if free_gib < min_free:
        _emit("GITHUB_ENV", "MODEL_CACHE_DOCKER_ARGS", "")
        print(
            "::warning::Runner free disk is below MODEL_CACHE_MIN_FREE_GIB even "
            "with an empty model cache; weights are not cached for this run"
        )
        return

    (cache_dir / DIRTY_FILE).write_text(f"{identity}\n")
    (cache_dir / IN_USE_FILE).write_text(f"{identity} {int(time.time())}\n")
    _emit(
        "GITHUB_ENV",
        "MODEL_CACHE_DOCKER_ARGS",
        f"-v {cache_dir}:{CONTAINER_CACHE_DIR} {CONTAINER_ENV}",
    )
    used_gib = _size_bytes(cache_dir) / 2**30
    print(
        f"Model cache {cache_dir}: {status}; "
        f"{used_gib:.1f} GiB cached, {free_gib:.1f} GiB free on disk"
    )


def _remove_own_marker(marker_path: Path) -> None:
    if not marker_path.is_file():
        return

    owner = marker_path.read_text().split()[:1]
    if owner != [_identity()]:
        return

    marker_path.unlink(missing_ok=True)


def _clean() -> None:
    # Runs only when the job was not cancelled, so no container was killed
    # while writing into the cache.
    _remove_own_marker(_cache_dir() / DIRTY_FILE)


def _release() -> None:
    _remove_own_marker(_cache_dir() / IN_USE_FILE)


if __name__ == "__main__":
    try:
        {
            "prepare": _prepare,
            "clean": _clean,
            "release": _release,
        }[os.environ["CI_MODEL_CACHE_ACTION"]]()
    except (ValueError, KeyError, OSError):
        # Do not render the exception, command or user input.
        print("::error::Model cache step failed; no fallback performed")
        raise SystemExit(1) from None
