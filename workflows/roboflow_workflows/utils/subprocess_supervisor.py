"""Bounded lifecycle management for trusted integration worker modules.

Workers receive JSON on stdin, write results in a private temporary directory,
and may report a bounded error.json. This is not an OS sandbox or a hard memory
limit. Callers own destination policy, result validation and remote cancellation.
Only the direct child is terminated; descendant cleanup needs explicit support.
"""

import json
import math
import os
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Sequence, TypeVar

_Result = TypeVar("_Result")
_MAX_ERROR_BYTES = 2048


def bounded_json_bytes(value: Any, *, max_bytes: int) -> bytes:
    """Serialize JSON only after checking its encoded size without copying strings.

    Args:
        value (Any): JSON-compatible data with string dictionary keys.
        max_bytes (int): Maximum UTF-8 serialized size.

    Returns:
        bytes: Compact UTF-8 JSON within the limit.

    Raises:
        ValueError: Data is oversized, too deeply nested or unsupported.
    """
    remaining = max_bytes

    def reserve(size: int) -> None:
        nonlocal remaining
        remaining -= size
        if remaining < 0:
            raise ValueError("Worker request exceeds its byte limit")

    def string_size(text: str) -> None:
        # Even unescaped ASCII needs this many bytes. Reject huge inputs in O(1).
        if len(text) + 2 > remaining:
            raise ValueError("Worker request exceeds its byte limit")

        reserve(2)
        for character in text:
            code = ord(character)
            if character in '"\\\b\f\n\r\t':
                reserve(2)
            elif code < 32:
                reserve(6)
            elif code < 128:
                reserve(1)
            elif code < 2048:
                reserve(2)
            elif 0xD800 <= code <= 0xDFFF:
                raise ValueError("Worker request contains invalid Unicode")
            else:
                reserve(3 if code < 65536 else 4)

    def measure(item: Any, depth: int = 0) -> None:
        if depth > 32:
            raise ValueError("Worker request nesting exceeds its limit")

        if isinstance(item, str):
            string_size(item)
        elif item is None:
            reserve(4)
        elif isinstance(item, bool):
            reserve(4 if item else 5)
        elif isinstance(item, int):
            if item.bit_length() > 64:
                raise ValueError("Worker request integer exceeds 64 bits")
            reserve(len(str(item)))
        elif isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("Worker request contains a non-finite number")
            reserve(len(json.dumps(item, allow_nan=False)))
        elif isinstance(item, (list, tuple, dict)):
            reserve(2 + max(0, len(item) - 1))
            if isinstance(item, dict):
                for key, entry in item.items():
                    if not isinstance(key, str):
                        raise ValueError("Worker request keys must be strings")
                    string_size(key)
                    reserve(1)
                    measure(entry, depth + 1)
            else:
                for entry in item:
                    measure(entry, depth + 1)
        else:
            raise ValueError("Worker request contains an unsupported value")

    measure(value)
    payload = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if len(payload) > max_bytes:
        raise ValueError("Worker request exceeds its byte limit")

    return payload


def _check_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError("Integration worker deadline expired")


def _stop_worker(process: subprocess.Popen, *, cleanup_deadline: float) -> None:
    if process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=max(0.0, cleanup_deadline - time.monotonic() - 2.0))
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=max(0.1, cleanup_deadline - time.monotonic()))


def _read_error(directory: Path, secrets: Sequence[str]) -> Exception:
    path = directory / "error.json"
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("Worker error must be a regular file")
            payload = stream.read(_MAX_ERROR_BYTES + 1)
        if len(payload) > _MAX_ERROR_BYTES:
            raise ValueError("Worker error exceeds its limit")
        error = json.loads(payload)
        if not isinstance(error, dict) or not isinstance(error.get("message"), str):
            raise ValueError("Invalid worker error")
    except (OSError, ValueError, RecursionError):
        return RuntimeError("Integration worker failed without a valid error report")

    message = error["message"]
    for secret in secrets:
        if secret:
            message = message.replace(secret, "***")
    exception_type = {"TimeoutError": TimeoutError, "ValueError": ValueError}.get(
        error.get("kind"), RuntimeError
    )
    return exception_type(message)


def run_in_subprocess(
    request: Any,
    *,
    worker_module: str,
    deadline: float,
    read_result: Callable[[Path], _Result],
    secrets: Sequence[str] = (),
    max_request_bytes: int = 1024 * 1024,
    cleanup_seconds: float = 35.0,
) -> _Result:
    """Supervise one trusted worker and validate results before deleting its files.

    Args:
        request (Any): JSON-compatible worker request.
        worker_module (str): Application-controlled Python module, never user input.
        deadline (float): Absolute monotonic deadline including result loading.
        read_result (Callable): Integration-specific validator/loader for the directory.
        secrets (Sequence[str]): Values to redact from worker error messages.
        max_request_bytes (int): Serialized request size limit.
        cleanup_seconds (float): Additional time allowed to terminate, kill and reap.

    Returns:
        Any: The validated result returned by read_result.

    Raises:
        ValueError: The request or worker data is invalid.
        TimeoutError: The deadline expires.
        RuntimeError: The worker fails.
        OSError: Worker launch or filesystem operations fail.
    """
    _check_deadline(deadline)
    payload = bounded_json_bytes(request, max_bytes=max_request_bytes)
    _check_deadline(deadline)
    with tempfile.TemporaryDirectory(prefix="workflow-worker-") as temporary:
        directory = Path(temporary)
        process = subprocess.Popen(
            [sys.executable, "-m", worker_module, str(directory)],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )
        try:
            try:
                process.communicate(
                    input=payload, timeout=max(0.0, deadline - time.monotonic())
                )
            except subprocess.TimeoutExpired:
                raise TimeoutError("Integration worker deadline expired") from None
            _check_deadline(deadline)
            if process.returncode != 0:
                raise _read_error(directory, secrets)

            result = read_result(directory)
            _check_deadline(deadline)
            return result
        finally:
            _stop_worker(process, cleanup_deadline=time.monotonic() + cleanup_seconds)
            if process.stdin is not None:
                process.stdin.close()
