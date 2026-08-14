"""Bounded IPC primitives for the staging per-job process experiment.

Video pixels and tensors deliberately never use this protocol. A child publishes
watched image output directly to MediaMTX. Small, image-redacted JSON workflow
results cross this boundary through a latest-value queue so browser consumers keep
working without allowing a slow reader to backpressure inference.
"""

from __future__ import annotations

import copy
import json
import math
import os
import threading
from typing import Any, Mapping

PROTOCOL_VERSION = 1
MAX_MESSAGE_BYTES = 64 * 1024
MAX_LOG_LINES = 40
MAX_LOG_LINE_BYTES = 1000

CHILD_EVENT_TYPES = frozenset(
    {"started", "status", "result", "failure", "completed", "stopped"}
)
PARENT_COMMAND_TYPES = frozenset({"watch", "stop"})
JOB_EXECUTION_THREAD = "thread"
JOB_EXECUTION_PROCESS = "process"
JOB_EXECUTION_MODES = (JOB_EXECUTION_THREAD, JOB_EXECUTION_PROCESS)
STATS_KEYS = frozenset(
    {
        "schemaVersion",
        "frames",
        "fps",
        "decodeToResultLatencyMs",
        "counters",
        "decodeToResultLatency",
        "timing",
        "pipelineStartS",
        "timeToFirstResultS",
    }
)
RUNTIME_KEYS = frozenset(
    {
        "schemaVersion",
        "processorId",
        "cell",
        "hostname",
        "processId",
        "image",
        "revision",
        "variant",
        "podUid",
        "gpuVisibleDevices",
        "videoIngestMode",
        "tensorRepresentationEnabled",
        "rtspLatencyMs",
        "videoProducer",
        "sourceStream",
        "tensorBridge",
        "hardwareDecodeVerified",
        "jobExecutionMode",
        "jobProcessProtocolVersion",
    }
)
SUPERVISOR_ENVIRONMENT_KEYS = (
    "VIDEO_PROC_SERVICE_SECRET",
    "PROCESSOR_PUBSUB_SUBSCRIPTION",
    "PROCESSOR_PUBLIC_URL",
    "GATEWAY_PUBLIC_BASE",
    "ROBOFLOW_API_KEY",
)
FORBIDDEN_IPC_FIELD_NAMES = frozenset(
    {
        "apikey",
        "token",
        "accesstoken",
        "processoraccesstoken",
        "sourceurl",
        "simpublishurl",
        "outpublishurl",
        "outwhipurl",
        "workflowspecification",
        "frame",
        "pixels",
        "tensor",
    }
)


def resolve_job_execution_mode(value=None) -> str:
    mode = str(value or os.getenv("PROCESSOR_JOB_EXECUTION_MODE", "thread"))
    mode = mode.strip().lower()
    if mode not in JOB_EXECUTION_MODES:
        raise ValueError(
            "PROCESSOR_JOB_EXECUTION_MODE must be one of "
            + ", ".join(JOB_EXECUTION_MODES)
        )
    return mode


def _encoded_size(value: Any) -> int:
    return len(json.dumps(value, default=str, separators=(",", ":")).encode())


def _contains_forbidden_field(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = "".join(
                character for character in str(key).lower() if character.isalnum()
            )
            if normalized in FORBIDDEN_IPC_FIELD_NAMES:
                return True
            if _contains_forbidden_field(nested):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_forbidden_field(item) for item in value)
    return False


def _is_bounded_json_value(value: Any, depth: int = 0) -> bool:
    """Reject pickle-capable objects even when ``json.dumps(default=str)`` works."""

    if depth > 16:
        return False
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_is_bounded_json_value(item, depth + 1) for item in value)
    if isinstance(value, Mapping):
        return all(
            isinstance(key, str) and _is_bounded_json_value(nested, depth + 1)
            for key, nested in value.items()
        )
    return False


def bounded_child_event(event: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one child event before the supervisor accepts it.

    The allowlist is intentionally small.  In particular, job payloads, API
    keys, source URLs, workflow definitions, frames, tensors, and arbitrary
    exception objects have no protocol field and cannot be reflected into
    status or persisted diagnostics by mistake.
    """

    event = dict(event)
    if _contains_forbidden_field(event):
        raise ValueError("job-process IPC event contains a forbidden field")
    if event.get("version") != PROTOCOL_VERSION:
        raise ValueError("unsupported job-process IPC version")
    if event.get("type") not in CHILD_EVENT_TYPES:
        raise ValueError("unsupported job-process IPC event")
    if event["type"] == "result":
        if set(event) != {"version", "type", "result"}:
            raise ValueError("job-process result event has unexpected fields")
        result = event.get("result")
        if not isinstance(result, dict) or set(result) != {
            "frameId",
            "timestamp",
            "latencyMs",
            "outputs",
        }:
            raise ValueError("job-process result event is invalid")
        if not isinstance(result["frameId"], (str, int)):
            raise ValueError("job-process result frame ID is invalid")
        if not isinstance(result["timestamp"], str) or len(result["timestamp"]) > 64:
            raise ValueError("job-process result timestamp is invalid")
        if result["latencyMs"] is not None and not isinstance(
            result["latencyMs"], (int, float)
        ):
            raise ValueError("job-process result latency is invalid")
        if not isinstance(result["outputs"], dict):
            raise ValueError("job-process result outputs are invalid")
        if not _is_bounded_json_value(result):
            raise ValueError("job-process result must contain only JSON values")
        if _encoded_size(event) > MAX_MESSAGE_BYTES:
            raise ValueError("job-process IPC event exceeds size limit")
        return copy.deepcopy(event)
    allowed = {
        "version",
        "type",
        "state",
        "stats",
        "runtime",
        "imageOutputs",
        "defaultImageOutput",
        "error",
        "logTail",
    }
    if set(event) - allowed:
        raise ValueError("job-process IPC event has unexpected fields")
    state = event.get("state")
    if state is not None and state not in {
        "starting",
        "running",
        "completed",
        "error",
        "stopped",
    }:
        raise ValueError("job-process IPC state is invalid")
    stats = event.get("stats")
    if stats is not None and (not isinstance(stats, dict) or set(stats) - STATS_KEYS):
        raise ValueError("job-process IPC stats have unexpected fields")
    runtime = event.get("runtime")
    if runtime is not None and (
        not isinstance(runtime, dict) or set(runtime) - RUNTIME_KEYS
    ):
        raise ValueError("job-process IPC runtime has unexpected fields")
    outputs = event.get("imageOutputs")
    if outputs is not None and (
        not isinstance(outputs, list)
        or len(outputs) > 64
        or any(not isinstance(output, str) or len(output) > 256 for output in outputs)
    ):
        raise ValueError("job-process IPC image outputs are invalid")
    default_output = event.get("defaultImageOutput")
    if default_output is not None and (
        not isinstance(default_output, str) or len(default_output) > 256
    ):
        raise ValueError("job-process IPC default image output is invalid")
    error = event.get("error")
    if error is not None and (not isinstance(error, str) or len(error) > 2000):
        raise ValueError("job-process IPC error is invalid")
    log_tail = event.get("logTail")
    if log_tail is not None and (
        not isinstance(log_tail, list)
        or len(log_tail) > MAX_LOG_LINES
        or any(
            not isinstance(line, str) or len(line.encode()) > MAX_LOG_LINE_BYTES
            for line in log_tail
        )
    ):
        raise ValueError("job-process IPC log tail is invalid")
    if _encoded_size(event) > MAX_MESSAGE_BYTES:
        raise ValueError("job-process IPC event exceeds size limit")
    return copy.deepcopy(event)


def bounded_parent_command(command: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the tiny parent-to-child command vocabulary."""

    command = dict(command)
    if command.get("version") != PROTOCOL_VERSION:
        raise ValueError("unsupported job-process command version")
    if command.get("type") not in PARENT_COMMAND_TYPES:
        raise ValueError("unsupported job-process command")
    allowed = {"version", "type", "watch"}
    if set(command) - allowed:
        raise ValueError("job-process command has unexpected fields")
    if command["type"] == "watch":
        watch = command.get("watch")
        if not isinstance(watch, dict):
            raise ValueError("watch command requires an object")
        # The platform watch contract currently has two bounded scalar fields.
        if set(watch) - {"output", "requested"}:
            raise ValueError("watch command has unexpected fields")
        output = watch.get("output")
        if output is not None and (not isinstance(output, str) or len(output) > 256):
            raise ValueError("watch output is invalid")
        requested = watch.get("requested")
        if requested is not None and not isinstance(requested, bool):
            raise ValueError("watch requested is invalid")
    if _encoded_size(command) > 4096:
        raise ValueError("job-process command exceeds size limit")
    return copy.deepcopy(command)


def send_parent_command(connection, command: Mapping[str, Any]) -> bool:
    """Best-effort send on the parent-to-child control pipe.

    A child can exit between ``is_alive()`` and ``send()``.  EOF or a broken
    pipe is therefore an expected crash-cleanup outcome, not evidence that the
    supervisor itself is wedged.  Callers still validate the complete command
    before attempting the write.
    """

    command = bounded_parent_command(command)
    if connection is None:
        return False
    try:
        connection.send(command)
    except (EOFError, BrokenPipeError, OSError, ValueError):
        return False
    return True


def wait_for_process_exit(process, timeout_s: float):
    """Boundedly reap a child and return its finalized exit status.

    Pipe EOF can become visible slightly before ``multiprocessing`` updates
    ``Process.exitcode``.  Joining here preserves the real signal/exit code in
    the failure report and prevents cleanup from acting on a transitional
    ``is_alive()`` result.
    """

    if process is None:
        return None
    try:
        process.join(timeout=max(0.0, float(timeout_s)))
        return process.exitcode
    except (AssertionError, OSError, ValueError):
        return None


def bounded_log_tail(lines) -> list[str]:
    return [
        str(line).encode()[:MAX_LOG_LINE_BYTES].decode(errors="replace")
        for line in list(lines or [])[-MAX_LOG_LINES:]
    ]


def remove_supervisor_credentials(environ=None) -> dict[str, str]:
    """Remove and return supervisor-only environment values."""

    environ = os.environ if environ is None else environ
    removed = {}
    for name in SUPERVISOR_ENVIRONMENT_KEYS:
        value = environ.pop(name, None)
        if value is not None:
            removed[name] = value
    return removed


def restore_supervisor_credentials(values, environ=None) -> None:
    environ = os.environ if environ is None else environ
    for name, value in dict(values).items():
        if name not in SUPERVISOR_ENVIRONMENT_KEYS:
            raise ValueError("refusing to restore an unknown environment key")
        environ[name] = value


def drop_supervisor_credentials(environ=None) -> None:
    """Remove credentials used only for claims/status from a spawned child.

    The job-scoped workspace API key remains in the in-memory job descriptor
    because the workflow/model loader requires it.  It never returns over IPC.
    """

    remove_supervisor_credentials(environ)


class RemoteStats:
    """Thread-safe parent-side view of the latest child telemetry snapshot."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot = {
            "schemaVersion": 2,
            "frames": 0,
            "fps": None,
            "decodeToResultLatencyMs": None,
            "counters": {},
        }

    def replace(self, value) -> None:
        if (
            not isinstance(value, dict)
            or set(value) - STATS_KEYS
            or _encoded_size(value) > MAX_MESSAGE_BYTES
        ):
            raise ValueError("invalid remote stats snapshot")
        with self._lock:
            self._snapshot = copy.deepcopy(value)

    def snapshot(self, runtime=None) -> dict[str, Any]:
        with self._lock:
            result = copy.deepcopy(self._snapshot)
        if runtime:
            result["runtime"] = copy.deepcopy(runtime)
        return result
