"""Security primitives for the video processor's job-addressed HTTP API.

This module intentionally uses only the Python standard library so its tenant
boundary can be tested without importing the inference runtime.
"""

import collections
import hashlib
import hmac
import os
import re
import threading
from urllib.parse import parse_qs, urlsplit, urlunsplit

JOB_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
CELL_ID_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_BEARER_VALUE_RE = re.compile(r"(?i)\bBearer\s+[^\s,;]+")
_SECRET_PARAM_RE = re.compile(
    r"(?i)(\b(?:api[_-]?key|token|access_token|signature|sig)=)[^\s&#]+"
)
_URL_RE = re.compile(r"(?P<url>(?:https?|rtsp)://[^\s\"'<>]+)", re.IGNORECASE)


class MissingJobAccessToken(ValueError):
    """A managed-pool claim did not include its browser-facing job token."""


class JobPlacementMismatch(ValueError):
    """A server-issued job placement is unsafe for this worker."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


def validate_cell_id(value, *, required=False):
    """Normalize a bounded deployment cell identifier.

    Cell identity is optional only for the single-cell migration path. Once a
    worker is configured, this value is read once at startup and job payloads
    cannot override it.
    """

    cell = str(value or "").strip()
    if not cell:
        if required:
            raise ValueError("processor cell is required")
        return None
    if not CELL_ID_RE.fullmatch(cell):
        raise ValueError("processor cell must be a lowercase DNS label")
    return cell


def validate_job_placement(job, processor_cell):
    """Fail closed on wrong-cell or implicit cross-cell claim payloads.

    Missing placement remains valid for legacy jobs during the one-cell
    migration. A placed job, however, requires a configured worker cell. Remote
    media consumption is accepted only when the control plane explicitly marks
    it as such.
    """

    job = job if isinstance(job, dict) else {}
    execution_cell = job.get("executionCell")
    source_cell = job.get("sourceCell")
    if execution_cell is None and source_cell is None:
        return
    try:
        execution_cell = validate_cell_id(execution_cell, required=True)
        source_cell = (
            validate_cell_id(source_cell, required=True)
            if source_cell is not None
            else None
        )
    except ValueError as error:
        raise JobPlacementMismatch("invalid_placement", str(error)) from error
    if processor_cell is None:
        raise JobPlacementMismatch(
            "processor_cell_missing",
            "placed job cannot run on a worker without cell identity",
        )
    if execution_cell != processor_cell:
        raise JobPlacementMismatch(
            "execution_cell_mismatch",
            "job execution cell does not match processor cell",
        )
    if (
        source_cell is not None
        and source_cell != execution_cell
        and job.get("remoteExecution") is not True
    ):
        raise JobPlacementMismatch(
            "implicit_cross_cell",
            "cross-cell media execution was not explicitly authorized",
        )


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def validate_job_id(job_id) -> str:
    value = str(job_id or "")
    if not JOB_ID_RE.fullmatch(value):
        raise ValueError("invalid job id")
    return value


def extract_access_token(authorization: str, request_path: str) -> str:
    """Read a job token from Bearer auth or the media-element query fallback."""
    header = (authorization or "").strip()
    scheme, separator, value = header.partition(" ")
    if separator and scheme.lower() == "bearer":
        return value.strip()
    query = parse_qs(urlsplit(request_path).query, keep_blank_values=True)
    return (query.get("access_token") or [""])[0]


def _redact_url(match) -> str:
    value = match.group("url")
    try:
        parsed = urlsplit(value)
        host = parsed.hostname or ""
        if parsed.port is not None:
            host = f"{host}:{parsed.port}"
        if parsed.username is not None or parsed.password is not None:
            host = f"[credentials-redacted]@{host}"
        query = "redacted" if parsed.query else ""
        return urlunsplit((parsed.scheme, host, parsed.path, query, ""))
    except (TypeError, ValueError):
        return "[url-redacted]"


def sanitize_diagnostic(value) -> str:
    """Remove credentials and signed-query material before persisting logs."""
    text = str(value)
    text = _BEARER_VALUE_RE.sub("Bearer [redacted]", text)
    text = _SECRET_PARAM_RE.sub(r"\1[redacted]", text)
    return _URL_RE.sub(_redact_url, text)


def format_inference_error(payload) -> str:
    """Turn an InferencePipeline error payload into a safe user-facing detail."""
    payload = payload if isinstance(payload, dict) else {}
    error_type = str(payload.get("error_type") or "InferenceError").strip()
    error_message = str(
        payload.get("error_message")
        or "the workflow inference thread stopped unexpectedly"
    ).strip()
    return sanitize_diagnostic(f"{error_type}: {error_message}")


class DiagnosticRing:
    """A small, explicitly job-scoped and credential-sanitized log tail."""

    def __init__(self, capacity=150):
        self._buf = collections.deque(maxlen=capacity)
        self._lock = threading.Lock()

    def note(self, line):
        parts = [
            sanitize_diagnostic(part) for part in str(line).splitlines() if part.strip()
        ]
        with self._lock:
            self._buf.extend(parts)

    def tail(self, n=40):
        with self._lock:
            return list(self._buf)[-n:]


class JobSecurityRegistry:
    """Maps opaque tokens and internally-created result paths to one job."""

    def __init__(self, require_tokens: bool):
        self.require_tokens = require_tokens
        self._tokens = {}
        self._result_paths = {}
        self._lock = threading.Lock()

    def register_job(self, job_id, token=None) -> str:
        job_id = validate_job_id(job_id)
        token = str(token or "").strip()
        if self.require_tokens and not token:
            raise MissingJobAccessToken(
                "managed video job claim is missing processorAccessToken"
            )
        digest = hashlib.sha256(token.encode()).digest() if token else None
        with self._lock:
            self._tokens[job_id] = digest
        return job_id

    def authorize(self, job_id, presented_token) -> bool:
        try:
            job_id = validate_job_id(job_id)
        except ValueError:
            return False
        with self._lock:
            if job_id not in self._tokens:
                return False
            expected_digest = self._tokens[job_id]
        if expected_digest is None:
            return not self.require_tokens
        presented = str(presented_token or "")
        presented_digest = hashlib.sha256(presented.encode()).digest()
        return bool(presented) and hmac.compare_digest(
            expected_digest, presented_digest
        )

    def register_result_paths(self, job_id, paths) -> None:
        job_id = validate_job_id(job_id)
        fixed_paths = {
            str(name): os.path.abspath(path) for name, path in dict(paths).items()
        }
        with self._lock:
            if job_id not in self._tokens:
                raise ValueError("job must be registered before its results")
            self._result_paths[job_id] = fixed_paths

    def result_path(self, job_id, filename):
        """Resolve only a path registered by the recorder, never a routed path."""
        try:
            job_id = validate_job_id(job_id)
        except ValueError:
            return None
        with self._lock:
            return self._result_paths.get(job_id, {}).get(str(filename))
