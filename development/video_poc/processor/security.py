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
_BEARER_RE = re.compile(r"^Bearer\s+(.+)$", re.IGNORECASE)
_BEARER_VALUE_RE = re.compile(r"(?i)\bBearer\s+[^\s,;]+")
_SECRET_PARAM_RE = re.compile(
    r"(?i)(\b(?:api[_-]?key|token|access_token|signature|sig)=)[^\s&#]+"
)
_URL_RE = re.compile(r"(?P<url>(?:https?|rtsp)://[^\s\"'<>]+)", re.IGNORECASE)


class MissingJobAccessToken(ValueError):
    """A managed-pool claim did not include its browser-facing job token."""


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
    match = _BEARER_RE.match((authorization or "").strip())
    if match:
        return match.group(1).strip()
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
