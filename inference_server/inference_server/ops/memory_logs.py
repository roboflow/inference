import logging
import re
from collections import deque
from datetime import datetime
from threading import Lock
from typing import Any, Deque, Dict, List, Optional
from urllib.parse import unquote

from inference_server import configuration
from inference_server.gateway import _redact_secrets
from inference_server.logging_config import (
    APPLICATION_LOGGER_NAME,
    PLAIN_LOG_FORMAT,
    UVICORN_ACCESS_LOGGER_NAME,
    structured_access_log_enabled,
)

MAX_LOG_ENTRIES = 1000
DEFAULT_LOG_LIMIT = 100
NON_PROPAGATING_LOGGER_CANDIDATES = (
    APPLICATION_LOGGER_NAME,
    "inference-models",
    "inference-models-verbose",
    "inference_model_manager",
    "roboflow_workflows",
    "streamvision",
)

PUBLIC_KEYS = ("timestamp", "level", "logger", "message", "module", "line")
MAX_DECODE_PASSES = 3
_QUOTED_AUTHORIZATION = re.compile(
    r"""((?<![A-Za-z0-9])authorization["']?[ \t]*:[ \t]*)(["'])[^"']*\2""",
    re.IGNORECASE,
)

_log_entries: Deque[Dict[str, Any]] = deque(maxlen=MAX_LOG_ENTRIES)
_log_lock = Lock()


def _sanitize(message: str) -> str:
    for _ in range(MAX_DECODE_PASSES):
        decoded = unquote(message)
        if decoded == message:
            break
        message = decoded

    return _redact_secrets(_QUOTED_AUTHORIZATION.sub(r"\1\2***\2", message))


class MemoryLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            log_entry = {
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": _sanitize(self.format(record)),
                "module": record.module or "",
                "line": record.lineno,
                "created": record.created,
            }
            with _log_lock:
                _log_entries.append(log_entry)
        except Exception:
            pass


_memory_handler = MemoryLogHandler(level=logging.DEBUG)
_memory_handler.setFormatter(logging.Formatter(PLAIN_LOG_FORMAT))


def _parse_since(since: str) -> Optional[float]:
    try:
        since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
    except ValueError:
        return None

    try:
        return since_dt.timestamp()
    except (OverflowError, OSError, ValueError):
        return None


def get_recent_logs(
    limit: int = DEFAULT_LOG_LIMIT,
    *,
    level: Optional[str] = None,
    since: Optional[str] = None,
) -> List[Dict[str, Any]]:
    with _log_lock:
        logs = list(_log_entries)

    if level:
        level_upper = level.upper()
        logs = [log for log in logs if log["level"] == level_upper]

    since_timestamp = _parse_since(since) if since else None
    if since_timestamp is not None:
        logs = [log for log in logs if log["created"] > since_timestamp]

    selected = logs[-limit:] if limit else logs

    return [{key: log[key] for key in PUBLIC_KEYS} for log in selected]


def _attach(target: logging.Logger) -> None:
    if _memory_handler not in target.handlers:
        target.addHandler(_memory_handler)


def setup_memory_logging() -> None:
    if not configuration.ENABLE_IN_MEMORY_LOGS:
        return

    _attach(logging.getLogger())
    for name in NON_PROPAGATING_LOGGER_CANDIDATES:
        candidate = logging.getLogger(name)
        if not candidate.propagate:
            _attach(candidate)

    if not structured_access_log_enabled():
        _attach(logging.getLogger(UVICORN_ACCESS_LOGGER_NAME))
