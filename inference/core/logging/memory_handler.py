"""
In-memory logging handler for dashboard log viewing.

This module provides a custom logging handler that stores log records in memory
for retrieval via the /logs API endpoint. It's designed to be used when
ENABLE_IN_MEMORY_LOGS environment variable is set to 'true'.
"""

import logging
from collections import deque
from datetime import datetime
from threading import Lock
from typing import Any, Dict, List

from inference.core.env import ENABLE_IN_MEMORY_LOGS

# Global log storage
_log_entries = deque(maxlen=1000)  # Keep last 1000 log entries
_log_lock = Lock()
logger = logging.getLogger(__name__)
_uvicorn_config_patched = False


class MemoryLogHandler(logging.Handler):
    """Custom log handler that stores log records in memory for dashboard access"""

    def emit(self, record):
        try:
            # Format the log entry for JSON serialization
            log_entry = {
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": self.format(record),
                "module": record.module or "",
                "line": record.lineno,
            }

            with _log_lock:
                _log_entries.append(log_entry)
        except Exception:
            # Silently handle any errors in logging to prevent recursion
            pass


def get_recent_logs(
    limit: int = 100, level: str = None, since: str = None
) -> List[Dict[str, Any]]:
    """Get recent log entries from memory"""
    with _log_lock:
        logs = list(_log_entries)

    # Filter by log level if specified
    if level:
        level_upper = level.upper()
        logs = [log for log in logs if log["level"] == level_upper]

    # Filter by timestamp if specified
    if since:
        try:
            since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
            logs = [
                log
                for log in logs
                if datetime.fromisoformat(log["timestamp"]) > since_dt
            ]
        except ValueError:
            pass  # Invalid since timestamp, ignore filter

    # Limit results
    return logs[-limit:] if limit else logs


def is_memory_logging_enabled() -> bool:
    return ENABLE_IN_MEMORY_LOGS


def setup_memory_logging() -> None:
    """Set up memory logging handler for the current logger hierarchy"""
    if not is_memory_logging_enabled():
        return
    logger.info("Setting up memory logging")
    memory_handler = MemoryLogHandler()
    memory_handler.setLevel(logging.DEBUG)  # Capture all levels
    memory_formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    memory_handler.setFormatter(memory_formatter)

    # Add to root logger to capture all logs immediately
    root_logger = logging.getLogger()
    if memory_handler not in root_logger.handlers:
        root_logger.addHandler(memory_handler)

    # Specifically add to uvicorn.access logger to ensure access logs are captured now
    access_logger = logging.getLogger("uvicorn.access")
    if memory_handler not in access_logger.handlers:
        access_logger.addHandler(memory_handler)

    # Also patch uvicorn's default LOGGING_CONFIG so when uvicorn applies dictConfig,
    # our in-memory handler remains attached
    global _uvicorn_config_patched
    if not _uvicorn_config_patched:
        try:
            from uvicorn.config import LOGGING_CONFIG as UVICORN_LOGGING_CONFIG

            # Modify in-place (safe: uvicorn makes a deep copy later)
            log_config = UVICORN_LOGGING_CONFIG

            log_config.setdefault("formatters", {})
            if "default" not in log_config["formatters"]:
                log_config["formatters"]["default"] = {
                    "()": "uvicorn.logging.DefaultFormatter",
                    "fmt": "%(levelprefix)s %(message)s",
                    "use_colors": None,
                }

            log_config.setdefault("handlers", {})["inmemory"] = {
                "class": "inference.core.logging.memory_handler.MemoryLogHandler",
                "level": "DEBUG",
                "formatter": "default",
            }

            log_config.setdefault("loggers", {})
            log_config["loggers"].setdefault(
                "uvicorn.access",
                {
                    "handlers": ["default"],
                    "level": "INFO",
                    "propagate": False,
                },
            )
            if "inmemory" not in log_config["loggers"]["uvicorn.access"]["handlers"]:
                log_config["loggers"]["uvicorn.access"]["handlers"].append("inmemory")

            log_config["loggers"].setdefault(
                "uvicorn", {"handlers": ["default"], "level": "INFO"}
            )
            log_config["loggers"].setdefault("uvicorn.error", {"level": "INFO"})

            root_cfg = log_config.setdefault(
                "root", {"handlers": ["default"], "level": "INFO"}
            )
            if "inmemory" not in root_cfg.get("handlers", []):
                root_cfg.setdefault("handlers", []).append("inmemory")

            _uvicorn_config_patched = True
            logger.info("Patched uvicorn LOGGING_CONFIG to include MemoryLogHandler")
        except Exception:
            # Avoid hard failure if uvicorn is not available
            pass

    return memory_handler
