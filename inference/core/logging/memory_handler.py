"""
In-memory logging handler for dashboard log viewing.

This module provides a custom logging handler that stores log records in memory
for retrieval via the /logs API endpoint. It's designed to be used when
ENABLE_IN_MEMORY_LOGS environment variable is set to 'true'.
"""

import logging
from collections import deque
from threading import Lock
from typing import List, Dict, Any
from datetime import datetime


# Global log storage
_log_entries = deque(maxlen=1000)  # Keep last 1000 log entries
_log_lock = Lock()


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


def setup_memory_logging():
    """Set up memory logging handler for the current logger hierarchy"""
    memory_handler = MemoryLogHandler()
    memory_handler.setLevel(logging.DEBUG)  # Capture all levels
    memory_formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    memory_handler.setFormatter(memory_formatter)

    # Add to root logger to capture all logs
    root_logger = logging.getLogger()
    root_logger.addHandler(memory_handler)

    return memory_handler
