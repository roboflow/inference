"""JSONL session logging for full audit trail and crash recovery."""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class SessionLog:
    """Append-only JSONL session transcript.

    Every tool call, result, and response is logged. This provides:
    - Full audit trail
    - Crash recovery (replay from last checkpoint)
    - Debugging and analytics
    """

    def __init__(self, sessions_dir: Path, session_id: Optional[str] = None):
        self._session_id = session_id or f"session_{uuid.uuid4().hex[:12]}"
        self._sessions_dir = sessions_dir
        self._sessions_dir.mkdir(parents=True, exist_ok=True)
        self._filepath = self._sessions_dir / f"{self._session_id}.jsonl"
        self._message_counter = 0

    @property
    def session_id(self) -> str:
        return self._session_id

    def log_user_message(self, content: str, attachments: Optional[list] = None) -> None:
        self._write_entry({
            "type": "user",
            "content": content,
            "has_attachments": bool(attachments),
            "attachment_count": len(attachments) if attachments else 0,
        })

    def log_assistant_response(self, content_blocks: list[dict]) -> None:
        # Filter out large data (images, base64) from the log
        sanitized = _sanitize_content_blocks(content_blocks)
        self._write_entry({
            "type": "assistant",
            "content": sanitized,
        })

    def log_tool_call(self, tool_name: str, tool_call_id: str, arguments: dict) -> None:
        self._write_entry({
            "type": "tool_call",
            "tool_name": tool_name,
            "tool_call_id": tool_call_id,
            "arguments": _truncate_values(arguments),
        })

    def log_tool_result(
        self,
        tool_name: str,
        tool_call_id: str,
        result: Any,
        is_error: bool = False,
    ) -> None:
        result_str = str(result)
        if len(result_str) > 2000:
            result_str = result_str[:2000] + "... (truncated)"
        self._write_entry({
            "type": "tool_result",
            "tool_name": tool_name,
            "tool_call_id": tool_call_id,
            "result": result_str,
            "is_error": is_error,
        })

    def log_event(self, event_type: str, data: Any) -> None:
        self._write_entry({
            "type": "event",
            "event_type": event_type,
            "data": str(data)[:1000],
        })

    def _write_entry(self, data: dict) -> None:
        self._message_counter += 1
        entry = {
            "id": f"msg_{self._message_counter:04d}",
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **data,
        }
        try:
            with open(self._filepath, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception as e:
            logger.error("Failed to write session log: %s", e)


def _sanitize_content_blocks(blocks: list[dict]) -> list[dict]:
    """Remove large binary data from content blocks before logging."""
    sanitized = []
    for block in blocks:
        if block.get("type") == "image":
            sanitized.append({"type": "image", "note": "(image data omitted from log)"})
        elif block.get("type") == "thinking":
            # Keep thinking but truncate if very long
            text = block.get("thinking", "")
            if len(text) > 500:
                text = text[:500] + "... (truncated)"
            sanitized.append({"type": "thinking", "thinking": text})
        else:
            sanitized.append(block)
    return sanitized


def _truncate_values(d: dict) -> dict:
    """Truncate long string values in a dict for logging."""
    result = {}
    for key, value in d.items():
        if isinstance(value, str) and len(value) > 500:
            result[key] = value[:500] + "... (truncated)"
        elif isinstance(value, dict):
            result[key] = _truncate_values(value)
        else:
            result[key] = value
    return result
