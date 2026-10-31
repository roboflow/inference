"""Workflow-scoped debug trace for custom Python blocks.

When the HTTP layer opts in to debug execution (``debug=True``), a
``WorkflowDebugTrace`` is activated for the run via ``register_debug_session()``.
Each custom Python block receives a module-level ``debug_traces`` proxy that appends
structured values into the active trace. The collected entries are returned
alongside ``python_blocks_output_streams`` in the workflow response.

Propagation model mirrors ``debug_logs``: the active trace lives in a
``ContextVar`` and the execution engine re-binds it (and the current step name)
inside every worker thread spawned by ``ThreadPoolExecutor``.
"""

import json
import threading
from contextvars import ContextVar
from datetime import datetime, timezone
from datetime import tzinfo as DatetimeTzInfo
from typing import Any, Dict, List, Optional, Union
from zoneinfo import ZoneInfo

current_debug_trace: ContextVar[Optional["WorkflowDebugTrace"]] = ContextVar(
    "current_debug_trace", default=None
)

current_debug_step_name: ContextVar[Optional[str]] = ContextVar(
    "current_debug_step_name", default=None
)

MAX_DEBUG_ENTRIES = 1_000
MAX_ENTRY_SERIALIZED_CHARS = 16_384
# Per-run budget aligned with ``DebugLogsCollector.MAX_TOTAL_CHARS``.
MAX_TOTAL_SERIALIZED_CHARS = 1_048_576
CAPACITY_EXCEEDED_MARKER = (
    "[debug trace limit reached - further entries of this run were dropped]"
)


def _serialize_debug_value(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value))
    except (TypeError, ValueError):
        return repr(value)


def _entry_serialized_size(entry: Dict[str, Any]) -> int:
    return len(json.dumps(entry, default=str))


def _capacity_marker_entry(
    step_name: Optional[str],
    max_entry_serialized_chars: int,
) -> Dict[str, Any]:
    entry = {"step": step_name, "value": CAPACITY_EXCEEDED_MARKER}
    if _entry_serialized_size(entry) <= max_entry_serialized_chars:
        return entry
    return {"step": None, "value": CAPACITY_EXCEEDED_MARKER}


def _resolve_timestamp_tz(
    timezone_name: Optional[Union[str, DatetimeTzInfo]],
) -> DatetimeTzInfo:
    if timezone_name is None:
        return timezone.utc
    if isinstance(timezone_name, DatetimeTzInfo):
        return timezone_name
    try:
        return ZoneInfo(timezone_name)
    except Exception as error:
        raise ValueError(f"Invalid timezone: {timezone_name!r}") from error


def _format_timestamp(
    timezone_name: Optional[Union[str, DatetimeTzInfo]] = None,
) -> tuple[str, str]:
    tz = _resolve_timestamp_tz(timezone_name)
    label = getattr(tz, "key", None) or str(tz)
    return datetime.now(tz).isoformat(), label


class WorkflowDebugTrace:
    """Thread-safe, append-only trace of intermediate state across blocks."""

    def __init__(
        self,
        max_entries: int = MAX_DEBUG_ENTRIES,
        max_entry_serialized_chars: int = MAX_ENTRY_SERIALIZED_CHARS,
        max_total_serialized_chars: int = MAX_TOTAL_SERIALIZED_CHARS,
    ) -> None:
        self._lock = threading.Lock()
        self._entries: List[Dict[str, Any]] = []
        self._max_entries = max_entries
        self._max_entry_serialized_chars = max_entry_serialized_chars
        self._max_total_serialized_chars = max_total_serialized_chars
        self._total_serialized_chars = 0
        self._capacity_exceeded = False

    def _append_capacity_marker(self, step_name: Optional[str]) -> None:
        entry = _capacity_marker_entry(step_name, self._max_entry_serialized_chars)
        self._total_serialized_chars += _entry_serialized_size(entry)
        self._entries.append(entry)

    def append(
        self,
        step_name: Optional[str],
        value: Any,
        *,
        add_timestamp: bool = False,
        timezone: Optional[Union[str, DatetimeTzInfo]] = None,
    ) -> None:
        serialized_value = _serialize_debug_value(value)
        entry: Dict[str, Any] = {"step": step_name, "value": serialized_value}
        if add_timestamp:
            timestamp, timezone_label = _format_timestamp(timezone)
            entry["timestamp"] = timestamp
            entry["timestamp_timezone"] = timezone_label
        if _entry_serialized_size(entry) > self._max_entry_serialized_chars:
            marker = "... [entry truncated]"
            text = (
                serialized_value
                if isinstance(serialized_value, str)
                else repr(serialized_value)
            )
            # Binary-search the longest value prefix that keeps the whole entry
            # (value + marker + metadata) within the cap. A char-count slice is
            # not a reliable size bound because json.dumps escaping can expand
            # characters several-fold; trimming one char at a time would
            # re-serialize the entry O(n) times for a large value. Serialized
            # size is monotonic in prefix length, so binary search needs only
            # O(log n) serializations.
            lo, hi, best = 0, len(text), 0
            while lo <= hi:
                mid = (lo + hi) // 2
                entry["value"] = text[:mid] + marker
                if _entry_serialized_size(entry) <= self._max_entry_serialized_chars:
                    best = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            entry["value"] = text[:best] + marker
        entry_size = _entry_serialized_size(entry)
        with self._lock:
            if self._capacity_exceeded:
                return
            if entry_size > self._max_entry_serialized_chars:
                # The value has been truncated as far as possible but the entry
                # still overflows the per-entry cap, so the metadata alone (e.g.
                # a very long, client-controlled step name) does not fit. Record
                # the capacity marker and stop instead of looping forever.
                self._capacity_exceeded = True
                self._append_capacity_marker(step_name)
                return
            if len(self._entries) >= self._max_entries:
                self._capacity_exceeded = True
                self._append_capacity_marker(step_name)
                return
            if (
                self._total_serialized_chars + entry_size
                > self._max_total_serialized_chars
            ):
                self._capacity_exceeded = True
                self._append_capacity_marker(step_name)
                return
            self._total_serialized_chars += entry_size
            self._entries.append(entry)

    def snapshot(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._entries)


class _DebugProxy:
    """No-op when debug mode is inactive; forwards to the active trace otherwise."""

    def append(
        self,
        value: Any,
        *,
        add_timestamp: bool = False,
        timezone: Optional[Union[str, DatetimeTzInfo]] = None,
    ) -> None:
        trace = get_active_debug_trace()
        if trace is None:
            return
        trace.append(
            step_name=current_debug_step_name.get(),
            value=value,
            add_timestamp=add_timestamp,
            timezone=timezone,
        )


debug_traces = _DebugProxy()


def get_active_debug_trace() -> Optional[WorkflowDebugTrace]:
    return current_debug_trace.get()
