"""Per-run debug capture for dynamic Python blocks (stdout/stderr and structured traces).

The HTTP layer opts in via ``debug=True`` on the workflow run request. That activates
a :class:`DebugSession` for the run through :func:`register_debug_session`, which
publishes both:

- a :class:`DebugLogsCollector` for stdout/stderr (returned as
  ``python_blocks_output_streams``), and
- a :class:`WorkflowDebugTrace` for ``debug_traces.append(...)`` calls (returned as
  ``python_blocks_debug_traces``).

Block runners look up the active session components through ContextVars after each
invocation.

Propagation model:
- Active session state is stored in ContextVars (``current_debug_collector``,
  ``current_debug_trace``) using the same pattern as ``execution_id``,
  ``remote_processing_times``, and ``apply_duration_minimum``. The engine captures
  values in the request thread and re-binds them inside each worker thread spawned
  by ``ThreadPoolExecutor`` (see ``safe_execute_step``).
- Only LOCAL execution is wired up today. Modal and OCI sandbox executors run the
  user code out of process and would need their own payload extension to bubble
  captured output back into the active session.
"""

import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Dict, Generator, List, Optional

from inference.core.workflows.execution_engine.v1.dynamic_blocks.workflow_debug import (
    WorkflowDebugTrace,
    current_debug_trace,
)

current_debug_collector: ContextVar[Optional["DebugLogsCollector"]] = ContextVar(
    "current_debug_collector", default=None
)

# Caps protecting response size and server memory: blocks printing in a loop
# over a large batch can otherwise produce megabytes per request.
MAX_CHARS_PER_STREAM = 16_384  # per stdout/stderr string in a single entry
MAX_TOTAL_CHARS = 1_048_576  # per run, across all entries
STREAM_TRUNCATION_MARKER = "\n... [output truncated]"
CAPACITY_EXCEEDED_MARKER = (
    "[log capture limit reached - further output of this run was dropped]"
)


def _truncate_stream(value: Optional[str], limit: int) -> Optional[str]:
    if value is None or len(value) <= limit:
        return value
    return value[:limit] + STREAM_TRUNCATION_MARKER


class DebugLogsCollector:
    """Thread-safe collector for stdout/stderr produced by Python blocks.

    Each stream of an entry is truncated to ``max_chars_per_stream``. Once the
    total collected size of the run exceeds ``max_total_chars``, a single marker
    entry is appended and all subsequent records are dropped.
    """

    def __init__(
        self,
        max_chars_per_stream: int = MAX_CHARS_PER_STREAM,
        max_total_chars: int = MAX_TOTAL_CHARS,
    ) -> None:
        self._lock = threading.Lock()
        self._entries: Dict[str, List[Dict[str, Optional[str]]]] = {}
        self._max_chars_per_stream = max_chars_per_stream
        self._max_total_chars = max_total_chars
        self._total_chars = 0
        self._capacity_exceeded = False

    def record(
        self,
        step_name: str,
        stdout: Optional[str],
        stderr: Optional[str],
    ) -> None:
        if stdout is None and stderr is None:
            return
        stdout = _truncate_stream(stdout, self._max_chars_per_stream)
        stderr = _truncate_stream(stderr, self._max_chars_per_stream)
        entry_chars = len(stdout or "") + len(stderr or "")
        with self._lock:
            if self._capacity_exceeded:
                return
            if self._total_chars + entry_chars > self._max_total_chars:
                self._capacity_exceeded = True
                self._entries.setdefault(step_name, []).append(
                    {"stdout": CAPACITY_EXCEEDED_MARKER, "stderr": None}
                )
                return
            self._total_chars += entry_chars
            self._entries.setdefault(step_name, []).append(
                {"stdout": stdout, "stderr": stderr}
            )

    def snapshot(self) -> Dict[str, List[Dict[str, Optional[str]]]]:
        with self._lock:
            return {step: list(entries) for step, entries in self._entries.items()}


@dataclass
class DebugSession:
    """Per-run debug state activated when ``debug=True`` on a workflow request."""

    output_streams: DebugLogsCollector
    debug_traces: WorkflowDebugTrace


@contextmanager
def register_debug_session() -> Generator[DebugSession, None, None]:
    """Activate stdout/stderr capture and structured debug traces for a run.

    Both collectors are published via ContextVars for the duration of the
    ``with`` block and removed on exit. Worker threads re-bind the ContextVars
    inside their own thread (the execution engine does this in
    ``safe_execute_step``).
    """
    session = DebugSession(
        output_streams=DebugLogsCollector(),
        debug_traces=WorkflowDebugTrace(),
    )
    token_collector = current_debug_collector.set(session.output_streams)
    token_trace = current_debug_trace.set(session.debug_traces)
    try:
        yield session
    finally:
        current_debug_collector.reset(token_collector)
        current_debug_trace.reset(token_trace)


def get_active_collector() -> Optional[DebugLogsCollector]:
    return current_debug_collector.get()
