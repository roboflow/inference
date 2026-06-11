"""Per-run capture of stdout/stderr emitted by dynamic Python blocks.

The HTTP layer can opt-in to "debug" execution. When that happens, it activates a
``DebugLogsCollector`` for the current workflow run via
``register_debug_collector()``; the local Python block runner looks up that
collector after each successful invocation and appends the captured
stdout/stderr. The collected logs are then returned alongside the normal
workflow outputs.

Propagation model:
- The active collector is stored in a ``ContextVar`` (``current_debug_collector``)
  so it follows the same propagation model as the other per-run signals already
  used by the execution engine (``execution_id``, ``remote_processing_times``,
  ``apply_duration_minimum``). The engine captures its value in the request
  thread and re-binds it inside each worker thread spawned by
  ``ThreadPoolExecutor`` (see ``safe_execute_step``).
- Only LOCAL execution is wired up today. Modal and OCI sandbox executors run
  the user code out of process and would need their own payload extension to
  bubble stdout/stderr back into the active collector.
"""

import threading
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Dict, Generator, List, Optional

current_debug_collector: ContextVar[Optional["DebugLogsCollector"]] = ContextVar(
    "current_debug_collector", default=None
)


class DebugLogsCollector:
    """Thread-safe collector for stdout/stderr produced by Python blocks."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: Dict[str, List[Dict[str, Optional[str]]]] = {}

    def record(
        self,
        step_name: str,
        stdout: Optional[str],
        stderr: Optional[str],
    ) -> None:
        if stdout is None and stderr is None:
            return
        with self._lock:
            self._entries.setdefault(step_name, []).append(
                {"stdout": stdout, "stderr": stderr}
            )

    def snapshot(self) -> Dict[str, List[Dict[str, Optional[str]]]]:
        with self._lock:
            return {step: list(entries) for step, entries in self._entries.items()}


@contextmanager
def register_debug_collector() -> Generator[DebugLogsCollector, None, None]:
    """Activate a fresh collector for the duration of the ``with`` block.

    The collector is published via the ``current_debug_collector`` ContextVar
    and removed on exit. Worker threads that need to see it must re-bind the
    ContextVar inside their own thread (the execution engine does this in
    ``safe_execute_step``).
    """
    collector = DebugLogsCollector()
    token = current_debug_collector.set(collector)
    try:
        yield collector
    finally:
        current_debug_collector.reset(token)


def get_active_collector() -> Optional[DebugLogsCollector]:
    return current_debug_collector.get()
