"""Execution context of the block call or constructor currently running.

The engine enters a context around each block constructor (``run_id=None``)
and each invocation (the actual ``run_id`` and the call's logical indices).
Code running inside, including submitted dynamic code, reads it::

    with use_execution_context(ExecutionContext(("child", "count"), "Counting", sid)):
        context = get_execution_context()   # the context above
    get_execution_context()                 # raises: nothing is active

The value lives in a ``ContextVar`` and is reset in ``finally``, so it never
leaks past an error or out of a nested call, and no process-global
"current step" record exists. Reading it outside an active call is an
explicit error rather than stale metadata.

``current_pulse_run_id()`` attributes observer callbacks: with pulses of a
pipelined run on several threads, it names the run of the pulse executing on
the calling thread (``None`` elsewhere, e.g. on a source reader).
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

from roboflow_workflows.execution_engine.v2.errors import StepPath, format_step_path

__all__ = [
    "ExecutionContext",
    "ExecutionContextReader",
    "NoExecutionContextError",
    "current_pulse_run_id",
    "get_execution_context",
    "use_execution_context",
    "use_pulse_run_id",
]


@dataclass(frozen=True)
class ExecutionContext:
    """Where the currently running block code executes.

    Args:
        step_path: Scope path of the step, e.g. ``("child", "count")``.
        block_type: Canonical block type of the step.
        session_id: Execution session constructing or running the step.
        run_id: Run of the session; ``None`` while the constructor runs.
        indices: Logical indices of this call: one index for a per-invocation
            call, every index of the domain for a batch-delivering call,
            ``()`` for the constructor.
    """

    step_path: StepPath
    block_type: str
    session_id: str
    run_id: Optional[str] = None
    indices: Tuple[Tuple[int, ...], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_path", tuple(self.step_path))
        object.__setattr__(
            self, "indices", tuple(tuple(index) for index in self.indices)
        )

    @property
    def step_selector(self) -> str:
        """Step selector, e.g. ``$steps.child/count``."""
        return format_step_path(self.step_path)


class NoExecutionContextError(RuntimeError):
    """Execution context was read outside a block constructor or call."""


_CURRENT: ContextVar[Optional[ExecutionContext]] = ContextVar(
    "workflows_v2_execution_context", default=None
)


def get_execution_context() -> ExecutionContext:
    """Return the context of the block constructor or call running now.

    Returns:
        The innermost active context.

    Raises:
        NoExecutionContextError: When no constructor or call is active.
    """
    context = _CURRENT.get()
    if context is None:
        raise NoExecutionContextError(
            "No V2 execution context is active: it exists only while the engine "
            "constructs a block or runs one of its calls"
        )

    return context


@contextmanager
def use_execution_context(context: ExecutionContext) -> Iterator[ExecutionContext]:
    """Make ``context`` current for the duration of the ``with`` block.

    Args:
        context: Context of the constructor or call about to run.

    Yields:
        ``context``. The previous value is restored on exit, also on errors.

    Raises:
        TypeError: When ``context`` is not an ``ExecutionContext``.
    """
    if not isinstance(context, ExecutionContext):
        raise TypeError(
            f"use_execution_context expects an ExecutionContext, got "
            f"{type(context).__name__}"
        )

    token = _CURRENT.set(context)
    try:
        yield context
    finally:
        _CURRENT.reset(token)


class ExecutionContextReader:
    """Gives block code ``self.execution_context``.

    Shared by ``Block`` and ``Implementation``, so an alternative
    implementation reads the same context as an ordinary block: the logical
    step, block type, session, run and indices of the current call.
    """

    @property
    def execution_context(self) -> ExecutionContext:
        """Context of the constructor or call running now (read-only).

        Available inside ``__init__`` (``run_id`` is ``None``), ``run`` and
        the phases a call runs.

        Raises:
            NoExecutionContextError: When read outside a constructor or call.
        """
        context = get_execution_context()

        return context


_PULSE_RUN_ID: ContextVar[Optional[str]] = ContextVar(
    "workflows_v2_pulse_run_id", default=None
)


def current_pulse_run_id() -> Optional[str]:
    """Return the run id of the pulse or passive run executing on this thread.

    Observer callbacks keep their signatures; a callback that needs to know
    which pulse a step notification belongs to reads this. Serial and
    pipelined runs both set it.

    Returns:
        ``PulseKey.run_id`` of the executing pulse, the run id of the
        executing passive run (serial or pipelined), or ``None`` elsewhere.
    """
    run_id = _PULSE_RUN_ID.get()

    return run_id


@contextmanager
def use_pulse_run_id(run_id: str) -> Iterator[str]:
    """Attribute the ``with`` block to one run (engine-internal).

    Args:
        run_id: Run id of the pulse or submission about to execute.

    Yields:
        ``run_id``. The previous value is restored on exit, also on errors.
    """
    token = _PULSE_RUN_ID.set(run_id)
    try:
        yield run_id
    finally:
        _PULSE_RUN_ID.reset(token)
