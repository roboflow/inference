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

During a call the context also answers, lazily per index, which source and
time the call's data belong to, and lets the block emit declared events::

    context.source_id            # the one source of a per-invocation call
    context.source_id_at(index)  # the source of one index (any call)
    context.sample_at(index)     # SampleContext of that index, or None
    context.temporal_at(index)   # TemporalContext of that index, or None
    context.cause                # the event that started this handler run, or None
    self.emit("crossed", count=n, frame=image)        # per-invocation call
    self.emit("crossed", at=frames.indices[k], ...)   # batch-delivering call
    context.set_machine_state("gate", "decide", "open")   # handler runs only

A call's context at ``index`` is the one its varying arguments share there
(``common_or_none``, the rule step outputs use); an explicit ``None`` stays
``None``. Only when no argument carries a context, the run's origin applies:
the source pulse's context, or the triggering event's one in a handler run.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import (
    Any,
    Collection,
    FrozenSet,
    Iterator,
    Mapping,
    Optional,
    Protocol,
    Set,
    Tuple,
)

from roboflow_workflows.execution_engine.v2.data import (
    Index,
    SampleContext,
    TemporalContext,
)
from roboflow_workflows.execution_engine.v2.errors import (
    ContractError,
    EventEmissionError,
    StepPath,
    format_step_path,
)

__all__ = [
    "CallScope",
    "ExecutionContext",
    "ExecutionContextReader",
    "NoExecutionContextError",
    "answer_wants",
    "current_pulse_run_id",
    "get_execution_context",
    "use_execution_context",
    "use_pulse_run_id",
]


class CallScope(Protocol):
    """Engine side of one step's calls: metadata lookup and event emission.

    The engine creates one per executed step; nothing is resolved until a
    block asks. Block code never uses it directly.
    """

    @property
    def cause(self) -> Optional[Any]:
        """``EventCause`` of the handler run executing the step, else ``None``."""

    def sample_at(self, index: Index) -> Optional[SampleContext]:
        """Source context of one admitted index of the step."""

    def temporal_at(self, index: Index) -> Optional[TemporalContext]:
        """Temporal context of one admitted index of the step."""

    def has_subscribers(self, event: str) -> bool:
        """Whether emitting ``event`` would reach a handler."""

    def wants(self, context: "ExecutionContext", output: str) -> bool:
        """Record that the call asked about ``output`` and answer whether it is demanded."""

    def emit(
        self,
        context: "ExecutionContext",
        event: str,
        fields: Mapping[str, Any],
        *,
        at: Optional[Index],
    ) -> None:
        """Validate and dispatch one event emitted by the call ``context``."""

    def set_machine_state(self, machine: str, transition: str, next_state: str) -> Any:
        """Apply a handler-selected transition for the running handler."""


@dataclass(frozen=True)
class ExecutionContext:
    """Where the currently running block code executes.

    Do not retain the context beyond the block invocation: it retains the call
    scope and input buffers. Save required IDs or timestamps instead.

    Args:
        step_path: Scope path of the step, e.g. ``("child", "count")``.
        block_type: Canonical block type of the step.
        session_id: Execution session constructing or running the step.
        run_id: Run of the session; ``None`` while the constructor runs.
        indices: Logical indices of this call: one index for a per-invocation
            call, every index of the domain for a batch-delivering call,
            ``()`` for the constructor.
        batched: Whether the call is batch-delivering, even with one or no
            member; such a call names its member explicitly (``at``,
            ``source_id_at``).
        call_scope: Engine-internal metadata and emission access of the
            step; ``None`` for a constructor or a hand-built context.
        wanted_outputs: Outputs of the step some reader demands in this run;
            ``None`` (a constructor or a hand-built context) wants every
            output. Fixed for the whole call, every phase included. Block
            code asks ``self.wants(name)`` instead of reading this: only an
            asked output may be left out of the result.
        queried_outputs: Outputs this call asked ``wants`` about; the engine
            lets the call leave out exactly the queried ones it answered
            ``False`` for. Engine-owned, per call, never shared.
        quality: Quality label honoured when the step's implementation was
            selected; ``None`` when no level asked for one.
    """

    step_path: StepPath
    block_type: str
    session_id: str
    run_id: Optional[str] = None
    indices: Tuple[Tuple[int, ...], ...] = ()
    batched: bool = False
    call_scope: Optional[CallScope] = field(default=None, compare=False, repr=False)
    wanted_outputs: Optional[FrozenSet[str]] = None
    queried_outputs: Set[str] = field(default_factory=set, compare=False, repr=False)
    quality: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_path", tuple(self.step_path))
        object.__setattr__(
            self, "indices", tuple(tuple(index) for index in self.indices)
        )
        if self.wanted_outputs is not None:
            object.__setattr__(self, "wanted_outputs", frozenset(self.wanted_outputs))

    @property
    def step_selector(self) -> str:
        """Step selector, e.g. ``$steps.child/count``."""
        return format_step_path(self.step_path)

    @property
    def cause(self) -> Optional[Any]:
        """The ``EventCause`` that started the handler run of this call.

        ``None`` in the main flow. Inside a handler run it keeps the
        original event, emitter, source and time even when the handler
        binds only scalar fields.
        """
        if self.call_scope is None:
            return None

        cause = self.call_scope.cause

        return cause

    @property
    def source_id(self) -> str:
        """Source of a per-invocation call.

        Raises:
            ContractError: For a batch-delivering call (always, even when its
                members share a source; use ``source_id_at``), a call without
                exactly one index (a constructor), or an index without a source.
        """
        if self.batched:
            raise ContractError(
                f"{self.step_selector} is a batch-delivering call, so it has no "
                "single source; name the member with source_id_at(index) "
                "(state.at(index))"
            )
        if len(self.indices) != 1:
            raise ContractError(
                f"{self.step_selector} has no call index here (a constructor has "
                "none), so it has no source"
            )

        source_id = self.source_id_at(self.indices[0])

        return source_id

    def source_id_at(self, index: Index) -> str:
        """Source of one of this call's indices.

        Args:
            index: One of ``indices`` (a full logical index).

        Returns:
            ``SampleContext.source_id`` at that index.

        Raises:
            ContractError: When ``index`` is not one of this call's indices or
                it has no single source (e.g. a mixed-source reduction).
        """
        sample = self.sample_at(index)
        if sample is None:
            raise ContractError(
                f"{self.step_selector}: index {list(index)} has no single source "
                "(its inputs carry none, or several)"
            )

        return sample.source_id

    def sample_at(self, index: Index) -> Optional[SampleContext]:
        """Source context of one of this call's indices.

        Args:
            index: One of ``indices`` (a full logical index).

        Returns:
            The context, or ``None`` when the index has no single source.

        Raises:
            ContractError: When ``index`` is not one of this call's indices.
        """
        scope = self._scope_at(index, what="sample_at")
        sample = scope.sample_at(tuple(index)) if scope is not None else None

        return sample

    def temporal_at(self, index: Index) -> Optional[TemporalContext]:
        """Temporal context of one of this call's indices.

        Args:
            index: One of ``indices`` (a full logical index).

        Returns:
            The context, or ``None`` when the index has no single time.

        Raises:
            ContractError: When ``index`` is not one of this call's indices.
        """
        scope = self._scope_at(index, what="temporal_at")
        temporal = scope.temporal_at(tuple(index)) if scope is not None else None

        return temporal

    def set_machine_state(self, machine: str, transition: str, next_state: str) -> Any:
        """Apply a handler-selected state machine transition.

        Only steps of a handler run may call it, and only for transitions
        whose ``handler`` is that handler. A run caused by an event of the
        same machine applies only while the machine still has the state and
        version that event announced; otherwise the result is ``stale``.

        Args:
            machine: Machine name relative to the handler's declaring scope,
                e.g. ``"gate"`` or ``"child/gate"``.
            transition: Transition name.
            next_state: One of the transition's ``to`` states.

        Returns:
            ``reactions.machines.TransitionResult``: ``applied``, ``ignored``
            or ``stale``, the resulting or found state and the stamp.

        Raises:
            ContractError: Outside a handler run, for an unknown or foreign
                transition or a state outside its ``to`` states.
            StateError: When the machine record cannot be read or updated.
        """
        if self.run_id is None or self.call_scope is None:
            raise ContractError(
                f"{self.step_selector}: set_machine_state works only while the "
                "engine runs a step of an event handler"
            )

        result = self.call_scope.set_machine_state(machine, transition, next_state)

        return result

    def _scope_at(self, index: Index, *, what: str) -> Optional[CallScope]:
        if tuple(index) not in self.indices:
            raise ContractError(
                f"{self.step_selector}: {what}({list(index)}) names no index of "
                f"this call; its indices are {[list(item) for item in self.indices]}"
            )

        return self.call_scope


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

    def emit(self, event: str, *, at: Optional[Index] = None, **fields: Any) -> None:
        """Emit one declared event of this block from the running call.

        Synchronous handlers run before ``emit`` returns, on this thread,
        while the step keeps its turn; asynchronous handlers receive an owned
        snapshot of the fields they bind, and ``emit`` may wait for queue
        space under the ``synchronous`` overflow policy. Fields no handler
        binds are not retained. Effects of handlers are not undone if the
        call later fails.

        Args:
            event: Name declared in the block's ``events``.
            at: Index the event belongs to; required by a batch-delivering
                call, even of one member.
            **fields: Every declared field of the event.

        Raises:
            EventEmissionError: Outside a block call, for an undeclared event,
                a missing or foreign ``at``, a field without a supported
                snapshot or after the run's reactions closed.
            EventPayloadError: On missing, unknown or ill-kinded fields.
            ReactionError: When a synchronous handler failed.
        """
        context = _call_context(f"emit({event!r})")
        context.call_scope.emit(context, event, fields, at=at)

    def has_subscribers(self, event: str) -> bool:
        """Whether emitting ``event`` now would reach any handler.

        Lets a block skip building costly fields nobody receives.

        Args:
            event: Name declared in the block's ``events``.

        Returns:
            ``True`` when at least one handler subscribes to the event.

        Raises:
            EventEmissionError: Outside a block call.
        """
        context = _call_context(f"has_subscribers({event!r})")
        subscribed = context.call_scope.has_subscribers(event)

        return subscribed

    def wants(self, output: str) -> bool:
        """Whether some reader of this run demands ``output`` of the running call.

        Lets a block skip building an output nobody reads. The answer is fixed
        for the whole call, every phase included, and belongs to this call
        only. A call may leave the key of an output out of its result only
        after asking about it here and receiving ``False``; every other
        declared output must still be returned, so a forgotten key is still
        an error.

        Args:
            output: Name declared in the block's ``outputs``.

        Returns:
            ``True`` when a retained step, a requested or recorded workflow
            output or an operator reads the output.

        Raises:
            EventEmissionError: Outside a block call.
            ContractError: For an undeclared output name.
        """
        context = _call_context(f"wants({output!r})", needs="demand answers")
        wanted = context.call_scope.wants(context, output)

        return wanted


def answer_wants(
    context: ExecutionContext, output: str, *, declared: Collection[str]
) -> bool:
    """Record that a call asked about ``output`` and answer from its demand.

    The engine and ``reactions.testing.block_call`` both answer ``wants``
    here, so a unit test sees exactly the engine's behaviour.

    Args:
        context: The running call's context.
        output: The asked output name.
        declared: Output names the step declares.

    Returns:
        Whether some reader demands ``output``; ``True`` when the context
        carries no demand.

    Raises:
        ContractError: When the step declares no such output.
    """
    if output not in declared:
        raise ContractError(
            f"{context.step_selector} ({context.block_type}) asked "
            f"wants({output!r}), but declares no such output; its outputs are "
            f"{sorted(declared)}"
        )
    context.queried_outputs.add(output)
    wanted = context.wanted_outputs is None or output in context.wanted_outputs

    return wanted


def _call_context(action: str, *, needs: str = "events") -> ExecutionContext:
    """The running call's context; constructors and plain calls have none."""
    context = _CURRENT.get()
    if context is None or context.run_id is None or context.call_scope is None:
        where = (
            "outside an engine call"
            if context is None
            else (f"in {context.step_selector} outside a block call")
        )
        raise EventEmissionError(
            f"{action} was called {where}; {needs} exist only while the engine "
            "runs the block. In a unit test, wrap the call in "
            "reactions.testing.block_call()"
        )

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
