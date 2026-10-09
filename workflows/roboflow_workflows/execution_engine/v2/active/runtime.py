"""Active runtime: independent source readers; a serial or a pipelined processor.

``ExecutionSession.start`` delegates here::

    run = start_session(session, inputs=static, handlers={"frames": on_frames})
    ...
    run.stop()      # optional: close admission, drain what was admitted
    run.wait()      # True when complete; raises the run's failure

Threads of one active run::

    reader[S]   (one per source)     serial (default)       pipelined (PipelineOptions)
    ----------------------------     --------------------   ----------------------------
    open(**params)                   processor:             dispatcher + max_in_flight
    loop: emission = read()            item = queue.get()     workers (pipeline.py);
          None -> end of S             run the pulse          each worker runs whole
          admit(S, emission)           (pulses.py)            pulses (pulses.py), in
    close()  exactly once                                     turn at every stage
    reader of S finished

A pulse runs its domain's route (``plan.route``) in plan order on one thread,
delivers each registered group as soon as its steps ran, then feeds the
operators consuming its domain; their returned pulses run depth first on the
same thread (``pulses.py``). Pipelined, different pulses overlap at
different stages; each stage (a step or one of its phases, a group delivery,
an operator) admits one call at a time in sequence order per domain.

Admission is the acceptance boundary: a reader takes one of its source's
``admission_bound`` slots and the pulse is admitted. The slot is released
after the pulse's handlers returned and the operators it fed returned and
their immediate pulses ran; what an operator retains afterwards is bounded
by the operator's own parameters, not by admission. Nothing waits for future
data: an operator returns what it can decide now. Per source at most
``admission_bound`` pulses are admitted and one more emission is read but not
admitted, so a slow synchronous handler backpressures the readers instead of
growing a queue. A pipelined source with overload ``latest`` never waits: its
reader keeps replacing one pending emission (the replaced ones are
``dropped``), and only the newest is admitted when a slot is free. Readers
never wait for each other.

A registered group is delivered as soon as every step its fields need has
run in the pulse (``PlannedOutputGroup.dependencies``), so a group reading the
source directly is delivered before an unrelated later action of the same
route; that action still runs once. An explicitly filtered emission delivers
every registered group of the source at pulse start, all fields filtered.

Lifecycle:

* End of one source ends only its reader. Once that source can produce no
  further pulse and none of its admitted pulses is outstanding, the operators
  it feeds learn that its inputs ended; an operator whose upstream domains all
  ended finishes once (``eof``) and its own domain ends once its final pulses,
  and any of its pulses still running, completed. The run completes when
  every reader has closed its source, every admitted pulse was processed and
  every operator was closed.
* ``stop()`` closes admission and sets the shared ``stop_event`` the sources
  see. A reader whose ``read()`` returns afterwards discards that emission
  and closes; a pending ``latest`` emission is discarded too. Admitted pulses
  are still processed and delivered exactly once; operators then finish with
  reason ``stop`` and apply their partial policies.
* A failure (open, read, emission, step, handler, observer, operator or
  close) closes admission, cancels admitted pulses not yet processed and the
  remaining steps, deliveries and operator pulses of the pulses in flight,
  closes every source whose ``open`` was attempted and every operator
  without finishing it (no partial emission), and is raised by ``wait()`` as
  one ``ActiveRunError``; later errors are kept in its ``suppressed``.
  Handlers already called are not undone. ``cancel()`` does the same without
  an error. Calls already running are never interrupted: a pipelined run
  waits until its workers are idle before it closes operators.
* ``stop()`` and ``cancel()`` never block and are safe inside a handler;
  ``wait()`` on any thread of the run (handler, observer, block call, source
  read) raises instead of waiting for itself. A handler may run before
  ``start()`` has returned; ``ExecutionSession.stop()`` (``stop_session``)
  stops the session's current run without a run handle.
* Source lifecycle calls and handlers are synchronous. Coroutine functions
  are rejected before any source opens; a call that returns an awaitable
  fails the run at its stage, the awaitable is closed, and no cleanup is
  reported that did not happen.
* A failure to start a reader thread fails the run: unstarted readers are
  marked done, started ones stop cooperatively and close, the run drains
  and ``start()`` raises the attributed failure.
* A later ``start()`` on the same session constructs new source and
  operator instances and reuses the session's block instances and their
  state. Nothing an operator retained survives its run.
* Event handlers (``reactions.runtime``) belong to the run. Asynchronous
  handler workers start at their first event; a stopped or completed run
  lets their queues drain, a failed or cancelled one discards queued events
  and wakes emitters waiting under ``synchronous`` overflow. Either way the
  run is done only once every accepted reaction call returned (also a
  ``signal()`` on its caller's thread) and every handler worker ended; a
  handler still running keeps ``wait(timeout)`` returning ``False``. Handler
  threads are threads of the run: ``wait()`` there raises.
* Reactions also cover external signals and state machines::

      start()     ... driver starts -> $system.events.started -> readers start
      signal()    on the caller's thread (owned meanwhile); rejected after stop()
      stop()      signals rejected, then pulse admission closes
      EOF/stop    pulses finish -> $system.events.ended -> drain -> done
      cancel      no ended; queued events discarded; running code finishes
                  -> done

  ``ended`` means the main flow ended, not that every reaction finished;
  the drain after it waits for accepted signals and every cascade they or
  the pulses caused. A failure of ``started`` or ``ended`` fails the run.

Serially, block calls and handlers run on the processor thread only, so
ordinary stateful blocks are never reentered. Pipelined, one step (or one
phase, with ``phase_overlap``) is never entered by two pulses at once, and
every handler and observer callback of the run is serialized.
"""

import contextlib
import functools
import importlib
import inspect
import queue
import threading
import time
import uuid
import weakref
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    Union,
)

from roboflow_workflows.execution_engine.v2.active.pulses import (
    OperatorSlot,
    PulseExecutor,
    Registered,
    SourcePulse,
    attributed,
    close_operator,
)
from roboflow_workflows.execution_engine.v2.context import (
    ExecutionContext,
    use_execution_context,
)
from roboflow_workflows.execution_engine.v2.controls import ControlSnapshot
from roboflow_workflows.execution_engine.v2.data import (
    SampleContext,
    TemporalContext,
    Timestamp,
)
from roboflow_workflows.execution_engine.v2.errors import (
    ActiveRunError,
    ContractError,
    EventEmissionError,
    GraphUpdateError,
    SessionClosedError,
    UpdateConflictError,
    UpdateTimeoutError,
    WorkflowInputError,
)
from roboflow_workflows.execution_engine.v2.execution.arguments import (
    arguments_for,
    validation_arguments,
)
from roboflow_workflows.execution_engine.v2.execution.entries import Entry
from roboflow_workflows.execution_engine.v2.execution.inputs import prepare_inputs
from roboflow_workflows.execution_engine.v2.execution.outputs import GroupResult
from roboflow_workflows.execution_engine.v2.locking import acquired
from roboflow_workflows.execution_engine.v2.operators import (
    OperatorCounters,
    TerminationReason,
)
from roboflow_workflows.execution_engine.v2.pipelining.options import PipelineOptions
from roboflow_workflows.execution_engine.v2.pipelining.stages import (
    SERIAL,
    Coordination,
    PipelineCounters,
    PipelinedCoordination,
)
from roboflow_workflows.execution_engine.v2.pipelining.workers import OwnedThreads
from roboflow_workflows.execution_engine.v2.plan import (
    Binding,
    Constant,
    ExecutionSession,
    PlannedOperator,
    PlannedSource,
    PulseKey,
)
from roboflow_workflows.execution_engine.v2.reactions.runtime import (
    HandlerCounters,
    ReactionOutcome,
    ReactionRuntime,
)
from roboflow_workflows.execution_engine.v2.sources import (
    Emission,
    engine_observation,
)

if TYPE_CHECKING:
    from roboflow_workflows.execution_engine.v2.plan import SessionGeneration
    from roboflow_workflows.execution_engine.v2.recording.compilation import Capture
    from roboflow_workflows.execution_engine.v2.updates.prepared import (
        ActiveUpdateReceipt,
        PreparedUpdate,
    )
    from roboflow_workflows.execution_engine.v2.updates.reset import Retirement

__all__ = [
    "ActiveRun",
    "ActiveRunError",
    "GroupHandler",
    "GroupResult",
    "OperatorCounters",
    "SourceCounters",
    "start_session",
    "stop_session",
]

PIPELINE_MODULE = "roboflow_workflows.execution_engine.v2.active.pipeline"
"""Pipelined driver, imported only by runs that pass ``PipelineOptions``."""

RECORDING_MODULE = "roboflow_workflows.execution_engine.v2.recording.compilation"
"""Capture of recorded groups, imported only by plans that declare ``recording``."""

UPDATES_MODULE = "roboflow_workflows.execution_engine.v2.updates.prepared"
"""Receipts of graph updates, imported only by ``ActiveRun.apply_update``."""

RESET_MODULE = "roboflow_workflows.execution_engine.v2.updates.reset"
"""Retirement of a reset's replaced processing, imported only by resets."""

DEFAULT_UPDATE_TIMEOUT = 30.0
"""Seconds ``ActiveRun.apply_update`` waits for the run's update boundary."""

GroupHandler = Callable[[GroupResult], None]
"""Synchronous callback receiving one ``GroupResult`` per delivered pulse."""


@dataclass
class SourceCounters:
    """What happened to one source during one active run.

    Once the run is done, ``read == admitted + dropped + unadmitted`` and
    ``admitted == processed + cancelled``.

    Args:
        read: Emissions ``read()`` returned (end of source excluded).
        admitted: Emissions that became pulses.
        dropped: Emissions a newer one replaced while they waited for
            admission (overload ``latest`` only); never admitted.
        unadmitted: Emissions read after admission closed and discarded,
            a pending ``latest`` emission included.
        processed: Pulses whose route ran and whose handlers returned.
        delivered: Group results handed to handlers that returned.
        omitted: Group results not delivered because every field of the
            group read steps of a disabled control (``controls``).
        cancelled: Admitted pulses that did not complete: dropped after a
            failure or cancellation, cut short by one, or the pulse that
            failed itself.
        peak_admitted: Most pulses of the source admitted and not yet
            completed at one time; at most ``admission_bound``.
        ended: Whether ``read()`` returned ``None``.
        opened: Whether ``open()`` returned.
        closed: Whether ``close()`` returned.
    """

    read: int = 0
    admitted: int = 0
    dropped: int = 0
    unadmitted: int = 0
    processed: int = 0
    delivered: int = 0
    omitted: int = 0
    cancelled: int = 0
    peak_admitted: int = 0
    ended: bool = False
    opened: bool = False
    closed: bool = False


@dataclass
class _SourceSlot:
    """One source of the run: its instance, resolved ``open`` arguments and state.

    ``next_sequence``, ``in_flight`` and the admission-side counters change
    only under the driver's admission lock. ``admission`` bounds ``block``
    sources (serial runs are always ``block``); a pipelined ``latest`` source
    is bounded by ``in_flight < admission_bound`` at dispatch instead.
    """

    planned: PlannedSource
    instance: Any
    arguments: Mapping[str, Any]
    admission: threading.Semaphore
    counters: SourceCounters = field(default_factory=SourceCounters)
    open_attempted: bool = False
    next_sequence: int = 0
    in_flight: int = 0

    @property
    def name(self) -> str:
        return self.planned.name


@dataclass(frozen=True)
class _ReaderDone:
    source: str
    controls: Optional[ControlSnapshot] = None
    # The source ended before a reset: its end is told to the new processing.
    replay: bool = False


class _UpdateToken:
    """One reservation of a run's update boundary (``ActiveRun.apply_update``).

    ``acked`` and ``rejected`` are written under ``ActiveRun._lock`` and
    waited for on its condition: the driver acknowledges once every admitted
    pulse and domain end completed, or rejects when the run concluded first.
    """

    def __init__(self) -> None:
        self.acked = False
        self.rejected: Optional[str] = None
        # time.monotonic() when the driver paused admission.
        self.cut_at = 0.0


class _Replacement:
    """A reset's new run-specific processing, built before the cut.

    Args:
        operators: The new operators, constructed but never called.
        reactions: The new reaction runtime; ``None`` without reactions.
    """

    def __init__(
        self,
        *,
        operators: Dict[str, OperatorSlot],
        reactions: Optional[ReactionRuntime],
    ) -> None:
        self.operators = operators
        self.reactions = reactions

    def discard(self) -> None:
        """The reset did not commit: close what was built; never raises."""
        for slot in self.operators.values():
            close_operator(slot)
        if self.reactions is not None:
            try:
                self.reactions.close()
            except Exception:
                pass


@dataclass(frozen=True)
class _Barrier:
    """Serial queue marker: everything queued before it has completed."""

    token: _UpdateToken


class _Driver(Protocol):
    """How an ``ActiveRun`` admits and executes pulses: serially or pipelined.

    ``ActiveRun`` and its readers call these methods. A driver may use
    from the run: ``run_id``, ``aborting``, ``pipeline_counters``,
    ``_slots``, ``_readers``, ``_owned`` (mark its own threads),
    ``_executor`` (pulses, seals and domain ends), ``_admitted`` (under its
    admission lock), ``_termination``, ``_fail``, ``_conclude``,
    ``_boundary_reached`` and ``_boundary_rejected``.

    Obligations of every driver:

    * call ``_conclude`` exactly once, after every reader called
      ``reader_finished`` and no pulse or domain end runs;
    * after an abort, close the operators (``_executor.close_operators()``)
      as soon as no work runs, without waiting for the readers;
    * call ``_conclude`` from its own thread. The pipelined driver concludes
      on its dispatcher after shutting down its workers and joining the
      readers. The serial driver concludes on its processor thread once
      every reader reported finished; it does not join the readers.

    A graph update pauses the driver (``pause``): admission of new pulses
    stops reversibly, nothing that was admitted is lost, a source that ends
    meanwhile is remembered but not sealed, and the driver reports
    ``_boundary_reached`` once no admitted pulse or domain end runs or
    waits. ``resume`` lifts the pause under whatever graph the session has
    then; a run that concluded before the boundary reports
    ``_boundary_rejected`` instead. Neither closes admission.
    """

    def start(self) -> None:
        """``ActiveRun._launch``, before any reader starts: start own threads."""

    def admit(
        self, slot: _SourceSlot, emission: Emission, *, observed: Timestamp
    ) -> bool:
        """A reader read ``emission``; ``False`` once admission is closed."""

    def reader_finished(self, source: str) -> None:
        """Exactly once per reader: it closed its source or never started."""

    def close_admission(self) -> bool:
        """``ActiveRun._close_admission``; ``True`` only for the closing call."""

    def end_domain_later(self, domain: str, reason: TerminationReason) -> None:
        """The executor sealed and drained ``domain``: end it now or on a worker."""

    def wake(self) -> None:
        """``ActiveRun._abort``: wake driver threads to observe the abort."""

    def pause(self, token: _UpdateToken) -> None:
        """Stop admitting new pulses reversibly; report the boundary for ``token``."""

    def resume(self, token: _UpdateToken) -> None:
        """Lift ``token``'s pause (another token's is left alone); admit again.

        Seals or sequences the sources that ended while paused.
        """

    def settle_workers(self, timeout: float) -> bool:
        """After the boundary: ``True`` once no worker thread executes work."""

    def readers_finished(self) -> bool:
        """Whether every reader reported finished (paused: sources ended)."""

    def frontiers(self) -> Dict[str, int]:
        """Next pulse sequence per source (under the admission lock)."""

    def restart_domains(self) -> None:
        """A reset's install: seal the sources that ended once more at ``resume``.

        Assignments only. The new processing's domain progress then learns
        about every ended source, and its operators get their input ends.
        """


_ACTIVE_RUNS: "weakref.WeakKeyDictionary[ExecutionSession, ActiveRun]" = (
    weakref.WeakKeyDictionary()
)
_ACTIVE_RUNS_LOCK = threading.Lock()


def start_session(
    session: ExecutionSession,
    *,
    inputs: Optional[Mapping[str, Any]] = None,
    handlers: Optional[Mapping[str, GroupHandler]] = None,
    admission_bound: int = 2,
    pipeline: Optional[PipelineOptions] = None,
    finalize: Optional[Callable[[], None]] = None,
) -> "ActiveRun":
    """Validate, construct the sources and start reading.

    Handlers, static inputs, pipeline options and every source's resolved
    parameters are checked and every source instance is constructed before
    any reader opens anything. A plan that declares ``recording`` creates
    its recording last, still before any reader opens; the engine records
    every recorded group, with or without a handler, and finalizes the
    recording before the run is done.

    Args:
        session: Session of an active plan (one with declared sources).
        inputs: Static input values by name; the plan's ungrouped inputs.
        handlers: Synchronous callback per output group name. Groups without
            a handler are neither built nor retained; their steps still run.
            A handler output group (``$handlers.<h>.<output>``) receives one
            result per completed run of its event handler, on the handler's
            thread, serialized with other reaction callbacks only.
        admission_bound: Pulses one source may have admitted but not yet
            processed; at least 1.
        pipeline: ``None`` runs serially (the reference). Options run the
            pulses on ``max_in_flight`` workers with per-stage order and the
            sources' overload policies.
        finalize: Engine-internal release of what the caller created for
            this run alone (a replay's session state); called once, before
            the run is done.

    Returns:
        The running active run.

    Raises:
        WorkflowInputError: When the plan has no sources or the inputs are
            invalid.
        ContractError: When a handler names an unknown group, is not
            callable, is a coroutine function, the bound or the pipeline
            options are invalid or the session already has an unfinished
            active run.
        ActiveRunError: When a source's resolved parameters violate its
            declaration, a source or operator constructor fails or the
            observer rejects the run (stage ``start``).
        RecordingError: When the recording destination cannot be created.
    """
    plan = session.plan
    if not plan.is_active:
        raise WorkflowInputError(
            "The plan declares no sources; run it passively with session.run(inputs)"
        )
    registered = _register_handlers(plan, handlers or {})
    if isinstance(admission_bound, bool) or not isinstance(admission_bound, int):
        raise ContractError(f"admission_bound must be an int, got {admission_bound!r}")
    if admission_bound < 1:
        raise ContractError(
            f"admission_bound must be at least 1, got {admission_bound}"
        )
    if pipeline is not None:
        if not isinstance(pipeline, PipelineOptions):
            raise ContractError(
                f"pipeline must be PipelineOptions or None, got "
                f"{type(pipeline).__name__}"
            )
        pipeline.check_sources(plan.sources)

    with _ACTIVE_RUNS_LOCK:
        previous = _ACTIVE_RUNS.get(session)
        if previous is not None and not previous.done:
            raise ContractError(
                f"Session {session.session_id} already has active run "
                f"{previous.run_id}; wait() for it before starting another"
            )
        if session.closed:
            raise SessionClosedError(
                f"Session {session.session_id} is closed; it cannot start. "
                "Create a new session."
            )
        if session.plan is not plan:
            raise ContractError(
                f"Session {session.session_id} switched to graph version "
                f"{session.graph_version} while this start prepared; call start again"
            )
        entries = prepare_inputs(plan, inputs or {}, controls=session.controls.current)
        run_id = uuid.uuid4().hex
        stop_event = threading.Event()
        slots = {
            planned.name: _prepare_source(
                session,
                planned,
                entries=entries,
                run_id=run_id,
                stop_event=stop_event,
                admission_bound=admission_bound,
            )
            for planned in plan.sources.values()
        }
        operators = _prepare_operators(plan.operators.values())
        capture = _start_capture(plan, entries=entries, operators=operators)
        if capture is not None:
            registered = _with_recorders(plan, registered, recorders=capture.recorders)
        run = ActiveRun(
            session,
            run_id=run_id,
            inputs=entries,
            registered=registered,
            slots=slots,
            operators=operators,
            stop_event=stop_event,
            admission_bound=admission_bound,
            pipeline=pipeline,
            handlers=handlers or {},
            capture=capture,
            finalize=finalize,
        )
        _ACTIVE_RUNS[session] = run
    run._launch()

    return run


def stop_session(session: ExecutionSession) -> None:
    """Request a graceful stop of the session's current active run.

    The run registered at the time of the call is captured under the
    registry lock and stopped after the lock is released, so a later
    ``start`` on the same session is never stopped by accident. Without an
    unfinished run this is a no-op. Safe inside a group handler, including
    one that runs before ``start`` has returned, because the run is
    registered before its readers launch.

    Args:
        session: Session of an active plan.

    Raises:
        WorkflowInputError: When the plan declares no sources.
    """
    if not session.plan.is_active:
        raise WorkflowInputError(
            "The plan declares no sources, so the session has no active run to stop"
        )

    with _ACTIVE_RUNS_LOCK:
        run = _ACTIVE_RUNS.get(session)
    if run is not None:
        run.stop()


@contextlib.contextmanager
def holding_run_registry(
    session: ExecutionSession, *, deadline: Optional[float] = None
) -> Iterator[Optional["ActiveRun"]]:
    """Hold the run registry and yield the session's unfinished active run.

    While the block runs, no ``start`` registers a run. A graph update and
    ``close`` change the session inside it, so a start sees either the old
    session or the changed one.

    Args:
        session: The session to change.
        deadline: ``time.monotonic()`` value at which waiting for the
            registry gives up; ``None`` waits.

    Yields:
        The unfinished active run of ``session``, or ``None``.

    Raises:
        UpdateTimeoutError: When the registry was not free before ``deadline``.
    """
    with acquired(_ACTIVE_RUNS_LOCK, deadline=deadline, what="the run registry"):
        run = _ACTIVE_RUNS.get(session)
        yield run if run is not None and not run.done else None


RECORDING_RUN = "recording_run"
SOURCES_ENDED = "sources_ended"


def reset_blockers(session: ExecutionSession) -> Tuple[Tuple[str, str], ...]:
    """Reasons the session's unfinished active run cannot take a reset now.

    Advisory, for ``assess_update``: ``ActiveRun.apply_update`` checks both
    again. Takes the run registry and then the driver's lock, one after the
    other, never nested; an idle session, or one whose run finished, has none.

    Returns:
        ``(reason, detail)`` pairs; empty when a reset can apply.
    """
    with _ACTIVE_RUNS_LOCK:
        run = _ACTIVE_RUNS.get(session)
    if run is None or run.done:
        return ()

    blockers = []
    if run._capture is not None:
        blockers.append(
            (
                RECORDING_RUN,
                f"run {run.run_id} records output groups, and records cannot be "
                "attributed to one processing across a reset; stop the run, then "
                "reset the idle session",
            )
        )
    if run._driver.readers_finished():
        blockers.append(
            (
                SOURCES_ENDED,
                f"every source of run {run.run_id} ended; the run completes with "
                "its current processing. Reset the session once it finished",
            )
        )

    return tuple(blockers)


def _is_async_callable(target: Any) -> bool:
    """Whether calling ``target`` would return a coroutine or async generator."""
    call = target if inspect.isroutine(target) else getattr(target, "__call__", None)
    is_async = call is not None and (
        inspect.iscoroutinefunction(call) or inspect.isasyncgenfunction(call)
    )

    return is_async


def _discard_awaitable(returned: Any) -> bool:
    """Close an awaitable a synchronous call returned; ``True`` when it was one."""
    if not inspect.isawaitable(returned):
        return False
    if inspect.iscoroutine(returned):
        returned.close()

    return True


def _register_handlers(
    plan: Any, handlers: Mapping[str, GroupHandler]
) -> List[Registered]:
    declared = {group.name: group for group in plan.output_groups}
    reacting = {group.name for group in plan.reactions.groups}
    unknown = sorted(set(handlers) - set(declared) - reacting)
    if unknown:
        raise ContractError(
            f"Handlers are registered for unknown output groups {unknown}; "
            f"declared groups: {list(declared) + sorted(reacting)}"
        )
    for name, handler in handlers.items():
        if not callable(handler):
            raise ContractError(
                f"Handler for output group {name!r} must be callable, got "
                f"{type(handler).__name__}"
            )
        if _is_async_callable(handler):
            raise ContractError(
                f"Handler for output group {name!r} is a coroutine function; active "
                "runs call synchronous handlers only"
            )

    registered = [
        Registered(
            group=group,
            handler=handlers[group.name],
            prerequisites=frozenset(group.dependencies),
        )
        for group in plan.output_groups
        if group.name in handlers
    ]

    return registered


def _start_capture(
    plan: Any, *, entries: Mapping[str, Entry], operators: Mapping[str, OperatorSlot]
) -> Optional["Capture"]:
    """Create the plan's recording; close the constructed operators if that fails."""
    if plan.recording is None:
        return None

    capture_module = importlib.import_module(RECORDING_MODULE)
    try:
        capture = capture_module.start_capture(plan, inputs=entries)
    except BaseException:
        for slot in operators.values():
            close_operator(slot)
        raise

    return capture


def _with_recorders(
    plan: Any, registered: List[Registered], *, recorders: Mapping[str, GroupHandler]
) -> List[Registered]:
    """Add the engine's recorders; a recorded group needs no host handler."""
    by_group = {item.group.name: item for item in registered}
    combined = []
    for group in plan.output_groups:
        item = by_group.get(group.name)
        recorder = recorders.get(group.name)
        if recorder is None:
            if item is not None:
                combined.append(item)
            continue
        combined.append(
            Registered(
                group=group,
                handler=item.handler if item is not None else None,
                prerequisites=frozenset(group.dependencies),
                recorder=recorder,
            )
        )

    return combined


def _prepare_source(
    session: ExecutionSession,
    planned: PlannedSource,
    *,
    entries: Mapping[str, Entry],
    run_id: str,
    stop_event: threading.Event,
    admission_bound: int,
) -> _SourceSlot:
    """Resolve and validate the source's ``open`` arguments, then construct it."""
    values = [_static_value(binding, entries) for binding in planned.bindings]
    try:
        planned.spec.validate_resolved_arguments(
            planned.params, validation_arguments(planned, values)
        )
    except ContractError as error:
        raise ActiveRunError(str(error), stage="start", source=planned.name) from error

    resources = {
        name: resolved.value
        for name, resolved in session.source_resources[planned.name].items()
    }
    context = ExecutionContext(
        step_path=planned.step_path,
        block_type=planned.spec.type,
        session_id=session.session_id,
        run_id=run_id,
    )
    try:
        with use_execution_context(context):
            instance = planned.spec.source_class(**resources)
    except Exception as error:
        raise ActiveRunError(
            f"constructor of {planned.spec.source_class.__qualname__} failed: "
            f"{type(error).__name__}: {error}",
            stage="start",
            source=planned.name,
        ) from error
    for method in ("open", "read", "close"):
        if _is_async_callable(getattr(instance, method)):
            raise ActiveRunError(
                f"{method}() of {planned.spec.source_class.__qualname__} is a "
                "coroutine function; source lifecycle calls are synchronous",
                stage="start",
                source=planned.name,
            )
    instance.source_name = planned.name
    instance.stop_event = stop_event
    slot = _SourceSlot(
        planned=planned,
        instance=instance,
        arguments=arguments_for(planned, values),
        admission=threading.Semaphore(admission_bound),
    )

    return slot


def _prepare_operators(
    planned_operators: Sequence[PlannedOperator],
) -> Dict[str, OperatorSlot]:
    """Construct a fresh instance of every operator; close them all on failure."""
    prepared: Dict[str, OperatorSlot] = {}
    for planned in planned_operators:
        operator_class = planned.spec.operator_class
        try:
            instance = operator_class(
                name=planned.name, params=planned.params, inputs=planned.inputs
            )
        except Exception as error:
            failure = ActiveRunError(
                f"constructor of {operator_class.__qualname__} failed: "
                f"{type(error).__name__}: {error}",
                stage="start",
                operator=planned.name,
            )
            failure.__cause__ = error
            for slot in prepared.values():
                close_error = close_operator(slot)
                if close_error is not None:
                    failure.suppressed += (close_error,)
            raise failure
        prepared[planned.name] = OperatorSlot(planned=planned, instance=instance)

    return prepared


def _close_replaced_operator(slot: OperatorSlot) -> None:
    """Close an operator a reset replaced; raise what its ``close`` raised."""
    failure = close_operator(slot)
    if failure is not None:
        raise failure.__cause__


def _retire_reactions(runtime: ReactionRuntime) -> None:
    """Close a reaction runtime a reset replaced and wait for its handler threads."""
    runtime.close()
    runtime.join()


def _static_value(binding: Binding, entries: Mapping[str, Entry]) -> Any:
    """Value of a source parameter selector: a literal or an ungrouped static input."""
    if isinstance(binding.source, Constant):
        return binding.source.value

    value = entries[binding.source.name].values[()]

    return value


class ActiveRun:
    """One execution of an active plan: its readers, processing and outcome.

    Create runs with ``ExecutionSession.start``. The run keeps no result
    history; hosts receive results through their handlers only.

    Args:
        session: Session whose block instances process the pulses.
        run_id: Identity of this run.
        inputs: Static input entries shared by every pulse.
        registered: Output groups with handlers, in plan order.
        slots: Constructed sources by name.
        operators: Constructed operators by name, in plan order.
        stop_event: Event the sources watch; set by ``stop()``, ``cancel()``
            and by failure.
        admission_bound: Pulses one source may have admitted at once.
        pipeline: Pipeline options; ``None`` processes serially.
        handlers: Every registered callback by group name; those of handler
            output groups (``$handlers.<h>.<output>``) receive one result per
            completed handler run.
        capture: The plan's open recording; ``None`` when it records nothing.
        finalize: Called once before the run is done; see ``start_session``.
    """

    def __init__(
        self,
        session: ExecutionSession,
        *,
        run_id: str,
        inputs: Mapping[str, Entry],
        registered: List[Registered],
        slots: Mapping[str, _SourceSlot],
        operators: Mapping[str, OperatorSlot],
        stop_event: threading.Event,
        admission_bound: int = 2,
        pipeline: Optional[PipelineOptions] = None,
        handlers: Optional[Mapping[str, GroupHandler]] = None,
        capture: Optional["Capture"] = None,
        finalize: Optional[Callable[[], None]] = None,
    ):
        self.session = session
        self._capture = capture
        self._finalize = finalize
        self._finished_capture = False
        self._stop_requested = False
        self._stopped_early = False
        self.run_id = run_id
        self.stop_event = stop_event
        self._slots = slots
        self._lock = threading.Lock()
        # Lifecycle changes (stop, cancel, failure, done) and update boundary
        # acknowledgements are signalled here; apply_update waits on it.
        self._changed = threading.Condition(self._lock)
        self._update_token: Optional[_UpdateToken] = None
        # Next emission ordinal of every operator name this run has had, so
        # a reset's operators number on and never repeat an ordinal.
        self._operator_next: Dict[str, int] = {}
        self._failure: Optional[ActiveRunError] = None
        self._cancelled = False
        self._done = threading.Event()
        self._releasing = False
        self._owned = OwnedThreads()
        # The run's callbacks by group name: the start's handlers, replaced
        # as a whole by every applied update. The reaction runtime reads it
        # at each delivery of a handler group.
        self._handlers: Dict[str, GroupHandler] = dict(handlers or {})
        self._coordination: Coordination = (
            PipelinedCoordination(session, options=pipeline)
            if pipeline is not None
            else SERIAL
        )
        self._observer = self._coordination.observer(session)
        self._reactions = self._reaction_runtime(
            session.plan,
            sessions=session.handler_sessions,
            managed_state=getattr(session, "managed_state", None),
        )
        self._executor = PulseExecutor(
            session,
            run_id=run_id,
            inputs=inputs,
            registered=registered,
            operators=operators,
            sources=frozenset(slots),
            coordination=self._coordination,
            fail=self._fail,
            aborting=lambda: self.aborting,
            schedule_end=self._end_domain_later,
            reactions=self._reactions,
        )
        self._driver: _Driver = (
            _SerialDriver(self)
            if pipeline is None
            else importlib.import_module(PIPELINE_MODULE).PipelinedDriver(
                self, options=pipeline, admission_bound=admission_bound
            )
        )
        self._readers = {
            slot.name: threading.Thread(
                target=self._read,
                args=(slot,),
                name=f"workflows-v2-source-{slot.name}",
                daemon=True,
            )
            for slot in slots.values()
        }

    @property
    def counters(self) -> Mapping[str, SourceCounters]:
        """Per-source counters; complete once ``done``."""
        return {name: slot.counters for name, slot in self._slots.items()}

    @property
    def operator_counters(self) -> Mapping[str, OperatorCounters]:
        """Per-operator counters, including bounded-state outcomes; complete once ``done``."""
        return {name: slot.counters for name, slot in self._executor.operators.items()}

    @property
    def pipeline_counters(self) -> Optional[PipelineCounters]:
        """Gauges, totals and stage counters of a pipelined run; ``None`` when serial."""
        if isinstance(self._coordination, PipelinedCoordination):
            return self._coordination.counters

        return None

    @property
    def recording_counters(self) -> Mapping[str, Mapping[str, int]]:
        """Chunks and bytes written per recorded group; empty without ``recording``."""
        if self._capture is None:
            return {}

        counters = self._capture.counters

        return counters

    @property
    def reaction_counters(self) -> Mapping[str, HandlerCounters]:
        """Per-handler event counters by handler selector; empty without handlers."""
        if self._reactions is None:
            return {}

        counters = self._reactions.counters

        return counters

    def reaction_outcomes(
        self, limit: Optional[int] = None
    ) -> Tuple[ReactionOutcome, ...]:
        """Recent handler outcomes, oldest first (a bounded log, metadata only).

        Args:
            limit: At most this many; every kept one when ``None``.

        Returns:
            Completed, failed, dropped and discarded events per handler.
        """
        if self._reactions is None:
            return ()

        outcomes = self._reactions.outcomes(limit)

        return outcomes

    @property
    def done(self) -> bool:
        """Whether every source is closed and every admitted pulse handled."""
        return self._done.is_set()

    @property
    def releasing(self) -> bool:
        """Whether the run no longer uses its session and only releases resources.

        Set after workers, reactions and operators settled, just before
        ``finalize``, also after a failed start; ``done`` follows. A
        replay's ``finalize`` closes its session then.
        """
        return self._releasing

    @property
    def failure(self) -> Optional[ActiveRunError]:
        """The run's terminal failure, once one happened."""
        return self._failure

    @property
    def aborting(self) -> bool:
        """Whether the run failed or was cancelled; nothing new starts."""
        return self._failure is not None or self._cancelled

    @property
    def state(self) -> str:
        """``running``, ``stopping``, ``finished``, ``failed`` or ``cancelled``."""
        if self._done.is_set():
            if self._failure is not None:
                return "failed"
            return "cancelled" if self._cancelled else "finished"
        if self.stop_event.is_set():
            return "stopping"

        return "running"

    def machine_state(
        self, machine: str, *, source_id: Optional[str] = None
    ) -> Tuple[str, int]:
        """Read one state machine instance of the session.

        Args:
            machine: Scoped machine name, e.g. ``"gate"`` or ``"child/gate"``.
            source_id: Source of a per-source machine; ``None`` for a global one.

        Returns:
            ``(state, version)``; version 0 before the first transition.

        Raises:
            ContractError: For an unknown machine or a plan without machines.
            StateScopeError: For a missing or unexpected ``source_id``.
        """
        if self._reactions is None:
            raise ContractError(
                f"Unknown state machine {machine!r}: the workflow declares none"
            )

        current = self._reactions.machine_state(machine, source_id=source_id)

        return current

    def machine_counters(self) -> Mapping[str, Any]:
        """Applied, ignored and stale attempts per ``<machine>.<transition>``.

        Returns:
            ``TransitionCounters`` by label, an independent snapshot; empty
            without state machines.
        """
        if self._reactions is None:
            return {}

        counters = self._reactions.machine_counters()

        return counters

    def signal(
        self,
        name: str,
        *,
        source_id: Optional[str] = None,
        sample: Optional[SampleContext] = None,
        temporal: Optional[TemporalContext] = None,
        **fields: Any,
    ) -> None:
        """Deliver one declared external signal to the run.

        Fixed transitions and synchronous handlers run on the calling thread
        before this returns; asynchronous handlers are admitted (and may make
        this wait under ``synchronous`` overflow). While it runs, the calling
        thread counts as a thread of the run. Accepted signals finish even if
        ``stop()`` follows; the run is done only after their work drained.

        Args:
            name: Signal name declared in the workflow's ``signals``.
            source_id: Source the signal belongs to; never inferred.
            sample: Full source context; its ``source_id`` must equal
                ``source_id`` when both are given.
            temporal: Temporal context the handlers see.
            **fields: Every declared field of the signal.

        Raises:
            EventEmissionError: For an undeclared signal, a per-source
                machine subscription without a source, or a run that is
                stopping, cancelled, failed or done.
            EventPayloadError: For missing, unknown or ill-kinded fields.
            ContractError: When ``sample`` and ``source_id`` disagree.
            ReactionError: When a synchronous handler failed; the run
                continues.
        """
        if sample is not None and source_id is not None:
            if sample.source_id != source_id:
                raise ContractError(
                    f"signal({name!r}) got source_id={source_id!r} and a sample "
                    f"of source {sample.source_id!r}; pass one, or equal ones"
                )
        elif source_id is not None:
            sample = SampleContext(source_id=source_id)
        if self._reactions is None:
            raise EventEmissionError(
                f"Signal {name!r} is not declared; the workflow declares no signals"
            )

        with self._owned.borrow():
            self._reactions.signal(name, fields, sample=sample, temporal=temporal)

    def stop(self) -> None:
        """Close admission and ask the sources to stop; never blocks.

        ``signal()`` is rejected from now on. Admitted pulses, accepted
        signals and the handler work they cause are still processed and
        delivered. Calling it again, or from inside a handler, has no
        further effect.
        """
        with self._lock:
            if not self._stop_requested:
                self._stop_requested = True
                # A source that returns None once asked to stop also ends;
                # only sources that ended before this request ended naturally.
                self._stopped_early = not all(
                    slot.counters.ended for slot in self._slots.values()
                )
                self._changed.notify_all()
        if self._reactions is not None:
            self._reactions.stop_ingress()
        self._close_admission()

    def cancel(self) -> None:
        """Stop without draining; never blocks.

        Admission closes, admitted pulses that have not started are
        cancelled, running pulses stop at their next step, delivery or
        operator boundary, and operators are closed without finishing.
        Calls already running are not interrupted. ``wait()`` returns
        ``True`` with state ``cancelled`` once they returned, accepted
        ``signal()`` calls included. No effect once the run failed or is
        done.
        """
        with self._lock:
            if self._failure is not None or self._done.is_set():
                return
            self._cancelled = True
            self._changed.notify_all()
        self._abort()

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Wait until the run is done.

        Args:
            timeout: Seconds to wait; forever when ``None``.

        Returns:
            ``True`` when the run completed or was cancelled, ``False`` on
            timeout.

        Raises:
            ActiveRunError: The run's failure, once it completed failed.
            ContractError: When called from a thread of this run (a handler,
                observer callback, block call or source read), which would
                wait for itself; call ``stop()`` there instead.
        """
        self._owned.reject_wait("ActiveRun.wait()")
        if not self._done.wait(timeout):
            return False
        if self._failure is not None:
            raise self._failure

        return True

    def __enter__(self) -> "ActiveRun":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._owned.reject_wait("ActiveRun context exit")
        self.stop()
        if exc_type is None:
            self.wait()
        else:
            self._done.wait()

    # Graph updates ---------------------------------------------------------

    def apply_update(
        self,
        update: "PreparedUpdate",
        *,
        handlers: Optional[Mapping[str, GroupHandler]] = None,
        timeout: float = DEFAULT_UPDATE_TIMEOUT,
    ) -> "ActiveUpdateReceipt":
        """Switch this running run to a prepared graph at a quiescent boundary.

        The run pauses admission, lets everything already admitted settle
        and then publishes the new graph::

            cut      readers park (``block``) or keep replacing their pending
                     emission (``latest``); ``signal()`` is rejected
            drain    admitted pulses, their deliveries and the operator
                     pulses they fed, scheduled domain ends, queued and
                     running handler events, accepted signals and the
                     cascades they cause all complete
            commit   the session's graph, this run's deliveries and callbacks,
                     the control snapshot and the stage frontiers switch
            resume   parked readers admit under the new graph

        Sources stay open and keep their slots and sequences; operators keep
        their instances, buffers and ended inputs (a partial window is not
        flushed); retained steps keep their state; the reaction runtime,
        its handler sessions and every model or resource stay. Nothing is
        restarted. Results delivered after the commit carry the new
        ``graph_version``; an operator result whose window spans the
        boundary keeps its old causes. Control values, version, epochs and
        a pending reset are kept; a control write during the drain lands
        before or after the commit, never half-way.

        A reset candidate (``prepare_update(plan, reset=True)``) replaces
        the processing instead: its operators and reaction runtime are built
        before the cut; the commit installs them with every step instance,
        fresh stage gates and domain progress. Sources, readers, driver and
        workers stay; operator ordinals continue per name; ended sources are
        told to the new operators; partial windows are dropped, never
        flushed; the replaced reactions refuse every later signal, and the
        new ones refuse signals until the resume, like the cut. The
        replaced processing closes on a thread started before the cut,
        after the resume (``receipt.cleanup``, an ``updates.Cleanup``);
        until it finished, the session refuses another reset, in this run
        or a later one. The new reaction runtime gets no ``started`` event
        (the run started once), so a handler subscribed to ``started``
        does not run again; its state machines initialize on their first
        event, as at start.

        The boundary is host-quiescent, not device-quiescent: every admitted
        pulse finished, including the futures its blocks returned, so the
        engine honors the completion each block declares. Device work a
        block queued without a future or a result that waits for it may
        still run; the engine adds no device synchronization. A reset's
        replaced instances are released on the cleanup thread after the
        resume, so their memory can be freed while the new processing runs.

        Handlers are a patch of the run's callbacks: a group named here gets
        this callback (replacing the one it had), a group not named keeps
        its callback, and a group the new plan no longer declares loses
        its callback. Every name is validated against the new plan before
        anything pauses. A new group without a callback is built but not
        delivered, as at ``start``.

        Call it from a host thread, never from a handler, block, source or
        reaction thread of this run: there it raises ``ContractError``, as
        it would wait for itself. Queue the request there and apply it from
        the thread that started the run. Prepare the candidate first with
        ``session.prepare_update``; its construction and this call may
        overlap with processing. One update of a run at a time; a second
        call while one is in progress is rejected.

        Args:
            update: An ``updates.PreparedUpdate`` of this run's session.
            handlers: Callbacks to add or replace, by output group name.
            timeout: Seconds for the cut, the drain and the commit's lock
                waits; positive and finite. A reset's construction before
                the cut is not counted, and a rejected reset closes the
                operators it built on this thread afterwards: both are block
                code, which no timeout bounds.

        Returns:
            The ``updates.ActiveUpdateReceipt``, with the raw stamps of the
            call, cut, drain and resume, the durations between them, and
            the domain frontiers the new stages started at (the next source
            read ordinal or operator emission ordinal per domain at the
            boundary).

        Raises:
            ContractError: When called from a thread of this run (a handler,
                observer, block, source or reaction thread, or a ``signal``
                caller), for an invalid ``timeout``, or for a handler that
                names an unknown group, is not callable or is asynchronous.
            GraphUpdateError: When the run records groups (``recording``):
                records would not be attributable to one graph version, so
                stop the run and start one from the updated session instead;
                when the operators or reactions of a reset candidate cannot
                be built, or no thread starts to close what it replaces. The
                candidate stays prepared.
            UpdateConflictError: When the candidate belongs to another
                session, was applied or discarded, or was prepared from an
                earlier graph version (it is discarded then); when another
                update of this run is in progress; when the run is
                stopping, cancelled, failed or done, also if that happened
                while this call waited; or when every source ended before
                the boundary; for a reset, while the session still retires
                what its last reset replaced. The run continues, or
                concludes, with its current graph.
            UpdateTimeoutError: When the boundary was not reached in time,
                or a lock of the commit (the session, its run registry, the
                candidate, the control panel or this run) was not free in
                time, e.g. a control write whose custom kind codec is slow
                holds the panel. The pause is lifted, nothing admitted is
                lost, the run keeps its graph and the candidate stays
                prepared: try again, or discard it when giving up.
            SessionClosedError: When the session was closed.
        """
        called = time.monotonic()
        self._owned.reject_wait(
            "ActiveRun.apply_update()",
            remedy=(
                "Hand the update to a host thread, e.g. put the plan on a queue "
                "that the thread which started the run reads"
            ),
        )
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not timeout > 0
            or timeout == float("inf")
        ):
            raise ContractError(
                f"timeout must be a positive finite number of seconds, got {timeout!r}"
            )
        if self._capture is not None:
            raise GraphUpdateError(
                f"run {self.run_id} records output groups; records cannot be "
                "attributed to one graph version across an update, so stop the "
                "run and start a new one from the updated session instead"
            )
        self._check_candidate(update)
        callbacks = self._patched_callbacks(update.plan, handlers)
        registered = _register_handlers(update.plan, callbacks)
        replacement = self._replacement(update) if update.reset else None

        retirement = None
        committed = False
        try:
            if replacement is not None:
                retirement = self._reserved_retirement()
            # The reservation starts here; the legacy durations count from it.
            reserving = time.monotonic()
            deadline = reserving + timeout
            token = self._reserve_boundary(deadline=deadline, reset=update.reset)
            try:
                self._wait_boundary(token, deadline=deadline)
                drained = time.monotonic()
                frontiers = self._frontiers()
                if replacement is None:
                    install = functools.partial(
                        self._rebind,
                        registered=registered,
                        callbacks=callbacks,
                        frontiers=frontiers,
                    )
                else:
                    frontiers = {
                        **{name: frontiers[name] for name in self._slots},
                        **self._numbered_on(replacement.operators, frontiers),
                    }
                    install = functools.partial(
                        self._install,
                        replacement,
                        token=token,
                        registered=registered,
                        callbacks=callbacks,
                        frontiers=frontiers,
                    )
                receipt, _ = self.session._commit_in_run(
                    update,
                    run=self,
                    check=lambda: self._check_boundary(token, deadline=deadline),
                    install=install,
                    retirement=retirement,
                    deadline=deadline,
                )
                committed = True
            finally:
                resumed = self._release_boundary(token)
        finally:
            if retirement is not None:
                if committed:
                    retirement.release()
                else:
                    retirement.cancel()
            if replacement is not None and not committed:
                replacement.discard()

        updates = importlib.import_module(UPDATES_MODULE)
        receipt = updates.ActiveUpdateReceipt(
            graph_version=receipt.graph_version,
            previous_version=receipt.previous_version,
            diff=receipt.diff,
            reset=receipt.reset,
            processing_version=receipt.processing_version,
            run_id=self.run_id,
            drained_seconds=drained - reserving,
            paused_seconds=resumed - reserving,
            frontiers=frontiers,
            called_at=called,
            cut_at=token.cut_at,
            drained_at=drained,
            resumed_at=resumed,
            run_build_seconds=reserving - called if replacement is not None else 0.0,
            cleanup=None if retirement is None else retirement.cleanup,
        )

        return receipt

    def _replacement(self, update: "PreparedUpdate") -> _Replacement:
        """Build a reset's operators and reactions; nothing pauses meanwhile."""
        parts = update._reset_parts
        try:
            operators = _prepare_operators(update.plan.operators.values())
        except ActiveRunError as error:
            raise GraphUpdateError(
                f"the reset was not applied; the run keeps its graph: {error}"
            ) from error
        try:
            reactions = self._reaction_runtime(
                update.plan,
                sessions=parts.handler_sessions,
                managed_state=parts.managed_state,
            )
        except BaseException:
            for slot in operators.values():
                close_operator(slot)
            raise
        replacement = _Replacement(operators=operators, reactions=reactions)

        return replacement

    def _reserved_retirement(self) -> "Retirement":
        """A reset's ``updates.Retirement`` with its closing thread started.

        Before the cut: a reset that gets no thread is refused while nothing
        changed, rather than closing arbitrary operators on this caller.
        """
        reset = importlib.import_module(RESET_MODULE)
        retirement = reset.Retirement()
        try:
            retirement.reserve(name=f"workflows-v2-cleanup-{self.run_id[:8]}")
        except RuntimeError as error:
            raise GraphUpdateError(
                "the reset was not applied; the run keeps its graph: no thread "
                f"starts to close the processing it would replace: {error}"
            ) from error

        return retirement

    def _numbered_on(
        self, operators: Mapping[str, OperatorSlot], frontiers: Mapping[str, int]
    ) -> Dict[str, int]:
        """Where each new operator numbers on: after every ordinal its name had."""
        numbered = {
            name: frontiers.get(name, self._operator_next.get(name, 0))
            for name in operators
        }

        return numbered

    def _install(
        self,
        replacement: _Replacement,
        retirement: Any,
        *,
        token: _UpdateToken,
        registered: List[Registered],
        callbacks: Dict[str, GroupHandler],
        frontiers: Mapping[str, int],
    ) -> None:
        """Run the reset's new processing: assignments only, nothing raises.

        Called like ``_rebind``, with the commit's ``updates.Retirement``.
        The replaced operators and reactions go into it; they are neither
        finished nor closed here (partial windows are dropped at the
        closing), and the replaced reactions refuse every later signal from
        now on. The new reactions are paused under ``token`` before they
        are reachable: ``signal()`` reaches them only after the session's
        publication, when ``_release_boundary`` resumes them.
        """
        for name, slot in self._executor.operators.items():
            self._operator_next[name] = slot.next_sequence
        for name, slot in replacement.operators.items():
            slot.next_sequence = frontiers[name]
        retirement.run_operators = {
            name: functools.partial(_close_replaced_operator, slot)
            for name, slot in self._executor.operators.items()
        }
        if self._reactions is not None:
            self._reactions.seal_replaced()
            retirement.run_reactions = functools.partial(
                _retire_reactions, self._reactions
            )
        if replacement.reactions is not None:
            replacement.reactions.pause_ingress(token)
        self._executor.restart(
            registered, operators=replacement.operators, reactions=replacement.reactions
        )
        self._reactions = replacement.reactions
        self._handlers = dict(callbacks)
        self._driver.restart_domains()
        if isinstance(self._coordination, PipelinedCoordination):
            self._coordination.restart_gates(frontiers)

    def _check_candidate(self, update: "PreparedUpdate") -> None:
        """Reject a candidate that can never commit, before anything pauses."""
        if update.session is not self.session:
            raise UpdateConflictError(
                f"the update was prepared for session {update.session.session_id}, "
                f"not {self.session.session_id}"
            )
        updates = importlib.import_module(UPDATES_MODULE)
        if update.state != updates.PREPARED:
            raise UpdateConflictError(f"the update was already {update.state}")
        if update.reset:
            self.session._raise_if_retiring()
        if update.base_version != self.session.graph_version:
            update.discard()
            raise UpdateConflictError(
                f"the update was prepared from graph version {update.base_version}, "
                f"but the session is at version {self.session.graph_version}; "
                "prepare it again"
            )

    def _patched_callbacks(
        self, plan: Any, handlers: Optional[Mapping[str, GroupHandler]]
    ) -> Dict[str, GroupHandler]:
        """The run's callbacks after the patch: kept, replaced, added, dropped."""
        declared = {group.name for group in plan.output_groups}
        declared.update(group.name for group in plan.reactions.groups)
        kept = {name: h for name, h in self._handlers.items() if name in declared}
        patched = {**kept, **dict(handlers or {})}

        return patched

    def _reserve_boundary(self, *, deadline: float, reset: bool) -> _UpdateToken:
        """Take the run's one update token and cut admission.

        Lock order: the session's lifecycle guard and run registry, then
        this run's lock, then the driver's admission lock and the reaction
        runtime's admission. No callback runs inside. Waiting for the guard,
        the registry or the lock (an idle update or a close of the session
        in progress) gives up at ``deadline``.
        """
        with acquired(self.session._use_lock, deadline=deadline, what="the session"):
            with holding_run_registry(self.session, deadline=deadline) as active:
                if self.session.closed:
                    raise SessionClosedError(
                        f"Session {self.session.session_id} is closed; it cannot "
                        "update its graph"
                    )
                if active is not self:
                    raise UpdateConflictError(
                        f"run {self.run_id} is done or no longer the active run of "
                        f"session {self.session.session_id}; it cannot be updated"
                    )
                with acquired(self._lock, deadline=deadline, what="the active run"):
                    if self._update_token is not None:
                        raise UpdateConflictError(
                            f"run {self.run_id} already has a graph update in progress"
                        )
                    if reset:
                        self.session._raise_if_retiring()
                    self._raise_if_terminal()
                    token = _UpdateToken()
                    self._update_token = token
                self._driver.pause(token)
                token.cut_at = time.monotonic()
                if self._reactions is not None:
                    self._reactions.pause_ingress(token)

        return token

    def _raise_if_terminal(self) -> None:
        """Lock held: a stopping, aborting or done run cannot switch graphs."""
        if self._done.is_set():
            state = "done"
        elif self._failure is not None:
            state = "failed"
        elif self._cancelled:
            state = "cancelled"
        elif self._stop_requested:
            state = "stopping"
        else:
            return

        raise UpdateConflictError(
            f"run {self.run_id} is {state}; it keeps its current graph"
        )

    def _wait_boundary(self, token: _UpdateToken, *, deadline: float) -> None:
        """Wait until nothing admitted runs: driver, workers, then reactions."""
        with self._lock:
            while not token.acked:
                if token.rejected is not None:
                    raise UpdateConflictError(
                        f"run {self.run_id} {token.rejected}; it keeps its "
                        "current graph"
                    )
                self._raise_if_terminal()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise self._timed_out("its admitted pulses")
                self._changed.wait(remaining)
        if not self._driver.settle_workers(max(deadline - time.monotonic(), 0.0)):
            raise self._timed_out("its worker threads")
        if self._reactions is not None:
            quiescent = self._reactions.wait_quiescent(
                max(deadline - time.monotonic(), 0.0), interrupted=self._interrupted
            )
            if not quiescent:
                with self._lock:
                    self._raise_if_terminal()
                raise self._timed_out("its reactions")
        in_flight = self.session.controls.in_flight.describe()
        if in_flight:
            raise ContractError(
                f"run {self.run_id} reached its update boundary with work still "
                f"counted in flight per control version: {in_flight}"
            )

    def _interrupted(self) -> bool:
        interrupted = (
            self._stop_requested or self._failure is not None or self._cancelled
        )

        return interrupted

    def _timed_out(self, what: str) -> UpdateTimeoutError:
        error = UpdateTimeoutError(
            f"run {self.run_id} did not settle {what} within the timeout; the "
            "pause was lifted and the run keeps its current graph"
        )

        return error

    def _frontiers(self) -> Dict[str, int]:
        """Next ordinal per source and operator domain at the boundary."""
        frontiers = self._driver.frontiers()
        for name, slot in self._executor.operators.items():
            frontiers[name] = slot.next_sequence

        return frontiers

    def _check_boundary(self, token: _UpdateToken, *, deadline: float) -> None:
        """The commit's last check of this run; raises to reject the commit.

        The session calls it under its commit locks and this run's lock,
        with the run paused and drained, right before ``_rebind`` and the
        publication under the same hold of the lock: a stop, cancel or
        failure of the run is decided either before this check or after the
        publication, never in between.
        """
        if self._update_token is not token:
            raise UpdateConflictError(
                f"the update of run {self.run_id} was withdrawn before its commit"
            )
        self._raise_if_terminal()
        if time.monotonic() > deadline:
            raise self._timed_out("its commit")
        if self._driver.readers_finished():
            raise UpdateConflictError(
                f"every source of run {self.run_id} ended before the update "
                "boundary; the run completes with its current graph"
            )

    def _rebind(
        self,
        retirement: None,
        *,
        registered: List[Registered],
        callbacks: Dict[str, GroupHandler],
        frontiers: Mapping[str, int],
    ) -> None:
        """Bind this run to the new graph: assignments only, nothing raises.

        The session calls it after ``_check_boundary`` passed, under the
        same hold of this run's lock, and publishes the generation next.
        A preserving update replaces nothing, so ``retirement`` is ``None``.
        """
        self._executor.rebind(registered)
        self._handlers = dict(callbacks)
        if isinstance(self._coordination, PipelinedCoordination):
            self._coordination.seed_frontiers(frontiers)

    def _release_boundary(self, token: _UpdateToken) -> float:
        """Lift this token's pause, then give the token up.

        The token is cleared last: no other update can reserve the run while
        its driver or reactions are still paused for this one, and a pause
        that already belongs to another token is left alone. Terminal state
        set meanwhile is never undone. Returns when admission resumed
        (``time.monotonic``).
        """
        self._driver.resume(token)
        resumed = time.monotonic()
        if self._reactions is not None:
            self._reactions.resume_ingress(token)
        with self._lock:
            if self._update_token is token:
                self._update_token = None

        return resumed

    def _boundary_reached(self, token: _UpdateToken) -> None:
        """The driver: nothing admitted runs or waits any more (driver thread)."""
        with self._lock:
            if self._update_token is token:
                token.acked = True
                self._changed.notify_all()

    def _boundary_rejected(self, token: _UpdateToken, reason: str) -> None:
        """The driver: the run concludes before the boundary (driver thread)."""
        with self._lock:
            if self._update_token is token:
                token.rejected = reason
                self._changed.notify_all()

    def _reaction_runtime(
        self, plan: Any, *, sessions: Mapping[Any, Any], managed_state: Any
    ) -> Optional[ReactionRuntime]:
        """The run's reactions, or ``None``: no locks or threads without them."""
        reactions = plan.reactions
        if not (reactions.handlers or reactions.machines or reactions.signals):
            return None

        def deliver(group: str, result: GroupResult) -> None:
            # Read at delivery: an applied update replaces self._handlers.
            callback = self._handlers.get(group)
            if callback is not None:
                callback(result)

        def fail(raised: BaseException, group: Optional[str]) -> None:
            stage = "handler" if group is not None else "observer"
            failure = ActiveRunError(
                f"{type(raised).__name__}: {raised}", stage=stage, group=group
            )
            failure.__cause__ = raised
            self._fail(failure)

        runtime = ReactionRuntime(
            plan,
            sessions=sessions,
            session_id=self.session.session_id,
            managed_state=managed_state,
            observer=getattr(self.session, "reaction_observer", None),
            owned=self._owned,
            active_run_id=self.run_id,
            deliver=deliver if reactions.groups else None,
            fail=fail,
            graph_version=lambda: self.session.graph_version,
            processing_version=lambda: self.session.processing_version,
            # Build a handler group's result only while a callback receives
            # it (an update may add or drop one at a quiescent boundary).
            subscribed=lambda group: group in self._handlers,
        )

        return runtime

    def _launch(self) -> None:
        """Notify the observer, start the processing and ``started``, then readers.

        Before processing starts, sources are constructed but not opened,
        so a failure only closes the constructed operators, releases the
        caller's resources and then the registration; the run stays
        registered until that cleanup settled. Once processing runs, a
        failure of the ``started`` reactions or of starting a reader is a
        failure of the run: the readers that never started are marked done
        so processing does not wait for them, the started readers stop
        cooperatively and close their sources, and the run is drained before
        the attributed failure is raised.
        """
        try:
            self._observer.on_run_started(
                session_id=self.session.session_id, run_id=self.run_id
            )
            self._driver.start()
        except Exception as raised:
            failure = attributed(raised, stage="start")
            for slot in self._executor.operators.values():
                close_error = close_operator(slot)
                if close_error is not None:
                    failure.suppressed += (close_error,)
            self._failure = failure
            self._release()
            raise failure from raised

        names = list(self._readers)
        if self._reactions is not None:
            try:
                with self._owned.borrow():
                    self._reactions.system("started")
            except Exception as raised:
                self._fail(attributed(raised, stage="start"))
                for name in names:
                    self._driver.reader_finished(name)
                self._done.wait()
                raise self._failure from raised

        for position, name in enumerate(names):
            try:
                self._readers[name].start()
            except Exception as raised:
                self._fail(attributed(raised, stage="start", source=name))
                for unstarted in names[position:]:
                    self._driver.reader_finished(unstarted)
                # Started readers are cooperative by contract; a source stuck
                # in a non-cooperative read keeps this wait open, as wait() would.
                self._done.wait()
                raise self._failure from raised

    def _unregister(self) -> None:
        with _ACTIVE_RUNS_LOCK:
            if _ACTIVE_RUNS.get(self.session) is self:
                del _ACTIVE_RUNS[self.session]

    # Reader threads -------------------------------------------------------

    def _read(self, slot: _SourceSlot) -> None:
        self._owned.mark()
        error: Optional[ActiveRunError] = None
        try:
            if not self.stop_event.is_set():
                self._open(slot)
            while not self.stop_event.is_set():
                emission = self._read_one(slot)
                if emission is None:
                    slot.counters.ended = True
                    break
                slot.counters.read += 1
                if not self._driver.admit(
                    slot, emission, observed=engine_observation()
                ):
                    break
        except Exception as raised:
            error = attributed(raised, stage="observer", source=slot.name)
            self._fail(error)
        finally:
            try:
                close_error = self._close(slot)
                self._observer.on_source_closed(
                    source=slot.name, error=error if error is not None else close_error
                )
            except Exception as raised:
                self._fail(attributed(raised, stage="observer", source=slot.name))
            finally:
                self._driver.reader_finished(slot.name)

    def _open(self, slot: _SourceSlot) -> None:
        slot.open_attempted = True
        try:
            returned = slot.instance.open(**slot.arguments)
        except Exception as error:
            raise ActiveRunError(
                f"open raised {type(error).__name__}: {error}",
                stage="open",
                source=slot.name,
            ) from error
        if _discard_awaitable(returned):
            raise ActiveRunError(
                "open() returned an awaitable that was never run; source lifecycle "
                "calls are synchronous",
                stage="open",
                source=slot.name,
            )
        slot.counters.opened = True
        self._observer.on_source_opened(source=slot.name)

    def _read_one(self, slot: _SourceSlot) -> Optional[Emission]:
        try:
            emission = slot.instance.read()
        except Exception as error:
            raise ActiveRunError(
                f"read raised {type(error).__name__}: {error}",
                stage="read",
                source=slot.name,
            ) from error
        if _discard_awaitable(emission):
            raise ActiveRunError(
                "read() returned an awaitable that was never run; source lifecycle "
                "calls are synchronous",
                stage="read",
                source=slot.name,
            )
        if emission is not None and not isinstance(emission, Emission):
            raise ActiveRunError(
                f"read() must return an Emission or None (end of source), got "
                f"{type(emission).__name__}",
                stage="read",
                source=slot.name,
            )

        return emission

    def _close(self, slot: _SourceSlot) -> Optional[ActiveRunError]:
        """Close a source whose ``open`` was attempted; exactly once per run."""
        if not slot.open_attempted:
            return None

        try:
            returned = slot.instance.close()
        except Exception as error:
            failure = ActiveRunError(
                f"close raised {type(error).__name__}: {error}",
                stage="close",
                source=slot.name,
            )
            failure.__cause__ = error
            self._fail(failure)
            return failure
        if _discard_awaitable(returned):
            failure = ActiveRunError(
                "close() returned an awaitable that was never run; the source was "
                "not cleaned up",
                stage="close",
                source=slot.name,
            )
            self._fail(failure)
            return failure
        slot.counters.closed = True

        return None

    # Driver side -------------------------------------------------------------
    # The members a driver may use and its obligations are listed on ``_Driver``.

    def _admitted(
        self, slot: _SourceSlot, emission: Emission, *, observed: Timestamp
    ) -> SourcePulse:
        """Turn a read emission into the source's next pulse (admission lock held)."""
        key = PulseKey(
            active_run_id=self.run_id, source=slot.name, sequence=slot.next_sequence
        )
        slot.next_sequence += 1
        slot.in_flight += 1
        slot.counters.admitted += 1
        slot.counters.peak_admitted = max(slot.counters.peak_admitted, slot.in_flight)
        self._executor.progress.add(slot.name)
        # The control snapshot is taken here, under the admission lock: pulses
        # admitted before an update keep the old version, later ones the new.
        pulse = SourcePulse(
            key=key,
            emission=emission,
            observed=observed,
            controls=self.session.controls.current,
        )
        # Counted in flight until the driver reports it processed or cancelled.
        self.session.controls.in_flight.enter(pulse.controls.version)

        return pulse

    def _termination(self, source: str) -> TerminationReason:
        """``eof`` when the source reached its end, ``stop`` when it was stopped."""
        ended = self._slots[source].counters.ended

        return "eof" if ended else "stop"

    def _end_domain_later(self, domain: str, reason: TerminationReason) -> None:
        self._driver.end_domain_later(domain, reason)

    def _conclude(self) -> None:
        """Settle reactions, close every operator, notify, release the session.

        After a completed or stopped run, ``ended`` is published and the
        reactions drain, cascades included. After a failure or cancellation
        queued events are discarded. Either way the run waits for every
        accepted reaction call still running (signal callers' threads
        included) and joins the workers before it is done and releases the
        session, so no handler outlives it.
        """
        try:
            if self._reactions is not None:
                if self.aborting:
                    # Idempotent; the aborting thread may not have reached it yet.
                    self._reactions.cancel()
                else:
                    self._end_reactions()
                self._reactions.settle()
                self._reactions.join()
            self._executor.close_operators()
            self._finish_capture()
            self._observer.on_run_finished(
                run_id=self.run_id, result=None, error=self._failure
            )
        except Exception as raised:
            self._fail(attributed(raised, stage="observer"))
        finally:
            try:
                self._executor.close_operators()
            finally:
                self._release()

    def _release(self) -> None:
        """Release the caller's resources, then the session; the run is done.

        Called once the run no longer uses the session: workers, reactions
        and operators have settled. ``releasing`` turns true first, so a
        replay's ``finalize`` may close the session. The registration ends
        and ``done`` is set on every exit. A release error is recorded before
        waiters can observe completion, beneath any earlier run failure.
        """
        self._releasing = True
        try:
            self._settle_resources()
        except Exception as raised:
            self._fail(attributed(raised, stage="finalize"))
        finally:
            self._unregister()
            self._done.set()
            with self._lock:
                self._changed.notify_all()

    def _settle_resources(self) -> None:
        """Finalize the recording, then release what the caller made for the run."""
        try:
            self._finish_capture()
        finally:
            if self._finalize is not None:
                finalize, self._finalize = self._finalize, None
                finalize()

    def _finish_capture(self) -> None:
        """Finalize the recording once, with the status the run ended in.

        ``failed`` and ``cancelled`` follow the run; otherwise it is
        ``stopped`` when ``stop()`` was requested before every source had
        reached its end, and ``complete`` when they all ended on their own.
        """
        if self._capture is None or self._finished_capture:
            return

        self._finished_capture = True
        error = None
        if self._failure is not None:
            status, error = "failed", str(self._failure)
        elif self._cancelled:
            status = "cancelled"
        elif self._stopped_early:
            status = "stopped"
        else:
            status = "complete"
        try:
            self._capture.finish(status, error=error)
        except Exception as raised:
            failure = ActiveRunError(
                f"finishing the recording raised {type(raised).__name__}: {raised}",
                stage="recording",
            )
            failure.__cause__ = raised
            self._fail(failure)

    def _end_reactions(self) -> None:
        """Publish ``ended``, then drain; a failure of ``ended`` fails the run."""
        try:
            self._reactions.system("ended")
        except Exception as raised:
            self._fail(attributed(raised, stage="handler"))
            return

        self._reactions.drain()

    # Shared ---------------------------------------------------------------

    def _close_admission(self) -> None:
        """Stop admitting pulses and wake every reader waiting for a slot."""
        if not self._driver.close_admission():
            return

        self.stop_event.set()
        for slot in self._slots.values():
            slot.admission.release()

    def _abort(self, cause: Optional[ActiveRunError] = None) -> None:
        """Close admission and wake every gate, so nothing new starts.

        ``cause`` is the run's failure; ``None`` for a cancellation.
        """
        self._close_admission()
        if self._reactions is not None:
            self._reactions.cancel()
        self._coordination.abort(cause)
        self._driver.wake()

    def _fail(self, failure: ActiveRunError) -> None:
        """Record the first failure as terminal, later ones as suppressed."""
        with self._lock:
            if self._failure is None:
                self._failure = failure
            elif failure is not self._failure:
                self._failure.suppressed += (failure,)
            self._changed.notify_all()
        self._abort(self._failure)


class _SerialDriver:
    """Readers queue admitted pulses; one processor thread runs them in order.

    The queue holds at most ``admission_bound`` pulses per source, because
    a reader takes one of its source's slots before admitting.

    A graph update pauses the driver: readers that hold a slot wait before
    admitting, a ``_Barrier`` is queued behind everything admitted so far,
    and the processor parks once it reaches it. A source that ends while
    paused is remembered and sequenced at ``resume``, so the run cannot
    conclude, nor an operator finish early, across the boundary.
    """

    def __init__(self, run: ActiveRun):
        self._run = run
        # Guards admission, the pause and the processor's parking.
        self._condition = threading.Condition()
        self._admission_open = True
        self._paused = False
        self._token: Optional[_UpdateToken] = None
        self._finished: Set[str] = set()
        self._deferred: List[str] = []
        self._replayed: List[str] = []
        self._queue: "queue.Queue[Union[SourcePulse, _ReaderDone, _Barrier]]" = (
            queue.Queue()
        )
        # Snapshot of the source end being processed; chained domain ends
        # (an operator finishing) run under it.
        self._ending: Optional[ControlSnapshot] = None
        self._processor = threading.Thread(
            target=self._process,
            name=f"workflows-v2-run-{run.run_id[:8]}",
            daemon=True,
        )

    def start(self) -> None:
        self._processor.start()

    def admit(
        self, slot: _SourceSlot, emission: Emission, *, observed: Timestamp
    ) -> bool:
        """Take one of the source's slots and queue the pulse.

        Returns ``False`` once admission is closed; the emission is then
        counted unadmitted and the reader stops. While a graph update pauses
        the run, the reader keeps the slot and the emission and waits here.
        """
        slot.admission.acquire()
        with self._condition:
            while self._paused and self._admission_open:
                self._condition.wait()
            if not self._admission_open:
                slot.counters.unadmitted += 1
                return False
            self._queue.put(self._run._admitted(slot, emission, observed=observed))

        return True

    def reader_finished(self, source: str) -> None:
        with self._condition:
            self._finished.add(source)
            if self._paused:
                # Sequenced at resume, under the graph published by then.
                self._deferred.append(source)
                return
            self._queue_done(source)

    def _queue_done(self, source: str, *, replay: bool = False) -> None:
        # The end of a source is sequenced like a pulse: its snapshot is taken
        # under the admission lock, so the queue stays version-monotonic.
        controls = self._run.session.controls.current
        self._run.session.controls.in_flight.enter(controls.version)
        self._queue.put(_ReaderDone(source, controls, replay=replay))

    def close_admission(self) -> bool:
        """Close admission; ``True`` only for the call that closed it."""
        with self._condition:
            if not self._admission_open:
                return False
            self._admission_open = False
            self._condition.notify_all()

        return True

    def end_domain_later(self, domain: str, reason: TerminationReason) -> None:
        # Serially there is no later: the processor ends the domain right away,
        # under the snapshot of the source end it is processing. Every serial
        # domain end is chained from a source end; taking the current snapshot
        # instead could carry a newer reset epoch and make this thread drain
        # (rule 4) for work it is itself running.
        if self._ending is None:
            raise ContractError(
                f"domain {domain!r} ended outside the processing of a source end; "
                "the serial driver has no admitted control snapshot for it"
            )
        self._run._executor.end_domain(domain, reason, controls=self._ending)

    def wake(self) -> None:
        """Wake parked readers and the processor so they observe the abort."""
        with self._condition:
            self._condition.notify_all()

    def pause(self, token: _UpdateToken) -> None:
        with self._condition:
            self._paused = True
            self._token = token
            self._queue.put(_Barrier(token))

    def resume(self, token: _UpdateToken) -> None:
        with self._condition:
            if self._token is not token:
                return
            self._paused = False
            self._token = None
            replayed, self._replayed = self._replayed, []
            for source in replayed:
                self._queue_done(source, replay=True)
            deferred, self._deferred = self._deferred, []
            for source in deferred:
                self._queue_done(source)
            self._condition.notify_all()

    def restart_domains(self) -> None:
        with self._condition:
            self._replayed = [
                name
                for name in self._run._slots
                if name in self._finished and name not in self._deferred
            ]

    def settle_workers(self, timeout: float) -> bool:
        """The processor is the only worker, and it is parked at the barrier."""
        return True

    def readers_finished(self) -> bool:
        with self._condition:
            finished = len(self._finished) == len(self._run._slots)

        return finished

    def frontiers(self) -> Dict[str, int]:
        with self._condition:
            frontiers = {
                name: slot.next_sequence for name, slot in self._run._slots.items()
            }

        return frontiers

    def _process(self) -> None:
        run = self._run
        run._owned.mark()
        executor = run._executor
        try:
            remaining = len(run._slots)
            in_flight = run.session.controls.in_flight
            while remaining:
                item = self._queue.get()
                if isinstance(item, _Barrier):
                    self._park(item.token)
                    continue
                if isinstance(item, _ReaderDone):
                    remaining -= 0 if item.replay else 1
                    self._ending = item.controls
                    try:
                        if not run.aborting:
                            executor.seal(item.source, run._termination(item.source))
                    finally:
                        self._ending = None
                        in_flight.leave(item.controls.version)
                else:
                    slot = run._slots[item.key.source]
                    try:
                        if run.aborting:
                            executor.count(slot.counters, "cancelled")
                        else:
                            executor.run_source_pulse(item, counters=slot.counters)
                    finally:
                        in_flight.leave(item.controls.version)
                    with self._condition:
                        slot.in_flight -= 1
                    slot.admission.release()
                if run.aborting:
                    # Release what operators retain now, not after the readers.
                    executor.close_operators()
            self._reject_pending_barriers()
        except Exception as raised:
            run._fail(attributed(raised, stage="observer"))
        finally:
            run._conclude()

    def _park(self, token: _UpdateToken) -> None:
        """Everything queued before the barrier completed: report, then park.

        A withdrawn token (the update timed out or failed) is passed over.
        Parking ends at ``resume``; a stop or abort meanwhile is observed by
        the updater, which resumes the driver.
        """
        with self._condition:
            if self._token is not token:
                return
        self._run._boundary_reached(token)
        with self._condition:
            while self._paused and self._token is token:
                self._condition.wait()

    def _reject_pending_barriers(self) -> None:
        """Every source ended before a queued barrier: the run concludes instead."""
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return
            if isinstance(item, _Barrier):
                self._run._boundary_rejected(
                    item.token, "concluded before the update boundary"
                )
