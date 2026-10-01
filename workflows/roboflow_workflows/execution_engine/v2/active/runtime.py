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

Serially, block calls and handlers run on the processor thread only, so
ordinary stateful blocks are never reentered. Pipelined, one step (or one
phase, with ``phase_overlap``) is never entered by two pulses at once, and
every handler and observer callback of the run is serialized.
"""

import importlib
import inspect
import queue
import threading
import uuid
import weakref
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Union,
)

from roboflow_workflows.execution_engine.v2.active.execution import engine_observation
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
from roboflow_workflows.execution_engine.v2.data import Timestamp
from roboflow_workflows.execution_engine.v2.errors import (
    ActiveRunError,
    ContractError,
    WorkflowInputError,
)
from roboflow_workflows.execution_engine.v2.execution.arguments import (
    arguments_for,
    validation_arguments,
)
from roboflow_workflows.execution_engine.v2.execution.entries import Entry
from roboflow_workflows.execution_engine.v2.execution.inputs import prepare_inputs
from roboflow_workflows.execution_engine.v2.execution.outputs import GroupResult
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
from roboflow_workflows.execution_engine.v2.sources import Emission

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


class _Driver(Protocol):
    """How an ``ActiveRun`` admits and executes pulses: serially or pipelined.

    ``ActiveRun`` and its readers call these six methods. A driver may use
    from the run: ``run_id``, ``aborting``, ``pipeline_counters``,
    ``_slots``, ``_readers``, ``_owned`` (mark its own threads),
    ``_executor`` (pulses, seals and domain ends), ``_admitted`` (under its
    admission lock), ``_termination``, ``_fail`` and ``_conclude``.

    Obligations of every driver:

    * call ``_conclude`` exactly once, after every reader called
      ``reader_finished`` and no pulse or domain end runs;
    * after an abort, close the operators (``_executor.close_operators()``)
      as soon as no work runs, without waiting for the readers;
    * call ``_conclude`` from its own thread. The pipelined driver concludes
      on its dispatcher after shutting down its workers and joining the
      readers. The serial driver concludes on its processor thread once
      every reader reported finished; it does not join the readers.
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
) -> "ActiveRun":
    """Validate, construct the sources and start reading.

    Handlers, static inputs, pipeline options and every source's resolved
    parameters are checked and every source instance is constructed before
    any reader opens anything.

    Args:
        session: Session of an active plan (one with declared sources).
        inputs: Static input values by name; the plan's ungrouped inputs.
        handlers: Synchronous callback per output group name. Groups without
            a handler are neither built nor retained; their steps still run.
        admission_bound: Pulses one source may have admitted but not yet
            processed; at least 1.
        pipeline: ``None`` runs serially (the reference). Options run the
            pulses on ``max_in_flight`` workers with per-stage order and the
            sources' overload policies.

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
        entries = prepare_inputs(plan, inputs or {})
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
    unknown = sorted(set(handlers) - set(declared))
    if unknown:
        raise ContractError(
            f"Handlers are registered for unknown output groups {unknown}; "
            f"declared groups: {list(declared)}"
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
    ):
        self.session = session
        self.run_id = run_id
        self.stop_event = stop_event
        self._slots = slots
        self._lock = threading.Lock()
        self._failure: Optional[ActiveRunError] = None
        self._cancelled = False
        self._done = threading.Event()
        self._owned = OwnedThreads()
        self._coordination: Coordination = (
            PipelinedCoordination(session, options=pipeline)
            if pipeline is not None
            else SERIAL
        )
        self._observer = self._coordination.observer(session)
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
    def done(self) -> bool:
        """Whether every source is closed and every admitted pulse handled."""
        return self._done.is_set()

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

    def stop(self) -> None:
        """Close admission and ask the sources to stop; never blocks.

        Admitted pulses are still processed and delivered. Calling it again,
        or from inside a handler, has no further effect.
        """
        self._close_admission()

    def cancel(self) -> None:
        """Stop without draining; never blocks.

        Admission closes, admitted pulses that have not started are
        cancelled, running pulses stop at their next step, delivery or
        operator boundary, and operators are closed without finishing.
        Calls already running are not interrupted. ``wait()`` then returns
        ``True`` with state ``cancelled``. No effect once the run failed or
        is done.
        """
        with self._lock:
            if self._failure is not None or self._done.is_set():
                return
            self._cancelled = True
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

    def _launch(self) -> None:
        """Notify the observer and start the processing, then every reader.

        Before processing starts, sources are constructed but not opened,
        so a failure only closes the constructed operators and releases the
        registration. Once readers are launching, a failure to start one of
        them is a failure of the run: the readers that never started are
        marked done so processing does not wait for them, the started
        readers stop cooperatively and close their sources, and the run is
        drained before the attributed failure is raised.
        """
        try:
            self._observer.on_run_started(
                session_id=self.session.session_id, run_id=self.run_id
            )
            self._driver.start()
        except Exception as raised:
            self._unregister()
            failure = attributed(raised, stage="start")
            for slot in self._executor.operators.values():
                close_error = close_operator(slot)
                if close_error is not None:
                    failure.suppressed += (close_error,)
            raise failure from raised

        names = list(self._readers)
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
        pulse = SourcePulse(key=key, emission=emission, observed=observed)

        return pulse

    def _termination(self, source: str) -> TerminationReason:
        """``eof`` when the source reached its end, ``stop`` when it was stopped."""
        ended = self._slots[source].counters.ended

        return "eof" if ended else "stop"

    def _end_domain_later(self, domain: str, reason: TerminationReason) -> None:
        self._driver.end_domain_later(domain, reason)

    def _conclude(self) -> None:
        """Close every operator, notify ``on_run_finished``, release the session."""
        try:
            self._executor.close_operators()
            self._observer.on_run_finished(
                run_id=self.run_id, result=None, error=self._failure
            )
        except Exception as raised:
            self._fail(attributed(raised, stage="observer"))
        finally:
            self._executor.close_operators()
            self._unregister()
            self._done.set()

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
        self._coordination.abort(cause)
        self._driver.wake()

    def _fail(self, failure: ActiveRunError) -> None:
        """Record the first failure as terminal, later ones as suppressed."""
        with self._lock:
            if self._failure is None:
                self._failure = failure
            elif failure is not self._failure:
                self._failure.suppressed += (failure,)
        self._abort(self._failure)


class _SerialDriver:
    """Readers queue admitted pulses; one processor thread runs them in order.

    The queue holds at most ``admission_bound`` pulses per source, because
    a reader takes one of its source's slots before admitting.
    """

    def __init__(self, run: ActiveRun):
        self._run = run
        self._lock = threading.Lock()
        self._admission_open = True
        self._queue: "queue.Queue[Union[SourcePulse, _ReaderDone]]" = queue.Queue()
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
        counted unadmitted and the reader stops.
        """
        slot.admission.acquire()
        with self._lock:
            if not self._admission_open:
                slot.counters.unadmitted += 1
                return False
            self._queue.put(self._run._admitted(slot, emission, observed=observed))

        return True

    def reader_finished(self, source: str) -> None:
        self._queue.put(_ReaderDone(source))

    def close_admission(self) -> bool:
        """Close admission; ``True`` only for the call that closed it."""
        with self._lock:
            if not self._admission_open:
                return False
            self._admission_open = False

        return True

    def end_domain_later(self, domain: str, reason: TerminationReason) -> None:
        # Serially there is no later: the processor ends the domain right away.
        self._run._executor.end_domain(domain, reason)

    def wake(self) -> None:
        """Nothing waits on anything but the queue, which readers always feed."""

    def _process(self) -> None:
        run = self._run
        run._owned.mark()
        executor = run._executor
        try:
            remaining = len(run._slots)
            while remaining:
                item = self._queue.get()
                if isinstance(item, _ReaderDone):
                    remaining -= 1
                    if not run.aborting:
                        executor.seal(item.source, run._termination(item.source))
                else:
                    slot = run._slots[item.key.source]
                    if run.aborting:
                        executor.count(slot.counters, "cancelled")
                    else:
                        executor.run_source_pulse(item, counters=slot.counters)
                    with self._lock:
                        slot.in_flight -= 1
                    slot.admission.release()
                if run.aborting:
                    # Release what operators retain now, not after the readers.
                    executor.close_operators()
        except Exception as raised:
            run._fail(attributed(raised, stage="observer"))
        finally:
            run._conclude()
