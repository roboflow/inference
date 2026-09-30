"""Active runtime: independent source readers, one sequential processor.

``ExecutionSession.start`` delegates here::

    run = start_session(session, inputs=static, handlers={"frames": on_frames})
    ...
    run.stop()      # optional: close admission, drain what was admitted
    run.wait()      # True when complete; raises the run's failure

Threads of one active run::

    reader[S]   (one per source)     processor   (one per run)
    ----------------------------     -----------------------------------------
    open(**params)                   item = queue.get()
    loop: emission = read()          pulse of S:
          None -> end of S             run = begin_pulse(...)      fresh RunState
          admit(S, emission)           for step in plan.route(S):  plan order
    close()  exactly once                  execute_step; deliver groups now ready
    queue "S done"                     feed operators of S (below)
                                       release S's admission slot
                                     "S done": end S's inputs of its operators

Operators turn pulses of upstream domains into pulses of their own domain,
depth first on the processor thread::

    feed(run of domain D):
        for operator in plan.consumers_of(D):          plan order
            emitted = operator.push(arrivals of D's pulse)
            for each emitted pulse P of the operator:
                run = begin_operator_pulse(...)        fresh RunState, causes
                route of the operator, groups, then feed(run)

    end(D):
        for operator in plan.consumers_of(D):
            end_input(each input of D) -> pulses run as above
            all upstream domains ended -> finish once -> pulses run, end(operator)

Admission is the acceptance boundary: a reader takes one of its source's
``admission_bound`` slots and, under the run's lock, queues the pulse. The
slot is released after the pulse's handlers returned and the operators it
fed returned and their immediate pulses ran; what an operator retains
afterwards is bounded by the operator's own parameters, not by admission.
Nothing waits for future data: an operator returns what it can decide
now. A reader holds at most
one read-but-unadmitted emission, so per source at most ``admission_bound + 1``
emissions exist outside the source at any time, and a slow synchronous
handler backpressures every reader instead of growing a queue. Readers never
wait for each other: a source blocked in ``read()`` does not delay another
source's pulses.

A registered group is delivered as soon as every step its fields need has
run in the pulse (``PlannedOutputGroup.dependencies``), so a group reading the source
directly is delivered before an unrelated later action of the same route;
that action still runs once. An explicitly filtered emission delivers every
registered group of the source at pulse start, all fields filtered.

Lifecycle:

* End of one source ends only its reader. Its end marker follows its
  admitted pulses through the queue; the operators it feeds then learn that
  its inputs ended, and an operator whose upstream domains all ended
  finishes once (``eof``) and ends its own domain after its final pulses.
  The run completes when every reader has closed its source, every admitted
  pulse was processed and every operator was closed.
* ``stop()`` closes admission and sets the shared ``stop_event`` the sources
  see. A reader whose ``read()`` returns afterwards discards that emission
  and closes. Admitted pulses are still processed and delivered exactly once;
  operators then finish with reason ``stop`` and apply their partial policies.
* A failure (open, read, emission, step, handler, observer, operator or
  close) closes admission, cancels admitted pulses not yet processed and the
  remaining steps, deliveries and operator pulses of the pulse in flight,
  closes every source whose ``open`` was attempted and every operator
  without finishing it (no partial emission), and is raised by ``wait()`` as
  one ``ActiveRunError``; later errors are kept in its ``suppressed``.
  Handlers already called are not undone.
* ``stop()`` never blocks and is safe inside a handler; ``wait()`` inside a
  handler raises instead of waiting for itself. A handler may run before
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

Block calls and handlers run on the processor thread only, so ordinary
stateful blocks are never reentered. This is sequential processing with
independent acquisition, not a pipelined scheduler.
"""

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
    FrozenSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Union,
)

from roboflow_workflows.execution_engine.v2.active.execution import (
    abandon_pulse,
    begin_operator_pulse,
    begin_pulse,
    engine_observation,
    group_result,
    operator_arrivals,
)
from roboflow_workflows.execution_engine.v2.context import (
    ExecutionContext,
    use_execution_context,
)
from roboflow_workflows.execution_engine.v2.data import Timestamp
from roboflow_workflows.execution_engine.v2.errors import (
    ActiveRunError,
    ActiveRunStage,
    ContractError,
    StepExecutionError,
    StepPath,
    WorkflowInputError,
)
from roboflow_workflows.execution_engine.v2.execution.arguments import (
    arguments_for,
    validation_arguments,
)
from roboflow_workflows.execution_engine.v2.execution.entries import Entry
from roboflow_workflows.execution_engine.v2.execution.inputs import prepare_inputs
from roboflow_workflows.execution_engine.v2.execution.outputs import GroupResult
from roboflow_workflows.execution_engine.v2.execution.steps import (
    RunState,
    execute_step,
)
from roboflow_workflows.execution_engine.v2.operators import (
    Operator,
    OperatorCounters,
    OperatorPulse,
    TerminationReason,
)
from roboflow_workflows.execution_engine.v2.plan import (
    Binding,
    Constant,
    ExecutionSession,
    PlannedOperator,
    PlannedOutputGroup,
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

GroupHandler = Callable[[GroupResult], None]
"""Synchronous callback receiving one ``GroupResult`` per delivered pulse."""


@dataclass
class SourceCounters:
    """What happened to one source during one active run.

    Args:
        read: Emissions ``read()`` returned (end of source excluded).
        admitted: Emissions that became pulses.
        unadmitted: Emissions read after admission closed and discarded.
        processed: Pulses whose route ran and whose handlers returned.
        delivered: Group results handed to handlers that returned.
        cancelled: Admitted pulses that did not complete: dropped after a
            failure, cut short by one, or the pulse that failed itself.
            ``admitted == processed + cancelled`` once the run is done.
        ended: Whether ``read()`` returned ``None``.
        opened: Whether ``open()`` returned.
        closed: Whether ``close()`` returned.
    """

    read: int = 0
    admitted: int = 0
    unadmitted: int = 0
    processed: int = 0
    delivered: int = 0
    cancelled: int = 0
    ended: bool = False
    opened: bool = False
    closed: bool = False


@dataclass
class _SourceSlot:
    """One source of the run: its instance, resolved ``open`` arguments and state."""

    planned: PlannedSource
    instance: Any
    arguments: Mapping[str, Any]
    admission: threading.Semaphore
    counters: SourceCounters = field(default_factory=SourceCounters)
    open_attempted: bool = False
    next_sequence: int = 0

    @property
    def name(self) -> str:
        return self.planned.name


@dataclass
class _OperatorSlot:
    """One operator of the run: its instance and which upstream domains ended."""

    planned: PlannedOperator
    instance: Operator
    next_sequence: int = 0
    ended: Set[str] = field(default_factory=set)
    stopped: bool = False
    close_attempted: bool = False
    error: Optional[ActiveRunError] = None

    @property
    def name(self) -> str:
        return self.planned.name

    @property
    def counters(self) -> OperatorCounters:
        return self.instance.counters


@dataclass(frozen=True)
class _Outcome:
    """How a pulse, or the operator work it fed, ended.

    Args:
        completed: Whether everything ran and every handler returned.
        error: The failure raised by this work, already recorded as the run's
            failure (or suppressed); ``None`` when it completed or a failure
            elsewhere cut it short.
    """

    completed: bool
    error: Optional[ActiveRunError] = None


_COMPLETED = _Outcome(completed=True)
_CANCELLED = _Outcome(completed=False)


@dataclass(frozen=True)
class _Pulse:
    key: PulseKey
    emission: Emission
    observed: Timestamp


@dataclass(frozen=True)
class _ReaderDone:
    source: str


@dataclass(frozen=True)
class _Registered:
    """An output group with a handler and the steps its fields wait for."""

    group: PlannedOutputGroup
    handler: GroupHandler
    prerequisites: FrozenSet[StepPath]


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
) -> "ActiveRun":
    """Validate, construct the sources and start reading.

    Handlers, static inputs and every source's resolved parameters are
    checked and every source instance is constructed before any reader
    opens anything.

    Args:
        session: Session of an active plan (one with declared sources).
        inputs: Static input values by name; the plan's ungrouped inputs.
        handlers: Synchronous callback per output group name. Groups without
            a handler are neither built nor retained; their steps still run.
        admission_bound: Pulses one source may have admitted but not yet
            processed; at least 1.

    Returns:
        The running active run.

    Raises:
        WorkflowInputError: When the plan has no sources or the inputs are
            invalid.
        ContractError: When a handler names an unknown group, is not
            callable, is a coroutine function, the bound is invalid or the
            session already has an unfinished active run.
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
) -> List[_Registered]:
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
        _Registered(
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
) -> Dict[str, _OperatorSlot]:
    """Construct a fresh instance of every operator; close them all on failure."""
    prepared: Dict[str, _OperatorSlot] = {}
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
                close_error = _close_operator(slot)
                if close_error is not None:
                    failure.suppressed += (close_error,)
            raise failure
        prepared[planned.name] = _OperatorSlot(planned=planned, instance=instance)

    return prepared


def _close_operator(slot: _OperatorSlot) -> Optional[ActiveRunError]:
    """Close an operator exactly once; return what ``close`` raised."""
    if slot.close_attempted:
        return None

    slot.close_attempted = True
    try:
        slot.instance.close()
    except Exception as error:
        failure = ActiveRunError(
            f"close raised {type(error).__name__}: {error}",
            stage="operator",
            operator=slot.name,
        )
        failure.__cause__ = error
        return failure
    slot.counters.closed = True

    return None


def _static_value(binding: Binding, entries: Mapping[str, Entry]) -> Any:
    """Value of a source parameter selector: a literal or an ungrouped static input."""
    if isinstance(binding.source, Constant):
        return binding.source.value

    value = entries[binding.source.name].values[()]

    return value


def _attributed(
    raised: Exception,
    *,
    stage: ActiveRunStage,
    source: Optional[str] = None,
    operator: Optional[str] = None,
    pulse: Optional[int] = None,
) -> ActiveRunError:
    """Wrap an exception of the run into its attributed terminal error."""
    if isinstance(raised, ActiveRunError):
        return raised
    if isinstance(raised, StepExecutionError):
        error = ActiveRunError(
            str(raised),
            stage="step",
            source=source,
            operator=operator,
            pulse=pulse,
            step_path=raised.step_path,
        )
    else:
        error = ActiveRunError(
            f"{type(raised).__name__}: {raised}",
            stage=stage,
            source=source,
            operator=operator,
            pulse=pulse,
        )
    error.__cause__ = raised

    return error


class ActiveRun:
    """One execution of an active plan: its readers, processor and outcome.

    Create runs with ``ExecutionSession.start``. The run keeps no result
    history; hosts receive results through their handlers only.

    Args:
        session: Session whose block instances process the pulses.
        run_id: Identity of this run.
        inputs: Static input entries shared by every pulse.
        registered: Output groups with handlers, in plan order.
        slots: Constructed sources by name.
        operators: Constructed operators by name, in plan order.
        stop_event: Event the sources watch; set by ``stop()`` and by failure.
    """

    def __init__(
        self,
        session: ExecutionSession,
        *,
        run_id: str,
        inputs: Mapping[str, Entry],
        registered: List[_Registered],
        slots: Mapping[str, _SourceSlot],
        operators: Mapping[str, _OperatorSlot],
        stop_event: threading.Event,
    ):
        self.session = session
        self.run_id = run_id
        self.stop_event = stop_event
        self._inputs = inputs
        self._registered = registered
        self._slots = slots
        self._operators = operators
        self._lock = threading.Lock()
        self._admission_open = True
        self._failure: Optional[ActiveRunError] = None
        self._queue: "queue.Queue[Union[_Pulse, _ReaderDone]]" = queue.Queue()
        self._done = threading.Event()
        self._processor = threading.Thread(
            target=self._process, name=f"workflows-v2-run-{run_id[:8]}", daemon=True
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
        return {name: slot.counters for name, slot in self._operators.items()}

    @property
    def done(self) -> bool:
        """Whether every source is closed and every admitted pulse handled."""
        return self._done.is_set()

    @property
    def failure(self) -> Optional[ActiveRunError]:
        """The run's terminal failure, once one happened."""
        return self._failure

    @property
    def state(self) -> str:
        """``running``, ``stopping``, ``finished`` or ``failed``."""
        if self._done.is_set():
            return "failed" if self._failure is not None else "finished"
        if self.stop_event.is_set():
            return "stopping"

        return "running"

    def stop(self) -> None:
        """Close admission and ask the sources to stop; never blocks.

        Admitted pulses are still processed and delivered. Calling it again,
        or from inside a handler, has no further effect.
        """
        self._close_admission()

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Wait until the run is done.

        Args:
            timeout: Seconds to wait; forever when ``None``.

        Returns:
            ``True`` when the run completed, ``False`` on timeout.

        Raises:
            ActiveRunError: The run's failure, once it completed failed.
            ContractError: When called from a handler, which would wait for
                itself; call ``stop()`` there instead.
        """
        if threading.current_thread() is self._processor:
            raise ContractError(
                "ActiveRun.wait() was called from an output handler and would wait "
                "for itself; call stop() there and wait() from the host thread"
            )
        if not self._done.wait(timeout):
            return False
        if self._failure is not None:
            raise self._failure

        return True

    def __enter__(self) -> "ActiveRun":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop()
        if exc_type is None:
            self.wait()
        else:
            self._done.wait()

    def _launch(self) -> None:
        """Notify the observer and start the processor, then every reader.

        Before the processor runs, sources are constructed but not opened,
        so a failure only closes the constructed operators and releases the
        registration. Once readers are launching, a failure to start one of
        them is a failure of the run: the readers that never started are
        marked done so the processor does not wait for them, the started
        readers stop cooperatively and close their sources, and the run is
        drained before the attributed failure is raised.
        """
        try:
            self.session.observer.on_run_started(
                session_id=self.session.session_id, run_id=self.run_id
            )
            self._processor.start()
        except Exception as raised:
            self._unregister()
            failure = _attributed(raised, stage="start")
            for slot in self._operators.values():
                close_error = _close_operator(slot)
                if close_error is not None:
                    failure.suppressed += (close_error,)
            raise failure from raised

        names = list(self._readers)
        for position, name in enumerate(names):
            try:
                self._readers[name].start()
            except Exception as raised:
                self._fail(_attributed(raised, stage="start", source=name))
                for unstarted in names[position:]:
                    self._queue.put(_ReaderDone(unstarted))
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
                if not self._admit(slot, emission, observed=engine_observation()):
                    slot.counters.unadmitted += 1
                    break
        except Exception as raised:
            error = _attributed(raised, stage="observer", source=slot.name)
            self._fail(error)
        finally:
            try:
                close_error = self._close(slot)
                self.session.observer.on_source_closed(
                    source=slot.name, error=error if error is not None else close_error
                )
            except Exception as raised:
                self._fail(_attributed(raised, stage="observer", source=slot.name))
            finally:
                self._queue.put(_ReaderDone(slot.name))

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
        self.session.observer.on_source_opened(source=slot.name)

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

    def _admit(
        self, slot: _SourceSlot, emission: Emission, *, observed: Timestamp
    ) -> bool:
        """Take one of the source's slots and queue the pulse under the guard.

        Returns ``False`` once admission is closed; the emission is then
        discarded by the caller.
        """
        slot.admission.acquire()
        with self._lock:
            if not self._admission_open:
                return False
            key = PulseKey(
                active_run_id=self.run_id, source=slot.name, sequence=slot.next_sequence
            )
            slot.next_sequence += 1
            slot.counters.admitted += 1
            self._queue.put(_Pulse(key=key, emission=emission, observed=observed))

        return True

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

    # Processor thread -----------------------------------------------------

    def _process(self) -> None:
        try:
            remaining = len(self._slots)
            while remaining:
                item = self._queue.get()
                if isinstance(item, _ReaderDone):
                    remaining -= 1
                    if self._failure is None:
                        self._end_domain(item.source, reason=self._termination(item))
                else:
                    slot = self._slots[item.key.source]
                    if self._failure is not None:
                        slot.counters.cancelled += 1
                    else:
                        self._process_source_pulse(slot, item)
                    slot.admission.release()
                if self._failure is not None:
                    # Release what operators retain now, not after the readers.
                    self._close_operators()
            self._close_operators()
            self.session.observer.on_run_finished(
                run_id=self.run_id, result=None, error=self._failure
            )
        except Exception as raised:
            self._fail(_attributed(raised, stage="observer"))
        finally:
            self._close_operators()
            self._unregister()
            self._done.set()

    def _termination(self, done: _ReaderDone) -> TerminationReason:
        """``eof`` when the source reached its end, ``stop`` when it was stopped."""
        ended = self._slots[done.source].counters.ended

        return "eof" if ended else "stop"

    def _process_source_pulse(self, slot: _SourceSlot, item: _Pulse) -> None:
        self._run_pulse(
            item.key,
            counters=slot.counters,
            begin=lambda: begin_pulse(
                self.session,
                pulse=item.key,
                emission=item.emission,
                inputs=self._inputs,
                observed=item.observed,
            ),
            begin_stage="emission",
            present=frozenset(item.emission.data),
            filtered=item.emission.is_filtered,
        )

    def _process_operator_pulse(
        self, slot: _OperatorSlot, emission: OperatorPulse
    ) -> _Outcome:
        key = PulseKey(
            active_run_id=self.run_id, source=slot.name, sequence=slot.next_sequence
        )
        slot.next_sequence += 1
        outcome = self._run_pulse(
            key,
            counters=slot.counters,
            begin=lambda: begin_operator_pulse(
                self.session, pulse=key, emission=emission, inputs=self._inputs
            ),
            begin_stage="operator",
            present=frozenset(emission.ports),
            filtered=False,
        )

        return outcome

    def _run_pulse(
        self,
        key: PulseKey,
        *,
        counters: Union[SourceCounters, OperatorCounters],
        begin: Callable[[], RunState],
        begin_stage: ActiveRunStage,
        present: FrozenSet[str],
        filtered: bool,
    ) -> _Outcome:
        """Run one pulse of a source or an operator, then the operators it feeds.

        A failure raised here is attributed to this pulse, recorded as the
        run's failure once and returned; a failure of work the pulse fed is
        returned as it was recorded there. Either way ``on_pulse_finished``
        of this pulse receives it. A pulse cut short by a failure elsewhere
        is cancelled with no error, as in M2.1.
        """
        run: Optional[RunState] = None
        stage: ActiveRunStage = "observer"
        try:
            self.session.observer.on_pulse_started(
                run_id=key.run_id, source=key.source, pulse=key
            )
            stage = begin_stage
            run = begin()
            stage = "step"
            if self._run_route(
                run, counters=counters, present=present, filtered=filtered
            ):
                outcome = self._feed_operators(run)
            else:
                outcome = _CANCELLED
        except Exception as raised:
            error = _attributed(raised, stage=stage, **self._where(key))
            self._fail(error)
            outcome = _Outcome(completed=False, error=error)
        finally:
            if run is not None:
                abandon_pulse(run)
        if outcome.completed:
            counters.processed += 1
        else:
            counters.cancelled += 1
        self._pulse_finished(key, error=outcome.error)

        return outcome

    def _pulse_finished(
        self, key: PulseKey, *, error: Optional[ActiveRunError]
    ) -> None:
        try:
            self.session.observer.on_pulse_finished(
                run_id=key.run_id, source=key.source, pulse=key, error=error
            )
        except Exception as raised:
            self._fail(_attributed(raised, stage="observer", **self._where(key)))

    def _feed_operators(self, run: RunState) -> _Outcome:
        """Push the pulse's values to each consuming operator; run what they emit."""
        for planned in self.session.plan.consumers_of(run.pulse.source):
            if self._failure is not None:
                return _CANCELLED
            slot = self._operators[planned.name]
            arrivals = operator_arrivals(run, planned)
            slot.counters.arrivals += len(arrivals)
            emitted = self._call_operator(slot, "push", arrivals, fed_by=run.pulse)
            outcome = self._run_emissions(slot, emitted)
            if not outcome.completed:
                return outcome

        return _COMPLETED

    def _end_domain(self, domain: str, *, reason: TerminationReason) -> None:
        """Tell the operators of ``domain`` that its inputs ended; finish the done ones.

        An operator whose upstream domains all ended finishes once, its final
        pulses run, and its own domain ends in turn, after them. Stops at the
        first failure, which is already recorded.
        """
        for planned in self.session.plan.consumers_of(domain):
            if self._failure is not None:
                return
            slot = self._operators[planned.name]
            slot.stopped = slot.stopped or reason == "stop"
            for item in planned.inputs_from(domain):
                emitted = self._call_operator(slot, "end_input", item.name)
                if not self._run_emissions(slot, emitted).completed:
                    return
            slot.ended.add(domain)
            if not slot.ended.issuperset(planned.upstream_domains):
                continue

            final: TerminationReason = "stop" if slot.stopped else "eof"
            emitted = self._call_operator(slot, "finish", final)
            if emitted is not None:
                slot.counters.finished = True
            if not self._run_emissions(slot, emitted).completed:
                return
            self._end_domain(planned.name, reason=final)

    def _call_operator(
        self,
        slot: _OperatorSlot,
        method: str,
        argument: Any,
        *,
        fed_by: Optional[PulseKey] = None,
    ) -> Optional[List[OperatorPulse]]:
        """Call ``push``, ``end_input`` or ``finish``; count what it returned.

        Returns ``None`` when the call raised or returned something other than
        a list of ``OperatorPulse``; that failure is attributed to the
        operator, kept as ``slot.error`` and recorded as the run's failure.
        Pulses an operator built but did not return were never emitted.
        """
        try:
            emitted = getattr(slot.instance, method)(argument)
            if not isinstance(emitted, (list, tuple)) or not all(
                isinstance(pulse, OperatorPulse) for pulse in emitted
            ):
                raise ContractError(
                    f"{method}() must return a list of OperatorPulse, got {emitted!r}"
                )
        except Exception as error:
            fed = ""
            where: Dict[str, Any] = {}
            if fed_by is not None and fed_by.source in self._slots:
                where = {"source": fed_by.source, "pulse": fed_by.sequence}
            elif fed_by is not None:
                fed = f" (fed by pulse {fed_by.source}#{fed_by.sequence})"
            failure = ActiveRunError(
                f"{method}{fed} raised {type(error).__name__}: {error}",
                stage="operator",
                operator=slot.name,
                **where,
            )
            failure.__cause__ = error
            slot.error = failure
            self._fail(failure)
            return None
        slot.counters.emitted += len(emitted)

        return list(emitted)

    def _run_emissions(
        self, slot: _OperatorSlot, emitted: Optional[Sequence[OperatorPulse]]
    ) -> _Outcome:
        """Run an operator's returned pulses in order until one does not complete.

        ``None`` is a failed call (see ``_call_operator``). A started pulse
        counts itself as processed or cancelled; returned pulses that never
        started because of a failure are counted cancelled here.
        """
        if emitted is None:
            return _Outcome(completed=False, error=slot.error)

        # A pulse that does not complete always leaves a recorded run failure.
        outcome = _COMPLETED
        started = 0
        for emission in emitted:
            if self._failure is not None:
                break
            outcome = self._process_operator_pulse(slot, emission)
            started += 1
        unstarted = len(emitted) - started
        slot.counters.cancelled += unstarted
        if unstarted and outcome.completed:
            outcome = _CANCELLED  # the run failed elsewhere before they started

        return outcome

    def _close_operators(self) -> None:
        """Close every operator once, releasing what it retained; notify the observer."""
        for slot in self._operators.values():
            if slot.close_attempted:
                continue
            close_error = _close_operator(slot)
            if close_error is not None:
                self._fail(close_error)
                slot.error = slot.error or close_error
            try:
                self.session.observer.on_operator_finished(
                    operator=slot.name, error=slot.error
                )
            except Exception as raised:
                self._fail(_attributed(raised, stage="observer", operator=slot.name))

    def _run_route(
        self,
        run: RunState,
        *,
        counters: Union[SourceCounters, OperatorCounters],
        present: FrozenSet[str],
        filtered: bool,
    ) -> bool:
        """Run the pulse's route, delivering each activated group once it is ready.

        A group is activated when its anchor port is ``present``, or for
        every group of the domain when the pulse is explicitly ``filtered``.
        Returns ``False`` when a failure elsewhere cancelled the rest of the
        pulse at a step or delivery boundary.
        """
        pending = [
            item
            for item in self._registered
            if item.group.source == run.pulse.source
            and (filtered or item.group.anchor.output in present)
        ]
        executed: Set[StepPath] = set()
        self._deliver_ready(
            run, pending, counters=counters, executed=executed, filtered=filtered
        )
        for step in run.plan.route(run.pulse.source):
            if self._failure is not None:
                return False
            execute_step(run, step)
            executed.add(step.path)
            self._deliver_ready(
                run, pending, counters=counters, executed=executed, filtered=filtered
            )
        completed = not pending

        return completed

    def _deliver_ready(
        self,
        run: RunState,
        pending: List[_Registered],
        *,
        counters: Union[SourceCounters, OperatorCounters],
        executed: Set[StepPath],
        filtered: bool,
    ) -> None:
        """Deliver, and drop from ``pending``, every group whose steps have run."""
        for item in list(pending):
            if not filtered and not item.prerequisites <= executed:
                continue
            if self._failure is not None:
                return
            self._deliver(run, item, counters=counters, filtered=filtered)
            pending.remove(item)

    def _deliver(
        self,
        run: RunState,
        item: _Registered,
        *,
        counters: Union[SourceCounters, OperatorCounters],
        filtered: bool,
    ) -> None:
        result = group_result(run, item.group, filtered=filtered)
        where = self._where(run.pulse)
        try:
            returned = item.handler(result)
        except Exception as error:
            raise ActiveRunError(
                f"handler raised {type(error).__name__}: {error}",
                stage="handler",
                group=item.group.name,
                **where,
            ) from error
        if inspect.isawaitable(returned):
            if inspect.iscoroutine(returned):
                returned.close()
            raise ActiveRunError(
                "handler returned an awaitable; active runs call synchronous "
                "handlers only and never await their results",
                stage="handler",
                group=item.group.name,
                **where,
            )
        counters.delivered += 1
        try:
            self.session.observer.on_group_delivered(
                run_id=run.run_id,
                group=item.group.name,
                source=run.pulse.source,
                pulse=run.pulse,
            )
        except Exception as error:
            raise ActiveRunError(
                f"on_group_delivered raised {type(error).__name__}: {error}",
                stage="observer",
                group=item.group.name,
                **where,
            ) from error

    def _where(self, pulse: PulseKey) -> Dict[str, Any]:
        """Attribution of a pulse: its source or operator, and its sequence."""
        domain = "operator" if pulse.source in self._operators else "source"
        where = {domain: pulse.source, "pulse": pulse.sequence}

        return where

    # Shared ---------------------------------------------------------------

    def _close_admission(self) -> None:
        """Stop admitting pulses and wake every reader waiting for a slot."""
        with self._lock:
            if not self._admission_open:
                return
            self._admission_open = False
        self.stop_event.set()
        for slot in self._slots.values():
            slot.admission.release()

    def _fail(self, failure: ActiveRunError) -> None:
        """Record the first failure as terminal, later ones as suppressed."""
        with self._lock:
            if self._failure is None:
                self._failure = failure
            elif failure is not self._failure:
                self._failure.suppressed += (failure,)
        self._close_admission()
