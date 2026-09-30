"""Active runtime: independent source readers, one sequential processor.

``ExecutionSession.start`` delegates here::

    run = start_session(session, inputs=static, handlers={"frames": on_frames})
    ...
    run.stop()      # optional: close admission, drain what was admitted
    run.wait()      # True when complete; raises the run's failure

Threads of one active run::

    reader[S]   (one per source)     processor   (one per run)
    ----------------------------     -----------------------------------------
    open(**params)                   pulse = queue.get()
    loop: emission = read()          run = begin_pulse(...)        fresh RunState
          None -> end of S           deliver groups that are ready
          admit(S, emission)         for step in plan.route(S):    plan order
    close()  exactly once                execute_step; deliver groups now ready
                                     release S's admission slot

Admission is the acceptance boundary: a reader takes one of its source's
``admission_bound`` slots and, under the run's lock, queues the pulse. The
slot is released after the pulse's handlers returned. A reader holds at most
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

* End of one source ends only its reader. The run completes when every
  reader has closed its source and every admitted pulse was processed.
* ``stop()`` closes admission and sets the shared ``stop_event`` the sources
  see. A reader whose ``read()`` returns afterwards discards that emission
  and closes. Admitted pulses are still processed and delivered exactly once.
* A failure (open, read, emission, step, handler, observer or close) closes
  admission, cancels admitted pulses not yet processed and the remaining
  steps and deliveries of the pulse in flight, closes every source whose
  ``open`` was attempted, and is raised by ``wait()`` as one
  ``ActiveRunError``; later errors are kept in its ``suppressed``. Handlers
  already called are not undone.
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
* A later ``start()`` on the same session constructs new source instances
  and reuses the session's block instances and their state.

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
from typing import Any, Callable, FrozenSet, List, Mapping, Optional, Set, Union

from roboflow_workflows.execution_engine.v2.active.execution import (
    abandon_pulse,
    begin_pulse,
    engine_observation,
    group_result,
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
from roboflow_workflows.execution_engine.v2.plan import (
    Binding,
    Constant,
    ExecutionSession,
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
            declaration, its constructor fails or the observer rejects the
            run (stage ``start``).
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
        run = ActiveRun(
            session,
            run_id=run_id,
            inputs=entries,
            registered=registered,
            slots=slots,
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
            pulse=pulse,
            step_path=raised.step_path,
        )
    else:
        error = ActiveRunError(
            f"{type(raised).__name__}: {raised}",
            stage=stage,
            source=source,
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
        stop_event: threading.Event,
    ):
        self.session = session
        self.run_id = run_id
        self.stop_event = stop_event
        self._inputs = inputs
        self._registered = registered
        self._slots = slots
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

        Before the processor runs nothing needs unwinding: sources are
        constructed but not opened, so a failure only releases the
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
            raise _attributed(raised, stage="start") from raised

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
                    continue

                slot = self._slots[item.key.source]
                if self._failure is not None:
                    slot.counters.cancelled += 1
                else:
                    self._process_pulse(slot, item)
                slot.admission.release()
            self.session.observer.on_run_finished(
                run_id=self.run_id, result=None, error=self._failure
            )
        except Exception as raised:
            self._fail(_attributed(raised, stage="observer"))
        finally:
            self._unregister()
            self._done.set()

    def _process_pulse(self, slot: _SourceSlot, item: _Pulse) -> None:
        key = item.key
        run: Optional[RunState] = None
        error: Optional[ActiveRunError] = None
        stage: ActiveRunStage = "observer"
        try:
            self.session.observer.on_pulse_started(
                run_id=key.run_id, source=key.source, pulse=key
            )
            stage = "emission"
            run = begin_pulse(
                self.session,
                pulse=key,
                emission=item.emission,
                inputs=self._inputs,
                observed=item.observed,
            )
            stage = "step"
            if self._run_route(slot, run, emission=item.emission):
                slot.counters.processed += 1
            else:
                slot.counters.cancelled += 1
        except Exception as raised:
            error = _attributed(
                raised, stage=stage, source=key.source, pulse=key.sequence
            )
            slot.counters.cancelled += 1
        finally:
            if run is not None:
                abandon_pulse(run)
        if error is not None:
            self._fail(error)
        try:
            self.session.observer.on_pulse_finished(
                run_id=key.run_id, source=key.source, pulse=key, error=error
            )
        except Exception as raised:
            self._fail(
                _attributed(
                    raised, stage="observer", source=key.source, pulse=key.sequence
                )
            )

    def _run_route(
        self, slot: _SourceSlot, run: RunState, *, emission: Emission
    ) -> bool:
        """Run the source route, delivering each activated group once it is ready.

        Returns ``False`` when a failure elsewhere cancelled the rest of the
        pulse at a step or delivery boundary.
        """
        filtered = not emission.data
        pending = [
            item
            for item in self._registered
            if item.group.source == run.pulse.source
            and (filtered or item.group.anchor.output in emission.data)
        ]
        executed: Set[StepPath] = set()
        self._deliver_ready(slot, run, pending, executed=executed, filtered=filtered)
        for step in run.plan.route(run.pulse.source):
            if self._failure is not None:
                return False
            execute_step(run, step)
            executed.add(step.path)
            self._deliver_ready(
                slot, run, pending, executed=executed, filtered=filtered
            )
        completed = not pending

        return completed

    def _deliver_ready(
        self,
        slot: _SourceSlot,
        run: RunState,
        pending: List[_Registered],
        *,
        executed: Set[StepPath],
        filtered: bool,
    ) -> None:
        """Deliver, and drop from ``pending``, every group whose steps have run."""
        for item in list(pending):
            if not filtered and not item.prerequisites <= executed:
                continue
            if self._failure is not None:
                return
            self._deliver(slot, run, item, filtered=filtered)
            pending.remove(item)

    def _deliver(
        self, slot: _SourceSlot, run: RunState, item: _Registered, *, filtered: bool
    ) -> None:
        result = group_result(run, item.group, filtered=filtered)
        try:
            returned = item.handler(result)
        except Exception as error:
            raise ActiveRunError(
                f"handler raised {type(error).__name__}: {error}",
                stage="handler",
                source=run.pulse.source,
                pulse=run.pulse.sequence,
                group=item.group.name,
            ) from error
        if inspect.isawaitable(returned):
            if inspect.iscoroutine(returned):
                returned.close()
            raise ActiveRunError(
                "handler returned an awaitable; active runs call synchronous "
                "handlers only and never await their results",
                stage="handler",
                source=run.pulse.source,
                pulse=run.pulse.sequence,
                group=item.group.name,
            )
        slot.counters.delivered += 1
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
                source=run.pulse.source,
                pulse=run.pulse.sequence,
                group=item.group.name,
            ) from error

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
