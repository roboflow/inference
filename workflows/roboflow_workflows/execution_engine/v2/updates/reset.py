"""Explicit processing reset: replace everything downstream of the sources.

``prepare_update(plan, reset=True)`` builds a complete new processing while
the session keeps running the old one::

    assess        same as session.assess_update; refused before anything
                  is built when a change or the session rules a reset out
    state         fresh (engine), retained or replaced (caller), or none
    construct     every step in a fresh resolver: caller values are passed
                  again, Factory values are created again; every handler
                  session
    candidate     PreparedUpdate(reset=True); a failure closes the state
                  the preparation created and leaves the session as it was

    apply (idle)  one commit publishes the new generation, its state, its
                  handler sessions and a reset control snapshot, and fills a
                  Retirement with what it replaced; that closes outside
                  every lock

Sources are kept: an idle session keeps the resources its sources were
resolved with, so the next run opens them as before. Caller resources stay
the caller's: the engine closes only state it created. Nothing that blocks
built or loaded is torn down: the engine drops its references, and objects
are freed when the last reference goes, including references a caller
keeps, e.g. in results.
"""

import importlib
import threading
import time
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
)

from roboflow_workflows.execution_engine.v2.errors import (
    IncompatibleUpdateError,
    StepPath,
    UpdateConflictError,
)
from roboflow_workflows.execution_engine.v2.plan import (
    STATE_SESSION_MODULE,
    construct_step,
    create_session,
)
from roboflow_workflows.execution_engine.v2.resources import (
    ResolvedResource,
    ResourceResolver,
)
from roboflow_workflows.execution_engine.v2.updates.assessment import (
    assess_reset,
    carried_controls,
)
from roboflow_workflows.execution_engine.v2.updates.prepared import (
    PreparedUpdate,
    ResetParts,
)

if TYPE_CHECKING:
    from roboflow_workflows.execution_engine.v2.plan import (
        CompiledWorkflow,
        ExecutionSession,
    )


Closer = Callable[[], None]

PENDING = "pending"
DONE = "done"
FAILED = "failed"


class Cleanup:
    """A reset's cleanup as the host sees it: ``receipt.cleanup``; read-only.

    It holds no part of the replaced processing; its ``Retirement`` reports
    the outcome here once, when the closing finished.
    """

    def __init__(self) -> None:
        self._finished = threading.Event()
        self._finished_at: Optional[float] = None
        self._errors: Tuple[str, ...] = ()

    @property
    def state(self) -> str:
        """``pending``, then ``done``, or ``failed`` when a close failed."""
        if not self._finished.is_set():
            return PENDING

        return FAILED if self._errors else DONE

    @property
    def errors(self) -> Tuple[str, ...]:
        """One message per failure; empty while pending or when done."""
        return self._errors if self._finished.is_set() else ()

    @property
    def finished_at(self) -> Optional[float]:
        """When the closing finished (``time.monotonic``); ``None`` while pending."""
        return self._finished_at if self._finished.is_set() else None

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Wait until the closing finished; ``True`` once done or failed.

        Args:
            timeout: Seconds to wait at most; ``None`` waits.
        """
        return self._finished.wait(timeout)

    def _finish(self, errors: Tuple[str, ...]) -> None:
        self._errors = errors
        self._finished_at = time.monotonic()
        self._finished.set()


class Retirement:
    """What a committed reset replaced; closed once, after the switch.

    The commit fills it under its locks, at the point where each part is
    replaced: the session sets ``owned_state``, ``session_reactions`` and
    ``dropped``; an active run sets ``run_reactions`` and ``run_operators``.
    Each part is a closer that raises on failure. ``close`` runs them in
    this order, then drops every reference::

        run_reactions      the run's reaction runtime; waits for its handlers
        run_operators      the run's operators, by name; partial windows
                           are dropped, never flushed
        session_reactions  the idle session's reaction runtime
        owned_state        the engine-owned managed state, last: the
                           handlers and blocks above may still use it
        dropped            only dropped, never closed: the old generation
                           and handler sessions, so block instances and
                           resources are freed here, not inside the commit

    The engine closes only lifecycle objects it owns; caller resources and
    a caller's managed state are never closed. A failure does not undo the
    commit: it is reported, and the next part still closes.

    An idle session calls ``close`` before ``apply_update`` returns. An
    active run ``reserve``s the closing thread before its cut, so a reset
    that gets no thread is refused while nothing changed, and ``release``s
    it after the resume; a rejected commit ``cancel``s it. Only the engine
    holds a ``Retirement``: the receipt and the session get its ``cleanup``,
    which reports the outcome and cannot close or finish anything. Nothing
    cancels a close that hangs: it stays ``pending``, and the session
    refuses another reset, in this run or a later one, until it finished.
    """

    def __init__(self) -> None:
        self.run_reactions: Optional[Closer] = None
        self.run_operators: Dict[str, Closer] = {}
        self.session_reactions: Optional[Closer] = None
        self.owned_state: Optional[Closer] = None
        self.dropped: Tuple[Any, ...] = ()
        self.cleanup = Cleanup()
        self._may_close = threading.Event()
        self._cancelled = False

    def close(self) -> Tuple[str, ...]:
        """Close every part in order, then drop every reference; call once.

        Returns:
            One message per failure, e.g.
            ``closing operator 'pair': RuntimeError: close failed``.
        """
        closers = [
            ("the run's old reaction runtime", self.run_reactions),
            *(
                (f"operator {name!r}", close)
                for name, close in self.run_operators.items()
            ),
            ("the old reaction runtime", self.session_reactions),
            ("the old managed state", self.owned_state),
        ]
        self.run_reactions = self.session_reactions = self.owned_state = None
        self.run_operators = {}
        errors: List[str] = []
        for what, close in closers:
            if close is None:
                continue
            try:
                close()
            except Exception as error:
                errors.append(f"closing {what}: {type(error).__name__}: {error}")
        self.dropped = ()
        self.cleanup._finish(tuple(errors))

        return self.cleanup.errors

    def reserve(self, *, name: str) -> None:
        """Start the thread that runs ``close`` after ``release``; call once.

        The thread closes nothing until ``release``, and nothing after
        ``cancel``.

        Args:
            name: Name of the thread.

        Raises:
            RuntimeError: When the thread cannot start.
        """
        thread = threading.Thread(
            target=self._close_when_allowed, name=name, daemon=True
        )
        thread.start()

    def release(self) -> None:
        """Let the reserved thread close: the commit published the reset."""
        self._may_close.set()

    def cancel(self) -> None:
        """End the reserved thread without closing: the commit was rejected."""
        self._cancelled = True
        self._may_close.set()

    def _close_when_allowed(self) -> None:
        self._may_close.wait()
        if not self._cancelled:
            self.close()


def prepare_reset(
    session: "ExecutionSession",
    plan: "CompiledWorkflow",
    *,
    resources: Optional[Mapping[str, Any]],
) -> PreparedUpdate:
    """Build a candidate that replaces the session's whole processing.

    One reset preparation runs per session at a time. Construction is
    sequential and cannot be cancelled.

    Args:
        session: The session to reset.
        plan: The new compiled plan.
        resources: Caller values for the reset; they may give an existing
            key a new value.

    Returns:
        The prepared reset candidate.

    Raises:
        SessionClosedError: When the session was closed.
        UpdateConflictError: When another reset of the session is being
            prepared.
        IncompatibleUpdateError: When a change or the session rules a reset
            out; its ``assessment`` names every reason.
        ResourceError: When a resource, a factory, the managed state or a
            constructor fails. State the preparation created is closed.
    """
    started = time.monotonic()
    if not session._reset_preparation.acquire(blocking=False):
        raise UpdateConflictError(
            f"another reset of session {session.session_id} is being prepared; "
            "apply or discard it first"
        )

    try:
        prepared = _prepare(session, plan, resources=resources, started=started)
    finally:
        session._reset_preparation.release()

    return prepared


def _prepare(
    session: "ExecutionSession",
    plan: "CompiledWorkflow",
    *,
    resources: Optional[Mapping[str, Any]],
    started: float,
) -> PreparedUpdate:
    assessment, decision = assess_reset(session, plan, resources=resources)
    if assessment.reset is None:
        reasons = "; ".join(
            f"{reason.name}: {reason.reason}" for reason in assessment.reset_blocked_by
        )
        raise IncompatibleUpdateError(
            f"a reset cannot replace graph version {assessment.base_version}: "
            f"{reasons}",
            diff=assessment.diff,
            assessment=assessment,
        )

    state_session = importlib.import_module(STATE_SESSION_MODULE)
    state = state_session.configure_reset_state(
        plan,
        decision=decision,
        service=session.managed_state,
        owned=session.owned_state,
        session_id=session.session_id,
    )
    owned = state.owned
    try:
        resolver = ResourceResolver(
            provided=state.resources, providers=plan.catalogue.providers
        )
        instances, chosen = _construct(plan, resolver=resolver, session=session)
        handler_sessions = {
            handler.path: create_session(handler.plan, resources=state.resources)
            for handler in plan.reactions.handlers
        }
    except BaseException:
        if owned is not None:
            owned.close()
        raise

    parts = ResetParts(
        managed_state=state.service,
        owned_state=owned,
        handler_sessions=MappingProxyType(handler_sessions),
        carried_controls=carried_controls(session.plan, plan),
    )
    prepared = PreparedUpdate(
        session=session,
        base_version=assessment.base_version,
        plan=plan,
        diff=assessment.diff,
        instances=instances,
        resources=chosen,
        resolver=resolver,
        reset_parts=parts,
        assessment=assessment,
        prepared_seconds=time.monotonic() - started,
    )

    return prepared


def _construct(
    plan: "CompiledWorkflow",
    *,
    resolver: ResourceResolver,
    session: "ExecutionSession",
) -> Tuple[Dict[StepPath, Any], Dict[StepPath, Mapping[str, ResolvedResource]]]:
    instances: Dict[StepPath, Any] = {}
    chosen: Dict[StepPath, Mapping[str, ResolvedResource]] = {}
    for step in plan.steps:
        instances[step.path], chosen[step.path] = construct_step(
            step, resolver=resolver, session_id=session.session_id
        )

    return instances, chosen
