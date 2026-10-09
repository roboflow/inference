"""A prepared graph update and the receipt of an applied one.

``prepare_update`` compares the session's current plan with the new one and
constructs only the added steps. Nothing in the session changes::

    session generation N ── prepare ──> PreparedUpdate(base_version=N)
                                          diff, added instances,
                                          forked resource resolver
    apply: same session, still at N, idle ──> generation N+1 (commit point)
    discard, stale or busy ──> session stays at N

The candidate resolves resources in a fork of the session's resolver: a
session ``Factory`` the session already created is reused, never created
again; one the candidate creates joins the session only when it is applied.
A discarded or stale candidate releases its instances, its factory values and
the caller values given to it without undoing their constructors' effects.
An applied candidate releases them too, so keeping it keeps nothing alive
that the session later replaces.

A reset candidate (``prepare_update(plan, reset=True)``, ``updates.reset``)
constructs every step and handler session of the new plan in a fresh
resolver, with the managed state its assessment decided on. A state the
candidate created is closed when the candidate is discarded or goes stale.
"""

import contextlib
import importlib
import threading
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    FrozenSet,
    Iterator,
    Mapping,
    Optional,
    Tuple,
)

from roboflow_workflows.execution_engine.v2.errors import (
    IncompatibleUpdateError,
    SessionClosedError,
    StepPath,
    UpdateConflictError,
)
from roboflow_workflows.execution_engine.v2.locking import acquired
from roboflow_workflows.execution_engine.v2.plan import (
    STATE_SESSION_MODULE,
    construct_step,
)
from roboflow_workflows.execution_engine.v2.resources import (
    ResolvedResource,
    ResourceResolver,
)
from roboflow_workflows.execution_engine.v2.updates.diff import (
    RESET,
    PlanDiff,
    compare_plans,
)

if TYPE_CHECKING:
    from roboflow_workflows.execution_engine.v2.plan import (
        CompiledWorkflow,
        ExecutionSession,
    )
    from roboflow_workflows.execution_engine.v2.updates.assessment import (
        UpdateAssessment,
    )
    from roboflow_workflows.execution_engine.v2.updates.reset import Cleanup

PREPARED = "prepared"
APPLIED = "applied"
DISCARDED = "discarded"


@dataclass(frozen=True)
class UpdateReceipt:
    """Outcome of an applied graph update.

    Args:
        graph_version: Graph version the session runs from now on.
        previous_version: Graph version the update replaced.
        diff: What the update retained, added and changed.
        reset: Whether the update reset the processing.
        processing_version: Processing version the session runs from now
            on; a reset adds one, a preserving update keeps it.
        cleanup: Read-only ``updates.Cleanup`` of a reset's replaced
            processing; ``None`` without a reset. An idle reset closed it
            before returning; an active run's closes on its own thread
            after the resume. The update applied all the same.
    """

    graph_version: int
    previous_version: int
    diff: PlanDiff
    reset: bool = field(default=False, kw_only=True)
    processing_version: int = field(default=0, kw_only=True)
    cleanup: Optional["Cleanup"] = field(default=None, kw_only=True)

    @property
    def cleanup_failures(self) -> Tuple[str, ...]:
        """Every cleanup failure known now: ``cleanup.errors`` once it finished."""
        failures = self.cleanup.errors if self.cleanup is not None else ()

        return failures

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-friendly description."""
        return {
            "graph_version": self.graph_version,
            "previous_version": self.previous_version,
            "reset": self.reset,
            "processing_version": self.processing_version,
            "cleanup": (
                None
                if self.cleanup is None
                else {
                    "state": self.cleanup.state,
                    "errors": list(self.cleanup_failures),
                }
            ),
            "diff": self.diff.describe(),
        }


@dataclass(frozen=True)
class ActiveUpdateReceipt(UpdateReceipt):
    """Outcome of an update applied to a running active run.

    The stamps are ``time.monotonic()`` values and authoritative. Each
    duration names the interval it measures::

        called_at                  apply_update was called
          checks; a reset builds the run's operators and
          reaction runtime (run_build_seconds)
          the lock waits of the reservation
        cut_at                     admission paused
          everything admitted settles
        drained_at
          the commit publishes the new graph
        resumed_at                 admission resumed under the new graph

        admission_pause_seconds    resumed_at - cut_at
        drain_after_cut_seconds    drained_at - cut_at
        call_to_resume_seconds     resumed_at - called_at

    ``drained_seconds`` and ``paused_seconds`` are kept from before resets
    existed and keep that meaning: they count from the start of the
    reservation, after the checks and the build, so they include the lock
    waits before the cut. The first new result and the completion gap are
    the host's to measure.

    Args:
        run_id: The run that switched graphs.
        drained_seconds: From the start of the reservation until
            everything admitted had settled: pulses, deliveries, reactions
            and accepted signals.
        paused_seconds: From the start of the reservation until admission
            resumed.
        frontiers: Next pulse ordinal per domain at the boundary: a source's
            next read ordinal, an operator's next emission ordinal. The
            stages this update added start there; a pulse of the domain with
            a lower ordinal ran under the previous graph. Each later update
            has its own frontiers; ``result.graph_version`` tells which graph
            produced any one result.
        called_at: When ``apply_update`` was called.
        cut_at: When admission paused.
        drained_at: When everything admitted had settled.
        resumed_at: When admission resumed under the new graph.
        run_build_seconds: From the call to the reservation: the call's
            checks, the construction of the run's new operators and reaction
            runtime, and the start of the cleanup thread, all before the
            cut; ``0.0`` without a reset.
    """

    run_id: str
    drained_seconds: float
    paused_seconds: float
    frontiers: Mapping[str, int]
    called_at: float = field(default=0.0, kw_only=True)
    cut_at: float = field(default=0.0, kw_only=True)
    drained_at: float = field(default=0.0, kw_only=True)
    resumed_at: float = field(default=0.0, kw_only=True)
    run_build_seconds: float = field(default=0.0, kw_only=True)

    @property
    def admission_pause_seconds(self) -> float:
        """How long admission was paused: ``resumed_at - cut_at``."""
        return self.resumed_at - self.cut_at

    @property
    def drain_after_cut_seconds(self) -> float:
        """How long the admitted work took to settle: ``drained_at - cut_at``."""
        return self.drained_at - self.cut_at

    @property
    def call_to_resume_seconds(self) -> float:
        """From the call until admission resumed: ``resumed_at - called_at``."""
        return self.resumed_at - self.called_at

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-friendly description."""
        description = {
            **super().describe(),
            "run_id": self.run_id,
            "admission_pause_seconds": self.admission_pause_seconds,
            "drain_after_cut_seconds": self.drain_after_cut_seconds,
            "call_to_resume_seconds": self.call_to_resume_seconds,
            "run_build_seconds": self.run_build_seconds,
            "drained_seconds": self.drained_seconds,
            "paused_seconds": self.paused_seconds,
            "stamps": {
                "called_at": self.called_at,
                "cut_at": self.cut_at,
                "drained_at": self.drained_at,
                "resumed_at": self.resumed_at,
            },
            "frontiers": dict(self.frontiers),
        }

        return description


@dataclass(frozen=True)
class ResetParts:
    """Session-level objects of a reset candidate, besides its step instances.

    Args:
        managed_state: The managed state of the new processing; ``None``
            when the new plan uses none.
        owned_state: State the candidate created; the session closes it
            from then on, the candidate when it is not applied.
        handler_sessions: A new handler session per handler path.
        carried_controls: Controls that keep their values at the commit.
    """

    managed_state: Any
    owned_state: Any
    handler_sessions: Mapping[StepPath, Any]
    carried_controls: FrozenSet[str]


class PreparedUpdate:
    """A candidate graph for one session, applicable once.

    Create it with ``ExecutionSession.prepare_update``; apply it with
    ``ExecutionSession.apply_update`` or drop it with ``discard``. Its
    attributes are read-only; only the session changes its ``state``.

    Args:
        session: The session the candidate was prepared for.
        base_version: The session's graph version at preparation.
        plan: The new plan.
        diff: Comparison of the base plan with ``plan``.
        instances: Constructed instances of the added steps; of every step
            for a reset.
        resources: Resources chosen for them.
        resolver: The session resolver's fork that resolved them; a fresh
            resolver for a reset.
        reset_parts: The session-level objects of a reset candidate;
            ``None`` for a preserving update.
        assessment: The assessment a reset candidate was prepared from.
        prepared_seconds: Duration of the preparation.
    """

    def __init__(
        self,
        *,
        session: "ExecutionSession",
        base_version: int,
        plan: "CompiledWorkflow",
        diff: PlanDiff,
        instances: Mapping[StepPath, Any],
        resources: Mapping[StepPath, Mapping[str, ResolvedResource]],
        resolver: ResourceResolver,
        reset_parts: Optional[ResetParts] = None,
        assessment: Optional["UpdateAssessment"] = None,
        prepared_seconds: float = 0.0,
    ):
        self._session = session
        self._base_version = base_version
        self._plan = plan
        self._diff = diff
        self._instances = MappingProxyType(dict(instances))
        self._resources = MappingProxyType(dict(resources))
        self._resolver: Optional[ResourceResolver] = resolver
        self._reset = reset_parts is not None
        self._reset_parts = reset_parts
        self._assessment = assessment
        self._prepared_seconds = prepared_seconds
        # Held for a whole commit, so discard() never sees half of one.
        self._lock = threading.Lock()
        self._state = PREPARED

    @property
    def session(self) -> "ExecutionSession":
        """The session the candidate was prepared for."""
        return self._session

    @property
    def base_version(self) -> int:
        """The session's graph version at preparation."""
        return self._base_version

    @property
    def plan(self) -> "CompiledWorkflow":
        """The new plan."""
        return self._plan

    @property
    def diff(self) -> PlanDiff:
        """Comparison of the base plan with ``plan``."""
        return self._diff

    @property
    def instances(self) -> Mapping[StepPath, Any]:
        """Instances of the added (reset: all) steps; empty once applied or discarded."""
        return self._instances

    @property
    def resources(self) -> Mapping[StepPath, Mapping[str, ResolvedResource]]:
        """Resources chosen for ``instances``; empty once applied or discarded."""
        return self._resources

    @property
    def reset(self) -> bool:
        """Whether the candidate resets the processing when applied."""
        return self._reset

    @property
    def assessment(self) -> Optional["UpdateAssessment"]:
        """The assessment of a reset candidate; ``None`` for a preserving one."""
        return self._assessment

    @property
    def prepared_seconds(self) -> float:
        """How long the preparation took, constructors included."""
        return self._prepared_seconds

    @property
    def state(self) -> str:
        """``PREPARED``, ``APPLIED`` or ``DISCARDED``."""
        return self._state

    def discard(self) -> None:
        """Drop the candidate and its added instances; safe to repeat.

        Waits for a commit of the candidate in progress.

        Raises:
            UpdateConflictError: When the candidate was already applied.
        """
        with self._lock:
            if self._state == APPLIED:
                raise UpdateConflictError(
                    "the update was already applied; it cannot be discarded"
                )
            self._drop()

    @contextlib.contextmanager
    def _committing(
        self,
        session: "ExecutionSession",
        *,
        graph_version: int,
        deadline: Optional[float] = None,
    ) -> Iterator[ResourceResolver]:
        # The session's commit only. Checks the candidate, then holds it for
        # the commit and yields the resolver the session adopts. A normal exit
        # marks it applied; an exception leaves it prepared. A stale
        # candidate can never apply, so it is discarded. Waiting for the
        # candidate (a discard in progress) gives up at ``deadline``.
        with acquired(self._lock, deadline=deadline, what="the update candidate"):
            if session is not self._session:
                raise UpdateConflictError(
                    f"the update was prepared for session {self._session.session_id}, "
                    f"not {session.session_id}"
                )
            if self._state != PREPARED:
                raise UpdateConflictError(f"the update was already {self._state}")
            if graph_version != self._base_version:
                self._drop()
                raise UpdateConflictError(
                    f"the update was prepared from graph version {self._base_version}, "
                    f"but the session is at version {graph_version}; prepare it again"
                )
            yield self._resolver
            self._state = APPLIED
            # The session owns everything now; the candidate keeps nothing
            # alive that a later reset replaces.
            self._release()

    def _drop(self) -> None:
        # Releases everything only the candidate holds: its instances, the
        # factory values it created and the caller values of its fork. Values
        # shared with the session stay referenced by the session's resolver.
        # State the candidate created is closed, once.
        self._state = DISCARDED
        owned = None if self._reset_parts is None else self._reset_parts.owned_state
        self._release()
        if owned is not None:
            owned.close()

    def _release(self) -> None:
        self._instances = MappingProxyType({})
        self._resources = MappingProxyType({})
        self._resolver = None
        self._reset_parts = None


def prepare_update(
    session: "ExecutionSession",
    plan: "CompiledWorkflow",
    *,
    resources: Optional[Mapping[str, Any]],
) -> PreparedUpdate:
    """Compare ``plan`` with the session's graph and construct its added steps.

    Args:
        session: The session to update.
        plan: The new compiled plan.
        resources: Caller values for new resource keys only.

    Returns:
        The prepared candidate.

    Raises:
        SessionClosedError: When the session was closed.
        IncompatibleUpdateError: When the comparison finds a breaking change.
        ContractError: When ``resources`` repeats a key the session has.
        ResourceError: When a new step requests another managed state service
            than the session's, or its resource or constructor fails.
    """
    if session.closed:
        raise SessionClosedError(
            f"Session {session.session_id} is closed; it cannot be updated"
        )

    generation = session.generation
    diff = compare_plans(generation.plan, plan)
    if not diff.compatible:
        reasons = "; ".join(
            f"{change.name}: {change.reason}" for change in diff.breaking
        )
        remedy = (
            "; a reset can apply it: prepare_update(plan, reset=True)"
            if diff.kind == RESET
            else ""
        )
        raise IncompatibleUpdateError(
            f"the update cannot reuse graph version {generation.graph_version}: "
            f"{reasons}{remedy}",
            diff=diff,
        )

    resolver = generation.resolver.fork(provided=resources)
    if session.managed_state is not None:
        # Before any construction: new steps share the session's service.
        state_session = importlib.import_module(STATE_SESSION_MODULE)
        state_keys = state_session.added_state_keys(
            plan, diff.added, resolver=resolver, service=session.managed_state
        )
        if state_keys:
            resolver = generation.resolver.fork(
                provided=resources, replacing=state_keys
            )
    instances: Dict[StepPath, Any] = {}
    chosen: Dict[StepPath, Mapping[str, ResolvedResource]] = {}
    for path in diff.added:
        step = plan.step(path)
        instances[path], chosen[path] = construct_step(
            step, resolver=resolver, session_id=session.session_id
        )

    prepared = PreparedUpdate(
        session=session,
        base_version=generation.graph_version,
        plan=plan,
        diff=diff,
        instances=instances,
        resources=chosen,
        resolver=resolver,
    )

    return prepared
