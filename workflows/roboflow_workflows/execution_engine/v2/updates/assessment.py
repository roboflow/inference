"""What an update of a session would do, decided before anything is built.

``ExecutionSession.assess_update(plan)`` constructs nothing, writes no state
and changes no session. It combines the pure plan comparison with what only
the session knows (its managed state and resources)::

    assessment.kind      preserve      a preserving update applies the plan
                         reset         only a reset applies it
                         unsupported   no update applies it
    preserve_blocked_by  the changes a preserving update rejects
    reset_blocked_by     the changes no update applies, then the session's
                         reasons against a reset (``blockers``)
    assessment.reset     what a reset would do; None when it cannot run

``kind`` follows from the two lists: ``preserve`` when nothing blocks a
preserving update, else ``reset`` when nothing blocks a reset, else
``unsupported``. A ``preserve`` assessment can still list reasons against a
reset; they matter only to a caller who asks for one.

A reset is always explicit: ``prepare_update(plan, reset=True)``. It is also
possible for a plan a preserving update could apply. The assessment is
advisory: preparation decides again, and the commit checks that the session
is still at ``base_version``.
"""

import importlib
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    FrozenSet,
    Iterator,
    Mapping,
    NamedTuple,
    Optional,
    Tuple,
)

from roboflow_workflows.execution_engine.v2.errors import (
    SessionClosedError,
    format_step_path,
)
from roboflow_workflows.execution_engine.v2.plan import (
    ACTIVE_RUNTIME_MODULE,
    STATE_SESSION_MODULE,
    requests_managed_state,
)
from roboflow_workflows.execution_engine.v2.resources import (
    MANAGED_STATE_RESOURCE,
    ResourceChoice,
    ResourceResolver,
    ResourceSpec,
)
from roboflow_workflows.execution_engine.v2.updates.diff import (
    PRESERVE,
    RESET,
    UNSUPPORTED,
    PlanChange,
    PlanDiff,
    compare_plans,
    same_control_declaration,
    same_value,
)

if TYPE_CHECKING:
    from roboflow_workflows.execution_engine.v2.plan import (
        CompiledWorkflow,
        ExecutionSession,
    )
    from roboflow_workflows.execution_engine.v2.state.session import ResetState

# Blocker reasons of component ``resource``: a retained source and the new
# processing would hold different values of one resource.
SOURCE_RESOURCE_SHARED = "source_resource_shared"
SOURCE_RESOURCE_OVERRIDDEN = "source_resource_overridden"


@dataclass(frozen=True)
class StateConsequence:
    """What a reset does with the session's managed state.

    Args:
        action: ``none``, ``fresh``, ``retained`` or ``replaced`` (see
            ``state.session.ResetState``).
        owner: ``engine`` or ``caller``; ``None`` for ``none``.
        detail: Human-readable explanation.
    """

    action: str
    owner: Optional[str]
    detail: str

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-friendly description."""
        return {"action": self.action, "owner": self.owner, "detail": self.detail}


@dataclass(frozen=True)
class ResetConsequences:
    """What a reset of the session would replace and what it would keep.

    A reset constructs every step and handler session of the new plan, so
    every block starts with fresh block-local state, also a step whose
    declaration did not change. Sources keep their contract and the
    resources they were resolved with.

    Args:
        sources_kept: Source names; they keep their instances while an active
            run runs, and their resolved resources.
        steps_constructed: Every step of the new plan, constructed fresh.
        steps_removed: Steps of the current plan the new plan drops.
        operators_constructed: Operators of the new plan; a running run
            discards the partial windows of the old ones.
        operators_removed: Operators the new plan drops.
        handlers_constructed: Handler sessions of the new plan.
        handlers_on_started: Handlers of the new plan subscribed to
            ``$system.events.started``. ``started`` is sent once per run, so
            after a reset of a running run they do not run before the next
            run starts; fresh handler state they set up there stays unset.
        controls_carried: Controls declared alike in both plans; they keep
            the value or enable state they have at the commit.
        controls_initialized: Other controls of the new plan; they start
            at their declared initial state.
        managed_state: What happens to managed state.
        resources: Per step, where each constructor resource comes from;
            a ``Factory`` is created again, a caller value is passed again.
        resources_replaced: Caller resource keys the reset's ``resources``
            give a new value.
    """

    sources_kept: Tuple[str, ...]
    steps_constructed: Tuple[str, ...]
    steps_removed: Tuple[str, ...]
    operators_constructed: Tuple[str, ...]
    operators_removed: Tuple[str, ...]
    handlers_constructed: Tuple[str, ...]
    handlers_on_started: Tuple[str, ...]
    controls_carried: Tuple[str, ...]
    controls_initialized: Tuple[str, ...]
    managed_state: StateConsequence
    resources: Mapping[str, Tuple[ResourceChoice, ...]]
    resources_replaced: Tuple[str, ...]

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-friendly description."""
        description = {
            "sources_kept": list(self.sources_kept),
            "steps_constructed": list(self.steps_constructed),
            "steps_removed": list(self.steps_removed),
            "operators_constructed": list(self.operators_constructed),
            "operators_removed": list(self.operators_removed),
            "handlers_constructed": list(self.handlers_constructed),
            "handlers_on_started": list(self.handlers_on_started),
            "controls_carried": list(self.controls_carried),
            "controls_initialized": list(self.controls_initialized),
            "managed_state": self.managed_state.describe(),
            "resources": {
                step: [choice.describe() for choice in choices]
                for step, choices in self.resources.items()
            },
            "resources_replaced": list(self.resources_replaced),
        }

        return description


@dataclass(frozen=True)
class UpdateAssessment:
    """What updating a session to a plan would do; nothing is built.

    Args:
        kind: ``preserve``, ``reset`` or ``unsupported``: the least
            disruptive update that applies the plan.
        base_version: The session's graph version when assessed.
        processing_version: The session's processing version when assessed.
        diff: The plan comparison; each change says what it requires.
        blockers: The session's reasons against a reset, each requiring
            ``unsupported``; they do not affect a preserving update.
        reset: What a reset would do; ``None`` when no reset can run.
    """

    kind: str
    base_version: int
    processing_version: int
    diff: PlanDiff
    blockers: Tuple[PlanChange, ...]
    reset: Optional[ResetConsequences]

    @property
    def preserve_blocked_by(self) -> Tuple[PlanChange, ...]:
        """The changes a preserving update rejects (``diff.breaking``)."""
        return self.diff.breaking

    @property
    def reset_blocked_by(self) -> Tuple[PlanChange, ...]:
        """The changes no update applies, then the session's ``blockers``."""
        blocked_by = (*self.diff.unsupported, *self.blockers)

        return blocked_by

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-friendly description."""
        description = {
            "kind": self.kind,
            "base_version": self.base_version,
            "processing_version": self.processing_version,
            "preserve_blocked_by": [
                change.describe() for change in self.preserve_blocked_by
            ],
            "reset_blocked_by": [change.describe() for change in self.reset_blocked_by],
            "reset": None if self.reset is None else self.reset.describe(),
            "diff": self.diff.describe(),
        }

        return description


def assess_update(
    session: "ExecutionSession",
    plan: "CompiledWorkflow",
    *,
    resources: Optional[Mapping[str, Any]] = None,
) -> UpdateAssessment:
    """Tell whether an update to ``plan`` preserves, resets or cannot apply.

    Args:
        session: The session to update.
        plan: The new compiled plan.
        resources: Caller values a reset would get; a reset may give an
            existing key a new value, e.g. an isolated ``managed_state``.

    Returns:
        The assessment.

    Raises:
        SessionClosedError: When the session was closed.
    """
    assessment, _ = assess_reset(session, plan, resources=resources)

    return assessment


def assess_reset(
    session: "ExecutionSession",
    plan: "CompiledWorkflow",
    *,
    resources: Optional[Mapping[str, Any]],
) -> Tuple[UpdateAssessment, "ResetState"]:
    """``assess_update`` plus the managed-state decision a reset carries out.

    Returns:
        The assessment and the ``state.session.ResetState`` decision.

    Raises:
        SessionClosedError: When the session was closed.
    """
    if session.closed:
        raise SessionClosedError(
            f"Session {session.session_id} is closed; it cannot be updated"
        )

    generation = session.generation
    old = generation.plan
    diff = compare_plans(old, plan)
    state_session = importlib.import_module(STATE_SESSION_MODULE)
    decision = state_session.plan_reset_state(
        plan,
        requested=requests_managed_state(plan),
        schema_changed=not (
            same_value(old.reactions.state, plan.reactions.state)
            and same_value(old.reactions.machines, plan.reactions.machines)
        ),
        service=session.managed_state,
        owned=session.owned_state,
        provided=generation.resolver.provided,
        resources=resources,
    )
    blockers = tuple(
        PlanChange(
            "managed_state",
            "managed_state",
            reason,
            detail=detail,
            requires=UNSUPPORTED,
        )
        for reason, detail in decision.blockers
    )
    if diff.kind != UNSUPPORTED:
        blockers += _source_resource_blockers(
            session, plan, provided=decision.resources
        )
    if old.is_active:
        runtime = importlib.import_module(ACTIVE_RUNTIME_MODULE)
        blockers += tuple(
            PlanChange(
                "run",
                "run",
                reason,
                detail=detail,
                requires=UNSUPPORTED,
            )
            for reason, detail in runtime.reset_blockers(session)
        )
    reset_blocked = bool(diff.unsupported or blockers)
    if diff.compatible:
        kind = PRESERVE
    elif not reset_blocked:
        kind = RESET
    else:
        kind = UNSUPPORTED
    consequences = None
    if not reset_blocked:
        consequences = _consequences(
            old,
            plan,
            decision=decision,
            provided=generation.resolver.provided,
            resources=resources or {},
        )
    assessment = UpdateAssessment(
        kind=kind,
        base_version=generation.graph_version,
        processing_version=generation.processing_version,
        diff=diff,
        blockers=blockers,
        reset=consequences,
    )

    return assessment, decision


def _source_resource_blockers(
    session: "ExecutionSession",
    plan: "CompiledWorkflow",
    *,
    provided: Mapping[str, Any],
) -> Tuple[PlanChange, ...]:
    """Reasons a reset would give a retained source and new processing two values.

    Sources keep the resources they were resolved with. A step shares a
    source's value when the session's resolver gives it the same object:
    a plain value under any key (``log`` and ``demo.log`` can hold one
    list), or the session ``Factory`` value of the source's own key.
    Handler sessions get the same plain values, but create their own
    ``Factory`` values. Every step and handler step of the new plan that
    shares a source's value now must get that same object from the reset's
    resolver. It does not when a session ``Factory`` would be created
    again, or when the reset's choice, by precedence (``namespace.name``
    over ``name``, caller over catalogue) or by a replaced caller value,
    is another object. Both are refused; the remedy is to pass the
    source's value. Nothing is created.
    """
    held = [
        (source, resolved)
        for source, resources in session.source_resources.items()
        for resolved in resources.values()
        if resolved.name != MANAGED_STATE_RESOURCE and resolved.source != "default"
    ]
    if not held:
        return ()

    by_key = {resolved.source: (source, resolved.value) for source, resolved in held}
    # ``None`` is no resource two consumers could share.
    by_identity = {
        id(resolved.value): (source, resolved.value)
        for source, resolved in held
        if resolved.value is not None
    }
    current = session.generation.resolver
    resolver = ResourceResolver(provided=provided, providers=plan.catalogue.providers)
    reasons: Dict[str, PlanChange] = {}
    for consumer in _consumers(plan):
        for spec in consumer.resources:
            bound, bound_value = current.candidate(spec, namespace=consumer.namespace)
            if bound.factory is None and bound.source not in ("default", "missing"):
                shared = by_identity.get(id(bound_value))
            elif bound.factory == "session" and not consumer.own_factories:
                shared = by_key.get(bound.source)
            else:
                # The source got a value of its own; nothing is shared.
                continue
            if shared is None or bound.source in reasons:
                continue
            source, value = shared
            choice, candidate = resolver.candidate(spec, namespace=consumer.namespace)
            if bound.factory == "session" and choice.factory is not None:
                reason = SOURCE_RESOURCE_SHARED
                detail = (
                    f"source {source!r} keeps the value its Factory created; a "
                    "reset would create another one for the new processing. Pass "
                    "the source's value as a caller resource: "
                    f"session.source_resources[{source!r}][{spec.name!r}].value"
                )
            elif choice.factory is not None or candidate is not value:
                reason = SOURCE_RESOURCE_OVERRIDDEN
                detail = (
                    f"source {source!r} keeps its value; the reset would give the "
                    f"new processing another one ({choice.source}). Keep the "
                    "value, or create a new session for a new source value"
                )
            else:
                continue
            reasons[bound.source] = PlanChange(
                "resource",
                bound.source,
                reason,
                detail=detail,
                requires=UNSUPPORTED,
            )

    return tuple(reasons.values())


class _Consumer(NamedTuple):
    """Resource declarations of one step of a plan or of a handler plan."""

    namespace: str
    resources: Tuple[ResourceSpec, ...]
    # A handler session's own resolver creates its Factory values.
    own_factories: bool


def _consumers(
    plan: "CompiledWorkflow", *, handler: bool = False
) -> Iterator[_Consumer]:
    for step in plan.steps:
        yield _Consumer(step.namespace, tuple(step.selected.resources), handler)
    for planned in plan.reactions.handlers:
        yield from _consumers(planned.plan, handler=True)


def carried_controls(
    old: "CompiledWorkflow", new: "CompiledWorkflow"
) -> FrozenSet[str]:
    """Names of the controls a reset carries: declared alike in both plans.

    Args:
        old: The session's current plan.
        new: The reset's plan.

    Returns:
        Control names whose declarations match, closures aside.
    """
    old_controls = old.controls.controls
    carried = frozenset(
        name
        for name, control in new.controls.controls.items()
        if name in old_controls
        and same_control_declaration(old_controls[name], control)
    )

    return carried


def _consequences(
    old: "CompiledWorkflow",
    new: "CompiledWorkflow",
    *,
    decision: "ResetState",
    provided: Mapping[str, Any],
    resources: Mapping[str, Any],
) -> ResetConsequences:
    new_paths = {step.path for step in new.steps}
    carried = carried_controls(old, new)
    # Managed state is reported on its own; its keys are configured later.
    resolver = ResourceResolver(
        provided=decision.resources, providers=new.catalogue.providers
    )
    choices = {
        format_step_path(step.path): tuple(
            choice
            for choice in resolver.choices(
                step.selected.resources, namespace=step.namespace
            )
            if choice.name != MANAGED_STATE_RESOURCE
        )
        for step in new.steps
    }
    consequences = ResetConsequences(
        sources_kept=tuple(new.sources),
        steps_constructed=tuple(format_step_path(step.path) for step in new.steps),
        steps_removed=tuple(
            format_step_path(step.path)
            for step in old.steps
            if step.path not in new_paths
        ),
        operators_constructed=tuple(new.operators),
        operators_removed=tuple(
            name for name in old.operators if name not in new.operators
        ),
        handlers_constructed=tuple(
            format_step_path(handler.path) for handler in new.reactions.handlers
        ),
        handlers_on_started=tuple(
            format_step_path(handler.path)
            for handler in new.reactions.handlers
            if handler.origin.kind == "system" and handler.origin.event == "started"
        ),
        controls_carried=tuple(
            name for name in new.controls.controls if name in carried
        ),
        controls_initialized=tuple(
            name for name in new.controls.controls if name not in carried
        ),
        managed_state=StateConsequence(
            action=decision.action, owner=decision.owner, detail=decision.detail
        ),
        resources=choices,
        resources_replaced=tuple(sorted(key for key in resources if key in provided)),
    )

    return consequences
