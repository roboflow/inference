"""Managed state of one execution session, chosen by resource resolution.

A session and its reaction handlers use one managed-state service. Every step,
source and handler step that requests ``managed_state`` resolves it with the
normal ``ResourceResolver`` precedence (``"<namespace>.managed_state"``,
``"managed_state"``, catalogue provider, constructor default). A ``Factory``
is created once. Choices that resolve to different objects are rejected
before any block is constructed. A ``None`` value is no choice. Without any
choice, the engine creates and owns an in-memory ``ManagedState``; a chosen
service, a constructor default included, stays the caller's.

The plan module imports this module lazily, only for plans that need state.
"""

from dataclasses import replace
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    Iterator,
    Mapping,
    NamedTuple,
    Optional,
    Tuple,
)

from roboflow_workflows.execution_engine.v2.errors import ResourceError, StepPath
from roboflow_workflows.execution_engine.v2.resources import (
    ResolvedResource,
    ResourceResolver,
    ResourceSpec,
)
from roboflow_workflows.execution_engine.v2.state.api import (
    MANAGED_STATE_RESOURCE,
    ManagedState,
)
from roboflow_workflows.execution_engine.v2.state.memory import InMemoryStateBackend

if TYPE_CHECKING:
    from roboflow_workflows.execution_engine.v2.plan import CompiledWorkflow

# Plans that only declare defaults or machines resolve the unscoped key.
_NO_DEMAND = ResourceSpec(name=MANAGED_STATE_RESOURCE, required=False, default=None)
_PROVIDED = "provided:"


class SessionState(NamedTuple):
    """Managed state chosen for one session.

    Args:
        service: Service given to every demand: the view with the workflow's
            declared defaults, or the chosen service when there are none.
        owned: ``ManagedState`` the engine created and the session closes;
            ``None`` when the service is the caller's, a provider's or a
            constructor default.
        resources: Caller resources where ``managed_state`` and every key a
            demand chose map to ``service``; other keys are unchanged.
    """

    service: Any
    owned: Any
    resources: Dict[str, Any]


class _Demand(NamedTuple):
    namespace: str
    step_path: StepPath
    block_type: str
    spec: ResourceSpec


def configure_session_state(
    plan: "CompiledWorkflow",
    *,
    resources: Optional[Mapping[str, Any]],
    session_id: str,
) -> SessionState:
    """Choose, create and seed the managed state of one session.

    Args:
        plan: Compiled plan whose session needs managed state.
        resources: Caller resources keyed by ``name`` or ``namespace.name``.
        session_id: Namespace of a state the engine creates.

    Returns:
        The configured service, the engine-owned original and the resources
        to construct the plan and its handler sessions with.

    Raises:
        ResourceError: When demands resolve to different service objects or a
            ``managed_state`` factory fails.
    """
    resources = dict(resources or {})
    resolver = ResourceResolver(provided=resources, providers=plan.catalogue.providers)
    demands = list(_demands(plan)) or [_Demand("", (), "", _NO_DEMAND)]
    choices = [(demand, _resolve(resolver, demand=demand)) for demand in demands]
    # A None value, defaulted or passed, is no choice: it gets the service.
    selected = [
        (demand, chosen) for demand, chosen in choices if chosen.value is not None
    ]
    for demand, chosen in selected[1:]:
        _check_same_service(selected[0], (demand, chosen))

    owned = None
    if selected:
        original = selected[0][1].value
    else:
        original = owned = ManagedState(namespace=session_id)
    service = _with_defaults(plan, original=original, owned=owned)

    chosen_keys = {
        chosen.source[len(_PROVIDED) :]
        for _, chosen in choices
        if chosen.source.startswith(_PROVIDED)
    }
    configured_resources = {
        **resources,
        **{key: service for key in chosen_keys},
        MANAGED_STATE_RESOURCE: service,
    }
    session_state = SessionState(
        service=service, owned=owned, resources=configured_resources
    )

    return session_state


def added_state_keys(
    plan: "CompiledWorkflow",
    paths: Iterable[StepPath],
    *,
    resolver: ResourceResolver,
    service: Any,
) -> Dict[str, Any]:
    """Check that the added steps of a graph update share the session service.

    Each added step that requests ``managed_state`` resolves it like at session
    creation. ``None`` is no choice and gets ``service``; any other object
    than ``service`` is rejected. Nothing is constructed or seeded.

    Args:
        plan: The update's new plan.
        paths: Paths of the steps the update adds.
        resolver: The update's resolver, before any added step is constructed.
        service: The session's managed state (``ExecutionSession.managed_state``).

    Returns:
        Every provided key the added demands chose, mapped to ``service``, so
        the update resolves them as session creation does.

    Raises:
        ResourceError: When a demand resolves to another service.
    """
    keys: Dict[str, Any] = {}
    for path in paths:
        step = plan.step(path)
        spec = _state_spec(step.selected.resources)
        if spec is None:
            continue
        demand = _Demand(step.namespace, step.path, step.block_type, spec)
        chosen = _resolve(resolver, demand=demand)
        if chosen.value is not None and chosen.value is not service:
            raise ResourceError(
                f"{_describe(demand, chosen)} is not the session's managed state "
                "service; a new step shares the session's service, so pass "
                "session.managed_state or no value",
                step_path=demand.step_path,
                block_type=demand.block_type,
                parameter=MANAGED_STATE_RESOURCE,
            )
        if chosen.source.startswith(_PROVIDED):
            keys[chosen.source[len(_PROVIDED) :]] = service

    return keys


class ResetState(NamedTuple):
    """What a reset does with managed state, decided before anything is built.

    Args:
        action: ``none`` (the new plan uses no managed state), ``fresh`` (the
            engine creates a new in-memory state), ``retained`` (the caller's
            service stays, with every stored value and machine record) or
            ``replaced`` (another caller or catalogue service).
        owner: ``engine`` or ``caller``; ``None`` for ``none``.
        detail: Human-readable explanation.
        blockers: ``(reason, detail)`` of every condition that rules the
            reset out; empty when it can run.
        resources: Caller resources to configure the new state from: the
            session's, without the old service unless it is retained, and
            the reset's own on top.
    """

    action: str
    owner: Optional[str]
    detail: str
    blockers: Tuple[Tuple[str, str], ...]
    resources: Dict[str, Any]


STATE_NONE = "none"
STATE_FRESH = "fresh"
STATE_RETAINED = "retained"
STATE_REPLACED = "replaced"

SOURCE_SHARES_STATE = "source_shares_state"
STATE_NOT_ISOLATED = "state_not_isolated"
STATE_BACKEND_CLOSES = "state_backend_closes"
STATE_ISOLATION_UNKNOWN = "state_isolation_unknown"
RETAINED_STATE_SCHEMA_CHANGED = "retained_state_schema_changed"


def plan_reset_state(
    plan: "CompiledWorkflow",
    *,
    requested: bool,
    schema_changed: bool,
    service: Any,
    owned: Any,
    provided: Mapping[str, Any],
    resources: Optional[Mapping[str, Any]],
) -> ResetState:
    """Decide the managed state of a reset without creating or writing anything.

    A caller's service is never cleared, and a source keeps the service it
    was resolved with. The first matching row decides::

        the new plan uses no managed state                  none
        the session uses a caller's service, and no
          supplied service stores anywhere else             retained (caller)
        a service is supplied, or the new plan chooses
          one itself (catalogue provider, default)          replaced (caller)
        otherwise                                           fresh (engine)

    Blockers rule the reset out instead of splitting or sharing state:

        retained         the plan changes state defaults or machines
        replaced, fresh  a source uses the current service; a supplied
                         service stores where the current one does; the
                         engine cannot tell where a supplied one stores;
                         a supplied service uses the backend of the
                         engine-owned state, which closes after the commit

    Args:
        plan: The new plan.
        requested: Whether ``plan`` needs managed state
            (``plan.requests_managed_state``).
        schema_changed: Whether ``plan`` declares other state defaults or
            machines than the session's plan.
        service: The session's managed state; ``None`` when it has none.
        owned: The engine-owned state of the session; ``None`` when the
            service is a caller's.
        provided: The session resolver's caller values.
        resources: Caller values of the reset; a ``managed_state`` key with
            another object than ``service`` replaces it.

    Returns:
        The decision with its blockers.
    """
    resources = dict(resources or {})
    # The session's caller values without the current service, then the
    # reset's own: what a new service is configured from.
    without_current = {
        **{
            key: value
            for key, value in provided.items()
            if service is None or value is not service
        },
        **resources,
    }
    if not requested:
        return ResetState(
            STATE_NONE, None, "the new plan uses no managed state", (), without_current
        )

    # Where each supplied service stores, relative to the current one.
    relations = [
        _storage_relation(value, service)
        for key, value in resources.items()
        if _is_state_key(key) and value is not None
    ]
    supplies_another = any(relation != _SAME for relation in relations)
    if service is not None and owned is None and not supplies_another:
        blockers = []
        if schema_changed:
            blockers.append(
                (
                    RETAINED_STATE_SCHEMA_CHANGED,
                    "the plan changes state defaults or machines, but the caller's "
                    "managed state stays with its stored values and machine "
                    "records; pass an isolated replacement under managed_state, "
                    "e.g. ManagedState(backend, namespace=<new name>)",
                )
            )
        detail = (
            "the caller's managed state stays: stored values and machine "
            "records are kept, never cleared"
        )
        # The session's keys already map to the service, the view with
        # defaults included; a supplied original must not bypass it.
        retained = {
            **dict(provided),
            **{
                key: value for key, value in resources.items() if not _is_state_key(key)
            },
        }
        return ResetState(STATE_RETAINED, "caller", detail, tuple(blockers), retained)

    # The processing gets a new service: replaced or fresh.
    blockers = []
    if service is not None and _sources_request_state(plan):
        blockers.append(
            (
                SOURCE_SHARES_STATE,
                "a source of this session uses its managed state and keeps it "
                "across a reset; the processing would no longer share state "
                "with the source. Create a new session instead",
            )
        )
    if _SAME in relations:
        blockers.append(
            (
                STATE_NOT_ISOLATED,
                "the engine-owned managed state cannot stay across a reset; pass "
                "no managed_state for a fresh one, or a service on a backend "
                "of its own",
            )
        )
    if _UNKNOWN in relations:
        blockers.append(
            (
                STATE_ISOLATION_UNKNOWN,
                "the engine cannot tell that the supplied managed state stores "
                "apart from the current one: another backend object with the "
                "same namespace may address the same store; pass "
                "ManagedState(backend, namespace=<new name>) to isolate it",
            )
        )
    # Another namespace stores apart, but on a backend the retirement closes.
    if owned is not None and any(
        _borrows_backend(value, owned) and _storage_relation(value, service) != _SAME
        for key, value in resources.items()
        if _is_state_key(key)
    ):
        blockers.append((STATE_BACKEND_CLOSES, _BACKEND_CLOSES_DETAIL))
    if supplies_another or _chooses_service(plan, provided=without_current):
        detail = (
            "the processing uses another caller service; previous engine-owned "
            "state is closed after commit, while previous caller-owned state "
            "is left untouched"
        )
        return ResetState(
            STATE_REPLACED, "caller", detail, tuple(blockers), without_current
        )

    detail = "the engine creates a new in-memory managed state"
    if owned is not None:
        detail += "; the previous one closes after the commit"

    return ResetState(STATE_FRESH, "engine", detail, tuple(blockers), without_current)


def configure_reset_state(
    plan: "CompiledWorkflow",
    *,
    decision: ResetState,
    service: Any,
    owned: Any,
    session_id: str,
) -> SessionState:
    """Create the managed state a reset decided on; seeds only new state.

    Args:
        plan: The new plan.
        decision: The ``plan_reset_state`` decision; it has no blockers.
        service: The session's current managed state.
        owned: The session's engine-owned state; ``None`` when it has none.
        session_id: Namespace of a state the engine creates.

    Returns:
        The state for the reset's steps and handler sessions. ``owned`` is
        the state the candidate created and closes when it is not applied.

    Raises:
        ResourceError: As ``configure_session_state``; when a ``Factory``
            created a service on the backend of ``owned``, which closes
            after the commit.
    """
    if decision.action == STATE_NONE:
        return SessionState(service=None, owned=None, resources=decision.resources)
    if decision.action == STATE_RETAINED:
        # The service already carries the unchanged defaults; nothing is seeded.
        return SessionState(service=service, owned=None, resources=decision.resources)

    session_state = configure_session_state(
        plan, resources=decision.resources, session_id=session_id
    )
    # The assessment saw supplied services only; a Factory's shows up here.
    if owned is not None and _borrows_backend(session_state.service, owned):
        raise ResourceError(_BACKEND_CLOSES_DETAIL, parameter=MANAGED_STATE_RESOURCE)

    return session_state


def _is_state_key(key: str) -> bool:
    return key == MANAGED_STATE_RESOURCE or key.endswith(f".{MANAGED_STATE_RESOURCE}")


def _sources_request_state(plan: "CompiledWorkflow") -> bool:
    requested = any(
        _state_spec(source.spec.resources) is not None
        for source in plan.sources.values()
    )

    return requested


_SAME = "same"
_ISOLATED = "isolated"
_UNKNOWN = "unknown"


def _storage_relation(first: Any, second: Any) -> str:
    """Whether two services store into the same place, another place, or unknown.

    The engine cannot see where an external backend stores, and two backend
    objects (e.g. two Redis clients) can address one store. So only these
    are known: the same object, or the same backend object and namespace,
    is the same storage; another namespace, or two private in-memory
    backends, is isolated storage. Anything else is unknown.
    """
    if first is second:
        return _SAME

    if first is None or second is None:
        return _ISOLATED

    try:
        backends = (first.backend, second.backend)
        namespaces = (first.namespace, second.namespace)
    except AttributeError:
        return _UNKNOWN

    if namespaces[0] != namespaces[1]:
        return _ISOLATED

    if backends[0] is backends[1]:
        return _SAME

    if all(isinstance(backend, InMemoryStateBackend) for backend in backends):
        return _ISOLATED

    return _UNKNOWN


_BACKEND_CLOSES_DETAIL = (
    "the supplied managed state uses the backend of the engine-owned state, "
    "which the reset closes after its commit; pass no managed_state for a "
    "fresh one, or a service on a backend of its own"
)


def _borrows_backend(service: Any, owned: Any) -> bool:
    """Whether ``service`` stores through the backend that ``owned`` closes."""
    borrows = (
        service is not owned and getattr(service, "backend", None) is owned.backend
    )

    return borrows


def _chooses_service(plan: "CompiledWorkflow", *, provided: Mapping[str, Any]) -> bool:
    """Whether resolution picks a service for some demand; creates nothing."""
    resolver = ResourceResolver(provided=provided, providers=plan.catalogue.providers)
    for demand in _demands(plan):
        spec = demand.spec
        if spec.required:
            spec = replace(spec, required=False, default=None)
        (choice,) = resolver.choices((spec,), namespace=demand.namespace)
        if choice.source.startswith(_PROVIDED):
            value = resolver.provided[choice.source[len(_PROVIDED) :]]
        elif choice.source == "default":
            value = spec.default
        else:
            value = plan.catalogue.providers[demand.namespace][spec.name]
        if value is not None:
            return True

    return False


def _demands(plan: "CompiledWorkflow", prefix: StepPath = ()) -> Iterator[_Demand]:
    """Yield every constructor request of ``managed_state``, handlers included."""
    for step in plan.steps:
        spec = _state_spec(step.selected.resources)
        if spec is not None:
            yield _Demand(step.namespace, (*prefix, *step.path), step.block_type, spec)
    for source in plan.sources.values():
        spec = _state_spec(source.spec.resources)
        if spec is not None:
            yield _Demand(
                source.namespace,
                (*prefix, *source.step_path),
                source.spec.type,
                spec,
            )
    for handler in plan.reactions.handlers:
        yield from _demands(handler.plan, prefix=(*prefix, *handler.path))


def _state_spec(specs: Iterable[ResourceSpec]) -> Optional[ResourceSpec]:
    return next((spec for spec in specs if spec.name == MANAGED_STATE_RESOURCE), None)


def _resolve(resolver: ResourceResolver, *, demand: _Demand) -> ResolvedResource:
    # The actual spec, so a constructor default is a choice like any other;
    # a required one without a value gets the engine's service.
    spec = demand.spec
    if spec.required:
        spec = replace(spec, required=False, default=None)
    resolved = resolver.resolve(
        (spec,),
        namespace=demand.namespace,
        step_path=demand.step_path,
        block_type=demand.block_type,
    )

    return resolved[MANAGED_STATE_RESOURCE]


def _check_same_service(
    first: Tuple[_Demand, ResolvedResource], other: Tuple[_Demand, ResolvedResource]
) -> None:
    if other[1].value is first[1].value:
        return

    demand = other[0]
    raise ResourceError(
        f"{_describe(*first)} and {_describe(*other)} are different managed "
        "state services; a session and its handlers share one, so pass the "
        "same object under every key",
        step_path=demand.step_path,
        block_type=demand.block_type or None,
        parameter=MANAGED_STATE_RESOURCE,
    )


def _describe(demand: _Demand, chosen: ResolvedResource) -> str:
    if chosen.source == "default":
        return f"the constructor default of {demand.block_type}"

    return chosen.source


def _with_defaults(plan: "CompiledWorkflow", *, original: Any, owned: Any) -> Any:
    defaults = plan.reactions.state
    if defaults.is_empty:
        return original

    # Seeding writes the global defaults and may fail on the backend.
    try:
        service = original.with_defaults(
            global_=defaults.global_, source=defaults.source
        )
    except BaseException:
        if owned is not None:
            owned.close()
        raise

    return service
