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
