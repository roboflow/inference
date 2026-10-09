"""Constructor resources of V2 blocks and their resolution.

A block declares resources as the keyword parameters of its ``__init__``::

    class Detect(Block):
        def __init__(self, *, model_manager, api_key=None): ...

Resources are never part of workflow JSON and are never copied. An execution
session resolves them once per step and constructs the block. For a resource
named ``api_key`` of a block registered in catalogue namespace ``core``, the
first matching source wins:

1. the caller's value under ``"core.api_key"``;
2. the caller's value under ``"api_key"``;
3. the catalogue's provider for ``"api_key"`` in namespace ``core``;
4. the constructor default.

A value wrapped in ``Factory`` is created lazily, also when it is the
constructor default: once per session and source (``scope="session"``; a
default's source is its block type and parameter) or once per step
(``scope="step"``). Every other
value, including a plain callable, is passed through unchanged and keeps its
identity. This differs from V1, which called any callable initializer and
ranked plugin initializers above the caller's unscoped values.
"""

import inspect
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Dict, Iterable, Literal, Mapping, Optional, Tuple

from roboflow_workflows.execution_engine.v2.errors import (
    ContractError,
    ResourceError,
    StepPath,
)

FactoryScope = Literal["session", "step"]

MANAGED_STATE_RESOURCE = "managed_state"
"""Reserved constructor resource name of managed state (``state.api`` owns it)."""

_NO_DEFAULT = inspect.Parameter.empty


@dataclass(frozen=True)
class ResourceSpec:
    """One constructor resource of a block.

    Args:
        name: Keyword parameter name of ``__init__``.
        required: Whether the parameter has no default.
        default: Constructor default; ``inspect.Parameter.empty`` if none.
        annotation: Declared annotation, for documentation only.
    """

    name: str
    required: bool
    default: Any = _NO_DEFAULT
    annotation: Any = _NO_DEFAULT

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-friendly description."""
        annotation = None
        if self.annotation is not _NO_DEFAULT:
            annotation = getattr(self.annotation, "__name__", repr(self.annotation))
        description = {
            "name": self.name,
            "required": self.required,
            "annotation": annotation,
        }

        return description


@dataclass(frozen=True)
class Factory:
    """A lazily created resource.

    Args:
        create: Zero-argument callable creating the resource.
        scope: ``session`` shares one created value across all steps of a
            session; ``step`` creates one value per step.

    Raises:
        ContractError: When ``create`` is not callable or the scope is unknown.
    """

    create: Callable[[], Any]
    scope: FactoryScope = "session"

    def __post_init__(self) -> None:
        if not callable(self.create):
            raise ContractError(f"Factory create must be callable, got {self.create!r}")
        if self.scope not in ("session", "step"):
            raise ContractError(
                f"Factory scope must be 'session' or 'step', got {self.scope!r}"
            )


@dataclass(frozen=True)
class ResolvedResource:
    """A resource value chosen for one step.

    Args:
        name: Constructor parameter name.
        value: Value passed to the constructor.
        source: Where it came from: ``"provided:<key>"``,
            ``"catalogue:<namespace>.<name>"`` or ``"default"``.
    """

    name: str
    value: Any
    source: str


@dataclass(frozen=True)
class ResourceChoice:
    """Where a resource of one step would come from; nothing is created.

    Args:
        name: Constructor parameter name.
        source: ``"provided:<key>"``, ``"catalogue:<namespace>.<name>"``,
            ``"default"``, or ``"missing"`` for a required resource without
            any value.
        factory: ``"session"`` or ``"step"`` when the value is a ``Factory``
            that construction creates; ``None`` for a value passed as is.
    """

    name: str
    source: str
    factory: Optional[FactoryScope] = None

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-friendly description."""
        return {"name": self.name, "source": self.source, "factory": self.factory}


def read_resource_specs(block_class: type) -> Tuple[ResourceSpec, ...]:
    """Read constructor resources from a block class's ``__init__``.

    Args:
        block_class: Block class to inspect. It is not instantiated.

    Returns:
        Resource declarations in signature order; empty without ``__init__``.

    Raises:
        ContractError: On ``*args``, ``**kwargs`` or positional-only parameters.
    """
    initializer = block_class.__init__
    if initializer is object.__init__:
        return ()

    specs = []
    parameters = list(inspect.signature(initializer).parameters.values())[1:]
    for parameter in parameters:
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.POSITIONAL_ONLY,
        ):
            raise ContractError(
                f"__init__ parameter {parameter.name!r} must be a named keyword "
                "parameter; resources are injected by name"
            )
        specs.append(
            ResourceSpec(
                name=parameter.name,
                required=parameter.default is _NO_DEFAULT,
                default=parameter.default,
                annotation=parameter.annotation,
            )
        )

    return tuple(specs)


class ResourceResolver:
    """Resolves constructor resources for the steps of one execution session.

    Args:
        provided: Caller values keyed by ``"name"`` or ``"namespace.name"``.
        providers: Catalogue providers: namespace to name to value.
    """

    def __init__(
        self,
        *,
        provided: Optional[Mapping[str, Any]] = None,
        providers: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ):
        self._provided = dict(provided or {})
        self._providers = {
            namespace: dict(values) for namespace, values in (providers or {}).items()
        }
        self._session_values: Dict[str, Any] = {}

    def fork(
        self,
        *,
        provided: Optional[Mapping[str, Any]] = None,
        replacing: Optional[Mapping[str, Any]] = None,
    ) -> "ResourceResolver":
        """Return a resolver that starts from this one and never changes it.

        The fork sees the same caller values, providers and session factory
        values. Values it creates later stay in the fork, so a graph update
        resolves its new steps in a fork and the session adopts the fork only
        when the update is applied.

        Args:
            provided: Additional caller values; their keys must be new.
            replacing: Values that replace keys in the fork only, applied
                after ``provided``; a graph update maps the managed-state keys
                its new steps chose to the session's service.

        Returns:
            The forked resolver.

        Raises:
            ContractError: When ``provided`` repeats a key of this resolver.
        """
        extra = dict(provided or {})
        repeated = sorted(key for key in extra if key in self._provided)
        if repeated:
            raise ContractError(
                f"resources {repeated} are already provided to this session; an "
                "update may add new resource keys only"
            )

        forked = ResourceResolver(
            provided={**self._provided, **extra, **dict(replacing or {})},
            providers=self._providers,
        )
        forked._session_values = dict(self._session_values)

        return forked

    @property
    def provided(self) -> Mapping[str, Any]:
        """Caller values of this resolver by key, read-only."""
        return MappingProxyType(self._provided)

    def choices(
        self, resources: Iterable[ResourceSpec], *, namespace: str
    ) -> Tuple[ResourceChoice, ...]:
        """Tell where each resource of one step would come from.

        Uses the precedence of ``resolve`` but creates nothing: a ``Factory``
        is reported, not called.

        Args:
            resources: The block's resource declarations.
            namespace: Catalogue namespace of the block; may be empty.

        Returns:
            One choice per declaration, in declaration order.
        """
        choices = tuple(
            self.candidate(resource, namespace=namespace)[0] for resource in resources
        )

        return choices

    def candidate(
        self, resource: ResourceSpec, *, namespace: str
    ) -> Tuple[ResourceChoice, Any]:
        """Tell where one resource would come from, and the value found there.

        Uses the precedence of ``resolve`` but creates nothing: a ``Factory``
        is returned as it is, not called.

        Args:
            resource: The block's resource declaration.
            namespace: Catalogue namespace of the block; may be empty.

        Returns:
            The choice and its value: the caller's or catalogue's value, the
            constructor default, or ``None`` for ``missing``.
        """
        chosen = next(iter(self._candidates(resource.name, namespace=namespace)), None)
        if chosen is not None:
            source, value = chosen
        elif resource.required:
            source, value = "missing", None
        else:
            source, value = "default", resource.default
        factory = value.scope if isinstance(value, Factory) else None
        choice = ResourceChoice(name=resource.name, source=source, factory=factory)

        return choice, value

    def resolve(
        self,
        resources: Iterable[ResourceSpec],
        *,
        namespace: str,
        step_path: StepPath,
        block_type: str,
    ) -> Dict[str, ResolvedResource]:
        """Choose a value for every resource of one step.

        Args:
            resources: The block's resource declarations.
            namespace: Catalogue namespace of the block; may be empty.
            step_path: Step being constructed, for errors and step factories.
            block_type: Block type of the step, for errors.

        Returns:
            Mapping of parameter name to the chosen value and its source.

        Raises:
            ResourceError: When a required resource has no source or a
                ``Factory`` raises.
        """
        resolved: Dict[str, ResolvedResource] = {}
        for resource in resources:
            candidates = self._candidates(resource.name, namespace=namespace)
            chosen = next(iter(candidates), None)
            if chosen is None:
                if resource.required:
                    raise ResourceError(
                        f"no value for required resource {resource.name!r}; "
                        f"provide {self._keys_for(resource.name, namespace=namespace)}",
                        step_path=step_path,
                        block_type=block_type,
                        parameter=resource.name,
                    )
                value = self._materialize(
                    resource.default,
                    cache_key=f"default:{block_type}.{resource.name}",
                    step_path=step_path,
                    block_type=block_type,
                    parameter=resource.name,
                )
                resolved[resource.name] = ResolvedResource(
                    name=resource.name, value=value, source="default"
                )
                continue

            source, value = chosen
            value = self._materialize(
                value,
                cache_key=source,
                step_path=step_path,
                block_type=block_type,
                parameter=resource.name,
            )
            resolved[resource.name] = ResolvedResource(
                name=resource.name, value=value, source=source
            )

        return resolved

    def _candidates(self, name: str, *, namespace: str) -> Iterable[Tuple[str, Any]]:
        if namespace and f"{namespace}.{name}" in self._provided:
            yield f"provided:{namespace}.{name}", self._provided[f"{namespace}.{name}"]
        if name in self._provided:
            yield f"provided:{name}", self._provided[name]
        catalogue_values = self._providers.get(namespace, {})
        if name in catalogue_values:
            yield f"catalogue:{namespace}.{name}", catalogue_values[name]

    def _keys_for(self, name: str, *, namespace: str) -> str:
        keys = [f"'{namespace}.{name}'", f"'{name}'"] if namespace else [f"'{name}'"]

        return " or ".join(keys)

    def _materialize(
        self,
        value: Any,
        *,
        cache_key: str,
        step_path: StepPath,
        block_type: str,
        parameter: str,
    ) -> Any:
        # cache_key names the source; a session factory is created once per key.
        if not isinstance(value, Factory):
            return value
        if value.scope == "session" and cache_key in self._session_values:
            return self._session_values[cache_key]

        try:
            created = value.create()
        except Exception as error:
            raise ResourceError(
                f"factory for resource {parameter!r} ({cache_key}) failed: {error}",
                step_path=step_path,
                block_type=block_type,
                parameter=parameter,
            ) from error
        if value.scope == "session":
            self._session_values[cache_key] = created

        return created
