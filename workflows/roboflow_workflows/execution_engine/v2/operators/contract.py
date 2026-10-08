"""Class-owned operator declarations and the push contract of the V2 engine.

An operator is the explicit transition between pulse domains. It reads values
at the end of pulses of its upstream domains (sources or other operators) and
emits pulses of its own domain, named like the operator. Like a block or a
source, one class declares everything and the catalogue only collects it::

    class Align(Operator):
        \"\"\"Pair each leader sample with the nearest follower samples.\"\"\"

        type = "v2/align@v1"
        input_roles = ("input",)

        class Params(OperatorParams):
            clock: str = Field(description="Clock compared by timestamps.")

        @classmethod
        def plan_ports(cls, name, params, inputs):
            return {item.name: OperatorPort(*item.kinds) for item in inputs}

        def push(self, arrivals): ...

Lifecycle, on the processor thread of one active run::

    operator = OperatorClass(name=..., params=..., inputs=...)   # per start()
    operator.push(arrivals)        # the inputs of one upstream pulse, together
    operator.end_input(name)       # that input's domain terminated
    operator.finish(reason)        # every upstream domain terminated; once
    operator.close()               # exactly once, also after a failure

``push``, ``end_input`` and ``finish`` return the pulses to emit now, in
order. The runtime executes each emitted pulse before it continues, so an
operator never waits for future data: whatever cannot be decided yet stays
retained under the operator's own declared bounds.
"""

import inspect
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    List,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Type,
)

from pydantic import ValidationError
from roboflow_workflows.execution_engine.v2._validation import clean_errors
from roboflow_workflows.execution_engine.v2.data import EntryLayout
from roboflow_workflows.execution_engine.v2.declaration import (
    _IDENTITY,
    RESERVED_PARAM_NAMES,
    BlockParams,
    _params_json_schema,
    _validate_aliases,
    _validate_compatibility,
    is_selector_segment,
)
from roboflow_workflows.execution_engine.v2.errors import (
    ContractError,
    ParamsValidationError,
    StepPath,
    WorkflowCompileError,
)
from roboflow_workflows.execution_engine.v2.kinds import Kind, normalize_kinds

if TYPE_CHECKING:
    # Runtime imports would cycle: the catalogue imports this module and the
    # plan and execution packages import the catalogue.
    from roboflow_workflows.execution_engine.v2.execution.entries import Entry
    from roboflow_workflows.execution_engine.v2.plan import PulseKey

__all__ = [
    "INPUT_MAP_ROLES",
    "Arrival",
    "InputRole",
    "Operator",
    "OperatorCounters",
    "OperatorDeclarationError",
    "OperatorInput",
    "OperatorParams",
    "OperatorPort",
    "OperatorPulse",
    "OperatorSpec",
    "TerminationReason",
    "operator_step_path",
    "spec_of_operator",
]

InputRole = Literal["input", "collect", "hold"]
"""How an operator consumes one named selector."""

INPUT_MAP_ROLES: Mapping[str, InputRole] = MappingProxyType(
    {"inputs": "input", "collect": "collect", "hold": "hold"}
)
"""Definition key of each ``{name: selector}`` map, and the role of its names."""

TerminationReason = Literal["eof", "stop"]
"""Why an operator finishes: its upstream domains ended, or the run stopped."""


class OperatorDeclarationError(ContractError):
    """An operator class declares an invalid or inconsistent contract.

    Raised while the class body is being created, so invalid operators fail
    at import time rather than during compilation or execution.

    Args:
        message: Human-readable explanation.
        operator_class: Name of the offending class, when known.
    """

    def __init__(self, message: str, *, operator_class: Optional[str] = None):
        prefix = f"Operator class {operator_class}: " if operator_class else ""
        super().__init__(prefix + message)
        self.operator_class = operator_class


class OperatorParams(BlockParams):
    """Base class of every operator's nested ``Params`` model.

    Operator parameters are literals: selectors are not accepted. Unknown
    parameters are rejected and validated instances are immutable.
    """


class OperatorInput(NamedTuple):
    """One named input as ``plan_ports`` and the constructor see it.

    Args:
        name: Key in the declaration's ``inputs``, ``collect`` or ``hold`` map.
        role: ``input``, ``collect`` or ``hold``.
        layout: Layout of the bound value.
        kinds: Kinds of the bound value.
    """

    name: str
    role: InputRole
    layout: EntryLayout
    kinds: Tuple[Kind, ...] = ()


@dataclass(frozen=True, init=False)
class OperatorPort:
    """One port an operator plans for its own domain.

    Args:
        *kinds: Kinds of the emitted payloads; none means the wildcard.
        layout: Layout of the port; axes the operator introduces use ids
            scoped as ``operators.<operator>:<local id>``.

    Raises:
        ContractError: On invalid kinds or a layout that is not an
            ``EntryLayout``.
    """

    kinds: Tuple[Kind, ...]
    layout: EntryLayout

    def __init__(self, *kinds: Kind, layout: EntryLayout = EntryLayout()):
        normalized_kinds = normalize_kinds(kinds, context="OperatorPort")
        if not isinstance(layout, EntryLayout):
            raise ContractError(
                f"OperatorPort layout must be an EntryLayout, got {layout!r}"
            )

        object.__setattr__(self, "kinds", normalized_kinds)
        object.__setattr__(self, "layout", layout)

    @property
    def kind_names(self) -> Tuple[str, ...]:
        """Declared kind names."""
        return tuple(kind.name for kind in self.kinds)


@dataclass(frozen=True)
class Arrival:
    """The value of one operator input at the end of one upstream pulse.

    Args:
        input: Name of the operator input.
        entry: The bound value; immutable, payloads shared and never copied.
            A filtered entry is an arrival carrying absence.
        pulse: The upstream pulse that produced the value.
    """

    input: str
    entry: "Entry"
    pulse: "PulseKey"


@dataclass(frozen=True)
class OperatorPulse:
    """One pulse an operator emits for its own domain.

    Args:
        ports: Entry per present port; an omitted port is terminally absent
            for the pulse.
        causes: Identities of the upstream pulses that contributed, in order
            and without repetition.
    """

    ports: Mapping[str, "Entry"]
    causes: Tuple["PulseKey", ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "ports", MappingProxyType(dict(self.ports)))
        object.__setattr__(self, "causes", tuple(self.causes))


@dataclass
class OperatorCounters:
    """What happened to one operator during one active run.

    Two owners, so an operator never maintains the common accounting::

        runtime   arrivals, emitted, processed, delivered, cancelled,
                  finished, closed          (every operator, automatically)
        operator  filtered, late, dropped, evicted, partial_dropped,
                  peak_retained             (its own policies; optional)

    An operator driven directly, outside an active run, therefore reports
    only its policy counters.

    Args:
        arrivals: Arrivals the runtime pushed, filtered and late ones included.
        filtered: Arrivals carrying absence (no usable value).
        late: Arrivals discarded for not advancing their input's timeline.
        dropped: Values discarded without contributing to an emission, for
            example a leader without a match under ``missing='drop'``.
        evicted: Retained values released once provably unusable.
        partial_dropped: Incomplete groups discarded at termination by policy.
        emitted: Pulses that ``push``, ``end_input`` or ``finish`` returned
            to the runtime. A pulse built inside a call that then raised was
            never emitted.
        processed: Emitted pulses whose route ran and whose handlers returned.
        delivered: Group results of emitted pulses handed to handlers that
            returned.
        omitted: Group results of emitted pulses not delivered because every
            field read steps of a disabled control (``controls``).
        cancelled: Emitted pulses that did not complete because the run
            failed, started or not. ``emitted == processed + cancelled`` once
            the run is done.
        peak_retained: Largest number of values held for one input at once.
        finished: Whether ``finish`` returned.
        closed: Whether ``close`` returned.
    """

    arrivals: int = 0
    filtered: int = 0
    late: int = 0
    dropped: int = 0
    evicted: int = 0
    partial_dropped: int = 0
    emitted: int = 0
    processed: int = 0
    delivered: int = 0
    omitted: int = 0
    cancelled: int = 0
    peak_retained: int = 0
    finished: bool = False
    closed: bool = False


@dataclass(frozen=True)
class OperatorSpec:
    """Validated, class-owned declaration of one operator type.

    Built once per concrete ``Operator`` subclass; read it with
    ``spec_of_operator``. Holds no instance.

    Args:
        type: Canonical identity used by ``operators`` declarations.
        aliases: Additional accepted identities.
        operator_class: The declaring class.
        params_model: The nested ``Params`` model.
        input_roles: Roles the operator accepts, e.g. ``("collect", "hold")``.
        engine_compatibility: PEP 440 specifier of supported engine versions.
        description: Operator documentation.
        metadata: Free-form UI and catalogue metadata.
        kinds: Kinds the declaration itself references; operators derive
            their port kinds from their inputs, so usually none.
    """

    type: str
    aliases: Tuple[str, ...]
    operator_class: type
    params_model: Type[OperatorParams]
    input_roles: Tuple[InputRole, ...]
    engine_compatibility: Optional[str]
    description: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    kinds: Tuple[Kind, ...] = ()

    @property
    def identities(self) -> Tuple[str, ...]:
        """Canonical type followed by aliases."""
        return (self.type,) + self.aliases

    def validate_params(
        self, raw: Mapping[str, Any], *, operator_name: str = ""
    ) -> OperatorParams:
        """Validate an operator declaration's literal parameters.

        ``type``, ``name`` and the ``inputs``/``collect``/``hold`` selector
        maps, when present, are ignored.

        Args:
            raw: Operator mapping of the definition, or its parameters.
            operator_name: Declared operator name for error messages.

        Returns:
            Validated, immutable parameters.

        Raises:
            ParamsValidationError: When the parameters are rejected, or a
                value is a selector.
        """
        owner = f"$operators.{operator_name} ({self.type})"
        path = operator_step_path(operator_name)
        values = {
            key: value
            for key, value in raw.items()
            if key not in RESERVED_PARAM_NAMES and key not in INPUT_MAP_ROLES
        }
        for key, value in values.items():
            if isinstance(value, str) and value.startswith("$"):
                raise ParamsValidationError(
                    f"{owner} parameter {key!r} is {value!r}; operator parameters "
                    "are literals, only the inputs/collect/hold maps take selectors",
                    step_path=path,
                    field_path=(key,),
                )

        try:
            params = self.params_model.model_validate(values)
        except ValidationError as error:
            located = clean_errors(error, values)
            details = "; ".join(
                f"{'.'.join(str(part) for part in location)}: {message}"
                for location, message in located
            )
            raise ParamsValidationError(
                f"{owner} has invalid parameters: {details}",
                step_path=path,
                field_path=located[0][0] if located else (),
            ) from error

        return params

    def params_schema(self) -> Dict[str, Any]:
        """Return the JSON schema of the parameters."""
        schema = _params_json_schema(self.params_model)

        return schema

    def plan_ports(
        self,
        name: str,
        params: OperatorParams,
        inputs: Sequence[Tuple[str, str, EntryLayout, Tuple[Kind, ...]]],
    ) -> Mapping[str, OperatorPort]:
        """Plan the ports of one declared operator at compile time.

        Args:
            name: Declared operator name.
            params: Parameters returned by ``validate_params``.
            inputs: ``(name, role, layout, kinds)`` per input, in declaration
                order.

        Returns:
            Port name to planned port, in the operator's order.

        Raises:
            WorkflowCompileError: When an input has a role the operator does
                not accept, or the operator rejects the inputs' structure.
        """
        described = tuple(OperatorInput(*item) for item in inputs)
        for item in described:
            if item.role not in self.input_roles:
                raise WorkflowCompileError(
                    f"$operators.{name} ({self.type}) does not accept {item.role} "
                    f"input {item.name!r}; accepted roles: {list(self.input_roles)}",
                    step_path=operator_step_path(name),
                )

        planned = self.operator_class.plan_ports(name, params, described)
        for port_name, port in planned.items():
            if not is_selector_segment(port_name) or not isinstance(port, OperatorPort):
                raise ContractError(
                    f"{self.operator_class.__qualname__}.plan_ports must map port "
                    f"names to OperatorPort, got {port_name!r}: {port!r}"
                )
        ports = MappingProxyType(dict(planned))

        return ports

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-friendly description without creating an instance."""
        description = {
            "type": self.type,
            "aliases": list(self.aliases),
            "class": (
                f"{self.operator_class.__module__}."
                f"{self.operator_class.__qualname__}"
            ),
            "description": self.description,
            "params_schema": self.params_schema(),
            "input_roles": list(self.input_roles),
            "engine_compatibility": self.engine_compatibility,
            "metadata": dict(self.metadata),
        }

        return description


class Operator:
    """Base class of V2 operators.

    A concrete operator sets ``type`` in its own class body. Classes without
    their own ``type`` are abstract bases and cannot be registered.

    Class attributes:
        type: Canonical identity, e.g. ``"v2/window@v1"``.
        aliases: Additional accepted identities.
        Params: Nested ``OperatorParams`` model of literal parameters.
        input_roles: Roles of the named inputs the operator accepts.
        engine_compatibility: PEP 440 specifier, e.g. ``">=2.0,<3"``.
        metadata: Free-form UI and catalogue metadata.

    An instance belongs to one active run and is used on its processor
    thread only; a later ``start`` constructs a fresh one. It keeps no
    ``RunState`` and retains only what its declared bounds allow.

    Args:
        name: Declared operator name, also the name of its pulse domain.
        params: Validated parameters.
        inputs: The operator's inputs in declaration order: records with
            ``name``, ``role`` and ``layout``.

    Attributes:
        counters: Counts of this instance. Update only the policy counters
            (see ``OperatorCounters``); the runtime keeps the common ones.
    """

    type: ClassVar[str]
    aliases: ClassVar[Tuple[str, ...]] = ()
    Params: ClassVar[Type[OperatorParams]] = OperatorParams
    input_roles: ClassVar[Tuple[InputRole, ...]] = ()
    engine_compatibility: ClassVar[Optional[str]] = None
    metadata: ClassVar[Mapping[str, Any]] = MappingProxyType({})

    __operator_spec__: ClassVar[Optional[OperatorSpec]] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.__operator_spec__ = None
        if "type" not in cls.__dict__:
            return

        cls.__operator_spec__ = _build_operator_spec(cls)

    def __init__(self, *, name: str, params: OperatorParams, inputs: Sequence[Any]):
        self.name = name
        self.params = params
        self.inputs = tuple(inputs)
        self.counters = OperatorCounters()

    @classmethod
    def plan_ports(
        cls, name: str, params: OperatorParams, inputs: Sequence[OperatorInput]
    ) -> Mapping[str, OperatorPort]:
        """Plan the operator's ports; concrete operators must override it.

        Args:
            name: Declared operator name; scope new axis ids with it.
            params: Validated parameters.
            inputs: Every input with its role, layout and kinds.

        Returns:
            Port name to planned port.

        Raises:
            WorkflowCompileError: When the inputs cannot be combined.
        """
        raise NotImplementedError(f"{cls.__name__} does not implement plan_ports()")

    def push(self, arrivals: Sequence[Arrival]) -> List[OperatorPulse]:
        """Take the inputs of one upstream pulse; concrete operators override it.

        Args:
            arrivals: One arrival per input bound to the pulse's domain, in
                declaration order, filtered ones included.

        Returns:
            Pulses to emit now, in order.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement push()")

    def end_input(self, name: str) -> List[OperatorPulse]:
        """Learn that input ``name`` receives no further arrivals.

        Args:
            name: Input whose domain terminated.

        Returns:
            Pulses that became decidable, in order.
        """
        return []

    def finish(self, reason: TerminationReason) -> List[OperatorPulse]:
        """Emit what the termination policy allows and release retained values.

        Called once, after every input ended, unless the run failed.

        Args:
            reason: ``eof`` when every upstream domain ended, ``stop`` when
                the run was stopped.

        Returns:
            Final pulses, in order.
        """
        return []

    def close(self) -> None:
        """Release everything retained; called exactly once per instance."""


def spec_of_operator(operator_class: Any) -> OperatorSpec:
    """Return the validated declaration of a concrete operator class.

    Args:
        operator_class: An ``Operator`` subclass that sets its own ``type``.

    Returns:
        The class's ``OperatorSpec``.

    Raises:
        OperatorDeclarationError: When ``operator_class`` is not a concrete
            operator class.
    """
    if not isinstance(operator_class, type) or not issubclass(operator_class, Operator):
        raise OperatorDeclarationError(
            f"Expected an Operator subclass, got {operator_class!r}"
        )

    spec = operator_class.__dict__.get("__operator_spec__")
    if spec is None:
        raise OperatorDeclarationError(
            "is abstract: set `type` in the class body to make it registrable",
            operator_class=operator_class.__name__,
        )

    return spec


def operator_step_path(operator_name: str) -> StepPath:
    """Return the structured location of an operator, ``("$operators", name)``.

    Args:
        operator_name: Declared operator name.

    Returns:
        The path compile errors carry.
    """
    return ("$operators", operator_name)


def _build_operator_spec(operator_class: type) -> OperatorSpec:
    class_name = operator_class.__name__

    def fail(message: str) -> OperatorDeclarationError:
        return OperatorDeclarationError(message, operator_class=class_name)

    operator_type = operator_class.__dict__["type"]
    if not isinstance(operator_type, str) or not _IDENTITY.fullmatch(operator_type):
        raise fail(
            f"type must be a non-empty identity of letters, digits and _-./@, "
            f"got {operator_type!r}"
        )

    aliases = _validate_aliases(
        operator_class.aliases, block_type=operator_type, fail=fail
    )

    params_model = operator_class.Params
    if not isinstance(params_model, type) or not issubclass(
        params_model, OperatorParams
    ):
        raise fail("Params must be a subclass of OperatorParams")
    if params_model.model_config.get("extra") != "forbid":
        raise fail("Params must keep extra='forbid' so unknown parameters are rejected")
    for field_name in params_model.model_fields:
        if field_name in RESERVED_PARAM_NAMES or field_name in INPUT_MAP_ROLES:
            raise fail(
                f"Params field {field_name!r} is reserved by the operator declaration"
            )

    roles = operator_class.input_roles
    if (
        not isinstance(roles, tuple)
        or not roles
        or any(role not in INPUT_MAP_ROLES.values() for role in roles)
    ):
        raise fail(
            f"input_roles must be a non-empty tuple of {list(INPUT_MAP_ROLES.values())}, "
            f"got {roles!r}"
        )

    if operator_class.plan_ports.__func__ is Operator.plan_ports.__func__:
        raise fail("does not implement plan_ports()")
    if operator_class.push is Operator.push:
        raise fail("does not implement push()")
    for method in ("push", "end_input", "finish", "close"):
        implementation = getattr(operator_class, method)
        if inspect.iscoroutinefunction(implementation) or inspect.isasyncgenfunction(
            implementation
        ):
            raise fail(
                f"{method}() must be a plain synchronous method; the engine calls "
                "operators on the run's processor thread"
            )

    metadata = operator_class.metadata
    if not isinstance(metadata, Mapping):
        raise fail(f"metadata must be a mapping, got {type(metadata).__name__}")

    spec = OperatorSpec(
        type=operator_type,
        aliases=aliases,
        operator_class=operator_class,
        params_model=params_model,
        input_roles=roles,
        engine_compatibility=_validate_compatibility(
            operator_class.engine_compatibility, fail=fail
        ),
        description=inspect.cleandoc(operator_class.__doc__ or ""),
        metadata=MappingProxyType(dict(metadata)),
    )

    return spec
