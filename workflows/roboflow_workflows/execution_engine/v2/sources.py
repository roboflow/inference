"""Class-owned source declarations and emissions of the V2 engine.

A source brings external data into an active workflow. Like a block, one class
declares everything and the catalogue only collects it::

    class CsvTemperature(Source):
        \"\"\"Finite temperature readings from a CSV file.\"\"\"

        type = "demo/csv_temperature@v1"
        outputs = {"temperature": SourceOutput(FLOAT_KIND)}

        class Params(SourceParams):
            path: str
            probe: str | Ref(STRING_KIND) = "probe-1"

        def open(self, *, path, probe):
            self._rows = iter(csv.DictReader(open(path)))

        def read(self):
            row = next(self._rows, None)
            if row is None:
                return None                                   # end of source
            pts = Timestamp(int(row["pts_ms"]), Fraction(1, 1000), "probe-clock")
            return Emission({"temperature": float(row["celsius"])}, media=pts)

        def close(self):
            ...

A source is not a block: it has no invocation, receives no selected data and
runs an acquisition lifecycle instead of ``run``. Its ``Params`` accept
literals or ``$inputs.<name>`` selectors of ungrouped inputs (static
configuration); the compiler rejects anything else. Constructor keyword
parameters are resources, resolved like a block's. The constructor must not
acquire the input itself; ``open`` does.

Lifecycle of one active run. The constructor runs inside ``start()`` on the
caller's thread; ``open``, ``read`` and ``close`` run on the source's own
reader thread::

    instance = SourceClass(**resources)      # start() caller; no acquisition here
    instance.source_name = declared_name     # set by the engine before open
    instance.stop_event = threading.Event()  # set by the engine before open
    instance.open(**params)                  # reader thread: open the file, connect
    while (emission := instance.read()) is not None:
        ...                                  # one terminal pulse per emission
    instance.close()                         # exactly once after open was attempted

``source_name`` is the declared name of the source in the workflow. It is not
available in the constructor. ``read`` blocks until the next emission. Once ``stop_event`` is
set it should return promptly (a timed source waits with
``self.stop_event.wait(delay)`` instead of sleeping); whatever it returns
afterwards is discarded. Authors read both attributes and never replace them;
they never set or clear the event. ``close`` runs exactly once after ``open``
was attempted, also when ``open`` raised part-way, so it must tolerate a
partial open. Interrupting a blocked native read is not promised.

Identity. By default, the engine gives each present port a ``SampleContext``
whose ``source_id`` is ``source_name``. A source that delivers several cameras
provides indexed ``EntryMetadata.sample`` contexts through ``InputValue``,
giving each camera a stable, distinct ``source_id`` (for example,
``f"{source_name}/{camera_index}"``). Per-source managed state is keyed by
these identifiers; reusing an identifier shares that state across cameras.

Timing. ``Emission.media`` and ``Emission.capture`` default to unknown; leave
``capture`` as ``None`` unless a real capture time is known, and give every
independent media timeline its own clock id. The engine stamps the pulse's
``observed`` time with ``engine_observation()`` when ``read`` returns. A
source that collects several members into one emission may stamp each
member's own arrival with ``engine_observation()`` in that member's
``TemporalContext``; the time is on ``ENGINE_CLOCK_ID``, the same clock as
the pulse's.

Emitted payloads are handed over to the engine: a source must not mutate or
reuse them after ``read`` returns.

Replay of a recording is the one exception to these port rules, and it is
private to the engine (``_ReplaySource``): the replay source of a
retrospective workflow restores recorded ports exactly as they were
delivered, time axes and filtered positions included. No other source can
declare a time axis or emit filtered positions.
"""

import inspect
import threading
import time
from dataclasses import dataclass, field
from fractions import Fraction
from types import MappingProxyType
from typing import Any, ClassVar, Dict, Mapping, Optional, Tuple, Type

from roboflow_workflows.execution_engine.v2._validation import ParamsValidator
from roboflow_workflows.execution_engine.v2.data import (
    Axis,
    EntryLayout,
    EntryMetadata,
    Index,
    InputValue,
    TimeCoverage,
    TimeSpan,
    Timestamp,
)
from roboflow_workflows.execution_engine.v2.declaration import (
    _IDENTITY,
    BlockParams,
    FieldSpec,
    SelectorUse,
    _analyze_fields,
    _collect_kinds,
    _find_selector_uses,
    _params_json_schema,
    _validate_aliases,
    _validate_compatibility,
    _validate_declared_params,
    _validate_keyword_signature,
    is_selector_segment,
)
from roboflow_workflows.execution_engine.v2.errors import ContractError, StepPath
from roboflow_workflows.execution_engine.v2.kinds import Kind, normalize_kinds
from roboflow_workflows.execution_engine.v2.resources import (
    ResourceSpec,
    read_resource_specs,
)

__all__ = [
    "ENGINE_CLOCK_ID",
    "Emission",
    "Source",
    "SourceDeclarationError",
    "SourceOutput",
    "SourceParams",
    "SourceSpec",
    "engine_observation",
    "spec_of_source",
]

ENGINE_CLOCK_ID = "engine.monotonic"
"""Clock of the observation timestamps the engine stamps (``time.monotonic_ns``)."""


def engine_observation() -> Timestamp:
    """Return the current time on the engine's monotonic clock.

    The engine stamps each pulse's ``observed`` time with it. A source may use
    it for a member's own arrival so both are on one comparable clock.

    Returns:
        A nanosecond timestamp on ``ENGINE_CLOCK_ID``. It is never comparable
        to a media clock and never derived from media ticks.
    """
    stamp = Timestamp(
        ticks=time.monotonic_ns(),
        time_base=Fraction(1, 10**9),
        clock_id=ENGINE_CLOCK_ID,
    )

    return stamp


class SourceDeclarationError(ContractError):
    """A source class declares an invalid or inconsistent contract.

    Raised while the class body is being created, so invalid sources fail at
    import time rather than during compilation or execution.

    Args:
        message: Human-readable explanation.
        source_class: Name of the offending class, when known.
    """

    def __init__(self, message: str, *, source_class: Optional[str] = None):
        prefix = f"Source class {source_class}: " if source_class else ""
        super().__init__(prefix + message)
        self.source_class = source_class


class SourceParams(BlockParams):
    """Base class of every source's nested ``Params`` model.

    Unknown parameters are rejected and validated instances are immutable.
    ``type`` and ``name`` belong to the source declaration in the definition
    and cannot be declared as fields.
    """


@dataclass(frozen=True, init=False)
class SourceOutput:
    """Declaration of one source port.

    Args:
        *kinds: Kinds of the emitted payloads; none means the wildcard.
        layout: Source-local axes of one emission. Empty for one payload per
            pulse. Axis ids are local to the source: two ports of one source
            sharing an axis id assert native correspondence, and the compiler
            scopes the ids so another source can never share them. A time axis
            is rejected; temporal grouping belongs to later operators.
        description: Human-readable meaning of the port.

    Raises:
        SourceDeclarationError: On invalid kinds or a layout with a time axis.
    """

    kinds: Tuple[Kind, ...]
    layout: EntryLayout
    description: str

    def __init__(
        self,
        *kinds: Kind,
        layout: EntryLayout = EntryLayout(),
        description: str = "",
    ):
        self._declare(kinds, layout=layout, description=description, restored=False)

    @classmethod
    def _restored(cls, *kinds: Kind, layout: EntryLayout) -> "SourceOutput":
        """Engine-private: a ``_ReplaySource`` port restoring a recorded layout.

        Unlike ``SourceOutput(...)``, the layout may keep a recorded time axis.
        """
        output = cls.__new__(cls)
        output._declare(
            kinds, layout=layout, description="Restored from a recording", restored=True
        )

        return output

    def _declare(
        self,
        kinds: Tuple[Kind, ...],
        *,
        layout: EntryLayout,
        description: str,
        restored: bool,
    ) -> None:
        """Validate and set the declaration; only a restored layout keeps time."""
        try:
            normalized_kinds = normalize_kinds(kinds, context="SourceOutput")
        except ContractError as error:
            raise SourceDeclarationError(str(error)) from error
        if not isinstance(layout, EntryLayout):
            raise SourceDeclarationError(
                f"SourceOutput layout must be an EntryLayout, got {layout!r}"
            )
        if layout.has_time and not restored:
            raise SourceDeclarationError(
                "SourceOutput layout cannot contain a time axis; a source emits "
                "individual samples and temporal grouping is not supported here"
            )

        object.__setattr__(self, "kinds", normalized_kinds)
        object.__setattr__(self, "layout", layout)
        object.__setattr__(self, "description", description)

    @property
    def kind_names(self) -> Tuple[str, ...]:
        """Declared kind names."""
        return tuple(kind.name for kind in self.kinds)

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-friendly description."""
        description = {
            "kinds": list(self.kind_names),
            "axes": list(self.layout.axis_ids),
            "description": self.description,
        }

        return description


@dataclass(frozen=True)
class Emission:
    """What one ``read`` call delivers: one terminal pulse of the source.

    ``data`` maps declared port names to payloads. A port that is not in the
    mapping is terminally absent for this pulse; it is never filled from an
    earlier emission. ``{}`` is an explicitly filtered pulse that keeps its
    pulse identity. ``None`` and ``[]`` are ordinary present payloads. A value
    may be an ``InputValue`` carrying indexed ``EntryMetadata``; supplied
    contexts are kept as given, including explicit ``None`` overrides. For a
    grouped port the value is a ``Batch`` following the port's local layout.

    Args:
        data: Payload or ``InputValue`` per present port.
        media: Position or interval on the source's media timeline, applied at
            ``()`` of every present port that supplies no temporal context.
        capture: Physical capture time or interval, likewise.
        source_metadata: Source-specific facts for this pulse, recorded in the
            engine-built ``SampleContext`` of every present port.

    Raises:
        ContractError: When ``data`` is not a mapping keyed by strings, a
            coverage is not a ``Timestamp``/``TimeSpan`` or
            ``source_metadata`` is not a mapping.
    """

    data: Mapping[str, Any]
    media: Optional[TimeCoverage] = None
    capture: Optional[TimeCoverage] = None
    source_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.data, Mapping):
            raise ContractError(
                "Emission data must map port names to payloads, got "
                f"{type(self.data).__name__}"
            )
        for name in self.data:
            if not isinstance(name, str) or not name:
                raise ContractError(
                    f"Emission port names must be strings, got {name!r}"
                )
        for label, coverage in (("media", self.media), ("capture", self.capture)):
            if coverage is not None and not isinstance(coverage, (Timestamp, TimeSpan)):
                raise ContractError(
                    f"Emission {label} must be a Timestamp or TimeSpan, got "
                    f"{type(coverage).__name__}"
                )
        if not isinstance(self.source_metadata, Mapping):
            raise ContractError(
                "Emission source_metadata must be a mapping, got "
                f"{type(self.source_metadata).__name__}"
            )

        object.__setattr__(self, "data", MappingProxyType(dict(self.data)))
        object.__setattr__(
            self, "source_metadata", MappingProxyType(dict(self.source_metadata))
        )

    @property
    def is_filtered(self) -> bool:
        """Whether this is an explicitly filtered pulse (no port present)."""
        return not self.data

    def payload(self, port: str) -> Any:
        """Return the bare payload of a present port, unwrapping ``InputValue``.

        Args:
            port: Port name present in ``data``.

        Returns:
            The payload as emitted.
        """
        value = self.data[port]
        if isinstance(value, InputValue):
            return value.data

        return value


@dataclass(frozen=True)
class SourceSpec:
    """Validated, class-owned declaration of one source type.

    Built once per concrete ``Source`` subclass; read it with
    ``spec_of_source``. Holds no instance and acquires nothing.

    Args:
        type: Canonical identity used by ``sources`` declarations.
        aliases: Additional accepted identities.
        source_class: The declaring class.
        params_model: The nested ``Params`` model.
        fields: Field declarations in model order, keyed by field name.
        outputs: Port declarations with their source-local layouts.
        resources: Constructor resources.
        engine_compatibility: PEP 440 specifier of supported engine versions.
        description: Source documentation.
        metadata: Free-form UI and catalogue metadata.
        kinds: Every kind referenced by fields and ports.
    """

    type: str
    aliases: Tuple[str, ...]
    source_class: type
    params_model: Type[SourceParams]
    fields: Mapping[str, FieldSpec]
    outputs: Mapping[str, SourceOutput]
    resources: Tuple[ResourceSpec, ...]
    engine_compatibility: Optional[str]
    description: str
    metadata: Mapping[str, Any]
    kinds: Tuple[Kind, ...]
    _validator: ParamsValidator = field(repr=False, compare=False)

    @property
    def identities(self) -> Tuple[str, ...]:
        """Canonical type followed by aliases."""
        return (self.type,) + self.aliases

    def validate_params(
        self, raw: Mapping[str, Any], *, source_name: str = ""
    ) -> SourceParams:
        """Validate a source declaration's parameters, keeping selectors as strings.

        ``type`` and ``name`` keys, when present, are ignored.

        Args:
            raw: Source mapping of the definition, or its parameters.
            source_name: Declared source name for error messages.

        Returns:
            Validated, immutable parameters.

        Raises:
            ParamsValidationError: When Pydantic rejects the parameters or an
                author validator raises an exception, retained as the cause.
            SelectorError: When a selector-capable position holds a string
                that starts like a selector but is malformed.
        """
        params = _validate_declared_params(
            self._validator,
            self.fields,
            raw,
            owner=f"$sources.{source_name} ({self.type})",
            step_path=source_step_path(source_name),
        )

        return params

    def find_selectors(self, params: SourceParams) -> Tuple[SelectorUse, ...]:
        """Find every selector at a declared position of validated params.

        Args:
            params: Parameters returned by ``validate_params``.

        Returns:
            Selector uses in field order, then list order or dict key order.
        """
        uses = _find_selector_uses(self.fields, params)

        return uses

    def validate_resolved_arguments(
        self, params: SourceParams, arguments: Mapping[str, Any]
    ) -> None:
        """Check the arguments of ``open`` after static selectors were resolved.

        Args:
            params: Parameters returned by ``validate_params``.
            arguments: One value per ``Params`` field: literals as validated
                and each selected leaf as its resolved input value.

        Raises:
            ResolvedParameterError: With ``field_path`` of the first violation.
        """
        selected = [(use.field_path, use.marker) for use in self.find_selectors(params)]
        self._validator.check_resolved(params, arguments, selected)

    def params_schema(self) -> Dict[str, Any]:
        """Return the JSON schema of the parameters, including selector metadata."""
        schema = _params_json_schema(self.params_model)

        return schema

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-friendly description without creating an instance."""
        description = {
            "type": self.type,
            "aliases": list(self.aliases),
            "class": f"{self.source_class.__module__}.{self.source_class.__qualname__}",
            "description": self.description,
            "params_schema": self.params_schema(),
            "fields": {name: spec.describe() for name, spec in self.fields.items()},
            "outputs": {
                name: output.describe() for name, output in self.outputs.items()
            },
            "resources": [resource.describe() for resource in self.resources],
            "engine_compatibility": self.engine_compatibility,
            "metadata": dict(self.metadata),
        }

        return description


class Source:
    """Base class of V2 sources.

    A concrete source sets ``type`` in its own class body. Classes without
    their own ``type`` are abstract bases and cannot be registered.

    Class attributes:
        type: Canonical identity, e.g. ``"demo/csv_temperature@v1"``.
        aliases: Additional accepted identities.
        Params: Nested ``SourceParams`` model; defaults to no parameters.
        outputs: Mapping of port name to ``SourceOutput``. Names use letters,
            digits, ``_`` and ``-``. At least one port is required.
        engine_compatibility: PEP 440 specifier, e.g. ``">=2.0,<3"``.
        metadata: Free-form UI and catalogue metadata.

    Resources are the keyword parameters of ``__init__``. The engine creates a
    fresh instance for every active run inside ``start()`` on the caller's
    thread, then drives ``open``, ``read`` and ``close`` on that run's reader
    thread for this source (module docstring).

    Attributes:
        source_name: Declared name of this source in the workflow. Set by the
            engine before ``open``; read it, never replace it.
        stop_event: Set by the engine before ``open`` and set when the run is
            stopping. Read it (``is_set()``, ``wait(timeout)``); never set,
            clear or replace it.
    """

    type: ClassVar[str]
    aliases: ClassVar[Tuple[str, ...]] = ()
    Params: ClassVar[Type[SourceParams]] = SourceParams
    outputs: ClassVar[Mapping[str, SourceOutput]] = MappingProxyType({})
    engine_compatibility: ClassVar[Optional[str]] = None
    metadata: ClassVar[Mapping[str, Any]] = MappingProxyType({})

    source_name: str
    stop_event: threading.Event

    __source_spec__: ClassVar[Optional[SourceSpec]] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.__source_spec__ = None
        if "type" not in cls.__dict__:
            return

        cls.__source_spec__ = _build_source_spec(cls)

    def open(self, **params: Any) -> None:
        """Acquire the input; concrete sources must override it.

        Args:
            **params: One keyword argument per ``Params`` field, with static
                selectors resolved.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement open()")

    def read(self) -> Optional[Emission]:
        """Block until the next emission; concrete sources must override it.

        Returns:
            The next ``Emission``, or ``None`` at the end of the source.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement read()")

    def close(self) -> None:
        """Release what ``open`` acquired; called exactly once after ``open``."""


def spec_of_source(source_class: Any) -> SourceSpec:
    """Return the validated declaration of a concrete source class.

    Args:
        source_class: A ``Source`` subclass that sets its own ``type``.

    Returns:
        The class's ``SourceSpec``.

    Raises:
        SourceDeclarationError: When ``source_class`` is not a concrete
            source class.
    """
    if not isinstance(source_class, type) or not issubclass(source_class, Source):
        raise SourceDeclarationError(
            f"Expected a Source subclass, got {source_class!r}"
        )

    spec = source_class.__dict__.get("__source_spec__")
    if spec is None:
        raise SourceDeclarationError(
            "is abstract: set `type` in the class body to make it registrable",
            source_class=source_class.__name__,
        )

    return spec


def source_step_path(source_name: str) -> StepPath:
    """Return the structured location of a source, ``("$sources", name)``.

    Compile errors, resource errors and the ``ExecutionContext`` of a source
    constructor use this path. The "$" marker cannot occur in a valid
    step name, so a nested workflow named "sources" remains distinct.

    Args:
        source_name: Declared source name.

    Returns:
        The path.
    """
    return ("$sources", source_name)


def _build_source_spec(source_class: type) -> SourceSpec:
    class_name = source_class.__name__

    def fail(message: str) -> SourceDeclarationError:
        return SourceDeclarationError(message, source_class=class_name)

    source_type = source_class.__dict__["type"]
    if not isinstance(source_type, str) or not _IDENTITY.fullmatch(source_type):
        raise fail(
            f"type must be a non-empty identity of letters, digits and _-./@, "
            f"got {source_type!r}"
        )

    aliases = _validate_aliases(source_class.aliases, block_type=source_type, fail=fail)

    params_model = source_class.Params
    if not isinstance(params_model, type) or not issubclass(params_model, SourceParams):
        raise fail("Params must be a subclass of SourceParams")
    if params_model.model_config.get("extra") != "forbid":
        raise fail("Params must keep extra='forbid' so unknown parameters are rejected")

    fields = _analyze_fields(params_model, fail=fail)
    for name, field_spec in fields.items():
        if field_spec.role == "step":
            raise fail(f"Params field {name!r} is a StepRef; sources control nothing")
        if field_spec.role == "group":
            raise fail(
                f"Params field {name!r} is a Group; source parameters are static "
                "configuration, bind a Ref to an ungrouped input instead"
            )
        if any(marker.batch != "never" for marker in field_spec.markers):
            raise fail(
                f"Params field {name!r} requests batch delivery; source parameters "
                "are static configuration resolved once before open()"
            )
    try:
        validator = ParamsValidator(params_model)
    except Exception as error:
        raise fail(f"Params cannot be projected for validation: {error}") from error

    outputs = _validate_source_outputs(source_class.outputs, fail=fail)
    _validate_lifecycle(source_class, fields=fields, fail=fail)
    engine_compatibility = _validate_compatibility(
        source_class.engine_compatibility, fail=fail
    )

    try:
        resources = read_resource_specs(source_class)
    except ContractError as error:
        raise fail(str(error)) from error

    metadata = source_class.metadata
    if not isinstance(metadata, Mapping):
        raise fail(f"metadata must be a mapping, got {type(metadata).__name__}")

    spec = SourceSpec(
        type=source_type,
        aliases=aliases,
        source_class=source_class,
        params_model=params_model,
        fields=MappingProxyType(fields),
        outputs=outputs,
        resources=resources,
        engine_compatibility=engine_compatibility,
        description=inspect.cleandoc(source_class.__doc__ or ""),
        metadata=MappingProxyType(dict(metadata)),
        kinds=_collect_kinds(fields=fields, outputs=outputs, fail=fail),
        _validator=validator,
    )

    return spec


def _validate_source_outputs(declared: Any, *, fail) -> Mapping[str, SourceOutput]:
    if not isinstance(declared, Mapping) or not declared:
        raise fail(
            "outputs must be a non-empty mapping of port name to SourceOutput, got "
            f"{declared!r}"
        )

    outputs: Dict[str, SourceOutput] = {}
    axes: Dict[str, Tuple[Axis, str]] = {}
    for name, output in declared.items():
        if not is_selector_segment(name):
            raise fail(
                f"output name {name!r} must use letters, digits, _ or - so a "
                "selector can address it"
            )
        if not isinstance(output, SourceOutput):
            raise fail(
                f"output {name!r} must be a SourceOutput, got {type(output).__name__}"
            )
        for axis in output.layout.axes:
            known_axis, known_port = axes.setdefault(axis.id, (axis, name))
            if known_axis != axis:
                raise fail(
                    f"output {name!r} declares axis {axis.id!r} differently from "
                    f"output {known_port!r}; one local axis id means one native "
                    "correspondence and must be declared identically"
                )
        outputs[name] = output

    frozen_outputs = MappingProxyType(outputs)

    return frozen_outputs


class _ReplaySource(Source):
    """Engine-private base of the replay source of a retrospective workflow.

    A new source creates new samples, so it can neither declare a time axis
    nor emit filtered positions. A replay source creates nothing: it restores
    ports that a run already delivered, time axes and filtered positions
    included. Its ports are declared with ``SourceOutput._restored`` and it
    emits ``_RestoredPort`` values; the engine builds their entries as
    recorded. Only ``recording.replay`` subclasses it.
    """


@dataclass(frozen=True)
class _RestoredPort:
    """One recorded port value, restored with its known structure.

    Only a ``_ReplaySource`` emits it; an ordinary source's payload is never
    read as one.

    Args:
        data: Surviving payload tree as delivered; ``None`` when nothing
            survived (``filtered`` then holds every known filtered node, or
            ``((),)`` for a port filtered as a whole).
        filtered: Minimal filtered index paths as delivered.
        metadata: Indexed context as delivered, explicit ``None`` included.
    """

    data: Any
    filtered: Tuple[Index, ...]
    metadata: EntryMetadata


def _validate_lifecycle(
    source_class: type, *, fields: Mapping[str, FieldSpec], fail
) -> None:
    if source_class.open is Source.open:
        raise fail("does not implement open()")
    if source_class.read is Source.read:
        raise fail("does not implement read()")
    for name in ("open", "read", "close"):
        method = getattr(source_class, name)
        if inspect.iscoroutinefunction(method) or inspect.isasyncgenfunction(method):
            raise fail(
                f"{name}() must be a plain synchronous method; the engine drives "
                "open, read and close on the source's reader thread and runs no "
                "event loop"
            )

    _validate_keyword_signature(
        source_class.open, name="open", fields=fields, fail=fail
    )
    read_parameters = list(inspect.signature(source_class.read).parameters.values())[1:]
    if any(
        parameter.default is inspect.Parameter.empty
        and parameter.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        for parameter in read_parameters
    ):
        raise fail("read() takes no required parameters; the engine calls read()")
