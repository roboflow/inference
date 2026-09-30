"""Class-owned block declarations for the V2 execution engine.

A block is one Python class. The class itself declares everything the engine
and tools need; nothing is restated in a separate registry::

    class Scale(Block):
        \"\"\"Multiply a number by a factor.\"\"\"

        type = "demo/scale@v1"
        outputs = {"scaled": Output(FLOAT_KIND)}

        class Params(BlockParams):
            value: Ref(FLOAT_KIND)
            factor: float | Ref(FLOAT_KIND) = 2.0

        def run(self, *, value: float, factor: float) -> dict:
            return {"scaled": value * factor}

Parameters live in a nested Pydantic model. The same field accepts a literal,
its default, a workflow-input selector or an upstream-output selector,
depending on its annotation. Selector annotations (``Ref``, ``Group``,
``StepRef``) may be the whole field or the elements of one ``list`` / values of
one ``dict`` level; each such leaf keeps its own role and batch mode. Strings
at undeclared positions always stay literal. Deeper nesting, selectors inside
nested models and two selector alternatives for one position are rejected, as
in V1's single container level. Names follow V1 selector segments (letters,
digits, ``_``, ``-``; ``$steps.parse.2026``) and complete strings are matched
with ``fullmatch``. Pydantic aliases decide the keys a step writes; each
``FieldSpec`` relates its field name to those keys (``input_paths``) and to
its JSON schema property (``schema_property``).

Each output declares its layout relative to the step's invocation level ``P``
and where its source/temporal context comes from:

* ``Output(...)``: one value per invocation, at ``P``.
* ``Output(..., expand="axis")``: a ``Batch`` of new children per invocation.
* ``Output(..., preserve="group_field")``: a ``Batch`` with one value per child
  of that ``Group`` field, keeping the group's existing axis and indices.
* ``source="field"``: take context from that field's bound values;
  ``context_policy="common_or_none"`` keeps a context only when all
  contributing values share it. Without ``source``, all data bindings of the
  invocation contribute under the same policy.

Collapse needs no keyword: a block with a ``Group`` field runs once per parent,
so its ordinary outputs sit at the parent level.

Invalid declarations raise ``DeclarationError`` while the class is created.
``spec_of(cls)`` returns the validated ``BlockSpec`` the compiler, executor,
catalogue and dynamic-block assembly consume.
"""

import inspect
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import (
    Any,
    ClassVar,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Tuple,
    Type,
    Union,
)

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from pydantic import (
    AliasChoices,
    AliasPath,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
)
from pydantic.fields import FieldInfo
from roboflow_workflows.execution_engine.entities.workload import (
    Discovery,
    declaration_failed_problem,
    declaration_unavailable_problem,
    normalize_declaration,
)
from roboflow_workflows.execution_engine.v2._selectors import (
    BATCH_MODES,
    DATA_SELECTOR_PATTERN,
    SELECTOR_PREFIXES,
    SELECTOR_SEGMENT,
    STEP_SELECTOR_PATTERN,
    BatchMode,
    ContainerKind,
    Group,
    ParsedSelector,
    Ref,
    SelectorMarker,
    SelectorRole,
    StepRef,
    analyze_annotation,
    field_annotation,
    is_selector_segment,
    parse_selector,
)
from roboflow_workflows.execution_engine.v2._validation import (
    ParamsValidator,
    clean_errors,
)
from roboflow_workflows.execution_engine.v2.context import (
    ExecutionContext,
    get_execution_context,
)
from roboflow_workflows.execution_engine.v2.errors import (
    ContractError,
    DeclarationError,
    FieldPath,
    ParamsValidationError,
    ResolvedParameterError,
    SelectorError,
    StepPath,
    format_step_path,
)
from roboflow_workflows.execution_engine.v2.kinds import Kind, normalize_kinds
from roboflow_workflows.execution_engine.v2.resources import (
    ResourceSpec,
    read_resource_specs,
)

__all__ = [
    "BATCH_MODES",
    "CONTEXT_POLICIES",
    "DATA_SELECTOR_PATTERN",
    "SELECTOR_SEGMENT",
    "STEP_SELECTOR_PATTERN",
    "BatchMode",
    "Block",
    "BlockParams",
    "BlockSpec",
    "ContextPolicy",
    "DependentResource",
    "FieldSpec",
    "Group",
    "InputPath",
    "Output",
    "OutputTransform",
    "ParsedSelector",
    "Ref",
    "Select",
    "SelectorMarker",
    "SelectorUse",
    "StepRef",
    "Stop",
    "WorkloadDeclaration",
    "is_selector_segment",
    "parse_selector",
    "spec_of",
]


OutputTransform = Literal["same", "expand", "preserve"]
ContextPolicy = Literal["common_or_none"]
InputPath = Tuple[Union[str, int], ...]
"""Location of a parameter in the step mapping: object keys and list indices."""

CONTEXT_POLICIES: Tuple[str, ...] = ("common_or_none",)
RESERVED_PARAM_NAMES: Tuple[str, ...] = ("type", "name")

_IDENTITY = re.compile(r"[A-Za-z0-9_\-./@]+")


class BlockParams(BaseModel):
    """Base class of every block's nested ``Params`` model.

    Unknown parameters are rejected and validated instances are immutable.
    ``type`` and ``name`` belong to the step, not to the block, and cannot be
    declared as fields.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


@dataclass(frozen=True, init=False)
class Output:
    """Declaration of one block output.

    Args:
        *kinds: Kinds of produced values; none means the wildcard.
        expand: Key of a new nesting axis. The block returns a ``Batch`` of
            children for each invocation.
        preserve: Name of a ``Group`` field. The block returns a ``Batch``
            with one value per child of that group, keeping its axis.
        stationary: With ``expand``, whether children keep stable identities
            across arrivals (for example configured static crops).
        source: Field whose bound values provide the output's source and
            temporal context. ``None`` uses every data binding.
        context_policy: How several contributing contexts combine;
            ``common_or_none`` keeps one only when all are equal.
        description: Human-readable meaning of the output.

    Raises:
        DeclarationError: On invalid kinds, both ``expand`` and ``preserve``,
            invalid names, ``stationary`` without ``expand``, an unknown
            context policy or a ``source`` differing from ``preserve``.
    """

    kinds: Tuple[Kind, ...]
    expand: Optional[str]
    preserve: Optional[str]
    stationary: bool
    source: Optional[str]
    context_policy: ContextPolicy
    description: str

    def __init__(
        self,
        *kinds: Kind,
        expand: Optional[str] = None,
        preserve: Optional[str] = None,
        stationary: bool = False,
        source: Optional[str] = None,
        context_policy: ContextPolicy = "common_or_none",
        description: str = "",
    ):
        try:
            normalized_kinds = normalize_kinds(kinds, context="Output")
        except ContractError as error:
            raise DeclarationError(str(error)) from error
        if expand is not None and preserve is not None:
            raise DeclarationError(
                "Output declares both expand and preserve; choose one layout"
            )
        if expand is not None and not is_selector_segment(expand):
            raise DeclarationError(
                f"Output expand axis key must use letters, digits, _ or -, got {expand!r}"
            )
        for label, value in (("preserve", preserve), ("source", source)):
            if value is not None and not (
                isinstance(value, str) and value.isidentifier()
            ):
                raise DeclarationError(
                    f"Output {label} must name a Params field, got {value!r}"
                )
        if not isinstance(stationary, bool):
            raise DeclarationError(
                f"Output stationary must be a bool, got {stationary!r}"
            )
        if stationary and expand is None:
            raise DeclarationError("Output stationary=True requires expand")
        if context_policy not in CONTEXT_POLICIES:
            raise DeclarationError(
                f"Output context_policy must be one of {list(CONTEXT_POLICIES)}, "
                f"got {context_policy!r}"
            )
        if preserve is not None and source not in (None, preserve):
            raise DeclarationError(
                f"Output preserving {preserve!r} takes its context from that group; "
                f"source {source!r} cannot differ"
            )

        values = {
            "kinds": normalized_kinds,
            "expand": expand,
            "preserve": preserve,
            "stationary": stationary,
            "source": preserve if preserve is not None else source,
            "context_policy": context_policy,
            "description": description,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)

    @property
    def kind_names(self) -> Tuple[str, ...]:
        """Declared kind names."""
        return tuple(kind.name for kind in self.kinds)

    @property
    def transform(self) -> OutputTransform:
        """Layout of the output relative to the step's invocation level."""
        if self.expand is not None:
            return "expand"
        if self.preserve is not None:
            return "preserve"

        return "same"

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-friendly description."""
        description = {
            "kinds": list(self.kind_names),
            "transform": self.transform,
            "expand": self.expand,
            "preserve": self.preserve,
            "stationary": self.stationary,
            "source": self.source,
            "context_policy": self.context_policy,
            "description": self.description,
        }

        return description


@dataclass(frozen=True, init=False)
class Select:
    """Result of a control block: the targets whose branches continue.

    Targets are selector strings exactly as the block received them from its
    ``StepRef`` fields. Targets that are not selected are denied for this
    invocation.

    Args:
        targets: One selector or an iterable of selectors; empty denies all.

    Raises:
        ContractError: When a target is not a complete step selector.
    """

    targets: Tuple[str, ...]

    def __init__(self, targets: Union[str, Iterable[str]] = ()):
        if isinstance(targets, str):
            targets = (targets,)
        normalized = tuple(targets)
        for target in normalized:
            try:
                parsed = parse_selector(target)
            except SelectorError as error:
                raise ContractError(
                    f"Select targets must be step selectors like '$steps.name', "
                    f"got {target!r}"
                ) from error
            if parsed.target != "step":
                raise ContractError(
                    f"Select targets must be step selectors like '$steps.name', "
                    f"got {target!r}"
                )

        object.__setattr__(self, "targets", normalized)


@dataclass(frozen=True, init=False)
class Stop(Select):
    """Control result that denies every target of this invocation."""

    def __init__(self):
        super().__init__(())


class DependentResource(BaseModel):
    """External resource a step needs, reported by workload introspection."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    resource_type: str = Field(
        description="Category of the resource, e.g. a model or a project.",
        examples=["roboflow_platform_model"],
    )
    identifier: str = Field(
        description=(
            "Resource identifier. A selector string when the identifier is "
            "only known at run time."
        ),
        examples=["yolov8n-640", "$inputs.model_id"],
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional JSON-friendly facts about the resource.",
    )


@dataclass(frozen=True)
class WorkloadDeclaration:
    """Normalized workload declarations of one configured step.

    Args:
        dependencies: External resources, as ``Discovery`` items.
        operations: Meaningful work performed, usually ``WorkOperation`` items.
        restrictions: ``RuntimeRestriction`` items with conditions intact.
    """

    dependencies: Discovery
    operations: Discovery
    restrictions: Discovery


@dataclass(frozen=True)
class FieldSpec:
    """Declared structure of one ``Params`` field.

    Args:
        name: Field name, identical to the ``run`` keyword argument.
        required: Whether the step must supply the field.
        has_default: Whether an omitted field takes a default.
        whole: Marker when the whole field may be a selector.
        leaves: Marker for list elements or dict values.
        container: ``"list"`` or ``"dict"`` when ``leaves`` is set.
        literal_allowed: Whether a non-selector value is accepted.
        description: Field description from the ``Params`` model.
        input_paths: Locations in the step mapping that Pydantic reads the
            field from, in its lookup order: the aliases (``alias``,
            ``validation_alias``, ``AliasChoices``, ``AliasPath``) when the
            model validates by alias, then the field name only when the model
            validates by name or the field has no alias. A step supplies one
            of them; supplying a second is rejected as an extra input. Paths
            under the step's own ``type``/``name`` keys are omitted, so a
            field may have none. Several fields may share a path.
        schema_property: Key of ``BlockSpec.params_schema()["properties"]``
            describing the field's value; several fields may share one. It is
            itself a writable key only when ``(schema_property,)`` is in
            ``input_paths``: Pydantic names the property of an
            ``AliasPath``-only field after the field.
    """

    name: str
    required: bool
    has_default: bool
    whole: Optional[SelectorMarker]
    leaves: Optional[SelectorMarker]
    container: Optional[ContainerKind]
    literal_allowed: bool
    description: Optional[str]
    input_paths: Tuple[InputPath, ...]
    schema_property: str

    @property
    def input_path(self) -> Optional[InputPath]:
        """First accepted location made of object keys only, or ``None``.

        A JSON writer nests one object per key and puts the value last.
        Locations with list indices are listed in ``input_paths`` only.
        """
        for path in self.input_paths:
            if all(isinstance(segment, str) for segment in path):
                return path

        return None

    @property
    def markers(self) -> Tuple[SelectorMarker, ...]:
        """Declared markers: whole field first, then container leaves."""
        return tuple(
            marker for marker in (self.whole, self.leaves) if marker is not None
        )

    @property
    def accepts_selectors(self) -> bool:
        """Whether any position of the field may hold a selector."""
        return bool(self.markers)

    @property
    def role(self) -> Optional[SelectorRole]:
        """Selector role of the field, or ``None`` for literal-only fields."""
        if not self.markers:
            return None

        return self.markers[0].role

    def marker_at(self, position: Tuple[Any, ...]) -> Optional[SelectorMarker]:
        """Return the marker governing ``()`` (whole field) or a leaf position.

        Args:
            position: ``()`` or ``(index_or_key,)``.

        Returns:
            The marker, or ``None`` when no selector is declared there.
        """
        marker = self.whole if not position else self.leaves

        return marker

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-friendly description."""
        description = {
            "required": self.required,
            "literal_allowed": self.literal_allowed,
            "whole_selector": self.whole.describe() if self.whole else None,
            "leaf_selector": self.leaves.describe() if self.leaves else None,
            "container": self.container,
            "description": self.description,
            "input_path": list(self.input_path) if self.input_path else None,
            "input_paths": [list(path) for path in self.input_paths],
            "schema_property": self.schema_property,
        }

        return description


@dataclass(frozen=True)
class SelectorUse:
    """One selector found in validated step parameters.

    Args:
        field: Field holding the selector.
        position: ``()`` for the whole field, ``(i,)`` for a list element or
            ``(key,)`` for a dict value.
        selector: Selector text as written.
        marker: Declaration of that position (role, kinds, batch mode).
    """

    field: str
    position: Tuple[Any, ...]
    selector: str
    marker: SelectorMarker

    @property
    def field_path(self) -> FieldPath:
        """Field name followed by the position inside it."""
        return (self.field,) + self.position


@dataclass(frozen=True)
class BlockSpec:
    """Validated, class-owned declaration of one block type.

    Built once per concrete ``Block`` subclass; read it with ``spec_of``.
    Holds no instance and allocates no resource.

    Args:
        type: Canonical identity used by workflow steps.
        aliases: Additional accepted identities.
        block_class: The declaring class.
        params_model: The nested ``Params`` model.
        fields: Field declarations in model order, keyed by field name (not
            by alias; see ``FieldSpec.input_paths``).
        outputs: Static output declarations.
        configured_outputs: Whether ``describe_outputs`` derives outputs from
            literal parameters.
        output_fields: Literal-only fields that shape configured outputs.
        resources: Constructor resources.
        is_control: Whether the block routes control via ``StepRef`` fields.
        accepts_batches: Capability: some selector declares ``always`` or
            ``if_varying`` batch delivery. Whether a particular step receives
            batches depends on its bindings (``PlannedStep.delivers_batches``).
        accepts_empty: Whether missing, filtered or ``None`` inputs still
            invoke the block.
        mutates: Fields whose bound payloads the block may modify in place.
        engine_compatibility: PEP 440 specifier of supported engine versions.
        description: Block documentation.
        metadata: Free-form UI and catalogue metadata.
        kinds: Every kind referenced by fields and static outputs.
    """

    type: str
    aliases: Tuple[str, ...]
    block_class: type
    params_model: Type[BlockParams]
    fields: Mapping[str, FieldSpec]
    outputs: Mapping[str, Output]
    configured_outputs: bool
    output_fields: Tuple[str, ...]
    resources: Tuple[ResourceSpec, ...]
    is_control: bool
    accepts_batches: bool
    accepts_empty: bool
    mutates: Tuple[str, ...]
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
        self,
        raw: Mapping[str, Any],
        *,
        step_path: StepPath = (),
    ) -> BlockParams:
        """Validate step parameters, keeping selectors as strings.

        ``type`` and ``name`` keys, when present, are ignored. Explicitly
        supplied values, including ``None``, are recorded in
        ``params.model_fields_set``; defaults are not.

        Args:
            raw: Step mapping or its parameters.
            step_path: Step location for error messages.

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
            owner=f"{format_step_path(step_path)} ({self.type})",
            step_path=step_path,
        )

        return params

    def find_selectors(self, params: BlockParams) -> Tuple[SelectorUse, ...]:
        """Find every selector at a declared position of validated params.

        Args:
            params: Parameters returned by ``validate_params``.

        Returns:
            Selector uses in field order, then list order or dict key order.
            Each keeps the leaf's role, kinds and batch mode.
        """
        uses = _find_selector_uses(self.fields, params)

        return uses

    def validate_resolved_arguments(
        self, params: BlockParams, arguments: Mapping[str, Any]
    ) -> None:
        """Check one logical invocation after its selectors were resolved.

        Call it once per logical invocation, before values are packed into
        batches. ``arguments`` holds one value per ``Params`` field: literals
        as validated in ``params`` (also a literal the engine casts into a
        group or batch, which stays the literal here), each selected ``Ref``
        leaf as its resolved payload for this invocation, each selected
        ``Group`` leaf as the ``Batch`` of its children, and ``None`` for an
        unavailable leaf.

        Selected payloads are checked against their ``Ref``/``Group`` kinds,
        independently of any literal alternative. Shared field constraints
        (outer ``Field(...)`` metadata) and the author's field and model
        validators then run on the resolved values; literal-only fields are
        not revalidated. Nothing is converted or copied: the call returns
        nothing and the caller delivers its own values.

        Args:
            params: Parameters returned by ``validate_params`` for the step.
            arguments: Resolved arguments of one invocation, by field name.

        Raises:
            ResolvedParameterError: With ``field_path`` of the first violation.
        """
        selected = [
            (use.field_path, use.marker)
            for use in self.find_selectors(params)
            if use.marker.role != "step"
        ]
        self._validator.check_resolved(params, arguments, selected)

    def validate_resolved_value(
        self,
        field_name: str,
        value: Any,
        *,
        position: Tuple[Any, ...] = (),
    ) -> Any:
        """Check one selected value against its position only.

        Applies the kinds of the selector at ``position`` and the constraints
        shared by that position (outer ``Field`` metadata), never a literal
        alternative. Field and model validators need the whole invocation;
        use ``validate_resolved_arguments`` for them.

        Args:
            field_name: Field holding the selector.
            value: Resolved value; ``None`` (unavailable) is not checked.
            position: ``()`` for the whole field or ``(index_or_key,)``.

        Returns:
            ``value`` itself, unchanged.

        Raises:
            ResolvedParameterError: On a wrong kind or violated constraint.
        """
        marker = self.fields[field_name].marker_at(tuple(position))
        if value is None or marker is None:
            return value

        path = (field_name,) + tuple(position)
        try:
            marker.check_payload(value)
        except ContractError as error:
            raise ResolvedParameterError(
                f"{self.type} parameter {'.'.join(map(str, path))} is not a valid "
                f"{list(marker.kind_names)}: {error}",
                field_path=path,
            ) from error
        self._validator.check_value(field_name, value, position=tuple(position))

        return value

    def resolve_outputs(self, params: BlockParams) -> Mapping[str, Output]:
        """Return the outputs of a configured step.

        Args:
            params: Validated parameters of the step.

        Returns:
            Static outputs, or those derived by ``describe_outputs``.

        Raises:
            DeclarationError: When ``describe_outputs`` returns an invalid
                declaration.
        """
        if not self.configured_outputs:
            return self.outputs

        declared = self.block_class.describe_outputs(params)
        outputs = _validate_outputs(
            declared,
            fields=self.fields,
            block_class=self.block_class.__name__,
            origin="describe_outputs()",
        )

        return outputs

    def describe_workload(
        self, params: BlockParams, *, node_id: str
    ) -> WorkloadDeclaration:
        """Normalize the block's workload hooks for one configured step.

        ``None`` from a hook means unknown and yields an incomplete discovery.
        A raising hook yields an incomplete discovery with a failure problem;
        the exception text is not reported.

        Args:
            params: Validated parameters of the step.
            node_id: Step selector used in reported problems.

        Returns:
            Dependencies, operations and restrictions.
        """
        declaration = WorkloadDeclaration(
            dependencies=self._discover(
                "discover_dependent_resources",
                params,
                node_id=node_id,
                domain="resources",
            ),
            operations=self._discover(
                "discover_work_operations", params, node_id=node_id, domain="operations"
            ),
            restrictions=self._discover(
                "discover_restrictions", params, node_id=node_id, domain="restrictions"
            ),
        )

        return declaration

    def params_schema(self) -> Dict[str, Any]:
        """Return the JSON schema of the parameters, including selector metadata.

        Property names follow the model's ``validate_by_alias`` setting. Each
        ``FieldSpec.schema_property`` names the property of its field.
        """
        schema = _params_json_schema(self.params_model)

        return schema

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-friendly description without creating an instance."""
        description = {
            "type": self.type,
            "aliases": list(self.aliases),
            "class": f"{self.block_class.__module__}.{self.block_class.__qualname__}",
            "description": self.description,
            "params_schema": self.params_schema(),
            "fields": {name: spec.describe() for name, spec in self.fields.items()},
            "outputs": {
                name: output.describe() for name, output in self.outputs.items()
            },
            "configured_outputs": self.configured_outputs,
            "output_fields": list(self.output_fields),
            "resources": [resource.describe() for resource in self.resources],
            "is_control": self.is_control,
            "accepts_batches": self.accepts_batches,
            "accepts_empty": self.accepts_empty,
            "mutates": list(self.mutates),
            "engine_compatibility": self.engine_compatibility,
            "metadata": dict(self.metadata),
        }

        return description

    def _discover(
        self, hook_name: str, params: BlockParams, *, node_id: str, domain: str
    ) -> Discovery:
        hook = getattr(self.block_class, hook_name)
        try:
            declared = hook(params)
            discovery = normalize_declaration(
                declared,
                declaration_unavailable_problem(
                    node_id=node_id, declaration=domain, block_type=self.type
                ),
            )
        except Exception:
            problem = declaration_failed_problem(
                node_id=node_id, declaration=domain, block_type=self.type
            )
            discovery = Discovery(items=[], complete=False, unknown_reasons=[problem])

        return discovery


def _container_items(value: Any, *, container: Optional[str]) -> List[Tuple[Any, Any]]:
    if container == "list" and isinstance(value, list):
        items = list(enumerate(value))
        return items
    if container == "dict" and isinstance(value, Mapping):
        items = list(value.items())
        return items

    return []


def _validate_declared_params(
    validator: ParamsValidator,
    fields: Mapping[str, FieldSpec],
    raw: Mapping[str, Any],
    *,
    owner: str,
    step_path: StepPath,
) -> BlockParams:
    """Validate a step's or source's parameters as written (shared by both specs).

    ``owner`` is the location text of the error messages, e.g.
    ``"$steps.scale (demo/scale@v1)"``; ``step_path`` is the structured
    location the errors carry.
    """
    values = {
        key: value for key, value in raw.items() if key not in RESERVED_PARAM_NAMES
    }
    try:
        params = validator.validate_definition(values)
    except ValidationError as error:
        located = clean_errors(error, values)
        details = "; ".join(
            f"{'.'.join(str(part) for part in path)}: {message}"
            for path, message in located
        )
        raise ParamsValidationError(
            f"{owner} has invalid parameters: {details}",
            step_path=step_path,
            field_path=located[0][0] if located else (),
        ) from error
    except TypeError as error:
        raise ParamsValidationError(
            f"{owner} declares a constraint that cannot apply to the given "
            f"literal: {error}",
            step_path=step_path,
        ) from error
    except Exception as error:
        raise ParamsValidationError(
            f"{owner} parameter validation raised {type(error).__name__}: {error}",
            step_path=step_path,
        ) from error

    for position_path, candidate, marker in _selector_candidates(fields, params):
        if not isinstance(candidate, str) or not candidate.startswith(
            SELECTOR_PREFIXES
        ):
            continue
        if marker.matches(candidate):
            continue
        expected = (
            "$steps.<step>"
            if marker.role == "step"
            else "$inputs.<name>, $steps.<step>.<output> or "
            "$sources.<source>.<output>"
        )
        raise SelectorError(
            f"{owner} parameter {'.'.join(str(part) for part in position_path)} "
            f"holds malformed selector {candidate!r}; expected {expected}",
            step_path=step_path,
            field_path=position_path,
        )

    return params


def _find_selector_uses(
    fields: Mapping[str, FieldSpec], params: BlockParams
) -> Tuple[SelectorUse, ...]:
    """Every selector at a declared position of validated params, in field order."""
    uses = tuple(
        SelectorUse(
            field=position_path[0],
            position=position_path[1:],
            selector=candidate,
            marker=marker,
        )
        for position_path, candidate, marker in _selector_candidates(fields, params)
        if marker.matches(candidate)
    )

    return uses


def _selector_candidates(
    fields: Mapping[str, FieldSpec], params: BlockParams
) -> List[Tuple[FieldPath, Any, SelectorMarker]]:
    """Values at declared selector positions, with their field paths."""
    candidates: List[Tuple[FieldPath, Any, SelectorMarker]] = []
    for name, field_spec in fields.items():
        value = getattr(params, name)
        if field_spec.whole is not None and isinstance(value, str):
            candidates.append(((name,), value, field_spec.whole))
            continue
        if field_spec.leaves is None:
            continue
        candidates.extend(
            ((name, position), leaf, field_spec.leaves)
            for position, leaf in _container_items(
                value, container=field_spec.container
            )
        )

    return candidates


class Block:
    """Base class of V2 blocks.

    A concrete block sets ``type`` in its own class body. Classes without their
    own ``type`` are abstract bases and cannot be registered.

    Class attributes:
        type: Canonical identity, e.g. ``"demo/scale@v1"``.
        aliases: Additional accepted identities.
        Params: Nested ``BlockParams`` model; defaults to no parameters.
        outputs: Mapping of output name to ``Output``. Names use letters,
            digits, ``_`` and ``-`` (``"class-name"``, ``"2026"``). Control
            blocks declare none; output-free side-effect blocks are valid.
        output_fields: Literal-only fields read by an overridden
            ``describe_outputs``.
        accepts_empty: ``False`` skips an invocation whose bound value is
            missing, filtered or ``None``, and a group whose children were all
            filtered. ``True`` invokes the block anyway: missing items arrive
            as ``None`` and groups contain only surviving children, possibly
            none. Genuinely empty groups reach the block under both settings.
        mutates: Fields whose bound payloads ``run`` may modify in place.
        engine_compatibility: PEP 440 specifier, e.g. ``">=2.0,<3"``.
        metadata: Free-form UI and catalogue metadata.

    Resources are the keyword parameters of ``__init__``. The engine creates
    one instance per step per execution session and keeps it across runs.

    ``run`` receives one keyword argument per ``Params`` field and returns a
    mapping of output name to value, or a ``Select`` for control blocks. A
    step that actually receives batches (see ``PlannedStep.delivers_batches``)
    is called once and returns a list with one such result per invocation.
    Values may be ``concurrent.futures.Future`` objects; the engine waits for
    them before dependent work and before returning results.
    """

    type: ClassVar[str]
    aliases: ClassVar[Tuple[str, ...]] = ()
    Params: ClassVar[Type[BlockParams]] = BlockParams
    outputs: ClassVar[Mapping[str, Output]] = MappingProxyType({})
    output_fields: ClassVar[Tuple[str, ...]] = ()
    accepts_empty: ClassVar[bool] = False
    mutates: ClassVar[Tuple[str, ...]] = ()
    engine_compatibility: ClassVar[Optional[str]] = None
    metadata: ClassVar[Mapping[str, Any]] = MappingProxyType({})

    __block_spec__: ClassVar[Optional[BlockSpec]] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.__block_spec__ = None
        if "type" not in cls.__dict__:
            return

        cls.__block_spec__ = _build_spec(cls)

    @classmethod
    def describe_outputs(cls, params: BlockParams) -> Mapping[str, Output]:
        """Derive outputs from literal parameters.

        Override together with ``output_fields`` when output names or kinds
        depend on configuration. Read only the fields named there.

        Args:
            params: Validated step parameters.

        Returns:
            Mapping of output name to ``Output``.
        """
        return cls.outputs

    @classmethod
    def discover_dependent_resources(
        cls, params: BlockParams
    ) -> Union[None, List[DependentResource], Discovery]:
        """Declare external resources the step needs; ``None`` means unknown.

        Args:
            params: Validated step parameters.

        Returns:
            ``None``, a complete list or a ``Discovery``.
        """
        return None

    @classmethod
    def discover_work_operations(
        cls, params: BlockParams
    ) -> Union[None, list, Discovery]:
        """Declare meaningful work performed; ``None`` means unknown.

        Args:
            params: Validated step parameters.

        Returns:
            ``None``, a complete list of ``WorkOperation`` or a ``Discovery``.
        """
        return None

    @classmethod
    def discover_restrictions(cls, params: BlockParams) -> Union[None, list, Discovery]:
        """Declare runtime restrictions; ``None`` means unknown.

        Args:
            params: Validated step parameters.

        Returns:
            ``None``, a complete list of ``RuntimeRestriction`` or a ``Discovery``.
        """
        return None

    @property
    def execution_context(self) -> ExecutionContext:
        """Context of the constructor or call running now (read-only).

        Available inside ``__init__`` (``run_id`` is ``None``) and ``run``.

        Raises:
            NoExecutionContextError: When read outside a constructor or call.
        """
        context = get_execution_context()

        return context

    def run(self, **kwargs: Any) -> Any:
        """Execute one invocation; concrete blocks must override it."""
        raise NotImplementedError(f"{type(self).__name__} does not implement run()")


def spec_of(block_class: Any) -> BlockSpec:
    """Return the validated declaration of a concrete block class.

    Args:
        block_class: A ``Block`` subclass that sets its own ``type``.

    Returns:
        The class's ``BlockSpec``.

    Raises:
        DeclarationError: When ``block_class`` is not a concrete block class.
    """
    if not isinstance(block_class, type) or not issubclass(block_class, Block):
        raise DeclarationError(f"Expected a Block subclass, got {block_class!r}")

    spec = block_class.__dict__.get("__block_spec__")
    if spec is None:
        raise DeclarationError(
            "is abstract: set `type` in the class body to make it registrable",
            block_class=block_class.__name__,
        )

    return spec


def _build_spec(block_class: type) -> BlockSpec:
    class_name = block_class.__name__

    def fail(message: str) -> DeclarationError:
        return DeclarationError(message, block_class=class_name)

    block_type = block_class.__dict__["type"]
    if not isinstance(block_type, str) or not _IDENTITY.fullmatch(block_type):
        raise fail(
            f"type must be a non-empty identity of letters, digits and _-./@, "
            f"got {block_type!r}"
        )

    aliases = _validate_aliases(block_class.aliases, block_type=block_type, fail=fail)

    params_model = block_class.Params
    if not isinstance(params_model, type) or not issubclass(params_model, BlockParams):
        raise fail("Params must be a subclass of BlockParams")
    if params_model.model_config.get("extra") != "forbid":
        raise fail("Params must keep extra='forbid' so unknown parameters are rejected")

    fields = _analyze_fields(params_model, fail=fail)
    try:
        validator = ParamsValidator(params_model)
    except Exception as error:
        raise fail(f"Params cannot be projected for validation: {error}") from error
    static_outputs = _validate_outputs(
        block_class.outputs, fields=fields, block_class=class_name, origin="outputs"
    )

    configured_outputs = (
        block_class.describe_outputs.__func__ is not Block.describe_outputs.__func__
    )
    output_fields = _validate_output_fields(
        block_class.output_fields,
        fields=fields,
        configured_outputs=configured_outputs,
        fail=fail,
    )

    is_control = any(spec.role == "step" for spec in fields.values())
    if is_control and (static_outputs or configured_outputs):
        raise fail(
            "declares StepRef control targets and outputs; control blocks return "
            "Select/Stop and must declare no outputs"
        )

    accepts_batches = any(
        marker.batch != "never" for spec in fields.values() for marker in spec.markers
    )

    accepts_empty = block_class.accepts_empty
    if not isinstance(accepts_empty, bool):
        raise fail(f"accepts_empty must be a bool, got {accepts_empty!r}")

    mutates = _validate_mutates(block_class.mutates, fields=fields, fail=fail)
    engine_compatibility = _validate_compatibility(
        block_class.engine_compatibility, fail=fail
    )
    _validate_run_signature(block_class, fields=fields, fail=fail)

    try:
        resources = read_resource_specs(block_class)
    except ContractError as error:
        raise fail(str(error)) from error

    metadata = block_class.metadata
    if not isinstance(metadata, Mapping):
        raise fail(f"metadata must be a mapping, got {type(metadata).__name__}")

    spec = BlockSpec(
        type=block_type,
        aliases=aliases,
        block_class=block_class,
        params_model=params_model,
        fields=MappingProxyType(fields),
        outputs=static_outputs,
        configured_outputs=configured_outputs,
        output_fields=output_fields,
        resources=resources,
        is_control=is_control,
        accepts_batches=accepts_batches,
        accepts_empty=accepts_empty,
        mutates=mutates,
        engine_compatibility=engine_compatibility,
        description=inspect.cleandoc(block_class.__doc__ or ""),
        metadata=MappingProxyType(dict(metadata)),
        kinds=_collect_kinds(fields=fields, outputs=static_outputs, fail=fail),
        _validator=validator,
    )

    return spec


def _validate_aliases(aliases: Any, *, block_type: str, fail) -> Tuple[str, ...]:
    if isinstance(aliases, str) or not isinstance(aliases, (tuple, list)):
        raise fail(f"aliases must be a tuple of identities, got {aliases!r}")

    normalized = tuple(aliases)
    for alias in normalized:
        if not isinstance(alias, str) or not _IDENTITY.fullmatch(alias):
            raise fail(f"alias {alias!r} is not a valid identity")
    if block_type in normalized:
        raise fail(f"alias repeats the canonical type {block_type!r}")
    if len(set(normalized)) != len(normalized):
        raise fail(f"aliases repeat an identity: {list(normalized)}")

    return normalized


def _analyze_fields(params_model: Type[BlockParams], *, fail) -> Dict[str, FieldSpec]:
    fields: Dict[str, FieldSpec] = {}
    for name, info in params_model.model_fields.items():
        if name in RESERVED_PARAM_NAMES:
            raise fail(f"Params field {name!r} is reserved for the step itself")

        structure = analyze_annotation(
            field_annotation(info), field_name=name, fail=fail
        )
        fields[name] = FieldSpec(
            name=name,
            required=info.is_required(),
            has_default=not info.is_required(),
            whole=structure.whole,
            leaves=structure.leaves,
            container=structure.container,
            literal_allowed=structure.literal_allowed,
            description=info.description,
            input_paths=_input_paths(name, info, config=params_model.model_config),
            schema_property=_schema_property(
                name, info, config=params_model.model_config
            ),
        )

    return fields


def _validates_by_alias(config: Mapping[str, Any]) -> bool:
    return config.get("validate_by_alias", True)


def _validates_by_name(config: Mapping[str, Any]) -> bool:
    # As in Pydantic: the legacy populate_by_name applies only when
    # validate_by_name is not set.
    validate_by_name = config.get("validate_by_name")
    if validate_by_name is None:
        return config.get("populate_by_name", False)

    return validate_by_name


def _params_json_schema(params_model: Type[BlockParams]) -> Dict[str, Any]:
    by_alias = _validates_by_alias(params_model.model_config)
    schema = params_model.model_json_schema(by_alias=by_alias)

    return schema


def _input_paths(
    name: str, info: FieldInfo, *, config: Mapping[str, Any]
) -> Tuple[InputPath, ...]:
    """Locations Pydantic validation reads the field from, in lookup order.

    Paths under ``type`` or ``name`` are left out: ``validate_params`` removes
    those keys before Pydantic sees them.
    """
    alias = info.validation_alias if info.validation_alias is not None else info.alias
    paths: List[InputPath] = []
    if alias is not None and _validates_by_alias(config):
        choices = alias.choices if isinstance(alias, AliasChoices) else [alias]
        paths.extend(
            tuple(choice.path) if isinstance(choice, AliasPath) else (choice,)
            for choice in choices
        )
    if alias is None or _validates_by_name(config):
        paths.append((name,))
    writable = [path for path in paths if path[0] not in RESERVED_PARAM_NAMES]

    return tuple(dict.fromkeys(writable))


def _schema_property(name: str, info: FieldInfo, *, config: Mapping[str, Any]) -> str:
    """Property naming of Pydantic's validation-mode JSON schema.

    A string alias names the property; ``AliasChoices`` contributes its first
    single-key choice; an ``AliasPath`` alone leaves the field name.
    """
    alias = info.validation_alias if info.validation_alias is not None else info.alias
    if not _validates_by_alias(config) or alias is None:
        return name
    if isinstance(alias, str):
        return alias
    if isinstance(alias, AliasChoices):
        for choice in alias.choices:
            path = choice.path if isinstance(choice, AliasPath) else [choice]
            if len(path) == 1 and isinstance(path[0], str):
                return path[0]

    return name


def _validate_outputs(
    declared: Any,
    *,
    fields: Mapping[str, FieldSpec],
    block_class: str,
    origin: str,
) -> Mapping[str, Output]:
    def fail(message: str) -> DeclarationError:
        return DeclarationError(f"{origin} {message}", block_class=block_class)

    if not isinstance(declared, Mapping):
        raise fail(
            f"must be a mapping of output name to Output, got {type(declared).__name__}"
        )

    outputs: Dict[str, Output] = {}
    for name, output in declared.items():
        if not is_selector_segment(name):
            raise fail(
                f"output name {name!r} must use letters, digits, _ or - so a "
                "selector can address it"
            )
        if not isinstance(output, Output):
            raise fail(
                f"output {name!r} must be an Output, got {type(output).__name__}"
            )
        if output.preserve is not None:
            group_field = fields.get(output.preserve)
            if group_field is None or group_field.role != "group":
                raise fail(
                    f"output {name!r} preserves {output.preserve!r}, which is not a "
                    "Group field"
                )
        if output.source is not None:
            source_field = fields.get(output.source)
            if source_field is None or source_field.role not in ("item", "group"):
                raise fail(
                    f"output {name!r} takes context from {output.source!r}, which "
                    "accepts no data selector"
                )
        outputs[name] = output

    frozen_outputs = MappingProxyType(outputs)

    return frozen_outputs


def _validate_output_fields(
    output_fields: Any,
    *,
    fields: Mapping[str, FieldSpec],
    configured_outputs: bool,
    fail,
) -> Tuple[str, ...]:
    if isinstance(output_fields, str) or not isinstance(output_fields, (tuple, list)):
        raise fail(
            f"output_fields must be a tuple of field names, got {output_fields!r}"
        )

    names = tuple(output_fields)
    if configured_outputs and not names:
        raise fail(
            "overrides describe_outputs but declares no output_fields; name the "
            "literal-only fields that shape its outputs"
        )
    if names and not configured_outputs:
        raise fail("declares output_fields without overriding describe_outputs")

    for name in names:
        field_spec = fields.get(name)
        if field_spec is None:
            raise fail(f"output_fields names unknown field {name!r}")
        if field_spec.accepts_selectors:
            raise fail(
                f"output field {name!r} accepts selectors; fields shaping outputs "
                "must be literal-only so topology is known at compile time"
            )

    return names


def _validate_mutates(
    mutates: Any, *, fields: Mapping[str, FieldSpec], fail
) -> Tuple[str, ...]:
    if isinstance(mutates, str) or not isinstance(mutates, (tuple, list)):
        raise fail(f"mutates must be a tuple of field names, got {mutates!r}")

    names = tuple(mutates)
    for name in names:
        field_spec = fields.get(name)
        if field_spec is None:
            raise fail(f"mutates names unknown field {name!r}")
        if field_spec.role not in ("item", "group"):
            raise fail(
                f"mutates names {name!r}, which accepts no data selector; only "
                "selector-bound payloads can be shared with other steps"
            )

    return names


def _validate_compatibility(value: Any, *, fail) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise fail(f"engine_compatibility must be a string, got {value!r}")

    try:
        SpecifierSet(value)
    except InvalidSpecifier as error:
        raise fail(
            f"engine_compatibility {value!r} is not a valid specifier"
        ) from error

    return value


def _validate_run_signature(
    block_class: type, *, fields: Mapping[str, FieldSpec], fail
) -> None:
    run = block_class.run
    if run is Block.run:
        raise fail("does not implement run()")

    _validate_keyword_signature(run, name="run", fields=fields, fail=fail)


def _validate_keyword_signature(
    method: Any, *, name: str, fields: Mapping[str, FieldSpec], fail
) -> None:
    """Check that ``method`` takes every ``Params`` field as a keyword argument."""
    parameters = list(inspect.signature(method).parameters.values())[1:]
    accepts_any_keyword = False
    accepted_names = set()
    for parameter in parameters:
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            accepts_any_keyword = True
            continue
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.POSITIONAL_ONLY,
        ):
            raise fail(
                f"{name}() parameter {parameter.name!r} must be passable by keyword"
            )
        accepted_names.add(parameter.name)
        if (
            parameter.name not in fields
            and parameter.default is inspect.Parameter.empty
        ):
            raise fail(
                f"{name}() requires {parameter.name!r}, which is not a Params field"
            )

    missing = [field_name for field_name in fields if field_name not in accepted_names]
    if missing and not accepts_any_keyword:
        raise fail(f"{name}() does not accept Params field(s) {missing}")


def _collect_kinds(
    *, fields: Mapping[str, FieldSpec], outputs: Mapping[str, Output], fail
) -> Tuple[Kind, ...]:
    kind_groups = [output.kinds for output in outputs.values()]
    kind_groups.extend(
        marker.kinds for field_spec in fields.values() for marker in field_spec.markers
    )
    collected: Dict[str, Kind] = {}
    for kinds in kind_groups:
        for kind in kinds:
            known = collected.setdefault(kind.name, kind)
            if known != kind:
                raise fail(
                    f"uses two different kinds named {kind.name!r}; import one "
                    "shared Kind object"
                )

    ordered = tuple(collected[name] for name in sorted(collected))

    return ordered
