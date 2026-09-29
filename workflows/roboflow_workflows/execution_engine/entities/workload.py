"""Dependency-light workload declaration primitives.

This module is the lowest layer of the workload-introspection contract. It is
imported by ``roboflow_workflows.prototypes.block`` (which re-exports the enums
and ``RuntimeRestriction`` for backwards compatibility) and by the response
schemas in
``roboflow_workflows.execution_engine.introspection.workload_entities``.

``RuntimeRestriction`` - the ONE entity a block authors a restriction with -
lives here rather than in ``prototypes.block`` so that the discovery machinery
in this module can key and project it without importing the framework layer.
``prototypes.block`` re-exports the very same class object, so
``from roboflow_workflows.prototypes.block import RuntimeRestriction`` is
unchanged.

Every workload DTO carries a versioned ``type`` discriminator (e.g.
``discovery_v1``). The suffix is the schema version of that one entity, so an
explicit tag of any other version is rejected; omitting ``type`` on
construction keeps the default.

Import discipline (an import-cycle trap otherwise): only the standard library,
``pydantic`` and ``typing_extensions`` may be imported here. Never import
``roboflow_workflows.prototypes.*``, ``core_steps.*``, ``execution_engine.v1.*``
or ``execution_engine.introspection.*`` from this module.
"""

import json
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Dict,
    Generic,
    Iterable,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationInfo,
    field_validator,
    model_validator,
)


class Severity(str, Enum):
    """Severity of a runtime restriction for a workflow block in a given runtime.

    SOFT: the block runs to completion and returns the right output shape,
    but the values are degraded or meaningless (e.g. tracker IDs reset across
    requests, cooldown does not throttle, file is written to ephemeral disk).

    HARD: the block does not run / raises / cannot produce a usable output
    in this runtime. The engine should refuse to compile or fail-fast.
    """

    SOFT = "soft"
    HARD = "hard"


class Runtime(str, Enum):
    """Canonical runtimes a workflow block can be executed in.

    Runtimes not covered by ``get_restrictions()`` are considered OK.
    """

    HOSTED_SERVERLESS = "hosted_serverless"
    DEDICATED_DEPLOYMENT = "dedicated_deployment"
    SELF_HOSTED_CPU = "self_hosted_cpu"
    SELF_HOSTED_GPU = "self_hosted_gpu"
    INFERENCE_PIPELINE = "inference_pipeline"


class RuntimeInputMode(str, Enum):
    """Workflow input modes for a restriction."""

    IMAGE = "image"
    VIDEO = "video"


class StepExecutionMode(Enum):
    """How a workflow step is dispatched at runtime.

    LOCAL: the step executes in-process inside the current Python interpreter.
    REMOTE: the step delegates execution to a remote inference service / HTTP
    runtime.

    Defined in this dependency-light module and re-exported from
    ``roboflow_workflows.prototypes.block`` (the historic home), so the
    framework layer keeps owning the enum while ``core_steps``, the executor
    and the compiler depend on ``prototypes`` rather than the other way
    around. Deliberately a plain ``Enum`` (not ``str``) - keep it that way.
    """

    LOCAL = "local"
    REMOTE = "remote"


class WorkOperation(str, Enum):
    """Meaningful categories of work a workflow step performs.

    Blocks declare members of this enum through
    ``WorkflowBlockManifest.discover_work_operations()``. The set is closed on
    purpose: there is no UNKNOWN/OTHER member - uncertainty is expressed through
    ``Discovery.complete`` / ``Discovery.unknown_reasons``, and adding a new
    category requires an Execution Engine update.
    """

    MODEL_INFERENCE = "model_inference"
    CUSTOM_PYTHON = "custom_python"
    IMAGE_RESIZE = "image_resize"
    IMAGE_CROP = "image_crop"
    IMAGE_TRANSFORM = "image_transform"
    IMAGE_COMPOSITION = "image_composition"
    IMAGE_FILTERING = "image_filtering"
    IMAGE_ANALYSIS = "image_analysis"
    IMAGE_ENCODING = "image_encoding"
    VISUALIZATION = "visualization"
    DETECTION_PROCESSING = "detection_processing"
    DETECTION_MATCHING = "detection_matching"
    TRACKING = "tracking"
    NUMERICAL_COMPUTATION = "numerical_computation"
    EXPRESSION_EVALUATION = "expression_evaluation"
    DATA_TRANSFORMATION = "data_transformation"
    DATA_AGGREGATION = "data_aggregation"
    TEMPORAL_BUFFERING = "temporal_buffering"
    FLOW_CONTROL = "flow_control"
    EXTERNAL_REQUEST = "external_request"
    STORAGE_READ = "storage_read"
    STORAGE_WRITE = "storage_write"
    CACHE_READ = "cache_read"
    CACHE_WRITE = "cache_write"
    ENVIRONMENT_READ = "environment_read"


T = TypeVar("T")

RESTRICTION_CODE_PATTERN = r"^[a-z][a-z0-9_]*$"


def _canonical_json(model: BaseModel) -> str:
    return json.dumps(model.model_dump(mode="json"), sort_keys=True)


def _sorted_unique_enum_axis(values: List[Enum], field_name: str) -> List[Enum]:
    if len(values) == 0:
        raise ValueError(
            f"`{field_name}` must be None (unrestricted) or a non-empty list; "
            f"an empty list is an ambiguous declaration and is rejected."
        )
    seen = set()
    for member in values:
        if member in seen:
            raise ValueError(
                f"`{field_name}` contains a duplicated entry: {member.value!r}."
            )
        seen.add(member)
    return sorted(values, key=lambda member: str(member.value))


class RestrictionCondition(BaseModel):
    """When a portable restriction applies.

    Semantics: AND across the axes and across ``configuration_equals``
    entries, OR within a single populated axis list. ``None`` on an axis means
    the axis does not restrict (unrestricted). An empty list is rejected: it
    would be an ambiguous declaration.

    The default instance (every axis ``None``, no configuration entries) means
    "always applies".
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["restriction_condition_v1"] = "restriction_condition_v1"
    runtimes: Optional[List[Runtime]] = None
    step_execution_modes: Optional[List[StepExecutionMode]] = None
    input_modes: Optional[List[RuntimeInputMode]] = None
    configuration_equals: Dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("runtimes", "step_execution_modes", "input_modes", mode="after")
    @classmethod
    def _normalise_axis(
        cls, value: Optional[List[Enum]], info: ValidationInfo
    ) -> Optional[List[Enum]]:
        if value is None:
            return None
        return _sorted_unique_enum_axis(values=value, field_name=str(info.field_name))

    @field_validator("configuration_equals", mode="after")
    @classmethod
    def _reject_blank_configuration_keys(
        cls, value: Dict[str, JsonValue]
    ) -> Dict[str, JsonValue]:
        for key in value:
            if not key.strip():
                raise ValueError(
                    "`configuration_equals` keys must be non-empty configuration "
                    "field names."
                )
        return value

    def __hash__(self) -> int:
        # `configuration_equals` is a dict, so pydantic's generated hash for
        # frozen models would fail; hash the canonical JSON form instead.
        return hash(_canonical_json(self))


class RestrictionMetadata(BaseModel):
    """Portable declaration of a runtime restriction.

    ``code`` is a stable machine-readable identifier authored from the
    restriction's meaning (never derived from a human note). Portable
    declarations intentionally carry no free-text note; human notes stay in
    the legacy ``get_restrictions()`` API.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["restriction_v1"] = "restriction_v1"
    code: str = Field(min_length=1, pattern=RESTRICTION_CODE_PATTERN)
    severity: Severity
    when: RestrictionCondition = RestrictionCondition()

    def __hash__(self) -> int:
        return hash((self.code, self.severity, self.when))


@dataclass(frozen=True)
class RuntimeRestriction:
    """A single caveat for a workflow block - the ONE entity blocks author.

    ``severity`` and ``note`` keep their historic meaning: ``note`` is a
    one-line, human-readable explanation of the failure mode or degraded
    behavior (e.g. "track_ids reset between requests", "raises RuntimeError"),
    not an abstract precondition.

    ``applies_to_runtimes`` / ``applies_to_step_execution_modes`` /
    ``applies_to_input_modes`` narrow the restriction to specific workflow
    runtimes, step execution modes and input modes. An axis left unset applies
    to all of them.

    ``code`` is a stable, machine-readable identifier authored from the
    restriction's MEANING (never derived from the note). The set is open -
    it is a plain string, and a plugin may add its own. The default,
    ``generic_restriction``, says "this caveat has no dedicated identifier".
    Within a ``Discovery[RuntimeRestriction]`` the note is part of a
    restriction's identity, so two ``generic_restriction`` entries explaining
    different failure modes stay two entries. That distinction is deliberately
    NOT preserved on the wire: ``RestrictionMetadata`` carries no note, so both
    project onto the same DTO and the portable discovery coalesces them under
    its own ``(code, severity, condition)`` identity.

    ``applies_to_configuration`` narrows the restriction to a target whose
    configuration matches every listed ``key == value`` pair. The keys are
    configuration field names (e.g. ``ENABLE_TENSOR_DATA_REPRESENTATION``);
    the values are the values the restriction applies to. A declaration is
    written for the TARGET configuration - the host answering an introspection
    call never bakes its own flags into it.

    Both new fields are defaulted and appended after the historic ones, so
    every existing positional and keyword constructor stays valid, and
    ``to_dict()`` (the legacy editor payload) is unchanged: it carries neither
    the code nor the configuration.
    """

    severity: Severity
    note: str
    applies_to_runtimes: Optional[List[Runtime]] = None
    applies_to_step_execution_modes: Optional[List[StepExecutionMode]] = None
    applies_to_input_modes: Optional[List[RuntimeInputMode]] = None
    code: str = "generic_restriction"
    applies_to_configuration: Optional[Dict[str, JsonValue]] = None

    def to_dict(self) -> Dict[str, Any]:
        """The legacy editor payload. Deliberately unchanged - `code` and
        `applies_to_configuration` are NOT part of it."""
        result: Dict[str, Any] = {"severity": self.severity.value, "note": self.note}
        if self.applies_to_runtimes is not None:
            result["applies_to_runtimes"] = [
                runtime.value for runtime in self.applies_to_runtimes
            ]
        if self.applies_to_step_execution_modes is not None:
            result["applies_to_step_execution_modes"] = [
                mode.value for mode in self.applies_to_step_execution_modes
            ]
        if self.applies_to_input_modes is not None:
            result["applies_to_input_modes"] = [
                mode.value for mode in self.applies_to_input_modes
            ]
        return result


def restriction_metadata_of(restriction: RuntimeRestriction) -> RestrictionMetadata:
    """Project the authored entity onto the portable workload DTO.

    The note is dropped on purpose - ``RestrictionMetadata`` is the wire form
    and carries no free text. The configuration map is COPIED, so the returned
    model never shares mutable state with a module-level preset.
    """
    return RestrictionMetadata(
        code=restriction.code,
        severity=restriction.severity,
        when=RestrictionCondition(
            runtimes=restriction.applies_to_runtimes,
            step_execution_modes=restriction.applies_to_step_execution_modes,
            input_modes=restriction.applies_to_input_modes,
            configuration_equals=dict(restriction.applies_to_configuration or {}),
        ),
    )


class DiscoveryProblemCode(str, Enum):
    """Why a discovery is incomplete. The set is closed and owned by the
    Execution Engine: a consumer branches on these codes, never on the
    human-readable ``description``.

    DECLARATION_UNAVAILABLE: the declaration is not available as a complete,
    portable statement. Three producers use it, told apart by ``details``: the
    block does not declare this domain at all (the hook returned ``None``, e.g.
    an unannotated third-party plugin); restrictions come only from the legacy
    ``get_restrictions()`` classmethod, which may already have filtered them
    for the answering host (``details.source == "get_restrictions"``); or a
    restriction's condition names configuration this process cannot evaluate
    (``details.configuration_keys``).
    DECLARATION_FAILED: the declaration hook raised or returned something that
    is not a valid declaration.
    UNRESOLVED_SELECTOR: a value the declaration depends on is a workflow
    selector, so it is only known at run time.
    INVALID_RESOURCE_IDENTIFIER: a literal resource identifier names nothing
    (blank / whitespace-only) - no identifier is ever fabricated.
    OPAQUE_REMOTE_WORKFLOW: the step dispatches a child workflow to a remote
    server, which compiles it; the child is not inspectable here.
    CUSTOM_PYTHON_INTERNALS_UNKNOWN: the step runs user-supplied Python; what
    the code does beyond the declared items is not statically analysable.
    """

    DECLARATION_UNAVAILABLE = "declaration_unavailable"
    DECLARATION_FAILED = "declaration_failed"
    UNRESOLVED_SELECTOR = "unresolved_selector"
    INVALID_RESOURCE_IDENTIFIER = "invalid_resource_identifier"
    OPAQUE_REMOTE_WORKFLOW = "opaque_remote_workflow"
    CUSTOM_PYTHON_INTERNALS_UNKNOWN = "custom_python_internals_unknown"


DeclarationDomain = Literal["resources", "operations", "restrictions"]


class DiscoveryProblem(BaseModel):
    """One structured reason why a ``Discovery`` is incomplete.

    ``code`` is what a consumer branches on, ``description`` is display text
    (never parsed), and ``details`` is an open JSON map understood per code -
    a loose-end contract: the conventions below are documented, not enforced
    by code-specific models, so a producer may add context without a schema
    change.

    Conventions of the built-in problems (see the workload introspection docs):

    * every built-in problem: ``node_id`` (canonical ``$steps.<name>`` id) and
      ``declaration`` (``resources`` / ``operations`` / ``restrictions``);
    * ``declaration_unavailable`` / ``declaration_failed``: ``block_type``
      where known;
    * ``unresolved_selector``: ``field`` and ``selector``, plus
      ``resource_type`` when the selector sits in a resource identity field
      (``field`` then names the resource metadata field);
    * ``invalid_resource_identifier``: ``field`` and ``resource_type`` - never
      the invalid value itself.

    Neither the description nor the details ever carry raw exception text,
    tracebacks, secret values or authorization headers.

    Two problems are the SAME problem when their ``code`` and their canonical
    (sorted-key) ``details`` JSON match; the wording of ``description`` is not
    part of that identity.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["discovery_problem_v1"] = "discovery_problem_v1"
    code: DiscoveryProblemCode
    description: str = Field(min_length=1)
    details: Dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("description", mode="after")
    @classmethod
    def _reject_blank_description(cls, value: str) -> str:
        if not value.strip():
            raise ValueError(
                "`description` must be a human-readable sentence, not blank."
            )
        return value

    @field_validator("details", mode="after")
    @classmethod
    def _reject_blank_detail_keys(
        cls, value: Dict[str, JsonValue]
    ) -> Dict[str, JsonValue]:
        for key in value:
            if not key.strip():
                raise ValueError("`details` keys must be non-empty names.")
        return value

    def identity(self) -> Tuple[str, str]:
        """What makes two problems the same one: the code and the canonical
        JSON form of the details - never the description."""
        return self.code.value, json.dumps(self.details, sort_keys=True)

    def __hash__(self) -> int:
        # `details` is a dict, so pydantic's generated hash for frozen models
        # would fail; hash the canonical JSON form instead. Deduplication goes
        # through `identity()` - never put this object in a set, its details
        # map stays mutable.
        return hash(_canonical_json(self))


def declaration_unavailable_problem(
    node_id: str, declaration: DeclarationDomain, block_type: Optional[str] = None
) -> DiscoveryProblem:
    """The block declares nothing about this domain (hook returned ``None``)."""
    details: Dict[str, JsonValue] = {"node_id": node_id, "declaration": declaration}
    if block_type is not None:
        details["block_type"] = block_type
    return DiscoveryProblem(
        code=DiscoveryProblemCode.DECLARATION_UNAVAILABLE,
        description=(
            f"Step `{node_id}` does not declare its {declaration}, so they are "
            f"unknown rather than absent."
        ),
        details=details,
    )


def declaration_failed_problem(
    node_id: str, declaration: DeclarationDomain, block_type: Optional[str] = None
) -> DiscoveryProblem:
    """The declaration hook raised or answered with an invalid shape."""
    details: Dict[str, JsonValue] = {"node_id": node_id, "declaration": declaration}
    if block_type is not None:
        details["block_type"] = block_type
    return DiscoveryProblem(
        code=DiscoveryProblemCode.DECLARATION_FAILED,
        description=(
            f"The {declaration} declaration of step `{node_id}` could not be "
            f"read: the block hook failed or returned an invalid declaration."
        ),
        details=details,
    )


def environment_filtered_declaration_problem(
    node_id: str, declaration: DeclarationDomain, block_type: Optional[str] = None
) -> DiscoveryProblem:
    """The only declaration available is the legacy ``get_restrictions()``.

    The default ``get_actual_restrictions()`` falls back to that classmethod:
    a block's own override, or the inherited ``[]`` when it never declared
    anything. A legacy override MAY already have filtered its entries against
    the flags of the host answering the call, so what comes back cannot be
    assumed environment-neutral; being a classmethod it also cannot refine the
    list from this step's own settings, so instance-specific restrictions may
    be missing; and the inherited default states nothing at all. Whatever the
    classmethod returned is kept - codes included, a legacy declaration may
    well carry a specific one - but the result is never claimed complete: an
    empty or short list is not proof of absence. Reported under the existing
    ``declaration_unavailable`` code, distinguished by ``source``.
    """
    details: Dict[str, JsonValue] = {
        "node_id": node_id,
        "declaration": declaration,
        "source": "get_restrictions",
    }
    if block_type is not None:
        details["block_type"] = block_type
    return DiscoveryProblem(
        code=DiscoveryProblemCode.DECLARATION_UNAVAILABLE,
        description=(
            f"Step `{node_id}` only declares its {declaration} through the "
            f"legacy `get_restrictions()` API, which may already have filtered "
            f"them for the host that answered and cannot report ones specific "
            f"to this step's settings, so whatever it returned is kept but "
            f"cannot be claimed complete."
        ),
        details=details,
    )


def unknown_configuration_problem(
    node_id: str,
    declaration: DeclarationDomain,
    configuration_keys: List[str],
    block_type: Optional[str] = None,
) -> DiscoveryProblem:
    """A declaration is conditioned on configuration this process cannot read.

    Only the KEY NAMES are reported - never the host's values. The entries
    concerned are kept, because an unreadable condition is uncertainty, not a
    reason to drop a restriction.

    Emitted only while the condition is still open: a restriction whose
    predicates are ANDed and which already has one evaluable predicate that
    does NOT hold here is definitively inactive, so it is dropped without a
    problem and its unreadable keys are never reported.
    """
    details: Dict[str, JsonValue] = {
        "node_id": node_id,
        "declaration": declaration,
        "configuration_keys": sorted(set(configuration_keys)),
    }
    if block_type is not None:
        details["block_type"] = block_type
    return DiscoveryProblem(
        code=DiscoveryProblemCode.DECLARATION_UNAVAILABLE,
        description=(
            f"Step `{node_id}` conditions part of its {declaration} on "
            f"configuration this process cannot evaluate, so those entries are "
            f"kept without being confirmed."
        ),
        details=details,
    )


def with_block_type(problem: DiscoveryProblem, block_type: str) -> DiscoveryProblem:
    """Add the reporter-known ``block_type`` to a declaration-level problem.

    A manifest hook knows its step, not the canonical block identifier the
    registry gave the block; the caller that forwards the problem does. Only
    ``declaration_unavailable`` / ``declaration_failed`` carry ``block_type``
    by convention, and an already-present value is never overwritten.
    """
    if problem.code not in (
        DiscoveryProblemCode.DECLARATION_UNAVAILABLE,
        DiscoveryProblemCode.DECLARATION_FAILED,
    ):
        return problem
    if "block_type" in problem.details:
        return problem
    return problem.model_copy(
        update={"details": {**problem.details, "block_type": block_type}}
    )


def unresolved_selector_problem(
    node_id: str,
    declaration: DeclarationDomain,
    field: str,
    selector: str,
    resource_type: Optional[str] = None,
) -> DiscoveryProblem:
    """A selector-valued field keeps the declaration from being complete."""
    details: Dict[str, JsonValue] = {
        "node_id": node_id,
        "declaration": declaration,
        "field": field,
        "selector": selector,
    }
    if resource_type is not None:
        details["resource_type"] = resource_type
        subject = f"Field `{field}` of the {resource_type} declared by step"
    else:
        subject = f"Field `{field}` of step"
    return DiscoveryProblem(
        code=DiscoveryProblemCode.UNRESOLVED_SELECTOR,
        description=(
            f"{subject} `{node_id}` is set by selector `{selector}`, which is "
            f"only known at run time, so the {declaration} are not fully known."
        ),
        details=details,
    )


def invalid_resource_identifier_problem(
    node_id: str, declaration: DeclarationDomain, field: str, resource_type: str
) -> DiscoveryProblem:
    """A literal resource identifier names nothing (blank / whitespace-only)."""
    return DiscoveryProblem(
        code=DiscoveryProblemCode.INVALID_RESOURCE_IDENTIFIER,
        description=(
            f"Step `{node_id}` declares a {resource_type} whose `{field}` is "
            f"blank, so the resource cannot be identified."
        ),
        details={
            "node_id": node_id,
            "declaration": declaration,
            "field": field,
            "resource_type": resource_type,
        },
    )


def opaque_remote_workflow_problem(
    node_id: str, declaration: DeclarationDomain
) -> DiscoveryProblem:
    """The child workflow is compiled by a remote server, not inspectable."""
    return DiscoveryProblem(
        code=DiscoveryProblemCode.OPAQUE_REMOTE_WORKFLOW,
        description=(
            f"Step `{node_id}` dispatches its child workflow to a remote "
            f"server, so the child's {declaration} are not visible here."
        ),
        details={"node_id": node_id, "declaration": declaration},
    )


def custom_python_internals_unknown_problem(
    node_id: str, declaration: DeclarationDomain
) -> DiscoveryProblem:
    """User-supplied Python may do more than the declared items say."""
    return DiscoveryProblem(
        code=DiscoveryProblemCode.CUSTOM_PYTHON_INTERNALS_UNKNOWN,
        description=(
            f"Step `{node_id}` runs custom Python code, so its internals may "
            f"add {declaration} beyond the declared ones."
        ),
        details={"node_id": node_id, "declaration": declaration},
    )


def _discovery_item_sort_key(item: Any) -> Tuple[str, ...]:
    if isinstance(item, Enum):
        return (str(item.value),)
    if isinstance(item, RestrictionMetadata):
        return (item.code, item.severity.value, _canonical_json(item.when))
    if isinstance(item, RuntimeRestriction):
        # The note is PART of the identity: two `generic_restriction` entries
        # explaining different failure modes are two restrictions, and merging
        # them by code would silently drop one. The axes are keyed without
        # going through `RestrictionMetadata`, so an invalid declaration still
        # sorts (it fails later, where it can be reported as a problem).
        return (
            str(item.code),
            item.severity.value,
            item.note,
            json.dumps(
                {
                    "applies_to_runtimes": item.applies_to_runtimes,
                    "applies_to_step_execution_modes": (
                        item.applies_to_step_execution_modes
                    ),
                    "applies_to_input_modes": item.applies_to_input_modes,
                    "applies_to_configuration": item.applies_to_configuration,
                },
                sort_keys=True,
                default=str,
            ),
        )
    if isinstance(item, BaseModel):
        return (_canonical_json(item),)
    try:
        return (json.dumps(item, sort_keys=True, default=str),)
    except TypeError:
        return (repr(item),)


def _deduplicate_and_sort_items(items: List[Any]) -> List[Any]:
    keyed: Dict[Tuple[str, ...], Any] = {}
    for item in items:
        key = _discovery_item_sort_key(item)
        if key not in keyed:
            keyed[key] = item
    return [keyed[key] for key in sorted(keyed)]


def _deduplicate_and_sort_problems(
    problems: List[DiscoveryProblem],
) -> List[DiscoveryProblem]:
    """Unique by ``(code, canonical details)`` and sorted by that same key.

    Neither the result order nor the surviving entry depends on the order the
    producers ran in, on the insertion order of a details map, or on how a
    description happens to be worded: identical identities with different
    wording keep the lexicographically first description.
    """
    keyed: Dict[Tuple[str, str], DiscoveryProblem] = {}
    for problem in problems:
        key = problem.identity()
        current = keyed.get(key)
        if current is None or problem.description < current.description:
            keyed[key] = problem
    return [keyed[key] for key in sorted(keyed)]


class Discovery(BaseModel, Generic[T]):
    """A set of discovered facts together with an honest completeness claim.

    * ``complete=True`` with no items is a known absence (the step truthfully
      does nothing of this kind). Complete results carry no ``unknown_reasons``.
    * ``complete=False`` may still list the items that ARE known, and MUST
      carry at least one reason. Reasons are ``DiscoveryProblem`` objects: a
      closed ``code`` a consumer branches on, a human-readable ``description``
      and open ``details`` - never exception traces or secrets.

    Items are de-duplicated and deterministically sorted (enum members by
    value, ``RestrictionMetadata`` by ``(code, severity, condition JSON)``,
    other models by their JSON dump with sorted keys); reasons are unique by
    ``(code, canonical details)`` and sorted by that key. Instances are frozen.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["discovery_v1"] = "discovery_v1"
    items: List[T]
    complete: bool
    unknown_reasons: List[DiscoveryProblem]

    @field_validator("items", mode="after")
    @classmethod
    def _canonicalise_items(cls, value: List[Any]) -> List[Any]:
        return _deduplicate_and_sort_items(items=value)

    @field_validator("unknown_reasons", mode="after")
    @classmethod
    def _canonicalise_reasons(
        cls, value: List[DiscoveryProblem]
    ) -> List[DiscoveryProblem]:
        return _deduplicate_and_sort_problems(problems=value)

    @model_validator(mode="after")
    def _enforce_completeness_contract(self) -> "Discovery[T]":
        if self.complete and self.unknown_reasons:
            raise ValueError(
                "A complete discovery must not carry `unknown_reasons`; "
                "set complete=False to report unknowns."
            )
        if not self.complete and not self.unknown_reasons:
            raise ValueError(
                "An incomplete discovery must state at least one reason in "
                "`unknown_reasons`."
            )
        return self


def complete_discovery(items: Iterable[T]) -> Discovery[T]:
    """Build a complete discovery (known absence when ``items`` is empty)."""
    return Discovery(items=list(items), complete=True, unknown_reasons=[])


def incomplete_discovery(
    items: Iterable[T], reasons: Iterable[DiscoveryProblem]
) -> Discovery[T]:
    """Build an incomplete discovery listing the known ``items`` and the
    ``DiscoveryProblem`` reasons why the rest is unknown (non-empty)."""
    return Discovery(items=list(items), complete=False, unknown_reasons=list(reasons))


def normalize_declaration(
    value: Union[None, Iterable[T], Discovery[T]], unknown_problem: DiscoveryProblem
) -> Discovery[T]:
    """Map a manifest hook return value onto the ``Discovery`` convention.

    * ``None`` -> unknown: ``Discovery(items=[], complete=False,
      unknown_reasons=[unknown_problem])``.
    * a plain list (any iterable) -> complete declaration (may be empty).
    * a ``Discovery`` -> re-validated copy of itself (item objects preserved,
      so in-process aids such as excluded resolver callbacks survive).
    """
    if value is None:
        return Discovery(items=[], complete=False, unknown_reasons=[unknown_problem])
    if isinstance(value, Discovery):
        return type(value)(
            items=list(value.items),
            complete=value.complete,
            unknown_reasons=list(value.unknown_reasons),
        )
    return complete_discovery(items=value)


class ModelMetadata(BaseModel):
    """Optional platform metadata about a referenced model; every field may be
    unknown (``None``)."""

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    type: Literal["model_metadata_v1"] = "model_metadata_v1"
    model_type: Optional[str] = None
    model_variant: Optional[str] = None
    task_type: Optional[str] = None

    def has_known_fields(self) -> bool:
        """True when at least one substantive field is known."""
        return any(
            value is not None
            for value in (self.model_type, self.model_variant, self.task_type)
        )


ModelMetadataStatus = Literal["available", "disabled", "unavailable"]


def ensure_model_metadata_status_consistent(
    status: str, metadata: Optional[ModelMetadata]
) -> None:
    """Shared rule: ``available`` <=> metadata present with >= 1 non-null
    field; ``disabled`` / ``unavailable`` => metadata is ``None``."""
    if status == "available":
        if metadata is None or not metadata.has_known_fields():
            raise ValueError(
                "metadata_status `available` requires metadata with at least one "
                "known field."
            )
        return None
    if metadata is not None:
        raise ValueError(
            f"metadata_status `{status}` requires metadata to be None; "
            f"use `available` when metadata is known."
        )
    return None


class ModelMetadataLookup(BaseModel):
    """Result of a host-side, metadata-only model lookup."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["model_metadata_lookup_v1"] = "model_metadata_lookup_v1"
    status: ModelMetadataStatus
    metadata: Optional[ModelMetadata] = None

    @model_validator(mode="after")
    def _enforce_status_metadata_consistency(self) -> "ModelMetadataLookup":
        ensure_model_metadata_status_consistent(
            status=self.status, metadata=self.metadata
        )
        return self


@runtime_checkable
class ModelMetadataProvider(Protocol):
    """Host-implemented, optional enrichment hook.

    Called once per unique ``(provider, model_id)`` with literal identifiers
    only - never with ``$``-prefixed selector values. Must not load models,
    download weights or register anything; it only looks metadata up.
    Exceptions are caught by the builder and yield an ``unavailable`` lookup.
    """

    def resolve_model_metadata(
        self, provider: str, model_id: str
    ) -> ModelMetadataLookup: ...
