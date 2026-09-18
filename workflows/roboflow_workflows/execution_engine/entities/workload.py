"""Dependency-light workload declaration primitives.

This module is the lowest layer of the workload-introspection contract. It is
imported by ``roboflow_workflows.prototypes.block`` (which re-exports the enums
for backwards compatibility) and by the response schemas in
``roboflow_workflows.execution_engine.introspection.workload_entities``.

Import discipline (an import-cycle trap otherwise): only the standard library,
``pydantic`` and ``typing_extensions`` may be imported here. Never import
``roboflow_workflows.prototypes.*``, ``core_steps.*``, ``execution_engine.v1.*``
or ``execution_engine.introspection.*`` from this module.
"""

import json
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

    type: Literal["restriction_condition"] = "restriction_condition"
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

    type: Literal["restriction"] = "restriction"
    code: str = Field(min_length=1, pattern=RESTRICTION_CODE_PATTERN)
    severity: Severity
    when: RestrictionCondition = RestrictionCondition()

    def __hash__(self) -> int:
        return hash((self.code, self.severity, self.when))


def _discovery_item_sort_key(item: Any) -> Tuple[str, ...]:
    if isinstance(item, Enum):
        return (str(item.value),)
    if isinstance(item, RestrictionMetadata):
        return (item.code, item.severity.value, _canonical_json(item.when))
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


class Discovery(BaseModel, Generic[T]):
    """A set of discovered facts together with an honest completeness claim.

    * ``complete=True`` with no items is a known absence (the step truthfully
      does nothing of this kind). Complete results carry no ``unknown_reasons``.
    * ``complete=False`` may still list the items that ARE known, and MUST
      carry at least one reason. Reasons are stable codes with context in the
      form ``<reason_code>:<context>`` (e.g.
      ``step_declaration_missing:$steps.crop``) - never exception traces or
      secrets.

    Items are de-duplicated and deterministically sorted (enum members by
    value, ``RestrictionMetadata`` by ``(code, severity, condition JSON)``,
    other models by their JSON dump with sorted keys); reasons are unique and
    sorted. Instances are frozen.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["discovery"] = "discovery"
    items: List[T]
    complete: bool
    unknown_reasons: List[str]

    @field_validator("items", mode="after")
    @classmethod
    def _canonicalise_items(cls, value: List[Any]) -> List[Any]:
        return _deduplicate_and_sort_items(items=value)

    @field_validator("unknown_reasons", mode="after")
    @classmethod
    def _canonicalise_reasons(cls, value: List[str]) -> List[str]:
        for reason in value:
            if not reason.strip():
                raise ValueError("`unknown_reasons` entries must be non-empty codes.")
        return sorted(set(value))

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


def incomplete_discovery(items: Iterable[T], reasons: Iterable[str]) -> Discovery[T]:
    """Build an incomplete discovery listing the known ``items`` and why the
    rest is unknown (``reasons`` must be non-empty)."""
    return Discovery(items=list(items), complete=False, unknown_reasons=list(reasons))


def normalize_declaration(
    value: Union[None, Iterable[T], Discovery[T]], unknown_reason: str
) -> Discovery[T]:
    """Map a manifest hook return value onto the ``Discovery`` convention.

    * ``None`` -> unknown: ``Discovery(items=[], complete=False,
      unknown_reasons=[unknown_reason])``.
    * a plain list (any iterable) -> complete declaration (may be empty).
    * a ``Discovery`` -> re-validated copy of itself (item objects preserved,
      so in-process aids such as excluded resolver callbacks survive).
    """
    if value is None:
        return Discovery(items=[], complete=False, unknown_reasons=[unknown_reason])
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

    type: Literal["model_metadata"] = "model_metadata"
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

    type: Literal["model_metadata_lookup"] = "model_metadata_lookup"
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
