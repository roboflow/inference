from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Type, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.json_schema import SkipJsonSchema
from roboflow_workflows.errors import BlockInterfaceError
from roboflow_workflows.execution_engine.entities.base import OutputDefinition
from roboflow_workflows.execution_engine.entities.workload import (
    DeclarationDomain,
    Discovery,
    DiscoveryProblem,
)
from roboflow_workflows.execution_engine.entities.workload import (  # noqa: F401 - compatibility re-export
    RestrictionCondition as RestrictionCondition,
)
from roboflow_workflows.execution_engine.entities.workload import (  # noqa: F401 - compatibility re-export
    RestrictionMetadata as RestrictionMetadata,
)
from roboflow_workflows.execution_engine.entities.workload import (
    Runtime,
    RuntimeInputMode,
    RuntimeRestriction,
    Severity,
    StepExecutionMode,
    WorkOperation,
    declaration_failed_problem,
    declaration_unavailable_problem,
    environment_filtered_declaration_problem,
    incomplete_discovery,
    normalize_declaration,
    restriction_metadata_of,
    unknown_configuration_problem,
)
from roboflow_workflows.execution_engine.introspection.restriction_environment import (
    ConfigurationMatch,
    evaluate_configuration_condition,
)
from roboflow_workflows.execution_engine.introspection.utils import get_full_type_name
from roboflow_workflows.execution_engine.v1.entities import FlowControl

BatchElementOutputs = Dict[str, Any]
BatchElementResult = Union[BatchElementOutputs, FlowControl]
BlockResult = Union[
    BatchElementResult, List[BatchElementResult], List[List[BatchElementResult]]
]


@dataclass(frozen=True)
class AirGappedAvailability:
    """Declares whether a block can operate without internet access.

    Blocks that require cloud APIs (e.g. OpenAI, Anthropic) return
    ``AirGappedAvailability(available=False, reason="requires_internet")``.
    Blocks that work fully offline return the default (available=True).
    """

    available: bool = True
    reason: Optional[str] = None


# ``RuntimeRestriction`` is defined in
# ``roboflow_workflows.execution_engine.entities.workload`` and re-exported
# above - the class object is the same one, so importing it from here is
# unchanged.


# ----------------------------------------------------------------------------
# Common block-restriction presets.
#
# Many blocks share the same failure mode (e.g. all stateful video blocks
# degrade the same way on stateless HTTP runtimes). Reusing these presets
# keeps the per-block overrides tight and the wording consistent across the
# codebase.
# ----------------------------------------------------------------------------


STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION = RuntimeRestriction(
    code="stateful_video_state_resets_on_stateless_http",
    severity=Severity.SOFT,
    note=(
        "Block keeps per-video state in process memory (keyed by "
        "video_metadata.video_identifier). With remote step execution on "
        "stateless or multi-replica HTTP runtimes, successive requests may "
        "be served by different worker processes, so the state resets "
        "between calls and the output is meaningless for tracking / "
        "counting / aggregation. Use local step execution in an "
        "InferencePipeline for stable cross-frame results."
    ),
    applies_to_runtimes=[Runtime.HOSTED_SERVERLESS, Runtime.DEDICATED_DEPLOYMENT],
    applies_to_step_execution_modes=[StepExecutionMode.REMOTE],
    applies_to_input_modes=[RuntimeInputMode.VIDEO],
)


COOLDOWN_HTTP_SOFT_RESTRICTION = RuntimeRestriction(
    code="cooldown_timer_resets_on_stateless_http",
    severity=Severity.SOFT,
    note=(
        "Cooldown / rate-limit timer is stored in process memory. With "
        "remote step execution on stateless or multi-replica HTTP runtimes "
        "each request gets a fresh worker, so cooldown does not throttle. "
        "Cooldown only behaves as documented with local step execution inside "
        "an InferencePipeline."
    ),
    applies_to_runtimes=[Runtime.HOSTED_SERVERLESS, Runtime.DEDICATED_DEPLOYMENT],
    applies_to_step_execution_modes=[StepExecutionMode.REMOTE],
)


STILL_IMAGE_INPUT_SOFT_RESTRICTION = RuntimeRestriction(
    code="temporal_block_no_benefit_on_still_image",
    severity=Severity.SOFT,
    note=(
        "Block depends on temporal context from video or repeated-frame "
        "workflows. With a still image/photo, there is no meaningful history "
        "to track, compare, aggregate, or visualize, so the block provides "
        "little or no benefit."
    ),
    applies_to_input_modes=[RuntimeInputMode.IMAGE],
)


# Portable (wire-shaped) twins of the three presets above. They are DERIVED,
# never separately authored, so the code and the axes cannot drift from the
# restriction they describe. Kept as compatibility exports for callers that
# already consume the `RestrictionMetadata` form; new code should declare the
# `RuntimeRestriction` preset and let the projection do this.
STATEFUL_VIDEO_HTTP_SOFT_PORTABLE_RESTRICTION = restriction_metadata_of(
    STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION
)


COOLDOWN_HTTP_SOFT_PORTABLE_RESTRICTION = restriction_metadata_of(
    COOLDOWN_HTTP_SOFT_RESTRICTION
)


STILL_IMAGE_INPUT_SOFT_PORTABLE_RESTRICTION = restriction_metadata_of(
    STILL_IMAGE_INPUT_SOFT_RESTRICTION
)


@dataclass(frozen=True)
class BlockAirGappedInfo:
    """Full air-gapped status for a block, as returned by the describe endpoint."""

    available: bool = True
    reason: Optional[str] = None
    model_id: Optional[str] = None
    compatible_task_types: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"available": self.available}
        if self.reason is not None:
            result["reason"] = self.reason
        if self.model_id is not None:
            result["model_id"] = self.model_id
        if self.compatible_task_types is not None:
            result["compatible_task_types"] = self.compatible_task_types
        return result


class DependentResourceType(str, Enum):
    ROBOFLOW_PLATFORM_MODEL = "roboflow_platform_model"
    ROBOFLOW_PLATFORM_PROJECT = "roboflow_platform_project"
    THIRD_PARTY_MODEL = "third_party_model"


class ModelRequiredAction(str, Enum):
    """What the declaring block needs from the model.

    ACCESS: the block only requires the model entity to be reachable on the
    platform (e.g. attaching monitoring metadata to it) — nothing executes.
    EXECUTION: the block executes the model — weights are pulled locally or
    inference is requested from a service.
    """

    ACCESS = "access"
    EXECUTION = "execution"


class ModelExecutionLocation(str, Enum):
    """Where a model declared with ``ModelRequiredAction.EXECUTION`` runs.

    LOCAL: always in-process.
    REMOTE: always on a remote service.
    ENVIRONMENT_DEFINED: decided at runtime by the step-execution-mode
    environment configuration (``WORKFLOWS_STEP_EXECUTION_MODE``) — not
    determinable at compile time.
    """

    LOCAL = "local"
    REMOTE = "remote"
    ENVIRONMENT_DEFINED = "environment_defined"


class RoboflowPlatformModelMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, protected_namespaces=())

    type: Literal["roboflow_platform_model"] = "roboflow_platform_model"
    model_id: str
    required_action: ModelRequiredAction = ModelRequiredAction.EXECUTION
    execution_location: Optional[ModelExecutionLocation] = None
    # In-process aid for callers resolving `$inputs`-fed declarations: a
    # closure turning the substituted input value into the final model id
    # (e.g. "ViT-B-16" -> "clip/ViT-B-16"), with everything it needs latched
    # inside. Returning None declares the value statically unresolvable (the
    # final id depends on more than this one value) — callers must skip and
    # let execution resolve it; raising means the value is invalid.
    # Deliberately excluded from serialization, JSON schema and equality —
    # it is not part of the envelope.
    model_id_resolver: SkipJsonSchema[Optional[Callable[[str], Optional[str]]]] = Field(
        default=None, exclude=True, repr=False
    )
    # Extra kwargs the model manager registration requires for this model —
    # model id and api key are injected by the Execution Engine, anything
    # block-specific (e.g. `endpoint_type` for core models) is declared here.
    # In-process aid — excluded from serialization, JSON schema and equality.
    model_registration_kwargs: SkipJsonSchema[Optional[Dict[str, Any]]] = Field(
        default=None, exclude=True, repr=False
    )
    # In-process loader-policy aid — excluded from serialization, JSON schema
    # and equality, like the two aids above.
    preloadable: SkipJsonSchema[bool] = Field(
        default=True,
        exclude=True,
        repr=False,
        description="Whether the generic Execution Engine model-manager "
        "preloader may register this model before the workflow runs. False for "
        "blocks that load and own their model in-process (e.g. streaming video "
        "trackers). It does not say whether the block loads weights or runs "
        "remotely.",
    )

    @model_validator(mode="before")
    @classmethod
    def _default_execution_location(cls, values: Any) -> Any:
        # Defaulted here instead of on the field, so serialized ACCESS entries
        # (which omit the field) deserialize back without gaining a location.
        if isinstance(values, dict) and "execution_location" not in values:
            action = values.get("required_action", ModelRequiredAction.EXECUTION)
            if action not in (
                ModelRequiredAction.ACCESS,
                ModelRequiredAction.ACCESS.value,
            ):
                values = {
                    **values,
                    "execution_location": ModelExecutionLocation.ENVIRONMENT_DEFINED,
                }
        return values

    @model_validator(mode="after")
    def _enforce_action_location_consistency(self) -> "RoboflowPlatformModelMetadata":
        if (
            self.required_action is ModelRequiredAction.EXECUTION
            and self.execution_location is None
        ):
            raise BlockInterfaceError(
                public_message="RoboflowPlatformModelMetadata with "
                "required_action=EXECUTION must define execution_location.",
                context="declaring_block_dependent_resources",
            )
        if (
            self.required_action is ModelRequiredAction.ACCESS
            and self.execution_location is not None
        ):
            raise BlockInterfaceError(
                public_message="RoboflowPlatformModelMetadata with "
                "required_action=ACCESS must not define execution_location.",
                context="declaring_block_dependent_resources",
            )
        return self

    def requires_runtime_resolution(self) -> bool:
        return is_workflow_selector(self.model_id)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, RoboflowPlatformModelMetadata):
            return NotImplemented
        return (self.model_id, self.required_action, self.execution_location) == (
            other.model_id,
            other.required_action,
            other.execution_location,
        )

    def __hash__(self) -> int:
        return hash((self.model_id, self.required_action, self.execution_location))


class RoboflowPlatformProjectMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["roboflow_platform_project"] = "roboflow_platform_project"
    project_url: str

    def requires_runtime_resolution(self) -> bool:
        return is_workflow_selector(self.project_url)


class ThirdPartyModelMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, protected_namespaces=())

    type: Literal["third_party_model"] = "third_party_model"
    provider: str
    model_id: str
    # See RoboflowPlatformModelMetadata.model_id_resolver — same contract.
    model_id_resolver: SkipJsonSchema[Optional[Callable[[str], Optional[str]]]] = Field(
        default=None, exclude=True, repr=False
    )

    def requires_runtime_resolution(self) -> bool:
        return is_workflow_selector(self.provider) or is_workflow_selector(
            self.model_id
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ThirdPartyModelMetadata):
            return NotImplemented
        return (self.provider, self.model_id) == (other.provider, other.model_id)

    def __hash__(self) -> int:
        return hash((self.provider, self.model_id))


REGISTERED_RESOURCE_METADATA_TYPES: Dict[DependentResourceType, Type[BaseModel]] = {
    DependentResourceType.ROBOFLOW_PLATFORM_MODEL: RoboflowPlatformModelMetadata,
    DependentResourceType.ROBOFLOW_PLATFORM_PROJECT: RoboflowPlatformProjectMetadata,
    DependentResourceType.THIRD_PARTY_MODEL: ThirdPartyModelMetadata,
}


class DependentResource(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["dependent_resource"] = "dependent_resource"
    resource_type: DependentResourceType
    metadata: Union[
        RoboflowPlatformModelMetadata,
        RoboflowPlatformProjectMetadata,
        ThirdPartyModelMetadata,
    ]

    @model_validator(mode="before")
    @classmethod
    def _resolve_metadata_by_resource_type(cls, values: Any) -> Any:
        # On deserialization, pick the metadata entity registered for the
        # declared resource_type instead of relying on union resolution.
        if not isinstance(values, dict):
            return values
        metadata = values.get("metadata")
        if not isinstance(metadata, dict):
            return values
        try:
            resource_type = DependentResourceType(values.get("resource_type"))
        except ValueError:
            return values
        expected_type = REGISTERED_RESOURCE_METADATA_TYPES.get(resource_type)
        if expected_type is None:
            return values
        # Old envelopes carry no `type`; an explicit one must agree with
        # `resource_type`, otherwise the declaration contradicts itself.
        declared_type = metadata.get("type")
        expected_discriminator = expected_type.model_fields["type"].default
        if declared_type is not None and declared_type != expected_discriminator:
            raise BlockInterfaceError(
                public_message=f"DependentResource of type {resource_type.value} "
                f"requires metadata of type `{expected_discriminator}`, got "
                f"`{declared_type}`.",
                context="declaring_block_dependent_resources",
            )
        return {**values, "metadata": expected_type.model_validate(metadata)}

    @model_validator(mode="after")
    def _enforce_metadata_matches_resource_type(self) -> "DependentResource":
        expected_type = REGISTERED_RESOURCE_METADATA_TYPES.get(self.resource_type)
        if expected_type is None or not isinstance(self.metadata, expected_type):
            raise BlockInterfaceError(
                public_message=f"DependentResource of type {self.resource_type} requires "
                f"metadata of type "
                f"{expected_type.__name__ if expected_type else '<unregistered>'}, "
                f"got {type(self.metadata).__name__}.",
                context="declaring_block_dependent_resources",
            )
        return self

    def to_dict(self) -> Dict[str, Any]:
        # Legacy envelope shape: no `type` discriminators at any level. Use
        # `model_dump(mode="json")` for the discriminated (introspection) form.
        return self.model_dump(
            mode="json",
            exclude_none=True,
            exclude={"type": True, "metadata": {"type": True}},
        )


def roboflow_platform_model(
    model_id: str,
    required_action: ModelRequiredAction = ModelRequiredAction.EXECUTION,
    execution_location: Optional[ModelExecutionLocation] = None,
    model_id_resolver: Optional[Callable[[str], Optional[str]]] = None,
    model_registration_kwargs: Optional[Dict[str, Any]] = None,
    *,
    preloadable: bool = True,
) -> DependentResource:
    """Declare a Roboflow platform model the step depends on.

    Args:
        model_id: Literal model id, or the workflow selector that feeds it.
        required_action: What the step needs from the model.
        execution_location: Where an EXECUTION model runs; defaults to
            ENVIRONMENT_DEFINED for EXECUTION when omitted.
        model_id_resolver: In-process aid turning a substituted `$inputs`
            value into the final model id.
        model_registration_kwargs: Extra model-manager registration kwargs.
        preloadable: In-process aid; False keeps the generic Execution Engine
            model-manager preloader away from this model. Never serialized.

    Returns:
        The dependent resource declaration.
    """
    metadata_kwargs: Dict[str, Any] = {
        "model_id": model_id,
        "required_action": required_action,
        "model_id_resolver": model_id_resolver,
        "model_registration_kwargs": model_registration_kwargs,
        "preloadable": preloadable,
    }
    if execution_location is not None:
        metadata_kwargs["execution_location"] = execution_location
    return DependentResource(
        resource_type=DependentResourceType.ROBOFLOW_PLATFORM_MODEL,
        metadata=RoboflowPlatformModelMetadata(**metadata_kwargs),
    )


def roboflow_platform_project(project_url: str) -> DependentResource:
    return DependentResource(
        resource_type=DependentResourceType.ROBOFLOW_PLATFORM_PROJECT,
        metadata=RoboflowPlatformProjectMetadata(project_url=project_url),
    )


def third_party_model(
    provider: str,
    model_id: str,
    model_id_resolver: Optional[Callable[[str], Optional[str]]] = None,
) -> DependentResource:
    return DependentResource(
        resource_type=DependentResourceType.THIRD_PARTY_MODEL,
        metadata=ThirdPartyModelMetadata(
            provider=provider,
            model_id=model_id,
            model_id_resolver=model_id_resolver,
        ),
    )


def is_workflow_selector(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("$")


class WorkflowBlockManifest(BaseModel, ABC):
    model_config = ConfigDict(
        validate_assignment=True,
    )

    type: str
    name: str = Field(
        title="Step Name", description="Enter a unique identifier for this step."
    )

    @classmethod
    @abstractmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        raise BlockInterfaceError(
            public_message=f"Class method `describe_outputs()` must be implemented "
            f"for {get_full_type_name(selected_type=cls)} to be valid "
            f"`WorkflowBlockManifest`.",
            context="getting_block_outputs",
        )

    def get_actual_outputs(self) -> List[OutputDefinition]:
        return self.describe_outputs()

    @classmethod
    def get_air_gapped_availability(cls) -> AirGappedAvailability:
        """Declare whether this block can operate without internet access.

        Override in subclasses that require cloud APIs to return
        ``AirGappedAvailability(available=False, reason="requires_internet")``.

        The default indicates the block works offline.
        """
        return AirGappedAvailability(available=True)

    @classmethod
    def get_restrictions(cls) -> List[RuntimeRestriction]:
        """Caveats for this block.

        Return restrictions describing where the block degrades
        (``Severity.SOFT``) or fails outright (``Severity.HARD``). Each
        restriction can scope itself to runtimes, step execution modes, and/or
        input modes.
        """
        return []

    @classmethod
    def get_supported_model_variants(cls) -> Optional[List[str]]:
        """Return model IDs whose cached weights enable this block to run offline.

        For foundation-model blocks, return the list of model variant IDs
        (e.g. ``["sam2/hiera_large", "sam2/hiera_small"]``).  The block is
        considered available if **any** variant has cached artifacts.

        Return ``None`` (the default) for blocks that do not depend on
        locally-cached model weights (pure logic blocks, cloud API blocks, etc.).
        """
        return None

    @classmethod
    def get_compatible_task_types(cls) -> Optional[List[str]]:
        """Return task types this block can process (e.g. ``["object-detection"]``).

        Used by the air-gapped builder to match user-trained models to
        compatible workflow blocks.  Return ``None`` (the default) for blocks
        that are not parameterised by a Roboflow model.
        """
        return None

    def discover_dependent_resources(self) -> Optional[List[DependentResource]]:
        """Declare external resources this step will pull at run time.

        Returns ``None`` (the default) when the block does not declare its
        dependencies — callers must treat that as *unknown*, which is distinct
        from ``[]`` (the block declares it needs no external resources).

        Field values that are workflow selectors (``$inputs.<name>`` /
        ``$steps.<name>.<property>``) are returned verbatim inside the
        metadata — the block does not resolve them. Callers may substitute
        ``$inputs`` references once runtime parameters are known; ``$steps``
        references are not statically resolvable.
        """
        return None

    def discover_work_operations(
        self,
    ) -> Optional[Union[List[WorkOperation], Discovery[WorkOperation]]]:
        """Declare the meaningful work this step performs.

        Return-value convention:

        * ``None`` (the default) - unknown: the block does not declare its
          operations (e.g. an unannotated plugin). Callers must treat it as an
          incomplete discovery, which is distinct from
        * ``[]`` / a plain ``list`` - a complete declaration; an empty list
          truthfully says the block performs no meaningful operation (noop).
        * a ``Discovery[WorkOperation]`` - explicit completeness with reasons,
          e.g. custom Python declares
          ``incomplete_discovery([WorkOperation.CUSTOM_PYTHON],
          [custom_python_internals_unknown_problem(
          node_id=f"$steps.{self.name}", declaration="operations")])``.

        Reasons are ``DiscoveryProblem`` objects (a closed ``code``, a
        human-readable ``description`` and open ``details``); build them with
        the factories in
        ``roboflow_workflows.execution_engine.entities.workload`` rather than
        writing free text, and never put an exception message, a traceback or
        a secret into them.

        Callers normalise every return value through
        ``roboflow_workflows.execution_engine.entities.workload.normalize_declaration``.
        Annotate meaningful block behaviour (what the step does), not every
        helper invoked internally; literal manifest settings may select the
        operations (e.g. resize vs rotate). Environment-dependent dispatch
        must not add ``EXTERNAL_REQUEST`` merely because this server is
        configured for remote execution.
        """
        return None

    def get_actual_restrictions(
        self, *, ignore_environment_restrictions: bool = False
    ) -> Discovery[RuntimeRestriction]:
        """This step's restrictions, as a ``Discovery[RuntimeRestriction]``.

        The single hook a block overrides to declare restrictions. It never
        executes the block, never resolves a selector and never invents a
        value. An override declares what applies CONDITIONALLY on the target
        deployment - runtimes, step execution modes, input modes and
        ``applies_to_configuration`` - refined by the instance's own literal
        settings; a field fed by a selector returns an ``incomplete_discovery``
        with an ``unresolved_selector_problem`` instead of a guess. Build the
        result with ``actual_restrictions_of()``, or return an empty complete
        discovery when the block declares nothing.

        ``ignore_environment_restrictions``:

        * ``False`` (default) - the HOST view: the ``applies_to_configuration``
          predicates are evaluated against the configuration installed in THIS
          process and entries that definitively do not apply here are removed.
          A predicate this process cannot evaluate keeps its entry and makes the
          result incomplete, unless another predicate of the same restriction
          already rules the ANDed condition out.
        * ``True`` - the PORTABLE view: no host evaluation, every declaration
          returned with its conditions intact.

        The runtime, input-mode and step-execution-mode axes are never
        evaluated in either mode.

        The default body serves a block that never adopted this API: it calls
        the legacy ``get_restrictions()`` classmethod - a block's own override
        through ordinary dispatch, otherwise the inherited ``[]`` - and wraps
        whatever comes back as INCOMPLETE. A legacy getter may already have
        filtered its list against the flags of the host that answered, and the
        inherited default declares nothing at all; neither a short list nor an
        empty one proves absence. ``ignore_environment_restrictions=True``
        cannot undo filtering that happened inside the classmethod, so the
        fallback stays incomplete in both views.
        """
        node_id = f"$steps.{getattr(self, 'name', '<undefined>')}"
        return actual_restrictions_of(
            declared=incomplete_discovery(
                items=list(self.get_restrictions()),
                reasons=[
                    environment_filtered_declaration_problem(
                        node_id=node_id, declaration=RESTRICTIONS_DECLARATION
                    )
                ],
            ),
            node_id=node_id,
            ignore_environment_restrictions=ignore_environment_restrictions,
        )

    @classmethod
    def get_input_dimensionality_offsets(cls) -> Dict[str, int]:
        return {}

    @classmethod
    def get_dimensionality_reference_property(cls) -> Optional[str]:
        return None

    @classmethod
    def get_output_dimensionality_offset(
        cls,
    ) -> int:
        return 0

    @classmethod
    def accepts_batch_input(cls) -> bool:
        return (
            len(cls.get_parameters_accepting_batches()) > 0
            or len(cls.get_parameters_accepting_batches_and_scalars()) > 0
        )

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return []

    @classmethod
    def get_parameters_accepting_batches_and_scalars(cls) -> List[str]:
        return []

    @classmethod
    def get_parameters_enforcing_auto_batch_casting(cls) -> List[str]:
        return []

    @classmethod
    def accepts_empty_values(cls) -> bool:
        return False

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return None


RESTRICTIONS_DECLARATION: DeclarationDomain = "restrictions"


def actual_restrictions_of(
    declared: Union[None, Iterable[RuntimeRestriction], Discovery[RuntimeRestriction]],
    *,
    node_id: str,
    ignore_environment_restrictions: bool,
) -> Discovery[RuntimeRestriction]:
    """Normalise restriction DATA a manifest has already computed.

    ``declared``: ``None`` is unknown, a list is a complete declaration (``[]``
    = declares none), a ``Discovery`` states its own completeness.

    ``ignore_environment_restrictions``: ``True`` returns the declaration
    untouched (the portable view), ``False`` evaluates its
    ``applies_to_configuration`` predicates against this process.

    ``RuntimeRestriction`` is a permissive dataclass, so each item is projected
    onto the portable DTO here. A declaration that projection rejects - a blank
    or non-identifier ``code``, an empty axis list, a blank configuration key -
    becomes a sanitised ``declaration_failed`` problem instead of a complete
    declaration the wire cannot carry.
    """
    try:
        normalised = normalize_declaration(
            declared,
            declaration_unavailable_problem(
                node_id=node_id, declaration=RESTRICTIONS_DECLARATION
            ),
        )
        # Re-typed deliberately: an item that is not a `RuntimeRestriction`
        # fails validation here and is never passed on.
        discovery = Discovery[RuntimeRestriction](
            items=list(normalised.items),
            complete=normalised.complete,
            unknown_reasons=list(normalised.unknown_reasons),
        )
        for restriction in discovery.items:
            restriction_metadata_of(restriction)
    except Exception:
        # The exception is never reported: a block may have put anything in it.
        return Discovery[RuntimeRestriction](
            items=[],
            complete=False,
            unknown_reasons=[
                declaration_failed_problem(
                    node_id=node_id, declaration=RESTRICTIONS_DECLARATION
                )
            ],
        )
    if ignore_environment_restrictions:
        return discovery
    return _evaluate_against_this_host(declared=discovery, node_id=node_id)


def _evaluate_against_this_host(
    declared: Discovery[RuntimeRestriction], node_id: str
) -> Discovery[RuntimeRestriction]:
    """Drop what definitively does not apply here; keep what is uncertain.

    Only ``applies_to_configuration`` is evaluated. The runtime, input-mode and
    step-execution-mode axes are left alone - this process is not the target
    and does not know them. Nothing is mutated and no condition is stripped:
    the surviving entries are the declarations themselves.
    """
    kept: List[RuntimeRestriction] = []
    unknown_keys: List[str] = []
    for restriction in declared.items:
        verdict, keys = evaluate_configuration_condition(restriction=restriction)
        if verdict is ConfigurationMatch.INACTIVE:
            continue
        kept.append(restriction)
        unknown_keys.extend(keys)
    reasons: List[DiscoveryProblem] = list(declared.unknown_reasons)
    if unknown_keys:
        reasons.append(
            unknown_configuration_problem(
                node_id=node_id,
                declaration=RESTRICTIONS_DECLARATION,
                configuration_keys=unknown_keys,
            )
        )
    return Discovery[RuntimeRestriction](
        items=kept,
        complete=declared.complete and not unknown_keys,
        unknown_reasons=reasons,
    )


class WorkflowBlock(ABC):

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return []

    @classmethod
    @abstractmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        raise BlockInterfaceError(
            public_message="Class method `get_manifest()` must be implemented for any entity "
            "deriving from WorkflowBlockManifest.",
            context="getting_block_manifest",
        )

    @abstractmethod
    def run(
        self,
        *args,
        **kwargs,
    ) -> BlockResult:
        pass
