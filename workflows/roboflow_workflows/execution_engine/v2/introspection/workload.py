"""Workload discovery: the resources, operations and restrictions of a plan.

Reads the class hooks every block declares (``discover_dependent_resources``,
``discover_work_operations``, ``discover_restrictions``) through
``BlockSpec.describe_workload`` and reports them with the neutral ``Discovery``
entities. No block, provider or submitted dynamic code runs.

Nothing is claimed that is not known:

==================================  ==========================================
Hook answer                         Reported
==================================  ==========================================
``None``                            incomplete, ``declaration_unavailable``
raises or returns a bad shape       incomplete, ``declaration_failed`` (no
                                    exception text)
a list                              complete
a ``Discovery``                     as declared
resource identifier is a selector   item kept, ``unresolved_selector``
resource identifier is blank        item kept, ``invalid_resource_identifier``
==================================  ==========================================

Hooks receive the step's validated parameters with compile-time constants
substituted: a nested workflow input bound to a literal, or defaulted, reaches
the hook as that literal. Root workflow inputs stay selectors even when they
have defaults, because the caller may override them. Hooks do not know their
step, so problems they report get the step's ``node_id`` added here.

Each step answers with the hooks of the implementation selected for the
plan's target; a hook the implementation does not override falls back to the
logical block's. The report names that implementation, the step's execution
mode and its constructor resources. Selection never constructs anything.

Declared sources have no workload hooks; the report lists each source's
constructor resources (what a session must provide before ``start``) read
from its class signature, without constructing anything.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel, ConfigDict, Field
from roboflow_workflows.execution_engine.entities.workload import (
    DeclarationDomain,
    Discovery,
    DiscoveryProblem,
    DiscoveryProblemCode,
    RestrictionMetadata,
    RuntimeRestriction,
    WorkOperation,
    declaration_failed_problem,
    invalid_resource_identifier_problem,
    restriction_metadata_of,
    unresolved_selector_problem,
)
from roboflow_workflows.execution_engine.v2.declaration import (
    BlockParams,
    DependentResource,
)
from roboflow_workflows.execution_engine.v2.errors import format_step_path
from roboflow_workflows.execution_engine.v2.introspection._sources import source_node
from roboflow_workflows.execution_engine.v2.plan import (
    CompiledWorkflow,
    Constant,
    PlannedStep,
)
from roboflow_workflows.execution_engine.v2.resources import ResourceSpec

_DECLARATION_PROBLEMS = (
    DiscoveryProblemCode.DECLARATION_UNAVAILABLE,
    DiscoveryProblemCode.DECLARATION_FAILED,
)


class ResourceUsage(BaseModel):
    """One literal external resource and the steps that need it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    resource_type: str = Field(
        description="Category of the resource, as declared by the blocks.",
        examples=["roboflow_platform_model", "python_module"],
    )
    identifier: str = Field(
        description="Literal resource identifier; selectors are never listed.",
        examples=["yolov8n-640"],
    )
    used_by_steps: List[str] = Field(
        description="Node ids of the steps declaring the resource, sorted.",
        examples=[["$steps.detect", "$steps.child/detect"]],
    )


@dataclass(frozen=True)
class StepWorkload:
    """Workload declarations of one step.

    Args:
        node_id: Step node id, e.g. ``$steps.child/detect``.
        block_type: Canonical block type.
        resources: ``Discovery`` of ``DependentResource``.
        operations: ``Discovery`` of ``WorkOperation``.
        restrictions: ``Discovery`` of portable ``RestrictionMetadata``.
        implementation: Name of the implementation selected for the plan's
            target; its hooks answered, falling back to the block's own.
        execution: ``run`` or ``phases``, as the step will execute.
        constructor_resources: Keyword resources of the selected
            implementation's ``__init__``, resolved by ``create_session``.
    """

    node_id: str
    block_type: str
    resources: Discovery
    operations: Discovery
    restrictions: Discovery
    implementation: str = "default"
    execution: str = "run"
    constructor_resources: Tuple[ResourceSpec, ...] = ()

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-friendly description."""
        description = {
            "block_type": self.block_type,
            "implementation": self.implementation,
            "execution": self.execution,
            "constructor_resources": [
                resource.describe() for resource in self.constructor_resources
            ],
            "resources": self.resources.model_dump(mode="json"),
            "operations": self.operations.model_dump(mode="json"),
            "restrictions": self.restrictions.model_dump(mode="json"),
        }

        return description


@dataclass(frozen=True)
class SourceWorkload:
    """Constructor resources of one declared source.

    Args:
        node_id: Source node id, e.g. ``$sources.camera``.
        source_type: Canonical source type.
        constructor_resources: Keyword resources of the source class's
            ``__init__``, resolved by ``create_session`` before any run.
    """

    node_id: str
    source_type: str
    constructor_resources: Tuple[ResourceSpec, ...]

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-friendly description."""
        description = {
            "source_type": self.source_type,
            "constructor_resources": [
                resource.describe() for resource in self.constructor_resources
            ],
        }

        return description


@dataclass(frozen=True)
class WorkloadReport:
    """Workload of a whole plan.

    Args:
        steps: Per-step declarations in execution order.
        resources: ``Discovery`` of ``ResourceUsage``: literal resources
            across steps; incomplete when any step's resources are.
        operations: Union of the steps' operations.
        restrictions: Union of the steps' restrictions.
        sources: Declared sources with their constructor resources, in
            declaration order; empty for a passive plan.
    """

    steps: Tuple[StepWorkload, ...]
    resources: Discovery
    operations: Discovery
    restrictions: Discovery
    sources: Tuple[SourceWorkload, ...] = ()

    def step(self, node_id: str) -> StepWorkload:
        """Return the workload of one step.

        Args:
            node_id: Step node id, e.g. ``$steps.child/detect``.

        Returns:
            The step's workload.

        Raises:
            KeyError: When the plan has no such step.
        """
        for step in self.steps:
            if step.node_id == node_id:
                return step

        raise KeyError(f"No step {node_id!r}; known: {[s.node_id for s in self.steps]}")

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-friendly description."""
        description = {
            "steps": {step.node_id: step.describe() for step in self.steps},
            "resources": self.resources.model_dump(mode="json"),
            "operations": self.operations.model_dump(mode="json"),
            "restrictions": self.restrictions.model_dump(mode="json"),
            "sources": {source.node_id: source.describe() for source in self.sources},
        }

        return description


def discover_workload(plan: CompiledWorkflow) -> WorkloadReport:
    """Collect the declared workload of every step of a compiled plan.

    Args:
        plan: Compiled plan.

    Returns:
        Per-step discoveries and their unions, plus the constructor resources
        of every declared source. Unknowns are reported as incomplete
        discoveries with reasons, never as absence.
    """
    steps = tuple(_step_workload(step, plan=plan) for step in plan.steps)
    report = WorkloadReport(
        steps=steps,
        resources=_resource_inventory(steps),
        operations=_union(Discovery[WorkOperation], [s.operations for s in steps]),
        restrictions=_union(
            Discovery[RestrictionMetadata], [s.restrictions for s in steps]
        ),
        sources=tuple(
            SourceWorkload(
                node_id=source_node(item.name),
                source_type=item.spec.type,
                constructor_resources=item.spec.resources,
            )
            for item in plan.sources.values()
        ),
    )

    return report


def _step_workload(step: PlannedStep, *, plan: CompiledWorkflow) -> StepWorkload:
    node_id = format_step_path(step.path)
    implementation = step.selected
    declared = step.spec.describe_workload(
        _params_with_constants(step, plan=plan),
        node_id=node_id,
        implementation=implementation,
    )

    def validated(
        discovery_type: Type[Discovery],
        discovery: Discovery,
        *,
        domain: DeclarationDomain,
        project: Optional[Callable[[Any], Any]] = None,
    ) -> Discovery:
        return _validated(
            discovery_type,
            discovery,
            domain=domain,
            project=project,
            node_id=node_id,
            block_type=step.block_type,
        )

    resources = validated(
        Discovery[DependentResource], declared.dependencies, domain="resources"
    )
    workload = StepWorkload(
        node_id=node_id,
        block_type=step.block_type,
        resources=_with_identity_problems(resources, node_id=node_id),
        operations=validated(
            Discovery[WorkOperation], declared.operations, domain="operations"
        ),
        restrictions=validated(
            Discovery[RestrictionMetadata],
            declared.restrictions,
            domain="restrictions",
            project=_portable_restriction,
        ),
        implementation=implementation.name,
        execution=step.execution,
        constructor_resources=implementation.resources,
    )

    return workload


def _params_with_constants(step: PlannedStep, *, plan: CompiledWorkflow) -> BlockParams:
    """Parameters with every compile-time constant substituted.

    A constant may be bound directly or reach the step through nested
    workflow input and output ports. It is passed as written; no kind decoder runs.
    """
    updates: Dict[str, Any] = {}
    for binding in step.bindings:
        origin = plan.origin(binding.source)
        if not isinstance(origin, Constant):
            continue
        if not binding.position:
            updates[binding.field] = origin.value
            continue
        container = updates.get(binding.field, getattr(step.params, binding.field))
        container = list(container) if isinstance(container, list) else dict(container)
        container[binding.position[0]] = origin.value
        updates[binding.field] = container

    if not updates:
        return step.params

    params = step.params.model_copy(update=updates)

    return params


def _validated(
    discovery_type: Type[Discovery],
    discovery: Discovery,
    *,
    domain: DeclarationDomain,
    project: Optional[Callable[[Any], Any]],
    node_id: str,
    block_type: str,
) -> Discovery:
    """Validate item types and add step context; a bad shape is a failure."""
    try:
        items = (
            [project(item) for item in discovery.items] if project else discovery.items
        )
        validated = discovery_type(
            items=list(items),
            complete=discovery.complete,
            unknown_reasons=[
                _with_step_context(reason, node_id=node_id, block_type=block_type)
                for reason in discovery.unknown_reasons
            ],
        )
    except Exception:
        # Like V1: a broken declaration must not hide the rest of the plan,
        # and exception text may carry anything, so it is not reported.
        validated = discovery_type(
            items=[],
            complete=False,
            unknown_reasons=[
                declaration_failed_problem(
                    node_id=node_id, declaration=domain, block_type=block_type
                )
            ],
        )

    return validated


def _portable_restriction(item: Any) -> RestrictionMetadata:
    if isinstance(item, RestrictionMetadata):
        return item
    if not isinstance(item, RuntimeRestriction):
        raise TypeError(f"not a restriction: {type(item).__name__}")

    portable = restriction_metadata_of(item)

    return portable


def _with_step_context(
    problem: DiscoveryProblem, *, node_id: str, block_type: str
) -> DiscoveryProblem:
    details = dict(problem.details)
    details.setdefault("node_id", node_id)
    if problem.code in _DECLARATION_PROBLEMS:
        details.setdefault("block_type", block_type)
    if details == problem.details:
        return problem

    located = problem.model_copy(update={"details": details})

    return located


def _with_identity_problems(resources: Discovery, *, node_id: str) -> Discovery:
    problems = []
    for resource in resources.items:
        if resource.identifier.startswith("$"):
            problems.append(
                unresolved_selector_problem(
                    node_id=node_id,
                    declaration="resources",
                    field="identifier",
                    selector=resource.identifier,
                    resource_type=resource.resource_type,
                )
            )
        elif not resource.identifier.strip():
            problems.append(
                invalid_resource_identifier_problem(
                    node_id=node_id,
                    declaration="resources",
                    field="identifier",
                    resource_type=resource.resource_type,
                )
            )
    if not problems:
        return resources

    incomplete = Discovery[DependentResource](
        items=list(resources.items),
        complete=False,
        unknown_reasons=list(resources.unknown_reasons) + problems,
    )

    return incomplete


def _resource_inventory(steps: Tuple[StepWorkload, ...]) -> Discovery:
    used_by: Dict[Tuple[str, str], List[str]] = {}
    reasons: List[DiscoveryProblem] = []
    for step in steps:
        reasons.extend(step.resources.unknown_reasons)
        for resource in step.resources.items:
            identifier = resource.identifier
            if identifier.startswith("$") or not identifier.strip():
                continue
            used_by.setdefault((resource.resource_type, identifier), []).append(
                step.node_id
            )

    inventory = Discovery[ResourceUsage](
        items=[
            ResourceUsage(
                resource_type=resource_type,
                identifier=identifier,
                used_by_steps=sorted(set(node_ids)),
            )
            for (resource_type, identifier), node_ids in used_by.items()
        ],
        complete=not reasons,
        unknown_reasons=reasons,
    )

    return inventory


def _union(discovery_type: Type[Discovery], discoveries: List[Discovery]) -> Discovery:
    reasons = [
        reason for discovery in discoveries for reason in discovery.unknown_reasons
    ]
    union = discovery_type(
        items=[item for discovery in discoveries for item in discovery.items],
        complete=not reasons,
        unknown_reasons=reasons,
    )

    return union
