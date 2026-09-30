"""Public response schemas of workload introspection.

These entities describe compile-time workload facts of a workflow: the
connectivity graph, per-step facts and summaries. They are an introspection
extension, not an estimator - no scores, weights, schedules or hardware
assumptions appear here.

Each entity carries a versioned ``type`` discriminator (e.g.
``workflow_introspection_v1``); there is no separate ``schema_version`` field.
``execution_engine_version`` is the version of the compiler that answered, not
a schema version.

Import discipline: this module may import ``prototypes.block`` and
``entities.workload``; it must never import ``execution_engine.v1.*`` or
``core_steps.*``. ``prototypes.block`` must never import this module
(``introspection/entities.py`` already imports prototypes - a cycle otherwise).
"""

from collections import Counter
from typing import Dict, List, Literal, Optional, Set, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from roboflow_workflows.execution_engine.entities.workload import (
    Discovery,
    ModelMetadata,
    ModelMetadataStatus,
    RestrictionMetadata,
    WorkOperation,
    ensure_model_metadata_status_consistent,
)
from roboflow_workflows.prototypes.block import DependentResource

GraphNodeKind = Literal["input", "step", "output"]
GraphEdgeKind = Literal["data", "control"]


class GraphNode(BaseModel):
    """A workflow input, step or output. ``id`` is the canonical selector
    (``$inputs.<name>``, ``$steps.<name>``, ``$outputs.<name>``). Synthetic
    compiler nodes never appear here."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["graph_node_v1"] = "graph_node_v1"
    id: str = Field(min_length=1)
    kind: GraphNodeKind


class GraphEdge(BaseModel):
    """Connectivity between two nodes. ``data`` edges carry values, ``control``
    edges express flow control between steps. A node pair may have both. The
    graph is connectivity, not a schedule."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["graph_edge_v1"] = "graph_edge_v1"
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    kind: GraphEdgeKind


class StepMetadata(BaseModel):
    """Compile-time facts about one compiled step.

    ``input_dimensionality`` is the effective compiled reference depth (control
    reference depth for input-less controlled steps), NOT executor loop depth;
    ``output_dimensionality`` is the compiled output depth.
    ``accepts_batch_input`` is the normalised manifest capability, not a
    promise of simultaneous computation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["step_metadata_v1"] = "step_metadata_v1"
    node_id: str = Field(min_length=1)
    block_type: str = Field(min_length=1)
    input_dimensionality: int = Field(ge=0)
    output_dimensionality: int = Field(ge=0)
    accepts_batch_input: bool
    resources: Discovery[DependentResource]
    restrictions: Discovery[RestrictionMetadata]
    operations: Discovery[WorkOperation]


def canonicalise_dimensionality_histogram(value: Dict[int, int]) -> Dict[int, int]:
    """Shared rule for every ``steps_by_dimensionality`` map: non-negative
    integer depths, positive counts (zero-count depths are omitted, never
    stored), keys sorted ascending."""
    for dimensionality, count in value.items():
        if dimensionality < 0:
            raise ValueError(
                "`steps_by_dimensionality` keys must be non-negative "
                f"dimensionalities, got {dimensionality}."
            )
        if count < 1:
            raise ValueError(
                "`steps_by_dimensionality` values must be positive counts "
                f"(omit zero-count dimensions), got {count} for "
                f"dimensionality {dimensionality}."
            )
    return dict(sorted(value.items()))


class ModelSummary(BaseModel):
    """One entry of the model inventory: a model reference and the steps that
    refer to it. Inventory membership is not a claim that the model executes;
    per-step resources keep ACCESS vs EXECUTION.

    ``steps_by_dimensionality`` maps input depth -> number of steps in
    ``used_by_steps`` compiled at that depth (same reference-depth semantics
    and string-keyed wire encoding as ``WorkflowSummary.steps_by_dimensionality``).
    Each referring step counts exactly once, whatever the number of resource
    declarations it makes for this model, so the counts always sum to
    ``len(used_by_steps)``. These are reference counts, not call estimates.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    type: Literal["model_summary_v1"] = "model_summary_v1"
    provider: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    used_by_steps: List[str]
    steps_by_dimensionality: Dict[int, int]
    metadata: Optional[ModelMetadata] = None
    metadata_status: ModelMetadataStatus

    @field_validator("used_by_steps", mode="after")
    @classmethod
    def _canonicalise_used_by_steps(cls, value: List[str]) -> List[str]:
        if len(value) == 0:
            raise ValueError("`used_by_steps` must name at least one step.")
        if len(set(value)) != len(value):
            raise ValueError("`used_by_steps` must not contain duplicates.")
        for step_id in value:
            if not step_id.strip():
                raise ValueError("`used_by_steps` entries must be non-empty ids.")
        return sorted(value)

    @field_validator("steps_by_dimensionality", mode="after")
    @classmethod
    def _validate_histogram(cls, value: Dict[int, int]) -> Dict[int, int]:
        return canonicalise_dimensionality_histogram(value)

    @model_validator(mode="after")
    def _enforce_cross_field_consistency(self) -> "ModelSummary":
        ensure_model_metadata_status_consistent(
            status=self.metadata_status, metadata=self.metadata
        )
        total = sum(self.steps_by_dimensionality.values())
        if total != len(self.used_by_steps):
            raise ValueError(
                "`steps_by_dimensionality` counts must sum to the number of "
                f"`used_by_steps` ({len(self.used_by_steps)}), got {total}."
            )
        return self


class WorkflowSummary(BaseModel):
    """Whole-graph summary. ``steps_by_dimensionality`` maps input depth ->
    number of steps at that depth (zero-count depths omitted; keys are strings
    on the wire). ``max_dimensionality`` considers the compiled graph's actual
    input/output depths, so a no-step input->output workflow is covered."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["workflow_summary_v1"] = "workflow_summary_v1"
    models: Discovery[ModelSummary]
    max_dimensionality: int = Field(ge=0)
    steps_by_dimensionality: Dict[int, int]

    @field_validator("steps_by_dimensionality", mode="after")
    @classmethod
    def _validate_histogram(cls, value: Dict[int, int]) -> Dict[int, int]:
        return canonicalise_dimensionality_histogram(value)


class WorkflowIntrospection(BaseModel):
    """Top-level workload introspection response.

    The versioned ``type`` identifies the response schema;
    ``execution_engine_version`` identifies the compiler that produced it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["workflow_introspection_v1"] = "workflow_introspection_v1"
    execution_engine_version: str = Field(min_length=1)
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    steps: List[StepMetadata]
    summary: WorkflowSummary

    @model_validator(mode="after")
    def _enforce_cross_field_consistency(self) -> "WorkflowIntrospection":
        node_kinds = _validate_nodes(nodes=self.nodes)
        _validate_edges(edges=self.edges, node_kinds=node_kinds)
        step_node_ids = {
            node_id for node_id, kind in node_kinds.items() if kind == "step"
        }
        _validate_steps(steps=self.steps, step_node_ids=step_node_ids)
        _validate_models(
            models=self.summary.models,
            step_node_ids=step_node_ids,
            input_dimensionality_by_step={
                step.node_id: step.input_dimensionality for step in self.steps
            },
        )
        _validate_summary_dimensionality(summary=self.summary, steps=self.steps)
        return self


def _validate_nodes(nodes: List[GraphNode]) -> Dict[str, str]:
    node_kinds: Dict[str, str] = {}
    for node in nodes:
        if node.id in node_kinds:
            raise ValueError(f"Duplicated graph node id: {node.id!r}.")
        node_kinds[node.id] = node.kind
    return node_kinds


def _validate_edges(edges: List[GraphEdge], node_kinds: Dict[str, str]) -> None:
    seen: Set[Tuple[str, str, str]] = set()
    for edge in edges:
        triple = (edge.source, edge.target, edge.kind)
        if triple in seen:
            raise ValueError(f"Duplicated graph edge: {triple!r}.")
        seen.add(triple)
        for endpoint in (edge.source, edge.target):
            if endpoint not in node_kinds:
                raise ValueError(
                    f"Graph edge {triple!r} references unknown node {endpoint!r}."
                )
        if edge.kind == "data" and node_kinds[edge.source] == "output":
            raise ValueError(f"Data edge {triple!r} must not start at an output node.")
        if edge.kind == "control" and (
            node_kinds[edge.source] != "step" or node_kinds[edge.target] != "step"
        ):
            raise ValueError(f"Control edge {triple!r} must connect two step nodes.")


def _validate_steps(steps: List[StepMetadata], step_node_ids: Set[str]) -> None:
    described: Set[str] = set()
    for step in steps:
        if step.node_id in described:
            raise ValueError(
                f"Step {step.node_id!r} has more than one StepMetadata entry."
            )
        if step.node_id not in step_node_ids:
            raise ValueError(
                f"StepMetadata {step.node_id!r} does not match any step node."
            )
        described.add(step.node_id)
    missing = step_node_ids - described
    if missing:
        raise ValueError(f"Step nodes without StepMetadata: {sorted(missing)!r}.")


def _validate_models(
    models: Discovery[ModelSummary],
    step_node_ids: Set[str],
    input_dimensionality_by_step: Dict[str, int],
) -> None:
    seen: Set[Tuple[str, str]] = set()
    for model in models.items:
        key = (model.provider, model.model_id)
        if key in seen:
            raise ValueError(f"Duplicated model inventory entry: {key!r}.")
        seen.add(key)
        unknown_steps = [
            step_id for step_id in model.used_by_steps if step_id not in step_node_ids
        ]
        if unknown_steps:
            raise ValueError(
                f"Model {key!r} is used by unknown steps: {unknown_steps!r}."
            )
        # every id is a known step node and `_validate_steps` guaranteed each
        # step node exactly one StepMetadata, so the lookup cannot miss
        histogram = Counter(
            input_dimensionality_by_step[step_id] for step_id in model.used_by_steps
        )
        if dict(histogram) != model.steps_by_dimensionality:
            raise ValueError(
                f"Model {key!r} `steps_by_dimensionality` must equal the histogram "
                "of the input dimensionalities of its `used_by_steps` "
                f"{dict(sorted(histogram.items()))!r}, got "
                f"{model.steps_by_dimensionality!r}."
            )


def _validate_summary_dimensionality(
    summary: WorkflowSummary, steps: List[StepMetadata]
) -> None:
    histogram = Counter(step.input_dimensionality for step in steps)
    if sum(summary.steps_by_dimensionality.values()) != len(steps):
        raise ValueError(
            "`steps_by_dimensionality` counts must sum to the number of steps "
            f"({len(steps)}), got {sum(summary.steps_by_dimensionality.values())}."
        )
    if dict(histogram) != summary.steps_by_dimensionality:
        raise ValueError(
            "`steps_by_dimensionality` must equal the histogram of step "
            f"input dimensionalities {dict(sorted(histogram.items()))!r}, got "
            f"{summary.steps_by_dimensionality!r}."
        )
    for step in steps:
        highest = max(step.input_dimensionality, step.output_dimensionality)
        if highest > summary.max_dimensionality:
            raise ValueError(
                f"`max_dimensionality` ({summary.max_dimensionality}) is lower "
                f"than the dimensionality of step {step.node_id!r} ({highest})."
            )
