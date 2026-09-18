"""Public response schemas of workload introspection.

These entities describe compile-time workload facts of a workflow: the
connectivity graph, per-step facts and summaries. They are an introspection
extension, not an estimator - no scores, weights, schedules or hardware
assumptions appear here.

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

WORKLOAD_INTROSPECTION_SCHEMA_VERSION = "1"


class GraphNode(BaseModel):
    """A workflow input, step or output. ``id`` is the canonical selector
    (``$inputs.<name>``, ``$steps.<name>``, ``$outputs.<name>``). Synthetic
    compiler nodes never appear here."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["graph_node"] = "graph_node"
    id: str = Field(min_length=1)
    kind: GraphNodeKind


class GraphEdge(BaseModel):
    """Connectivity between two nodes. ``data`` edges carry values, ``control``
    edges express flow control between steps. A node pair may have both. The
    graph is connectivity, not a schedule."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["graph_edge"] = "graph_edge"
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

    type: Literal["step_metadata"] = "step_metadata"
    node_id: str = Field(min_length=1)
    block_type: str = Field(min_length=1)
    input_dimensionality: int = Field(ge=0)
    output_dimensionality: int = Field(ge=0)
    accepts_batch_input: bool
    resources: Discovery[DependentResource]
    restrictions: Discovery[RestrictionMetadata]
    operations: Discovery[WorkOperation]


class ModelSummary(BaseModel):
    """One entry of the model inventory: a model reference and the steps that
    refer to it. Inventory membership is not a claim that the model executes;
    per-step resources keep ACCESS vs EXECUTION."""

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    type: Literal["model_summary"] = "model_summary"
    provider: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    used_by_steps: List[str]
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

    @model_validator(mode="after")
    def _enforce_status_metadata_consistency(self) -> "ModelSummary":
        ensure_model_metadata_status_consistent(
            status=self.metadata_status, metadata=self.metadata
        )
        return self


class WorkflowSummary(BaseModel):
    """Whole-graph summary. ``steps_by_dimensionality`` maps input depth ->
    number of steps at that depth (zero-count depths omitted; keys are strings
    on the wire). ``max_dimensionality`` considers the compiled graph's actual
    input/output depths, so a no-step input->output workflow is covered."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["workflow_summary"] = "workflow_summary"
    models: Discovery[ModelSummary]
    max_dimensionality: int = Field(ge=0)
    steps_by_dimensionality: Dict[int, int]

    @field_validator("steps_by_dimensionality", mode="after")
    @classmethod
    def _validate_histogram(cls, value: Dict[int, int]) -> Dict[int, int]:
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


class WorkflowIntrospection(BaseModel):
    """Top-level workload introspection response (schema version 1)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["workflow_introspection"] = "workflow_introspection"
    schema_version: Literal["1"] = WORKLOAD_INTROSPECTION_SCHEMA_VERSION
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
        _validate_models(models=self.summary.models, step_node_ids=step_node_ids)
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


def _validate_models(models: Discovery[ModelSummary], step_node_ids: Set[str]) -> None:
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
