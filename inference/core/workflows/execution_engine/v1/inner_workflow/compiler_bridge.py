"""
Compile-time helpers for nested workflows (composition validation and output projection).

Called from compile_workflow_graph; does not import block implementations.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Set, Tuple, Union

from inference.core.env import WORKFLOWS_MAX_INNER_WORKFLOW_DEPTH
from inference.core.workflows.errors import WorkflowDefinitionError
from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import WILDCARD_KIND, Kind
from inference.core.workflows.execution_engine.v1.compiler.entities import (
    GraphCompilationResult,
    ParsedWorkflowDefinition,
)
from inference.core.workflows.execution_engine.v1.inner_workflow.composition import (
    validate_inner_workflow_composition,
)
from inference.core.workflows.execution_engine.v1.inner_workflow.constants import (
    USE_INNER_WORKFLOW_BLOCK_TYPE,
)
from inference.core.workflows.prototypes.block import WorkflowBlockManifest


def workflow_identity_fingerprint(workflow_dict: Dict[str, Any]) -> str:
    """Stable opaque id for composition graph nodes."""
    payload = json.dumps(workflow_dict, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def collect_composition_edges_from_workflow_dict(
    workflow_dict: Dict[str, Any],
) -> List[Tuple[str, str]]:
    edges: List[Tuple[str, str]] = []

    def visit(wf: Dict[str, Any]) -> None:
        fp = workflow_identity_fingerprint(wf)
        for step in wf.get("steps", []):
            if step.get("type") != USE_INNER_WORKFLOW_BLOCK_TYPE:
                continue
            child_wf = step.get("workflow")
            if not isinstance(child_wf, dict):
                continue
            child_fp = workflow_identity_fingerprint(child_wf)
            edges.append((fp, child_fp))
            visit(child_wf)

    visit(workflow_dict)
    return edges


def validate_inner_workflow_composition_from_workflow_dict(
    workflow_dict: Dict[str, Any],
) -> None:
    edges = collect_composition_edges_from_workflow_dict(workflow_dict)
    root_fp = workflow_identity_fingerprint(workflow_dict)
    validate_inner_workflow_composition(
        containment_edges=edges,
        root_workflow_id=root_fp,
        max_nesting_depth=WORKFLOWS_MAX_INNER_WORKFLOW_DEPTH,
    )


def _normalize_kinds(kinds: List[Union[str, Kind]]) -> List[Kind]:
    result: List[Kind] = []
    for k in kinds:
        if isinstance(k, Kind):
            result.append(k)
        elif isinstance(k, dict):
            result.append(Kind.model_validate(k))
        elif isinstance(k, str):
            result.append(WILDCARD_KIND)
        else:
            raise WorkflowDefinitionError(
                public_message=f"Unexpected kind entry in workflow input: {k!r}",
                context="workflow_compilation | inner_workflow_output_projection",
            )
    return result


def kinds_for_workflow_output_selector(
    parsed: ParsedWorkflowDefinition,
    selector: str,
) -> List[Kind]:
    if selector.startswith("$inputs."):
        name = selector.split(".", 1)[1]
        for inp in parsed.inputs:
            if inp.name == name:
                return _normalize_kinds(inp.kind)
        raise WorkflowDefinitionError(
            public_message=f"Inner workflow output references unknown input `{name}` in selector `{selector}`.",
            context="workflow_compilation | inner_workflow_output_projection",
        )
    if not selector.startswith("$steps."):
        raise WorkflowDefinitionError(
            public_message=f"Unsupported output selector for inner workflow projection: `{selector}`.",
            context="workflow_compilation | inner_workflow_output_projection",
        )
    rest = selector[len("$steps.") :]
    step_name, _, prop = rest.partition(".")
    if not step_name or not prop:
        raise WorkflowDefinitionError(
            public_message=f"Invalid step output selector `{selector}` for inner workflow projection.",
            context="workflow_compilation | inner_workflow_output_projection",
        )
    for sm in parsed.steps:
        if sm.name != step_name:
            continue
        for od in sm.get_actual_outputs():
            if od.name == prop:
                return list(od.kind)
        raise WorkflowDefinitionError(
            public_message=f"Step `{step_name}` has no output `{prop}` (selector `{selector}`).",
            context="workflow_compilation | inner_workflow_output_projection",
        )
    raise WorkflowDefinitionError(
        public_message=f"Unknown step `{step_name}` in selector `{selector}`.",
        context="workflow_compilation | inner_workflow_output_projection",
    )


def max_projection_output_lift_from_child_workflow(
    child_graph: GraphCompilationResult,
) -> int:
    """
    Largest positive ``get_output_dimensionality_offset()`` among child steps referenced
    by the child workflow's JsonField outputs (e.g. ``dynamic_crop`` -> 1).
    """
    from inference.core.workflows.execution_engine.v1.compiler.entities import StepNode
    from inference.core.workflows.execution_engine.v1.compiler.graph_constructor import (
        NODE_COMPILATION_OUTPUT_PROPERTY,
    )
    from inference.core.workflows.execution_engine.v1.compiler.utils import (
        construct_step_selector,
    )

    lift = 0
    graph = child_graph.execution_graph
    for jf in child_graph.parsed_workflow_definition.outputs:
        sel = jf.selector
        if not isinstance(sel, str) or not sel.startswith("$steps."):
            continue
        rest = sel[len("$steps.") :]
        step_name, _, prop = rest.partition(".")
        if not step_name or not prop:
            continue
        step_selector = construct_step_selector(step_name=step_name)
        if step_selector not in graph.nodes:
            continue
        comp = graph.nodes[step_selector].get(NODE_COMPILATION_OUTPUT_PROPERTY)
        if not isinstance(comp, StepNode):
            continue
        o = comp.step_manifest.get_output_dimensionality_offset()
        if o > lift:
            lift = o
    return lift


def derive_resolved_outputs_for_child_workflow(
    child_graph: GraphCompilationResult,
) -> List[OutputDefinition]:
    parsed = child_graph.parsed_workflow_definition
    result: List[OutputDefinition] = []
    for jf in parsed.outputs:
        kinds = kinds_for_workflow_output_selector(parsed, jf.selector)
        result.append(OutputDefinition(name=jf.name, kind=kinds))
    return result


def validate_parameter_bindings_against_child(
    *,
    bindings: Dict[str, str],
    child_parsed: ParsedWorkflowDefinition,
    step_name: str,
) -> None:
    expected: Set[str] = {inp.name for inp in child_parsed.inputs}
    got = set(bindings.keys())
    if got != expected:
        raise WorkflowDefinitionError(
            public_message=(
                f"inner_workflow step `{step_name}` parameter_bindings keys {sorted(got)} "
                f"do not match child workflow inputs {sorted(expected)}."
            ),
            context="workflow_compilation | inner_workflow_parameter_bindings",
        )


def resolve_inner_workflow_steps_in_parsed_definition(
    parsed: ParsedWorkflowDefinition,
    raw_workflow_definition: Dict[str, Any],
    *,
    compile_workflow_graph_fn,
    available_blocks: Any,
    execution_engine_version: Any,
    init_parameters: Dict[str, Any],
    profiler: Any,
) -> ParsedWorkflowDefinition:
    """
    For each ``inner_workflow`` step, compile the child workflow (for cache + output kinds)
    and attach resolved_child_outputs on the manifest copy.
    """
    from inference.core.workflows.execution_engine.v1.compiler.entities import (
        ParsedWorkflowDefinition as PWD,
    )

    new_steps: List[WorkflowBlockManifest] = []
    for step in parsed.steps:
        if getattr(step, "type", None) != USE_INNER_WORKFLOW_BLOCK_TYPE:
            new_steps.append(step)
            continue
        child_wf = step.workflow
        if not isinstance(child_wf, dict):
            raise WorkflowDefinitionError(
                public_message=f"inner_workflow step `{step.name}` requires `workflow` object.",
                context="workflow_compilation | inner_workflow_resolution",
            )
        child_result = compile_workflow_graph_fn(
            workflow_definition=child_wf,
            execution_engine_version=execution_engine_version,
            profiler=profiler,
            init_parameters=init_parameters,
        )
        validate_parameter_bindings_against_child(
            bindings=dict(step.parameter_bindings),
            child_parsed=child_result.parsed_workflow_definition,
            step_name=step.name,
        )
        resolved = derive_resolved_outputs_for_child_workflow(child_result)
        lift = max_projection_output_lift_from_child_workflow(child_result)
        new_steps.append(
            step.model_copy(
                update={
                    "resolved_child_outputs": resolved,
                    "nested_output_dimensionality_lift": lift,
                },
            ),
        )
    return PWD(
        version=parsed.version,
        inputs=parsed.inputs,
        steps=new_steps,
        outputs=parsed.outputs,
    )
