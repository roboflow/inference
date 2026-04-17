"""
Compile-time helpers for nested workflows (composition validation and parameter bindings).

``inner_workflow`` steps are expanded into ordinary steps before parsing; see ``inline.py``.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Set, Tuple

from inference.core.env import WORKFLOWS_MAX_INNER_WORKFLOW_DEPTH
from inference.core.workflows.errors import WorkflowDefinitionError
from inference.core.workflows.execution_engine.entities.base import InputType, WorkflowParameter
from inference.core.workflows.execution_engine.v1.compiler.entities import (
    ParsedWorkflowDefinition,
)
from inference.core.workflows.execution_engine.v1.inner_workflow.composition import (
    validate_inner_workflow_composition,
)
from inference.core.workflows.execution_engine.v1.inner_workflow.constants import (
    USE_INNER_WORKFLOW_BLOCK_TYPE,
)


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
        for step in wf.get("steps") or []:
            if not isinstance(step, dict):
                continue
            if step.get("type") != USE_INNER_WORKFLOW_BLOCK_TYPE:
                continue
            child_wf = step.get("workflow_definition")
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


def _child_workflow_input_requires_parameter_binding(child_input: InputType) -> bool:
    """
    Bindings may omit ``WorkflowParameter`` / ``InferenceParameter`` inputs that declare a
    non-null ``default_value``; those values are supplied when assembling the child's runtime
    input (see :func:`assemble_inference_parameter`).
    """
    if isinstance(child_input, WorkflowParameter):
        return child_input.default_value is None
    return True


def validate_parameter_bindings_against_child(
    *,
    bindings: Dict[str, str],
    child_parsed: ParsedWorkflowDefinition,
    step_name: str,
) -> None:
    expected: Set[str] = {inp.name for inp in child_parsed.inputs}
    got = set(bindings.keys())
    unknown = got - expected
    if unknown:
        raise WorkflowDefinitionError(
            public_message=(
                f"inner_workflow step `{step_name}` parameter_bindings references unknown "
                f"child workflow input names {sorted(unknown)}. "
                f"Valid input names: {sorted(expected)}."
            ),
            context="workflow_compilation | inner_workflow_parameter_bindings",
        )
    missing_required = sorted(
        inp.name
        for inp in child_parsed.inputs
        if _child_workflow_input_requires_parameter_binding(inp) and inp.name not in got
    )
    if missing_required:
        raise WorkflowDefinitionError(
            public_message=(
                f"inner_workflow step `{step_name}` is missing parameter_bindings for required "
                f"child workflow inputs {missing_required}. "
                f"Omitting a binding is allowed only for `WorkflowParameter` / `InferenceParameter` "
                f"inputs with a non-null `default_value` in the child workflow definition."
            ),
            context="workflow_compilation | inner_workflow_parameter_bindings",
        )
