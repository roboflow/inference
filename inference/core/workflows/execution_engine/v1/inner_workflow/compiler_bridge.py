"""
Compile-time helpers for nested workflows (composition validation and parameter bindings).

``inner_workflow`` steps are expanded into ordinary steps before parsing; see ``inline.py``.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Set, Tuple

from inference.core.env import (
    WORKFLOWS_MAX_INNER_WORKFLOW_COUNT,
    WORKFLOWS_MAX_INNER_WORKFLOW_DEPTH,
)
from inference.core.workflows.execution_engine.entities.base import (
    InputType,
    WorkflowParameter,
)
from inference.core.workflows.execution_engine.v1.compiler.entities import (
    ParsedWorkflowDefinition,
)
from inference.core.workflows.execution_engine.v1.inner_workflow.composition import (
    validate_inner_workflow_composition,
)
from inference.core.workflows.execution_engine.v1.inner_workflow.constants import (
    USE_INNER_WORKFLOW_BLOCK_TYPE,
)
from inference.core.workflows.execution_engine.v1.inner_workflow.errors import (
    InnerWorkflowInvalidStepEntryError,
    InnerWorkflowParameterBindingsMissingRequiredError,
    InnerWorkflowParameterBindingsUnknownInputError,
)


def validate_inner_workflow_composition_from_raw_workflow_definition(
    raw_workflow_definition: Dict[str, Any],
) -> None:
    edges = _collect_composition_edges_from_raw_workflow_definition(
        raw_workflow_definition
    )
    root_workflow_id = _workflow_identity_fingerprint(raw_workflow_definition)
    validate_inner_workflow_composition(
        containment_edges=edges,
        root_workflow_id=root_workflow_id,
        max_nesting_depth=WORKFLOWS_MAX_INNER_WORKFLOW_DEPTH,
        max_inner_workflow_count=WORKFLOWS_MAX_INNER_WORKFLOW_COUNT,
    )


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
        raise InnerWorkflowParameterBindingsUnknownInputError(
            f"inner_workflow step `{step_name}` parameter_bindings references unknown "
            f"child workflow input names {sorted(unknown)}. "
            f"Valid input names: {sorted(expected)}."
        )

    missing_required = sorted(
        inp.name
        for inp in child_parsed.inputs
        if _child_workflow_input_requires_parameter_binding(inp) and inp.name not in got
    )
    if missing_required:
        raise InnerWorkflowParameterBindingsMissingRequiredError(
            f"inner_workflow step `{step_name}` is missing parameter_bindings for required "
            f"child workflow inputs {missing_required}. "
            f"Omitting a binding is allowed only for `WorkflowParameter` / `InferenceParameter` "
            f"inputs with a non-null `default_value` in the child workflow definition."
        )


def _workflow_identity_fingerprint(raw_workflow_definition: Dict[str, Any]) -> str:
    """Stable opaque id for composition graph nodes."""
    payload = json.dumps(
        raw_workflow_definition, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _collect_composition_edges_from_raw_workflow_definition(
    raw_workflow_definition: Dict[str, Any],
) -> List[Tuple[str, str]]:
    edges: List[Tuple[str, str]] = []

    def visit(workflow_definition: Dict[str, Any]) -> None:
        _id = _workflow_identity_fingerprint(workflow_definition)

        for index, step in enumerate(workflow_definition.get("steps") or []):
            if not isinstance(step, dict):
                raise InnerWorkflowInvalidStepEntryError(
                    f"Invalid workflow step at index {index}: each step must be a JSON object "
                    f"(dict), got {type(step).__name__}."
                )

            if step.get("type") != USE_INNER_WORKFLOW_BLOCK_TYPE:
                continue

            inner_workflow_definition = step.get("workflow_definition")
            if not isinstance(inner_workflow_definition, dict):
                raise InnerWorkflowInvalidStepEntryError(
                    f"Invalid workflow step at index {index}: `workflow_definition` must be a JSON object "
                    f"(dict), got {type(inner_workflow_definition).__name__}."
                )

            child_id = _workflow_identity_fingerprint(inner_workflow_definition)
            edges.append((_id, child_id))

            visit(inner_workflow_definition)

    visit(raw_workflow_definition)

    return edges


def _child_workflow_input_requires_parameter_binding(child_input: InputType) -> bool:
    """
    Bindings may omit ``WorkflowParameter`` / ``InferenceParameter`` inputs that declare a
    non-null ``default_value``; those values are supplied when assembling the child's runtime
    input (see :func:`assemble_inference_parameter`).
    """
    if isinstance(child_input, WorkflowParameter):
        return child_input.default_value is None
    return True
