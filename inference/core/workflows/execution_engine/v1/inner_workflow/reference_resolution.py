"""
Resolve ``roboflow_core/inner_workflow@v1`` steps that reference a saved workflow by id into inline
``workflow_definition`` payloads before parsing / composition validation.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, Optional, Tuple

from inference.core.workflows.errors import WorkflowDefinitionError
from inference.core.workflows.execution_engine.v1.inner_workflow.constants import (
    USE_INNER_WORKFLOW_BLOCK_TYPE,
)

WORKFLOWS_CORE_INNER_WORKFLOW_SPEC_RESOLVER = (
    "workflows_core.inner_workflow_spec_resolver"
)

InnerWorkflowSpecResolver = Callable[
    [str, str, Optional[str], Dict[str, Any]],
    Dict[str, Any],
]


def default_inner_workflow_spec_resolver(
    workspace_id: str,
    workflow_id: str,
    workflow_version_id: Optional[str],
    init_parameters: Dict[str, Any],
) -> Dict[str, Any]:
    from inference.core.roboflow_api import get_workflow_specification

    api_key = init_parameters.get("workflows_core.api_key")
    if workspace_id != "local" and not api_key:
        raise WorkflowDefinitionError(
            public_message=(
                "Resolving an `inner_workflow` step by workflow id requires a Roboflow API key. "
                "Set `workflows_core.api_key` in workflow init_parameters, inject "
                "`workflows_core.inner_workflow_spec_resolver`, or use "
                '`workflow_workspace_id` `"local"` with a matching on-disk workflow '
                "definition."
            ),
            context="workflow_compilation | inner_workflow_spec_resolution",
        )
    return get_workflow_specification(
        api_key=api_key,
        workspace_id=workspace_id,
        workflow_id=workflow_id,
        workflow_version_id=workflow_version_id,
    )


def get_inner_workflow_spec_resolver(
    init_parameters: Dict[str, Any],
) -> InnerWorkflowSpecResolver:
    resolver = init_parameters.get(WORKFLOWS_CORE_INNER_WORKFLOW_SPEC_RESOLVER)
    if resolver is not None:
        return resolver
    return default_inner_workflow_spec_resolver


def _strip_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        return s if s else None
    return str(value)


def _inner_workflow_step_has_reference(step: Dict[str, Any]) -> bool:
    ws = _strip_optional_str(step.get("workflow_workspace_id"))
    wf = _strip_optional_str(step.get("workflow_id"))
    return ws is not None and wf is not None


def _inner_workflow_step_has_nonempty_workflow_definition(
    step: Dict[str, Any],
) -> bool:
    wf = step.get("workflow_definition")
    return isinstance(wf, dict) and len(wf) > 0


def workflow_definition_contains_unresolved_inner_workflow_reference(
    workflow_definition: Dict[str, Any],
) -> bool:
    """True if any ``inner_workflow`` step (at any depth) still needs reference resolution."""

    def visit(wf: Dict[str, Any]) -> bool:
        for step in wf.get("steps", []) or []:
            if not isinstance(step, dict):
                continue
            if step.get("type") != USE_INNER_WORKFLOW_BLOCK_TYPE:
                continue
            if _inner_workflow_step_has_reference(step):
                return True
            child = step.get("workflow_definition")
            if isinstance(child, dict) and visit(child):
                return True
        return False

    return visit(workflow_definition)


def _reference_cache_key(
    workspace_id: str,
    saved_workflow_id: str,
    workflow_version_id: Optional[str],
) -> Tuple[str, str, Optional[str]]:
    return (workspace_id, saved_workflow_id, workflow_version_id)


def _strip_reference_fields_from_step(step: Dict[str, Any]) -> None:
    step.pop("workflow_workspace_id", None)
    step.pop("workflow_id", None)
    step.pop("workflow_version_id", None)


def _normalize_inner_workflow_refs_in_workflow_dict(
    workflow_dict: Dict[str, Any],
    init_parameters: Dict[str, Any],
    resolver: InnerWorkflowSpecResolver,
    fetch_memo: Dict[Tuple[str, str, Optional[str]], Dict[str, Any]],
) -> None:
    for step in workflow_dict.get("steps", []) or []:
        if not isinstance(step, dict):
            continue
        if step.get("type") != USE_INNER_WORKFLOW_BLOCK_TYPE:
            continue

        has_ref = _inner_workflow_step_has_reference(step)
        has_inline = _inner_workflow_step_has_nonempty_workflow_definition(step)

        if has_ref and has_inline:
            step_name = step.get("name", "<unknown>")
            raise WorkflowDefinitionError(
                public_message=(
                    f"inner_workflow step `{step_name}` must not set both `workflow_definition` "
                    f"and reference fields (`workflow_workspace_id` / `workflow_id`)."
                ),
                context="workflow_compilation | inner_workflow_spec_resolution",
            )

        if has_ref:
            workspace_id = _strip_optional_str(step["workflow_workspace_id"])
            saved_workflow_id = _strip_optional_str(step["workflow_id"])
            version_raw = step.get("workflow_version_id")
            workflow_version_id = (
                None if version_raw is None else _strip_optional_str(version_raw)
            )

            assert workspace_id is not None and saved_workflow_id is not None
            cache_key = _reference_cache_key(
                workspace_id, saved_workflow_id, workflow_version_id
            )
            if cache_key not in fetch_memo:
                fetch_memo[cache_key] = resolver(
                    workspace_id,
                    saved_workflow_id,
                    workflow_version_id,
                    init_parameters,
                )
            step["workflow_definition"] = copy.deepcopy(fetch_memo[cache_key])
            _strip_reference_fields_from_step(step)
        elif not has_inline:
            step_name = step.get("name", "<unknown>")
            raise WorkflowDefinitionError(
                public_message=(
                    f"inner_workflow step `{step_name}` requires a non-empty `workflow_definition` object or "
                    f"reference fields `workflow_workspace_id` and `workflow_id`."
                ),
                context="workflow_compilation | inner_workflow_spec_resolution",
            )

        child_wf = step.get("workflow_definition")
        if isinstance(child_wf, dict):
            _normalize_inner_workflow_refs_in_workflow_dict(
                child_wf, init_parameters, resolver, fetch_memo
            )


def normalize_inner_workflow_references_in_definition(
    workflow_definition: Dict[str, Any],
    init_parameters: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Return a workflow definition suitable for parsing: all ``inner_workflow`` reference fields
    are resolved to inline ``workflow_definition`` (recursively). The input dict is never mutated.
    """
    if not workflow_definition_contains_unresolved_inner_workflow_reference(
        workflow_definition
    ):
        return workflow_definition

    result = copy.deepcopy(workflow_definition)
    resolver = get_inner_workflow_spec_resolver(init_parameters)
    fetch_memo: Dict[Tuple[str, str, Optional[str]], Dict[str, Any]] = {}
    _normalize_inner_workflow_refs_in_workflow_dict(
        result, init_parameters, resolver, fetch_memo
    )
    return result
