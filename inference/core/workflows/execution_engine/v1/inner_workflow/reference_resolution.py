"""
Resolve ``use_subworkflow`` steps that reference a saved workflow by id into inline
``embedded_workflow`` definitions before parsing / composition validation.
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
# Legacy init parameter key (still honored by :func:`get_inner_workflow_spec_resolver`).
LEGACY_WORKFLOWS_CORE_SUBWORKFLOW_SPEC_RESOLVER = (
    "workflows_core.subworkflow_spec_resolver"
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
                "Resolving a `use_subworkflow` step by workflow id requires a Roboflow API key. "
                "Set `workflows_core.api_key` in workflow init_parameters, inject "
                "`workflows_core.inner_workflow_spec_resolver` (or legacy "
                "`workflows_core.subworkflow_spec_resolver`), or use "
                '`embedded_workflow_workspace_id` `"local"` with a matching on-disk workflow '
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
    if resolver is None:
        resolver = init_parameters.get(LEGACY_WORKFLOWS_CORE_SUBWORKFLOW_SPEC_RESOLVER)
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
    ws = _strip_optional_str(step.get("embedded_workflow_workspace_id"))
    wf = _strip_optional_str(step.get("embedded_workflow_id"))
    return ws is not None and wf is not None


def _inner_workflow_step_has_nonempty_embedded(step: Dict[str, Any]) -> bool:
    emb = step.get("embedded_workflow")
    return isinstance(emb, dict) and len(emb) > 0


def workflow_definition_contains_unresolved_inner_workflow_reference(
    workflow_definition: Dict[str, Any],
) -> bool:
    """True if any ``use_subworkflow`` step (at any depth) still needs reference resolution."""

    def visit(wf: Dict[str, Any]) -> bool:
        for step in wf.get("steps", []) or []:
            if not isinstance(step, dict):
                continue
            if step.get("type") != USE_INNER_WORKFLOW_BLOCK_TYPE:
                continue
            if _inner_workflow_step_has_reference(step):
                return True
            emb = step.get("embedded_workflow")
            if isinstance(emb, dict) and visit(emb):
                return True
        return False

    return visit(workflow_definition)


def _reference_cache_key(
    workspace_id: str,
    workflow_id: str,
    workflow_version_id: Optional[str],
) -> Tuple[str, str, Optional[str]]:
    return (workspace_id, workflow_id, workflow_version_id)


def _strip_reference_fields_from_step(step: Dict[str, Any]) -> None:
    step.pop("embedded_workflow_workspace_id", None)
    step.pop("embedded_workflow_id", None)
    step.pop("embedded_workflow_version_id", None)


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
        has_embedded = _inner_workflow_step_has_nonempty_embedded(step)

        if has_ref and has_embedded:
            step_name = step.get("name", "<unknown>")
            raise WorkflowDefinitionError(
                public_message=(
                    f"use_subworkflow step `{step_name}` must not set both `embedded_workflow` "
                    f"and reference fields (`embedded_workflow_workspace_id` / "
                    f"`embedded_workflow_id`)."
                ),
                context="workflow_compilation | inner_workflow_spec_resolution",
            )

        if has_ref:
            workspace_id = _strip_optional_str(step["embedded_workflow_workspace_id"])
            workflow_id = _strip_optional_str(step["embedded_workflow_id"])
            version_raw = step.get("embedded_workflow_version_id")
            workflow_version_id = (
                None if version_raw is None else _strip_optional_str(version_raw)
            )

            assert workspace_id is not None and workflow_id is not None
            cache_key = _reference_cache_key(
                workspace_id, workflow_id, workflow_version_id
            )
            if cache_key not in fetch_memo:
                fetch_memo[cache_key] = resolver(
                    workspace_id,
                    workflow_id,
                    workflow_version_id,
                    init_parameters,
                )
            step["embedded_workflow"] = copy.deepcopy(fetch_memo[cache_key])
            _strip_reference_fields_from_step(step)
        elif not has_embedded:
            step_name = step.get("name", "<unknown>")
            raise WorkflowDefinitionError(
                public_message=(
                    f"use_subworkflow step `{step_name}` requires a non-empty "
                    f"`embedded_workflow` object or reference fields "
                    f"`embedded_workflow_workspace_id` and `embedded_workflow_id`."
                ),
                context="workflow_compilation | inner_workflow_spec_resolution",
            )

        embedded = step.get("embedded_workflow")
        if isinstance(embedded, dict):
            _normalize_inner_workflow_refs_in_workflow_dict(
                embedded, init_parameters, resolver, fetch_memo
            )


def normalize_inner_workflow_references_in_definition(
    workflow_definition: Dict[str, Any],
    init_parameters: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Return a workflow definition suitable for parsing: all ``use_subworkflow`` reference fields
    are resolved to inline ``embedded_workflow`` (recursively). The input dict is never mutated.
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
