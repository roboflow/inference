"""
Honour steps disabled in the Workflow builder at runtime.

The builder stores "disable block" as a UI-only flag at
``metadata.ui.nodes["$steps.<name>"].disabled`` and strips such steps client-side
before preview runs. The persisted specification still contains them, so any
runtime that fetches the workflow by id (inference server, edge devices, Dedicated
Deployments, serverless) would otherwise compile and execute every step - including
loading model weights for disabled model blocks.

This module mirrors the builder's ``stripDisabledForExecution`` logic:
  * seed: every step manually flagged ``disabled: true``
  * cascade (a): a step whose required (no-default) field would be emptied entirely
    by removing references to disabled steps is itself disabled
  * cascade (b): a step gated only by conditional-flow blocks (``next_steps``)
    that are all disabled is itself disabled
  * strip: drop disabled steps, remove references to them from surviving steps,
    drop outputs whose selector points at a disabled step
"""

import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Type

from pydantic.fields import FieldInfo
from typing_extensions import get_args

from inference.core.workflows.execution_engine.v1.compiler.entities import (
    BlockSpecification,
)
from inference.core.workflows.prototypes.block import WorkflowBlockManifest

STEP_REF_PATTERN = re.compile(r"\$steps\.([^.\s\]\[\"']+)")
NEXT_STEPS_FIELD = "next_steps"
RESERVED_STEP_KEYS = {"type", "name", "id"}


def strip_disabled_steps(
    workflow_definition: Dict[str, Any],
    available_blocks: Iterable[BlockSpecification],
) -> Dict[str, Any]:
    """Return a copy of ``workflow_definition`` with disabled steps removed.

    Returns the input object untouched when nothing is disabled, so the common
    path is free of copies.
    """
    manually_disabled = _collect_manually_disabled_step_names(workflow_definition)
    if not manually_disabled:
        return workflow_definition
    steps = workflow_definition.get("steps") or []
    manifests_by_type = _index_manifests_by_type(available_blocks)
    disabled = _compute_disabled_step_names(
        steps=steps,
        seed=manually_disabled,
        manifests_by_type=manifests_by_type,
    )
    disabled_node_ids = {f"$steps.{name}" for name in disabled}
    surviving_steps = []
    for step in steps:
        if _step_name(step) in disabled:
            continue
        surviving_steps.append(_strip_references(step, disabled_node_ids))
    surviving_outputs = [
        output
        for output in workflow_definition.get("outputs") or []
        if not (
            isinstance(output, dict)
            and isinstance(output.get("selector"), str)
            and _selector_points_at_any(output["selector"], disabled_node_ids)
        )
    ]
    result = dict(workflow_definition)
    result["steps"] = surviving_steps
    result["outputs"] = surviving_outputs
    return result


def _collect_manually_disabled_step_names(
    workflow_definition: Dict[str, Any],
) -> Set[str]:
    metadata = workflow_definition.get("metadata")
    if not isinstance(metadata, dict):
        return set()
    ui = metadata.get("ui")
    if not isinstance(ui, dict):
        return set()
    nodes = ui.get("nodes")
    if not isinstance(nodes, dict):
        return set()
    result = set()
    for node_id, node_meta in nodes.items():
        if not isinstance(node_meta, dict) or node_meta.get("disabled") is not True:
            continue
        if not isinstance(node_id, str) or not node_id.startswith("$steps."):
            continue
        result.add(node_id[len("$steps.") :])
    return result


def _compute_disabled_step_names(
    steps: List[Dict[str, Any]],
    seed: Set[str],
    manifests_by_type: Dict[str, Type[WorkflowBlockManifest]],
) -> Set[str]:
    disabled = set(seed)
    control_predecessors = _build_control_predecessor_map(steps)
    changed = True
    while changed:
        changed = False
        disabled_node_ids = {f"$steps.{name}" for name in disabled}
        for step in steps:
            name = _step_name(step)
            if not name or name in disabled:
                continue
            if _has_fully_disabled_required_field(
                step=step,
                disabled_node_ids=disabled_node_ids,
                manifests_by_type=manifests_by_type,
            ) or _has_only_disabled_control_predecessors(
                step_name=name,
                control_predecessors=control_predecessors,
                disabled=disabled,
            ):
                disabled.add(name)
                changed = True
    return disabled


def _has_fully_disabled_required_field(
    step: Dict[str, Any],
    disabled_node_ids: Set[str],
    manifests_by_type: Dict[str, Type[WorkflowBlockManifest]],
) -> bool:
    manifest_class = manifests_by_type.get(step.get("type"))
    if manifest_class is None:
        return False
    for field_name, field_info in manifest_class.model_fields.items():
        if field_name in RESERVED_STEP_KEYS or field_name == NEXT_STEPS_FIELD:
            continue
        if not _is_required(field_info):
            continue
        value = step.get(field_name)
        if value is None and field_info.alias:
            value = step.get(field_info.alias)
        if value is None:
            continue
        if not _find_step_refs(value):
            continue
        cleaned, _ = _clean_value(value, disabled_node_ids)
        if cleaned is _DELETE:
            return True
    return False


def _is_required(field_info: FieldInfo) -> bool:
    try:
        return field_info.is_required()
    except AttributeError:  # pragma: no cover - pydantic v1 fallback
        return getattr(field_info, "required", False) is True


def _build_control_predecessor_map(
    steps: List[Dict[str, Any]],
) -> Dict[str, Set[str]]:
    result: Dict[str, Set[str]] = {}
    for step in steps:
        source = _step_name(step)
        if not source or NEXT_STEPS_FIELD not in step:
            continue
        for target in _find_step_refs(step.get(NEXT_STEPS_FIELD)):
            result.setdefault(target, set()).add(source)
    return result


def _has_only_disabled_control_predecessors(
    step_name: str,
    control_predecessors: Dict[str, Set[str]],
    disabled: Set[str],
) -> bool:
    predecessors = control_predecessors.get(step_name)
    if not predecessors:
        return False
    return all(predecessor in disabled for predecessor in predecessors)


def _strip_references(
    step: Dict[str, Any], disabled_node_ids: Set[str]
) -> Dict[str, Any]:
    result = {}
    for key, value in step.items():
        if key in RESERVED_STEP_KEYS:
            result[key] = value
            continue
        cleaned, _ = _clean_value(value, disabled_node_ids)
        if cleaned is _DELETE:
            continue
        result[key] = cleaned
    return result


class _Delete:
    pass


_DELETE = _Delete()


def _clean_value(value: Any, disabled_node_ids: Set[str]) -> Tuple[Any, bool]:
    """Return (cleaned_value, changed). ``_DELETE`` means drop the value."""
    if isinstance(value, str):
        if _string_references_disabled(value, disabled_node_ids):
            return _DELETE, True
        return value, False
    if isinstance(value, list):
        changed = False
        filtered = []
        for item in value:
            cleaned, item_changed = _clean_value(item, disabled_node_ids)
            if cleaned is _DELETE:
                changed = True
                continue
            changed = changed or item_changed
            filtered.append(cleaned)
        if not filtered:
            return _DELETE, True
        return (filtered if changed else value), changed
    if isinstance(value, dict):
        changed = False
        out = {}
        for key, item in value.items():
            cleaned, item_changed = _clean_value(item, disabled_node_ids)
            if cleaned is _DELETE:
                changed = True
                continue
            changed = changed or item_changed
            out[key] = cleaned
        if not out:
            return _DELETE, True
        return (out if changed else value), changed
    return value, False


def _string_references_disabled(value: str, disabled_node_ids: Set[str]) -> bool:
    return any(
        value == node_id or value.startswith(f"{node_id}.")
        for node_id in disabled_node_ids
    )


def _selector_points_at_any(selector: str, disabled_node_ids: Set[str]) -> bool:
    return _string_references_disabled(selector, disabled_node_ids)


def _find_step_refs(value: Any) -> List[str]:
    if isinstance(value, str):
        return STEP_REF_PATTERN.findall(value)
    if isinstance(value, list):
        return [ref for item in value for ref in _find_step_refs(item)]
    if isinstance(value, dict):
        return [ref for item in value.values() for ref in _find_step_refs(item)]
    return []


def _step_name(step: Any) -> Optional[str]:
    if not isinstance(step, dict):
        return None
    name = step.get("name") or step.get("id")
    return name if isinstance(name, str) else None


def _index_manifests_by_type(
    available_blocks: Iterable[BlockSpecification],
) -> Dict[str, Type[WorkflowBlockManifest]]:
    result: Dict[str, Type[WorkflowBlockManifest]] = {}
    for block in available_blocks:
        manifest_class = block.manifest_class
        type_field = manifest_class.model_fields.get("type")
        if type_field is None:
            continue
        for type_identifier in get_args(type_field.annotation):
            if isinstance(type_identifier, str):
                result[type_identifier] = manifest_class
    return result
