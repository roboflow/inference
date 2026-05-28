"""
Collect ``dynamic_blocks_definitions`` from a workflow and nested inner workflows.

Inner workflows may declare custom Python blocks on their own definition object. The
compiler must discover all of them before ``compile_dynamic_blocks`` and inlining.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from inference.core.workflows.execution_engine.v1.inner_workflow.constants import (
    USE_INNER_WORKFLOW_BLOCK_TYPE,
)


def _dynamic_block_type(definition: Dict[str, Any]) -> Optional[str]:
    """Return ``manifest.block_type`` when present, else ``None``."""
    manifest = definition.get("manifest")
    if not isinstance(manifest, dict):
        return None

    block_type = manifest.get("block_type")
    if isinstance(block_type, str) and block_type:
        return block_type

    return None


def collect_dynamic_blocks_definitions_from_workflow_definition(
    workflow_definition: Dict[str, Any],
) -> List[dict]:
    """Collect dynamic block definitions from a workflow and nested inner workflows.

    Walks ``workflow_definition`` depth-first. For each level, appends entries from
    ``dynamic_blocks_definitions``; then recurses into ``inner_workflow`` steps via
    ``workflow_definition``.

    When the same ``manifest.block_type`` appears more than once, the first occurrence
    is kept (parent definitions win over nested children). Definitions without a
    ``block_type`` are still included and are not deduplicated.

    Args:
        workflow_definition: Raw workflow JSON (``steps``, optional nested definitions).

    Returns:
        Merged list of dynamic block definition dicts in discovery order.
    """
    collected: List[dict] = []
    seen_block_types: Set[str] = set()

    def append_level(definitions: Any) -> None:
        if not definitions:
            return

        if not isinstance(definitions, list):
            return

        for definition in definitions:
            if not isinstance(definition, dict):
                continue

            block_type = _dynamic_block_type(definition)
            if block_type is not None:
                if block_type in seen_block_types:
                    continue

                seen_block_types.add(block_type)

            collected.append(definition)

    def visit(workflow: Dict[str, Any]) -> None:
        append_level(workflow.get("dynamic_blocks_definitions"))

        for step in workflow.get("steps") or []:
            if not isinstance(step, dict):
                continue

            if step.get("type") != USE_INNER_WORKFLOW_BLOCK_TYPE:
                continue

            child = step.get("workflow_definition")
            if isinstance(child, dict):
                visit(child)

    visit(workflow_definition)

    return collected


def apply_collected_dynamic_blocks_definitions_to_workflow_root(
    workflow_definition: Dict[str, Any],
) -> List[dict]:
    """Hoist collected dynamic block definitions onto the root workflow dict.

    Calls :func:`collect_dynamic_blocks_definitions_from_workflow_definition` and, when
    the result is non-empty, sets ``workflow_definition["dynamic_blocks_definitions"]``
    to that merged list (mutates ``workflow_definition`` in place).

    Args:
        workflow_definition: Raw workflow JSON to update and scan for definitions.

    Returns:
        The merged dynamic block definition list (possibly empty).
    """
    merged = collect_dynamic_blocks_definitions_from_workflow_definition(
        workflow_definition=workflow_definition,
    )

    if merged:
        workflow_definition["dynamic_blocks_definitions"] = merged

    return merged
