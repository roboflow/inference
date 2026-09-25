"""
Collect ``dynamic_blocks_definitions`` from a workflow and embedded inner workflows.

Embedded workflows may declare custom Python blocks on their own definition object. The
compiler must discover all of them before ``compile_dynamic_blocks`` and inlining.
Dispatched workflow definitions remain opaque because their target server compiles them.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from roboflow_workflows._compat_names import get_logger
from roboflow_workflows.execution_engine.v1.inner_workflow.constants import (
    INNER_WORKFLOW_EXECUTION_MODE_REMOTE_DISPATCH,
    USE_INNER_WORKFLOW_BLOCK_TYPE,
)

logger = get_logger(__name__)


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
    warn_on_duplicates: bool = True,
) -> List[Any]:
    """Collect dynamic block definitions from a workflow and embedded inner workflows.

    Walks ``workflow_definition`` depth-first. For each level, appends entries from
    ``dynamic_blocks_definitions``; then recurses into embedded ``inner_workflow``
    steps via ``workflow_definition``. Dispatched steps are skipped.

    When the same ``manifest.block_type`` appears more than once, the first occurrence
    is kept (parent definitions win over nested children) and, when ``warn_on_duplicates``
    is ``True``, a warning is logged for each skipped duplicate. The warning names the
    skipped and the retained definition only by structural position, for example
    ``steps[2].workflow_definition.dynamic_blocks_definitions[1]``; indexes refer to the
    original arrays. It never includes block types or other request-provided strings.
    Definitions without a ``block_type`` are still included and are not deduplicated.

    Malformed entries (non-list ``dynamic_blocks_definitions``, non-dict list items)
    are passed through as-is so :func:`compile_dynamic_blocks` can validate them.

    Args:
        workflow_definition: Raw workflow JSON (``steps``, optional nested definitions).
        warn_on_duplicates: Whether to log a warning for each skipped duplicate. The
            compiler collects definitions twice per cold compile (once before
            normalisation, once after); the pre-resolution call passes ``False`` so
            the warning is logged only once.

    Returns:
        Merged list of dynamic block definition dicts in discovery order.
    """
    collected: List[Any] = []
    # Locations are built only from fixed field labels and integer indexes, so they
    # are safe to log; block types are request-provided and must never be logged.
    first_location_by_block_type: Dict[str, str] = {}

    def append_definition(definition: Any, *, location: str) -> None:
        block_type = None
        if isinstance(definition, dict):
            block_type = _dynamic_block_type(definition)

        if block_type is not None:
            first_location = first_location_by_block_type.get(block_type)
            if first_location is not None:
                if warn_on_duplicates:
                    logger.warning(
                        "Skipping duplicate dynamic block definition at %s; keeping %s.",
                        location,
                        first_location,
                    )
                return

            first_location_by_block_type[block_type] = location

        collected.append(definition)

    def append_level(definitions: Any, *, location: str) -> None:
        if not definitions:
            return

        if not isinstance(definitions, list):
            append_definition(definitions, location=location)
            return

        for index, definition in enumerate(definitions):
            append_definition(definition, location=f"{location}[{index}]")

    def visit(workflow: Dict[str, Any], *, location_prefix: str) -> None:
        append_level(
            workflow.get("dynamic_blocks_definitions"),
            location=f"{location_prefix}dynamic_blocks_definitions",
        )

        for step_index, step in enumerate(workflow.get("steps") or []):
            if not isinstance(step, dict):
                continue

            if step.get("type") != USE_INNER_WORKFLOW_BLOCK_TYPE:
                continue
            if (
                step.get("execution_mode")
                == INNER_WORKFLOW_EXECUTION_MODE_REMOTE_DISPATCH
            ):
                continue

            child = step.get("workflow_definition")
            if isinstance(child, dict):
                visit(
                    child,
                    location_prefix=(
                        f"{location_prefix}steps[{step_index}].workflow_definition."
                    ),
                )

    visit(workflow_definition, location_prefix="")

    return collected


def apply_collected_dynamic_blocks_definitions_to_workflow_root(
    workflow_definition: Dict[str, Any],
) -> List[Any]:
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
