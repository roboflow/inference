"""Shared helpers for the block workload-declaration tests.

Blocks author restrictions as ``RuntimeRestriction`` and publish them through
the public ``get_actual_restrictions()`` hook; the workload document carries
the projected ``RestrictionMetadata`` DTO. Most declaration tests are about the
PORTABLE semantics (code, severity, condition), so they go through
``portable_restrictions()``: it exercises the real public hook and hands back
exactly what the wire would carry.

``ignore_environment_restrictions=True`` on purpose - a declaration test asks
what the block declares, never what the machine running the test is configured
for.
"""

from typing import List

from roboflow_workflows.execution_engine.entities.workload import (
    Discovery,
    RestrictionMetadata,
    RuntimeRestriction,
    restriction_metadata_of,
)
from roboflow_workflows.prototypes.block import WorkflowBlockManifest


def declared_restrictions(
    manifest: WorkflowBlockManifest,
) -> List[RuntimeRestriction]:
    """The restrictions a step declares, as the AUTHORED entity.

    Use this when the expectation is a shared preset (identity of the authored
    object, notes included); use ``portable_restrictions()`` when the
    expectation is about the wire semantics (code, severity, condition).
    """
    return list(
        manifest.get_actual_restrictions(ignore_environment_restrictions=True).items
    )


def portable_restrictions_discovery(
    manifest: WorkflowBlockManifest,
) -> Discovery[RestrictionMetadata]:
    """The portable view of a step's restrictions, completeness included."""
    declared = manifest.get_actual_restrictions(ignore_environment_restrictions=True)
    return Discovery[RestrictionMetadata](
        items=[restriction_metadata_of(item) for item in declared.items],
        complete=declared.complete,
        unknown_reasons=list(declared.unknown_reasons),
    )


def portable_restrictions(
    manifest: WorkflowBlockManifest,
) -> List[RestrictionMetadata]:
    """Just the declared items, as the wire DTO."""
    return list(portable_restrictions_discovery(manifest=manifest).items)


def restriction_codes(manifest: WorkflowBlockManifest) -> List[str]:
    """The codes a step declares, in the canonical discovery order."""
    return [item.code for item in portable_restrictions(manifest=manifest)]
