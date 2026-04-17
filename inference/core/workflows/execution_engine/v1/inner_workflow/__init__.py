"""
Inner workflow support (composition validation and compile-time inlining).

Design: docs/workflows/inner_workflow_design.md
"""

from inference.core.workflows.execution_engine.v1.inner_workflow.composition import (
    assert_composition_acyclic,
    build_composition_digraph,
    find_composition_cycles,
    max_nesting_depth_from_root,
    validate_inner_workflow_composition,
)
from inference.core.workflows.execution_engine.v1.inner_workflow.errors import (
    InnerWorkflowCompositionCycleError,
    InnerWorkflowCompositionError,
    InnerWorkflowNestingDepthError,
)

__all__ = [
    "InnerWorkflowCompositionCycleError",
    "InnerWorkflowCompositionError",
    "InnerWorkflowNestingDepthError",
    "assert_composition_acyclic",
    "build_composition_digraph",
    "find_composition_cycles",
    "max_nesting_depth_from_root",
    "validate_inner_workflow_composition",
]
