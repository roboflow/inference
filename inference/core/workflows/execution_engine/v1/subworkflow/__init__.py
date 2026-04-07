"""
Sub-workflow support (composition validation and execution strategy hooks).

Design: docs/workflows/subworkflow_design.md
"""

from inference.core.workflows.execution_engine.v1.subworkflow.composition import (
    assert_composition_acyclic,
    build_composition_digraph,
    find_composition_cycles,
    max_nesting_depth_from_root,
    validate_subworkflow_composition,
)
from inference.core.workflows.execution_engine.v1.subworkflow.errors import (
    SubworkflowCompositionCycleError,
    SubworkflowCompositionError,
    SubworkflowNestingDepthError,
)
from inference.core.workflows.execution_engine.v1.subworkflow.runner import (
    LocalSubworkflowRunner,
    SubworkflowExecutionMode,
    SubworkflowRunner,
)

__all__ = [
    "LocalSubworkflowRunner",
    "SubworkflowCompositionCycleError",
    "SubworkflowCompositionError",
    "SubworkflowExecutionMode",
    "SubworkflowNestingDepthError",
    "SubworkflowRunner",
    "assert_composition_acyclic",
    "build_composition_digraph",
    "find_composition_cycles",
    "max_nesting_depth_from_root",
    "validate_subworkflow_composition",
]
