"""Backwards-compatibility re-exports.

``StepExecutionMode`` used to live in this module. It has since moved to
``inference.core.workflows.prototypes.block`` so the framework layer
(``prototypes``) owns the enum and higher-level packages (``core_steps``,
executor, compiler) can depend on ``prototypes`` instead of the other way
around. This module is kept purely as a re-export shim so existing imports
``from inference.core.workflows.core_steps.common.entities import
StepExecutionMode`` keep working.
"""

from inference.core.workflows.prototypes.block import StepExecutionMode

__all__ = ["StepExecutionMode"]
