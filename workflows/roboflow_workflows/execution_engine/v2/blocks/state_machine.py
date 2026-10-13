"""Deprecated import path for ``roboflow_workflows.v2.blocks.state.machines``.

Kept so existing V2 alpha callers keep working; it re-exports the same
objects. Import from the canonical module instead.
"""

from roboflow_workflows.v2.blocks.state.machines import (
    StateMachineSetBlock,
)

__all__ = [
    "StateMachineSetBlock",
]
