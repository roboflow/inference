"""Deprecated import path for ``roboflow_workflows.v2.blocks.control``.

Kept so existing V2 alpha callers keep working; it re-exports the same
objects. Import from the canonical module instead.
"""

from roboflow_workflows.v2.blocks.control import (
    ContinueIfBlock,
)

__all__ = [
    "ContinueIfBlock",
]
